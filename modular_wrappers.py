import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from models import bottle
from utils import imagine_ahead, lambda_return, FreezeParameters

import slots.debugging_utils as du
import slots.interfaces as itf
import slots.training_utils as tu
import slots.latent_variable_model as lvm


# class MonolithicTransitionWrapper(nn.Module):
#     pass

# class MonolithicObservationWrapper(nn.Module):
#     pass

# class MonolithicRewardWrapper(nn.Module):
#     pass

class MonolithicModelWrapper(nn.Module):
    def __init__(self, encoder, transition, observation, reward, optimizer, args):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.transition_model = transition
        self.observation_model = observation
        self.reward = reward
        self.optimizer = optimizer
        self.args = args  # later this should be changed to model_self.args

    def main(self, observations, actions, rewards, nonterminals, free_nats, global_prior, param_list):

        beliefs, prior_states, prior, posterior_states, posterior = self.generate_trace(observations, actions, nonterminals)

        # print(beliefs.shape)
        # print(posterior_states.shape)
        # assert False

        observation_loss, reward_loss, kl_loss = self.compute_feedback(observations, rewards, free_nats, global_prior, beliefs, prior, posterior_states, posterior)


        # Calculate latent overshooting objective for t > 0
        if not self.args.lvm_only:
            if self.args.overshooting_kl_beta != 0:

                overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior = self.generate_overshooting_trace(actions, nonterminals, rewards, beliefs, prior_states, posterior)

                reward_loss, kl_loss = self.compute_overshooting_feedback(overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior, free_nats, reward_loss, kl_loss)

        # Apply linearly ramping learning rate schedule
        if self.args.learning_rate_schedule != 0:
            for group in self.optimizer.param_groups:
                group['lr'] = min(group['lr'] + self.args.model_learning_rate / self.args.learning_rate_schedule, self.args.model_learning_rate)

        model_loss = observation_loss + reward_loss + kl_loss

        self.update(model_loss, param_list)

        return beliefs, posterior_states, observation_loss, reward_loss, kl_loss

    def generate_trace(self, observations, actions, nonterminals):
        # Create initial belief and state for time t = 0
        init_belief, init_state = self.transition_model.initial_step(observation=observations[0], device=self.args.device)

        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        beliefs, prior_states, prior, posterior_states, posterior = self.transition_model.filter(
                prev_state=init_state, 
                actions=actions[:-1], 
                prev_belief=init_belief, 
                observations=bottle(self.encoder, (observations[1:], )), 
                nonterminals=nonterminals[:-1])

        return beliefs, prior_states, prior, posterior_states, posterior

    def generate_overshooting_trace(self, actions, nonterminals, rewards, beliefs, prior_states, posterior):
        overshooting_vars = []  # Collect variables for overshooting to process in batch
        for t in range(1, self.args.chunk_size - 1):
            d = min(t + self.args.overshooting_distance, self.args.chunk_size - 1)  # Overshooting distance
            t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
            seq_pad = (0, 0, 0, 0, 0, t - d + self.args.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
            # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
            overshooting_vars.append(
                itf.Overshooting(
                    actions=F.pad(actions[t:d], seq_pad), 
                    nonterminals=F.pad(nonterminals[t:d], seq_pad), 
                    rewards=F.pad(rewards[t:d], seq_pad[2:]), 
                    beliefs=beliefs[t_], 
                    prior_states=prior_states[t_], 
                    posterior_means=F.pad(posterior.loc[t_ + 1:d_ + 1].detach(), seq_pad), 
                    posterior_std_devs=F.pad(posterior.scale[t_ + 1:d_ + 1].detach(), seq_pad, value=1), 
                    masks=F.pad(torch.ones(d - t, self.args.batch_size, self.args.state_size, device=self.args.device), seq_pad)
                ))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences

        overshooting_vars = itf.Overshooting(*zip(*overshooting_vars))
        # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
        # just added ovsht_ as a prefix
        ovsht_beliefs, ovsht_prior_states, ovsht_prior = self.transition_model.generate(
            prev_state=torch.cat(overshooting_vars.prior_states, dim=0), 
            actions=torch.cat(overshooting_vars.actions, dim=1), 
            prev_belief=torch.cat(overshooting_vars.beliefs, dim=0), 
            observations=None, 
            nonterminals=torch.cat(overshooting_vars.nonterminals, dim=1))

        ### TODO ###
        ovsht_posterior = Normal(torch.cat(overshooting_vars.posterior_means, dim=1), torch.cat(overshooting_vars.posterior_std_devs, dim=1))
        ############

        return overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior




    def compute_feedback(self, observations, rewards, free_nats, global_prior, beliefs, prior, posterior_states, posterior):
        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        if self.args.worldmodel_LogProbLoss:
            observation_dist = Normal(bottle(self.observation_model, (beliefs, posterior_states)), 1)
            observation_loss = -observation_dist.log_prob(observations[1:]).sum(dim=2 if self.args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
        else: 
            observation_loss = F.mse_loss(bottle(self.observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum(dim=2 if self.args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))

        if self.args.worldmodel_LogProbLoss:
            reward_dist = Normal(bottle(self.reward, (beliefs, posterior_states)),1)
            reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
        else:
            reward_loss = F.mse_loss(bottle(self.reward, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0,1))

        # transition loss
        div = kl_divergence(posterior, prior).sum(dim=2)
        kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
        if self.args.global_kl_beta != 0:
            kl_loss += self.args.global_kl_beta * kl_divergence(posterior, global_prior).sum(dim=2).mean(dim=(0, 1))


        if self.args.lvm_only:
            reward_loss *= 0

        return observation_loss, reward_loss, kl_loss

    def compute_overshooting_feedback(self, overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior, free_nats, reward_loss, kl_loss):
        seq_mask = torch.cat(overshooting_vars.masks, dim=1)
        # Calculate overshooting KL loss with sequence mask
        kl_loss += (1 / self.args.overshooting_distance) * self.args.overshooting_kl_beta * torch.max((kl_divergence(ovsht_posterior, ovsht_prior) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (self.args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence) 

        # Calculate overshooting reward prediction loss with sequence mask
        if self.args.overshooting_reward_scale != 0: 
            reward_loss += (1 / self.args.overshooting_distance) * self.args.overshooting_reward_scale * F.mse_loss(bottle(self.reward, (ovsht_beliefs, ovsht_prior_states)) * seq_mask[:, :, 0], torch.cat(overshooting_vars.rewards, dim=1), reduction='none').mean(dim=(0, 1)) * (self.args.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence) 

        return reward_loss, kl_loss




    def update(self, model_loss, param_list):
        # Update model parameters
        self.optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(param_list, self.args.grad_clip_norm, norm_type=2)
        self.optimizer.step()



# class SlotsModelWrapper(nn.Module):
class SlotsModelWrapper(lvm.RSSMLVM):
    def __init__(self, encoder, transition, observation, reward, optimizer, args):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.transition_model = transition
        self.observation_model = observation

        self.reward = reward

        # self.optimizer = optimizer

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)#args.lr)  # just replace it with its own optimizer for now. Might need to reconcile with the previous implementation though because it seems like they do something with freezing parameters. 

        self.args = args  # later this should be changed to model_self.args

        self.optim_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_gamma)
        self.dkl_scheduler = tu.GeometricScheduler(initial_value=args.kl_coeff, final_value=args.dkl_coeff, max_steps=args.dkl_steps)
        self.i = 0


    def get_beliefs_and_states(self, posteriors):
        # beliefs = torch.stack([prior.belief for prior in priors])  # or what about prior?
        beliefs = torch.stack([posterior.belief for posterior in posteriors])
        posterior_states = torch.stack([posterior.sample for posterior in posteriors])
        return beliefs, posterior_states

    def main(self, observations, actions, rewards, nonterminals, free_nats, global_prior, param_list):

        # beliefs, prior_states, prior, posterior_states, posterior = self.generate_trace(observations, actions, nonterminals)

        batch = itf.InteractiveBatchData(obs=observations, act=actions)

        priors, posteriors = self.transition_model.forward(batch)

        beliefs, posterior_states = self.get_beliefs_and_states(posteriors)
        preds = self.predict(priors, posteriors, observations.shape)

        observation_loss, reward_loss, kl_loss = self.compute_feedback(observations, rewards, free_nats, global_prior, beliefs, priors, posterior_states, posteriors, preds)

        # Calculate latent overshooting objective for t > 0
        if not self.args.lvm_only:
            if self.args.overshooting_kl_beta != 0:

                overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior = self.generate_overshooting_trace(actions, nonterminals, rewards, beliefs, prior_states, posterior)

                reward_loss, kl_loss = self.compute_overshooting_feedback(overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior, free_nats, reward_loss, kl_loss)

        # # Apply linearly ramping learning rate schedule
        # if self.args.learning_rate_schedule != 0:
        #     for group in self.optimizer.param_groups:
        #         group['lr'] = min(group['lr'] + self.args.model_learning_rate / self.args.learning_rate_schedule, self.args.model_learning_rate)

        model_loss = observation_loss + reward_loss + kl_loss

        self.update(model_loss, param_list)

        return beliefs, posterior_states, observation_loss, reward_loss, kl_loss

    def generate_trace(self, observations, actions, nonterminals):
        # Create initial belief and state for time t = 0
        init_belief, init_state = self.transition_model.initial_step(observation=observations[0], device=self.args.device)

        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        beliefs, prior_states, prior, posterior_states, posterior = self.transition_model.filter(
                prev_state=init_state, 
                actions=actions[:-1], 
                prev_belief=init_belief, 
                observations=bottle(self.encoder, (observations[1:], )), 
                nonterminals=nonterminals[:-1])

        return beliefs, prior_states, prior, posterior_states, posterior

    def generate_overshooting_trace(self, actions, nonterminals, rewards, beliefs, prior_states, posterior):
        overshooting_vars = []  # Collect variables for overshooting to process in batch
        for t in range(1, self.args.chunk_size - 1):
            d = min(t + self.args.overshooting_distance, self.args.chunk_size - 1)  # Overshooting distance
            t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
            seq_pad = (0, 0, 0, 0, 0, t - d + self.args.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
            # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
            overshooting_vars.append(
                itf.Overshooting(
                    actions=F.pad(actions[t:d], seq_pad), 
                    nonterminals=F.pad(nonterminals[t:d], seq_pad), 
                    rewards=F.pad(rewards[t:d], seq_pad[2:]), 
                    beliefs=beliefs[t_], 
                    prior_states=prior_states[t_], 
                    posterior_means=F.pad(posterior.loc[t_ + 1:d_ + 1].detach(), seq_pad), 
                    posterior_std_devs=F.pad(posterior.scale[t_ + 1:d_ + 1].detach(), seq_pad, value=1), 
                    masks=F.pad(torch.ones(d - t, self.args.batch_size, self.args.state_size, device=self.args.device), seq_pad)
                ))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences

        overshooting_vars = itf.Overshooting(*zip(*overshooting_vars))
        # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
        # just added ovsht_ as a prefix
        ovsht_beliefs, ovsht_prior_states, ovsht_prior = self.transition_model.generate(
            prev_state=torch.cat(overshooting_vars.prior_states, dim=0), 
            actions=torch.cat(overshooting_vars.actions, dim=1), 
            prev_belief=torch.cat(overshooting_vars.beliefs, dim=0), 
            observations=None, 
            nonterminals=torch.cat(overshooting_vars.nonterminals, dim=1))

        ### TODO ###
        ovsht_posterior = Normal(torch.cat(overshooting_vars.posterior_means, dim=1), torch.cat(overshooting_vars.posterior_std_devs, dim=1))
        ############

        return overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior




    #############################################################################################


    # later will get rid of the batch argument
    def predict(self, priors, posteriors, obs_shape):
        # t, b, c, h, w = batch.obs.shape
        t, b, c, h, w = obs_shape
        k = self.transition_model.recognition_model.slot_attn.num_slots

        preds = itf.DSAPred(
            pbd=torch.empty(t, b, c, h, w).to(self.args.device),
            pad=torch.empty(t-1, b, c, h, w).to(self.args.device),
            cbd=torch.empty(t, b, k, c, h, w).to(self.args.device),
            cad=torch.empty(t-1, b, k, c, h, w).to(self.args.device))

        for step in range(len(priors)):
            preds.cbd[step], preds.pbd[step] = self.decode(posteriors[step])
            if step < t - 1:
                preds.cad[step], preds.pad[step] = self.decode(priors[step+1])

        # maybe we should actually be computing the losses here too. 
        return preds

    def decode(self, rssm_state):
        return self.observation_model.decode(lvm.get_obs_input(rssm_state, self.args.grndrate))

    def compute_filtering_feedback(self, preds, priors, posteriors, obs, args):
        t, b, c, h, w = obs.shape
        k = self.transition_model.recognition_model.slot_attn.num_slots

        kl = torch.empty(t, b, k).to(self.args.device)
        for step in range(len(priors)):
            # kl[step] = self.kl(posteriors[step].dist, priors[step].dist)
            kl[step] = lvm.averaged_kl(posteriors[step].dist, priors[step].dist)

        initial_kl = kl[0].mean()
        dynamic_kl = kl[1:].mean()

        loss_bd = F.mse_loss(preds.pbd, obs)
        loss_ad = F.mse_loss(preds.pad, obs[1:])
        dkl_coeff = self.dkl_scheduler.get_value()

        observation_loss = loss_bd + loss_ad
        kl_loss = args.kl_coeff*initial_kl + dkl_coeff*dynamic_kl**args.dkl_pwr


        # loss = loss_bd + loss_ad + args.kl_coeff*initial_kl + dkl_coeff*dynamic_kl**args.dkl_pwr # note that there are now twice as many recon loss terms as kl, so maybe kl_coeff needs to be double what it usually is in static case?

        if self.args.observation_consistency:
            loss_consistency = F.mse_loss(preds.cad, preds.cbd[1:])
            observation_loss += loss_consistency


        loss = observation_loss + kl_loss


        loss_trace = itf.LossTrace(
            loss=loss,
            kl=kl,
            initial_kl=initial_kl, 
            dynamic_kl=dynamic_kl, 
            loss_bd=loss_bd, 
            loss_ad=loss_ad,
            loss_consistency=loss_consistency)
        # return loss, loss_trace, dkl_coeff
        return observation_loss, kl_loss, loss_trace, dkl_coeff





    #############################################################################################



    def compute_feedback(self, observations, rewards, free_nats, global_prior, beliefs, prior, posterior_states, posterior, preds):
        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        if self.args.worldmodel_LogProbLoss:

            raise NotImplementedError

            observation_dist = Normal(bottle(self.observation_model, (beliefs, posterior_states)), 1)

            observation_loss = -observation_dist.log_prob(observations[1:]).sum(dim=2 if self.args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
        else: 
            # preds = self.predict(prior, posterior, observations.shape)
            # loss, loss_trace, dkl_coeff = self.compute_filtering_feedback(preds, prior, posterior, observations, self.args)


            observation_loss, kl_loss, loss_trace, dkl_coeff = self.compute_filtering_feedback(preds, prior, posterior, observations, self.args)

            log_string = 'Batch: {}\n\tLoss:\t{}\n\tLoss Before Dynamics:\t{}\n\tLoss After Dynamics:\t{}\n\tKL Initial:\t{}\n\tKL Dynamic:\t{}\n\tPrevious DKL:\t{}\n\tCurrent DKL:\t{}'.format(self.i, loss_trace.loss.item(), loss_trace.loss_bd.item(), loss_trace.loss_ad.item(), loss_trace.initial_kl.item(), loss_trace.dynamic_kl.item(), dkl_coeff, self.dkl_scheduler.get_value())
            if self.args.observation_consistency:
                log_string += '\n\tObservation Consistency:\t{}'.format(loss_trace.loss_consistency.item())
            # print(log_string)

        if self.args.lvm_only:
            # reward_loss = torch.zeros([1]).to(observations.device)

            if self.args.detach_latents_for_reward:
                reward_loss = F.mse_loss(bottle(self.reward, 
                    (beliefs[:-1], posterior_states[:-1])), 
                    # (beliefs[:-1].detach(), posterior_states[:-1].detach())), 
                    rewards[:-1], reduction='none').mean(dim=(0,1))  # NOTE THIS IS DIFFERENT FROM MONOLITHIC BECAUSE WE ARE TAKING [:-1] fomr beliefs and posterior_states!

            else:
                reward_loss = F.mse_loss(bottle(self.reward, 
                    # (beliefs[:-1], posterior_states[:-1])), 
                    (beliefs[:-1].detach(), posterior_states[:-1].detach())), 
                    rewards[:-1], reduction='none').mean(dim=(0,1))  # NOTE THIS IS DIFFERENT FROM MONOLITHIC BECAUSE WE ARE TAKING [:-1] fomr beliefs and posterior_states!





            log_string += '\n\tReward Loss:\t{}'.format(reward_loss.item())
            print(log_string)

        else:

            if self.args.worldmodel_LogProbLoss:
                reward_dist = Normal(bottle(self.reward, (beliefs, posterior_states)),1)
                reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
            else:
                reward_loss = F.mse_loss(bottle(self.reward, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0,1))

            # transition loss
            div = kl_divergence(posterior, prior).sum(dim=2)
            kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
            if self.args.global_kl_beta != 0:
                kl_loss += self.args.global_kl_beta * kl_divergence(posterior, global_prior).sum(dim=2).mean(dim=(0, 1))






        return observation_loss, reward_loss, kl_loss

    def compute_overshooting_feedback(self, overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior, free_nats, reward_loss, kl_loss):
        seq_mask = torch.cat(overshooting_vars.masks, dim=1)
        # Calculate overshooting KL loss with sequence mask
        kl_loss += (1 / self.args.overshooting_distance) * self.args.overshooting_kl_beta * torch.max((kl_divergence(ovsht_posterior, ovsht_prior) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (self.args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence) 

        # Calculate overshooting reward prediction loss with sequence mask
        if self.args.overshooting_reward_scale != 0: 
            reward_loss += (1 / self.args.overshooting_distance) * self.args.overshooting_reward_scale * F.mse_loss(bottle(self.reward, (ovsht_beliefs, ovsht_prior_states)) * seq_mask[:, :, 0], torch.cat(overshooting_vars.rewards, dim=1), reduction='none').mean(dim=(0, 1)) * (self.args.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence) 

        return reward_loss, kl_loss




    # def update(self, model_loss, param_list):
    #     # Update model parameters
    #     self.optimizer.zero_grad()
    #     model_loss.backward()
    #     nn.utils.clip_grad_norm_(param_list, self.args.grad_clip_norm, norm_type=2)
    #     self.optimizer.step()

    def update(self, model_loss, param_list):
        self.optimizer.zero_grad()
        model_loss.backward()
        self.optimizer.step()
        self.dkl_scheduler.step()

        if self.i >= self.args.lr_decay_after:
            before_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.optim_scheduler.step()
            after_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            if before_lr != after_lr:
                print('Batch: {}\tLR Previously: {}\tLR Now: {}'.format(self.i, before_lr, after_lr))
        self.i += 1








# class SlotsModelWrapper(nn.Module):
#     def __init__(self, encoder, transition, observation, reward, optimizer, args):
#         nn.Module.__init__(self)
#         self.encoder = encoder
#         self.transition_model = transition
#         self.observation_model = observation

#         self.reward = reward

#         # self.optimizer = optimizer

#         self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)#args.lr)  # just replace it with its own optimizer for now. Might need to reconcile with the previous implementation though because it seems like they do something with freezing parameters. 

#         self.args = args  # later this should be changed to model_self.args

#         self.optim_scheduler = torch.optim.lr_scheduler.StepLR(
#             self.optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_gamma)
#         self.dkl_scheduler = tu.GeometricScheduler(initial_value=args.kl_coeff, final_value=args.dkl_coeff, max_steps=args.dkl_steps)
#         self.i = 0


#     def get_beliefs_and_states(self, posteriors):
#         # beliefs = torch.stack([prior.belief for prior in priors])  # or what about prior?
#         beliefs = torch.stack([posterior.belief for posterior in posteriors])
#         posterior_states = torch.stack([posterior.sample for posterior in posteriors])
#         return beliefs, posterior_states

#     def main(self, observations, actions, rewards, nonterminals, free_nats, global_prior, param_list):

#         # beliefs, prior_states, prior, posterior_states, posterior = self.generate_trace(observations, actions, nonterminals)

#         batch = itf.InteractiveBatchData(obs=observations, act=actions)

#         priors, posteriors = self.transition_model.forward(batch)

#         beliefs, posterior_states = self.get_beliefs_and_states(posteriors)
#         preds = self.predict(priors, posteriors, observations.shape)

#         observation_loss, reward_loss, kl_loss = self.compute_feedback(observations, rewards, free_nats, global_prior, beliefs, priors, posterior_states, posteriors, preds)

#         # Calculate latent overshooting objective for t > 0
#         if not self.args.lvm_only:
#             if self.args.overshooting_kl_beta != 0:

#                 overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior = self.generate_overshooting_trace(actions, nonterminals, rewards, beliefs, prior_states, posterior)

#                 reward_loss, kl_loss = self.compute_overshooting_feedback(overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior, free_nats, reward_loss, kl_loss)

#         # # Apply linearly ramping learning rate schedule
#         # if self.args.learning_rate_schedule != 0:
#         #     for group in self.optimizer.param_groups:
#         #         group['lr'] = min(group['lr'] + self.args.model_learning_rate / self.args.learning_rate_schedule, self.args.model_learning_rate)

#         model_loss = observation_loss + reward_loss + kl_loss

#         self.update(model_loss, param_list)

#         return beliefs, posterior_states, observation_loss, reward_loss, kl_loss

#     def generate_trace(self, observations, actions, nonterminals):
#         # Create initial belief and state for time t = 0
#         init_belief, init_state = self.transition_model.initial_step(observation=observations[0], device=self.args.device)

#         # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
#         beliefs, prior_states, prior, posterior_states, posterior = self.transition_model.filter(
#                 prev_state=init_state, 
#                 actions=actions[:-1], 
#                 prev_belief=init_belief, 
#                 observations=bottle(self.encoder, (observations[1:], )), 
#                 nonterminals=nonterminals[:-1])

#         return beliefs, prior_states, prior, posterior_states, posterior

#     def generate_overshooting_trace(self, actions, nonterminals, rewards, beliefs, prior_states, posterior):
#         overshooting_vars = []  # Collect variables for overshooting to process in batch
#         for t in range(1, self.args.chunk_size - 1):
#             d = min(t + self.args.overshooting_distance, self.args.chunk_size - 1)  # Overshooting distance
#             t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
#             seq_pad = (0, 0, 0, 0, 0, t - d + self.args.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
#             # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
#             overshooting_vars.append(
#                 itf.Overshooting(
#                     actions=F.pad(actions[t:d], seq_pad), 
#                     nonterminals=F.pad(nonterminals[t:d], seq_pad), 
#                     rewards=F.pad(rewards[t:d], seq_pad[2:]), 
#                     beliefs=beliefs[t_], 
#                     prior_states=prior_states[t_], 
#                     posterior_means=F.pad(posterior.loc[t_ + 1:d_ + 1].detach(), seq_pad), 
#                     posterior_std_devs=F.pad(posterior.scale[t_ + 1:d_ + 1].detach(), seq_pad, value=1), 
#                     masks=F.pad(torch.ones(d - t, self.args.batch_size, self.args.state_size, device=self.args.device), seq_pad)
#                 ))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences

#         overshooting_vars = itf.Overshooting(*zip(*overshooting_vars))
#         # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
#         # just added ovsht_ as a prefix
#         ovsht_beliefs, ovsht_prior_states, ovsht_prior = self.transition_model.generate(
#             prev_state=torch.cat(overshooting_vars.prior_states, dim=0), 
#             actions=torch.cat(overshooting_vars.actions, dim=1), 
#             prev_belief=torch.cat(overshooting_vars.beliefs, dim=0), 
#             observations=None, 
#             nonterminals=torch.cat(overshooting_vars.nonterminals, dim=1))

#         ### TODO ###
#         ovsht_posterior = Normal(torch.cat(overshooting_vars.posterior_means, dim=1), torch.cat(overshooting_vars.posterior_std_devs, dim=1))
#         ############

#         return overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior




#     #############################################################################################


#     # later will get rid of the batch argument
#     def predict(self, priors, posteriors, obs_shape):
#         # t, b, c, h, w = batch.obs.shape
#         t, b, c, h, w = obs_shape
#         k = self.transition_model.recognition_model.slot_attn.num_slots

#         preds = itf.DSAPred(
#             pbd=torch.empty(t, b, c, h, w).to(self.args.device),
#             pad=torch.empty(t-1, b, c, h, w).to(self.args.device),
#             cbd=torch.empty(t, b, k, c, h, w).to(self.args.device),
#             cad=torch.empty(t-1, b, k, c, h, w).to(self.args.device))

#         for step in range(len(priors)):
#             preds.cbd[step], preds.pbd[step] = self.decode(posteriors[step])
#             if step < t - 1:
#                 preds.cad[step], preds.pad[step] = self.decode(priors[step+1])

#         # maybe we should actually be computing the losses here too. 
#         return preds

#     def decode(self, rssm_state):
#         return self.observation_model.decode(lvm.get_obs_input(rssm_state, self.args.grndrate))

#     def compute_filtering_feedback(self, preds, priors, posteriors, obs, args):
#         t, b, c, h, w = obs.shape
#         k = self.transition_model.recognition_model.slot_attn.num_slots

#         kl = torch.empty(t, b, k).to(self.args.device)
#         for step in range(len(priors)):
#             # kl[step] = self.kl(posteriors[step].dist, priors[step].dist)
#             kl[step] = lvm.averaged_kl(posteriors[step].dist, priors[step].dist)

#         initial_kl = kl[0].mean()
#         dynamic_kl = kl[1:].mean()

#         loss_bd = F.mse_loss(preds.pbd, obs)
#         loss_ad = F.mse_loss(preds.pad, obs[1:])
#         dkl_coeff = self.dkl_scheduler.get_value()

#         observation_loss = loss_bd + loss_ad
#         kl_loss = args.kl_coeff*initial_kl + dkl_coeff*dynamic_kl**args.dkl_pwr


#         # loss = loss_bd + loss_ad + args.kl_coeff*initial_kl + dkl_coeff*dynamic_kl**args.dkl_pwr # note that there are now twice as many recon loss terms as kl, so maybe kl_coeff needs to be double what it usually is in static case?

#         if self.args.observation_consistency:
#             loss_consistency = F.mse_loss(preds.cad, preds.cbd[1:])
#             observation_loss += loss_consistency


#         loss = observation_loss + kl_loss


#         loss_trace = itf.LossTrace(
#             loss=loss,
#             kl=kl,
#             initial_kl=initial_kl, 
#             dynamic_kl=dynamic_kl, 
#             loss_bd=loss_bd, 
#             loss_ad=loss_ad,
#             loss_consistency=loss_consistency)
#         # return loss, loss_trace, dkl_coeff
#         return observation_loss, kl_loss, loss_trace, dkl_coeff





#     #############################################################################################



#     def compute_feedback(self, observations, rewards, free_nats, global_prior, beliefs, prior, posterior_states, posterior, preds):
#         # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
#         if self.args.worldmodel_LogProbLoss:

#             raise NotImplementedError

#             observation_dist = Normal(bottle(self.observation_model, (beliefs, posterior_states)), 1)

#             observation_loss = -observation_dist.log_prob(observations[1:]).sum(dim=2 if self.args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
#         else: 
#             # preds = self.predict(prior, posterior, observations.shape)
#             # loss, loss_trace, dkl_coeff = self.compute_filtering_feedback(preds, prior, posterior, observations, self.args)


#             observation_loss, kl_loss, loss_trace, dkl_coeff = self.compute_filtering_feedback(preds, prior, posterior, observations, self.args)

#             log_string = 'Batch: {}\n\tLoss:\t{}\n\tLoss Before Dynamics:\t{}\n\tLoss After Dynamics:\t{}\n\tKL Initial:\t{}\n\tKL Dynamic:\t{}\n\tPrevious DKL:\t{}\n\tCurrent DKL:\t{}'.format(self.i, loss_trace.loss.item(), loss_trace.loss_bd.item(), loss_trace.loss_ad.item(), loss_trace.initial_kl.item(), loss_trace.dynamic_kl.item(), dkl_coeff, self.dkl_scheduler.get_value())
#             if self.args.observation_consistency:
#                 log_string += '\n\tObservation Consistency:\t{}'.format(loss_trace.loss_consistency.item())
#             # print(log_string)

#         if self.args.lvm_only:
#             # reward_loss = torch.zeros([1]).to(observations.device)

#             if self.args.detach_latents_for_reward:
#                 reward_loss = F.mse_loss(bottle(self.reward, 
#                     (beliefs[:-1], posterior_states[:-1])), 
#                     # (beliefs[:-1].detach(), posterior_states[:-1].detach())), 
#                     rewards[:-1], reduction='none').mean(dim=(0,1))  # NOTE THIS IS DIFFERENT FROM MONOLITHIC BECAUSE WE ARE TAKING [:-1] fomr beliefs and posterior_states!

#             else:
#                 reward_loss = F.mse_loss(bottle(self.reward, 
#                     # (beliefs[:-1], posterior_states[:-1])), 
#                     (beliefs[:-1].detach(), posterior_states[:-1].detach())), 
#                     rewards[:-1], reduction='none').mean(dim=(0,1))  # NOTE THIS IS DIFFERENT FROM MONOLITHIC BECAUSE WE ARE TAKING [:-1] fomr beliefs and posterior_states!





#             log_string += '\n\tReward Loss:\t{}'.format(reward_loss.item())
#             print(log_string)

#         else:

#             if self.args.worldmodel_LogProbLoss:
#                 reward_dist = Normal(bottle(self.reward, (beliefs, posterior_states)),1)
#                 reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
#             else:
#                 reward_loss = F.mse_loss(bottle(self.reward, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0,1))

#             # transition loss
#             div = kl_divergence(posterior, prior).sum(dim=2)
#             kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
#             if self.args.global_kl_beta != 0:
#                 kl_loss += self.args.global_kl_beta * kl_divergence(posterior, global_prior).sum(dim=2).mean(dim=(0, 1))






#         return observation_loss, reward_loss, kl_loss

#     def compute_overshooting_feedback(self, overshooting_vars, ovsht_beliefs, ovsht_prior_states, ovsht_prior, ovsht_posterior, free_nats, reward_loss, kl_loss):
#         seq_mask = torch.cat(overshooting_vars.masks, dim=1)
#         # Calculate overshooting KL loss with sequence mask
#         kl_loss += (1 / self.args.overshooting_distance) * self.args.overshooting_kl_beta * torch.max((kl_divergence(ovsht_posterior, ovsht_prior) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (self.args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence) 

#         # Calculate overshooting reward prediction loss with sequence mask
#         if self.args.overshooting_reward_scale != 0: 
#             reward_loss += (1 / self.args.overshooting_distance) * self.args.overshooting_reward_scale * F.mse_loss(bottle(self.reward, (ovsht_beliefs, ovsht_prior_states)) * seq_mask[:, :, 0], torch.cat(overshooting_vars.rewards, dim=1), reduction='none').mean(dim=(0, 1)) * (self.args.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence) 

#         return reward_loss, kl_loss




#     # def update(self, model_loss, param_list):
#     #     # Update model parameters
#     #     self.optimizer.zero_grad()
#     #     model_loss.backward()
#     #     nn.utils.clip_grad_norm_(param_list, self.args.grad_clip_norm, norm_type=2)
#     #     self.optimizer.step()

#     def update(self, model_loss, param_list):
#         self.optimizer.zero_grad()
#         model_loss.backward()
#         self.optimizer.step()
#         self.dkl_scheduler.step()

#         if self.i >= self.args.lr_decay_after:
#             before_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
#             self.optim_scheduler.step()
#             after_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
#             if before_lr != after_lr:
#                 print('Batch: {}\tLR Previously: {}\tLR Now: {}'.format(self.i, before_lr, after_lr))
#         self.i += 1





# class MonolithicActorWrapper(nn.Module):
#     pass

# class MonolithicValueWrapper(nn.Module):
#     pass


class MonolithicPolicyWrapper(nn.Module):
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer, args):
        nn.Module.__init__(self)
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.args = args  # later this should be changed to model_self.args

    def generate_trace(self, beliefs, posterior_states, model_modules, transition_model):
        #Dreamer implementation: actor loss calculation and optimization    
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()
        with FreezeParameters(model_modules):
            imagination_traj = imagine_ahead(actor_states, actor_beliefs, self.actor, transition_model, self.args.planning_horizon)
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
        return imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs

    def compute_feedback_actor(self, imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs, model_modules, reward_model):
        with FreezeParameters(model_modules + self.critic.modules):
            imged_reward = bottle(reward_model, (imged_beliefs, imged_prior_states))
            value_pred = bottle(self.critic, (imged_beliefs, imged_prior_states))
        returns = lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=self.args.discount, lambda_=self.args.disclam)
        actor_loss = -torch.mean(returns)
        return returns, actor_loss

    def compute_feedback_critic(self, imged_beliefs, imged_prior_states, returns):
        #Dreamer implementation: value loss calculation and optimization
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target_return = returns.detach()
        value_dist = Normal(bottle(self.critic, (value_beliefs, value_prior_states)),1) # detach the input tensor from the transition network.
        value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1)) 
        return value_loss

    def update_actor(self, actor_loss):
        # Update model parameters
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip_norm, norm_type=2)
        self.actor_optimizer.step()

    def update_critic(self, value_loss):
        # Update model parameters
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_clip_norm, norm_type=2)
        self.critic_optimizer.step()

    def main(self, beliefs, posterior_states, model_modules, transition_model, reward_model):
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = self.generate_trace(beliefs, posterior_states, model_modules, transition_model)
        returns, actor_loss = self.compute_feedback_actor(imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs, model_modules, reward_model)
        value_loss = self.compute_feedback_critic(imged_beliefs, imged_prior_states, returns)


        if self.args.lvm_only:
            actor_loss *= 0
            value_loss *= 0

        # LVM DEBUG
        # self.update_actor(actor_loss)
        # self.update_critic(value_loss)
        return actor_loss, value_loss
















