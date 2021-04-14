import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from models import bottle
from utils import imagine_ahead, lambda_return, FreezeParameters

import slots.interfaces as itf


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
        self.transition = transition
        self.observation = observation
        self.reward = reward
        self.optimizer = optimizer
        self.args = args  # later this should be changed to model_self.args

    def main(self, observations, actions, rewards, nonterminals, free_nats, global_prior, param_list):

        encoder = self.encoder
        transition_model = self.transition
        observation_model = self.observation
        reward_model = self.reward
        model_optimizer = self.optimizer


        # Create initial belief and state for time t = 0
        init_belief, init_state = transition_model.initial_step(observation=observations[0], device=self.args.device)

        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        beliefs, prior_states, prior, posterior_states, posterior = transition_model.filter(
                prev_state=init_state, 
                actions=actions[:-1], 
                prev_belief=init_belief, 
                observations=bottle(encoder, (observations[1:], )), 
                nonterminals=nonterminals[:-1])

        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        if self.args.worldmodel_LogProbLoss:
            observation_dist = Normal(bottle(observation_model, (beliefs, posterior_states)), 1)
            observation_loss = -observation_dist.log_prob(observations[1:]).sum(dim=2 if self.args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
        else: 
            observation_loss = F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum(dim=2 if self.args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))

        if self.args.worldmodel_LogProbLoss:
            reward_dist = Normal(bottle(reward_model, (beliefs, posterior_states)),1)
            reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
        else:
            reward_loss = F.mse_loss(bottle(reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0,1))

        # transition loss
        div = kl_divergence(posterior, prior).sum(dim=2)
        kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
        if self.args.global_kl_beta != 0:
            kl_loss += self.args.global_kl_beta * kl_divergence(posterior, global_prior).sum(dim=2).mean(dim=(0, 1))


        # Calculate latent overshooting objective for t > 0
        if self.args.overshooting_kl_beta != 0:
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
            ovsht_beliefs, ovsht_prior_states, ovsht_prior = transition_model.generate(
                prev_state=torch.cat(overshooting_vars.prior_states, dim=0), 
                actions=torch.cat(overshooting_vars.actions, dim=1), 
                prev_belief=torch.cat(overshooting_vars.beliefs, dim=0), 
                observations=None, 
                nonterminals=torch.cat(overshooting_vars.nonterminals, dim=1))


            ### TODO ###
            ovsht_posterior = Normal(torch.cat(overshooting_vars.posterior_means, dim=1), torch.cat(overshooting_vars.posterior_std_devs, dim=1))
            ############

            seq_mask = torch.cat(overshooting_vars.masks, dim=1)
            # Calculate overshooting KL loss with sequence mask
            kl_loss += (1 / self.args.overshooting_distance) * self.args.overshooting_kl_beta * torch.max((kl_divergence(ovsht_posterior, ovsht_prior) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (self.args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence) 

            # Calculate overshooting reward prediction loss with sequence mask
            if self.args.overshooting_reward_scale != 0: 
                reward_loss += (1 / self.args.overshooting_distance) * self.args.overshooting_reward_scale * F.mse_loss(bottle(reward_model, (ovsht_beliefs, ovsht_prior_states)) * seq_mask[:, :, 0], torch.cat(overshooting_vars.rewards, dim=1), reduction='none').mean(dim=(0, 1)) * (self.args.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence) 

        # Apply linearly ramping learning rate schedule
        if self.args.learning_rate_schedule != 0:
            for group in model_optimizer.param_groups:
                group['lr'] = min(group['lr'] + self.args.model_learning_rate / self.args.learning_rate_schedule, self.args.model_learning_rate)
        model_loss = observation_loss + reward_loss + kl_loss
        # Update model parameters
        model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(param_list, self.args.grad_clip_norm, norm_type=2)
        model_optimizer.step()

        return beliefs, posterior_states, observation_loss, reward_loss, kl_loss

    def generate_trace(self):
        pass

    def compute_feedback(self):
        pass

    def update(self):
        pass

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
        return returns

    def compute_feedback_critic(self, imged_beliefs, imged_prior_states, returns):
        #Dreamer implementation: value loss calculation and optimization
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target_return = returns.detach()
        value_dist = Normal(bottle(self.critic, (value_beliefs, value_prior_states)),1) # detach the input tensor from the transition network.
        return value_dist, target_return


    def update_actor(self, returns):
        actor_loss = -torch.mean(returns)
        # Update model parameters
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip_norm, norm_type=2)
        self.actor_optimizer.step()
        return actor_loss

    def update_critic(self, value_dist, target_return):
        value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1)) 
        # Update model parameters
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_clip_norm, norm_type=2)
        self.critic_optimizer.step()
        return value_loss

    def main(self, beliefs, posterior_states, model_modules, transition_model, reward_model):
        imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = self.generate_trace(beliefs, posterior_states, model_modules, transition_model)
        returns = self.compute_feedback_actor(imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs, model_modules, reward_model)
        value_dist, target_return = self.compute_feedback_critic(imged_beliefs, imged_prior_states, returns)
        actor_loss = self.update_actor(returns)
        value_loss = self.update_critic(value_dist, target_return)
        return actor_loss, value_loss
















