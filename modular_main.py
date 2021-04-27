# -*- coding: UTF-8 -*-

import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher, SIMPLE_ENTITY_ENVS, SimpleEntityEnv
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel
from planner import MPCPlanner
from utils import lineplot, write_video, imagine_ahead, lambda_return, FreezeParameters, ActivateParameters
from tensorboardX import SummaryWriter

import torchvision
import slots.interfaces as itf
import slots.latent_variable_model as lvm
import slots.static_slot_attention_modules as ssa
import slots.debugging_utils as du
import slots.dynamics_models as dm
import slots.policy as pol
import slots.visualize as vis
import modular_wrappers as mw
from entity_memory import EntityExperienceReplay


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
parser.add_argument('--algo', type=str, default='dreamer', help='planet or dreamer')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS + SIMPLE_ENTITY_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--cnn-activation-function', type=str, default='relu', choices=dir(F), help='Model activation function for a convolution layer')
parser.add_argument('--dense-activation-function', type=str, default='elu', choices=dir(F), help='Model activation function a dense layer')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--worldmodel-LogProbLoss', action='store_true', help='use LogProb loss for observation_model and reward_model training')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--model_learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate') 
parser.add_argument('--actor_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate') 
parser.add_argument('--value_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate') 
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value') 
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=25, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=10, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')


parser.add_argument('--slots', action='store_true', help='object-centric')
parser.add_argument('--num_slots', type=int, default=1)
parser.add_argument('--lvm_only', action='store_true', help='turn off loss for reward, actor, value')

args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))


# Setup
results_dir = os.path.join('results', '{}_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
    print("using CUDA")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
    os.environ['MUJOCO_GL'] = 'egl'


    # if 'vdisplay' not in globals():
    #     # start a virtual X display for MAGICAL rendering
    #     import xvfbwrapper
    #     vdisplay = xvfbwrapper.Xvfb()
    #     vdisplay.start()

    # # print(os.environ)


else:
    print("using CPU")
    args.device = torch.device('cpu')
os.environ["SDL_VIDEODRIVER"] = "dummy"
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 
                     'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': []}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))

# Initialise training environment and experience replay memory
print('Initializing environment...')
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
print('Initializing memory...')
if args.experience_replay != '' and os.path.exists(args.experience_replay):
    D = torch.load(args.experience_replay)
    metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
    if args.slots:
        D = EntityExperienceReplay(args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device)
    else:
        D = ExperienceReplay(args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device)
    # Initialise dataset D with S random seed episodes
    for s in tqdm(range(1, args.seed_episodes + 1)):
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            D.append(observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)

# close rendering window
env.close()

# Initialise model parameters randomly
print('Initializing model...')



args.lr_decay_after = 0
args.observation_consistency = True
args.dkl_pwr = 1
args.grndrate = 1
args.lr = 0.001  # doesn't affect the lr.
# args.lr = 0.0005
args.lr_decay_every = int(1e4)
args.lr_decay_gamma = 0.95
args.kl_coeff = 1e-4
args.dkl_coeff = 1e-4
args.dkl_steps = int(5e4)
dynamics_model_builder = lambda state_dim: dm.SlotDynamicsModel(state_dim, 
            action_dim=5, hid_dim=128, interaction_type='pairwise')
# torch.manual_seed(0)
slot_dynamic_autoencoder = lvm.RSSMLVM(
    recognition_model=ssa.RecognitionModel(
        num_slots=5, 
        interface_dim=32, 
        slot_dim=128,
        iters=3,
        slot_temp=1), 
    dynamics_model=dm.RSSM(stoch_dim=128, model=dynamics_model_builder(state_dim=128)),
    observation_model=ssa.ObservationModel(indim=128, num_slots=5, scale=10),
    mode='dynamics',
    device=args.device,
    args=args).to(args.device)













####################################
if args.slots:
    # transition_model = lvm.SlotTransitionModel(
    #     recognition_model=ssa.RecognitionModel(
    #         num_slots=args.num_slots,
    #         interface_dim=args.belief_size//args.num_slots,
    #         slot_dim=args.belief_size//args.num_slots,
    #         iters=3,
    #         slot_temp=0.1
    #         ),
    #     dynamics_model=dm.RSSM(
    #         stoch_dim=args.state_size//args.num_slots,
    #         model=dm.SlotDynamicsModel(
    #             state_dim=args.belief_size//args.num_slots, 
    #             action_dim=env.action_size, 
    #             hid_dim=args.hidden_size//args.num_slots, 
    #             interaction_type='pairwise')),
    #     rssm_head=dm.RSSMHead(
    #         indim=args.belief_size//args.num_slots,
    #         outdim=args.state_size//args.num_slots,
    #         ),
    #     mode='dynamics',
    #     device=args.device
    #     ).to(device=args.device)

    transition_model = slot_dynamic_autoencoder.transition_model


else:
    transition_model = TransitionModel(args.belief_size, args.state_size, env.action_size, args.hidden_size, args.embedding_size, args.dense_activation_function).to(device=args.device)


# du.visualize_parameters(transition_model, print)

# --------------


####################################
if args.slots:
    # observation_model = ssa.ObservationModel(indim=args.belief_size//args.num_slots+args.state_size//args.num_slots, num_slots=args.num_slots, scale=0.1).to(device=args.device)

    # print(observation_model.modules)

    observation_model = slot_dynamic_autoencoder.observation_model





else:
    observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.belief_size, args.state_size, args.embedding_size, args.cnn_activation_function).to(device=args.device)


if args.slots:
    reward_model = pol.EntityRewardModel(args.num_slots, args.belief_size//args.num_slots, args.state_size//args.num_slots, args.hidden_size//args.num_slots, args.dense_activation_function).to(device=args.device)
else:
    reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)

####################################
if args.slots:
    encoder = lvm.IdentityEncoder().to(device=args.device)
else:
    encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.cnn_activation_function).to(device=args.device)
# --------------
# encoder = lvm.IdentityEncoder()
####################################


if args.slots:
    actor_model = pol.EntityActorModel(args.num_slots, args.belief_size//args.num_slots, args.state_size//args.num_slots, args.hidden_size//args.num_slots, env.action_size, args.dense_activation_function).to(device=args.device)
else:
    actor_model = ActorModel(args.belief_size, args.state_size, args.hidden_size, env.action_size, args.dense_activation_function).to(device=args.device)


if args.slots:
    value_model = pol.EntityValueModel(args.num_slots, args.belief_size//args.num_slots, args.state_size//args.num_slots, args.hidden_size//args.num_slots, args.dense_activation_function).to(device=args.device)
else:

    value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)






param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(reward_model.parameters()) + list(encoder.parameters())
value_actor_param_list = list(value_model.parameters()) + list(actor_model.parameters())
params_list = param_list + value_actor_param_list
model_optimizer = optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.model_learning_rate, eps=args.adam_epsilon)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate, eps=args.adam_epsilon)
value_optimizer = optim.Adam(value_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate, eps=args.adam_epsilon)
if args.models != '' and os.path.exists(args.models):
    model_dicts = torch.load(args.models)
    transition_model.load_state_dict(model_dicts['transition_model'])
    observation_model.load_state_dict(model_dicts['observation_model'])
    reward_model.load_state_dict(model_dicts['reward_model'])
    encoder.load_state_dict(model_dicts['encoder'])
    actor_model.load_state_dict(model_dicts['actor_model'])
    value_model.load_state_dict(model_dicts['value_model'])
    model_optimizer.load_state_dict(model_dicts['model_optimizer'])
if args.algo=="dreamer":
    print("DREAMER")
    planner = actor_model
else:
    planner = MPCPlanner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model)
global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
free_nats = torch.full((1, ), args.free_nats, device=args.device)  # Allowed deviation in KL divergence


def update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation, explore=False):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    belief, posterior_state = transition_model.filter_step(
        prev_state=posterior_state, 
        actions=action.unsqueeze(dim=0), 
        prev_belief=belief, 
        observations=encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension

    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
    if args.algo=="dreamer":
        action = planner.get_action(belief, posterior_state, det=not(explore))
    else:
        action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore:
        action = torch.clamp(Normal(action, args.action_noise).rsample(), -1, 1) # Add gaussian exploration noise on top of the sampled action
        # action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
    return belief, posterior_state, action, next_observation, reward, done


# Testing only
if args.test:
    # Set models to eval mode
    transition_model.eval()
    reward_model.eval()
    encoder.eval()
    with torch.no_grad():
        total_reward = 0
        for _ in tqdm(range(args.test_episodes)):
            observation = env.reset()
            belief, posterior_state = transition_model.initial_step(observation=observation, device=args.device)
            action = torch.zeros(1, env.action_size, device=args.device)
            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action, observation, reward, done = update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device))
                total_reward += reward
                if args.render:
                    env.render()
                if done:
                    pbar.close()
                    break
    print('Average Reward:', total_reward / args.test_episodes)
    env.close()
    quit()

if args.slots:
    monolithic_model = mw.SlotsModelWrapper(
        encoder=encoder,
        transition=transition_model,
        observation=observation_model,
        reward=reward_model,
        optimizer=model_optimizer,
        args=args)
else:
    monolithic_model = mw.MonolithicModelWrapper(
        encoder=encoder,
        transition=transition_model,
        observation=observation_model,
        reward=reward_model,
        optimizer=model_optimizer,
        args=args)



monolithic_policy = mw.MonolithicPolicyWrapper(
    actor=actor_model,
    critic=value_model,
    actor_optimizer=actor_optimizer,
    critic_optimizer=value_optimizer,
    args=args)


print(reward_model)
print(transition_model)
print(observation_model)






# args.lr = 0.0001
# args.lr_decay_every = int(1e4)
# args.lr_decay_gamma = 0.95
# args.kl_coeff = 1e-4
# args.dkl_coeff = 1e-4
# args.dkl_steps = int(5e4)
# dynamics_model_builder = lambda state_dim: dm.SlotDynamicsModel(state_dim, 
#             action_dim=5, hid_dim=128, interaction_type='pairwise')
# torch.manual_seed(0)
# slot_dynamic_autoencoder = lvm.RSSMLVM(
#     recognition_model=ssa.RecognitionModel(
#         num_slots=5, 
#         interface_dim=32, 
#         slot_dim=128,
#         iters=3,
#         slot_temp=1), 
#     dynamics_model=dm.RSSM(stoch_dim=128, model=dynamics_model_builder(state_dim=128)),
#     observation_model=ssa.ObservationModel(indim=128, num_slots=5, scale=10),
#     mode='dynamics',
#     device=torch.device('cpu'),
#     args=args).to(torch.device('cpu'))


# monolithic_model.observation_model_model = slot_dynamic_autoencoder.observation_model
# monolithic_model.transition_model_model = slot_dynamic_autoencoder.transition_model
# transition_model = slot_dynamic_autoencoder.transition_model
# observation_model = slot_dynamic_autoencoder.observation_model


print('obs model')
du.visualize_parameters(monolithic_model.observation_model, print)
print(du.count_parameters(monolithic_model.observation_model))
print('trans model')
du.visualize_parameters(monolithic_model.transition_model, print)
print(du.count_parameters(monolithic_model.transition_model))


print('obs model', du.count_parameters(monolithic_model.observation_model))
print('trans model', du.count_parameters(monolithic_model.transition_model))
print('dynamics_model', du.count_parameters(monolithic_model.transition_model.dynamics_model))
print('rssm_head', du.count_parameters(monolithic_model.transition_model.rssm_head))

print('\n')

# assert False





# Training (and testing)
# for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):
for episode in range(metrics['episodes'][-1] + 1, args.episodes + 1):
    # Model fitting
    losses = []
    model_modules = transition_model.modules+encoder.modules+observation_model.modules+reward_model.modules

    print("training loop")
    for s in tqdm(range(args.collect_interval)):

        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size) # Transitions start at time t = 0

        # # print(observations[0])
        # plt.imsave('kik.png', observations[0,0].permute((1,2,0)).numpy())
        # # print(observation.shape)
        # print(args.chunk_size)
        # print(args.max_episode_length)
        # assert False


        # print(observations.shape)
        # torch.save(dict(observations=observations, actions=actions), 'debug_lvm.pt')

        # ckpt = torch.load('debug_lvm.pt')
        # observations, actions = ckpt['observations'], ckpt['actions']




        # print('obs model')
        # du.visualize_parameters(monolithic_model.observation_model, print)
        # print(du.count_parameters(monolithic_model.observation_model))
        # print('trans model')
        # du.visualize_parameters(monolithic_model.transition_model, print)
        # print(du.count_parameters(monolithic_model.transition_model))

        # print(reloaded_observations.shape)


        # assert False


        # torch.manual_seed(0)
        beliefs, posterior_states, observation_loss, reward_loss, kl_loss = monolithic_model.main(observations, actions, rewards, nonterminals, free_nats, global_prior, param_list)



        # assert False

        actor_loss, value_loss = monolithic_policy.main(beliefs, posterior_states, model_modules, monolithic_model.transition_model, monolithic_model.reward)


        # print([observation_loss.item(), reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item()])
        # assert False
        
        # # Store (0) observation loss (1) reward loss (2) KL loss (3) actor loss (4) value loss
        losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item()])


    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['kl_loss'].append(losses[2])
    metrics['actor_loss'].append(losses[3])
    metrics['value_loss'].append(losses[4])
    lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['actor_loss']):], metrics['actor_loss'], 'actor_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)


    # if args.lvm_only:
    #     print('NOT COLLECTING NEW DATA')
    # else:
    # Data collection
    print("Data collection")
    with torch.no_grad():
        observation, total_reward = env.reset(), 0

        belief, posterior_state = transition_model.initial_step(observation=observation.to(device=args.device), device=args.device)
        action = torch.zeros(1, env.action_size, device=args.device)

        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:
            belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device), explore=True)
            D.append(observation, action.cpu(), reward, done)
            total_reward += reward
            observation = next_observation
            if args.render:
                env.render()
            if done:
                pbar.close()
                break
        
        # Update and plot train reward metrics
        metrics['steps'].append(t + metrics['steps'][-1])
        metrics['episodes'].append(episode)
        metrics['train_rewards'].append(total_reward)
        lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)


    # Test model
    if episode % args.test_interval == 0:
        print("Test model")
        # Set models to eval mode
        transition_model.eval()
        observation_model.eval()
        reward_model.eval() 
        encoder.eval()
        actor_model.eval()
        value_model.eval()
        # Initialise parallelised test environments
        test_envs = EnvBatcher(Env, (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth), {}, args.test_episodes)
        
        with torch.no_grad():

            if args.lvm_only:
                test_observations = []
                test_actions = []

                # while not done:
                #     action = env.sample_random_action()
                #     next_observation, reward, done = env.step(action)
                #     D.append(observation, action, reward, done)
                #     observation = next_observation
                #     t += 1

                observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes, )), []
                for t in range(args.max_episode_length):
                    action = test_envs.sample_random_action()
                    next_observation, reward, done = test_envs.step(action)
                    # print(t, done)

                    test_observations.append(observation)
                    test_actions.append(action)
                    total_rewards += 0

                    observation = next_observation

                # print(test_observations[0].shape)
                # assert False

                test_observations = torch.stack(test_observations).to(args.device)
                test_actions = torch.stack(test_actions).to(args.device)

                test_priors, test_posteriors = monolithic_model.transition_model.forward(itf.InteractiveBatchData(obs=test_observations, act=test_actions))
                test_beliefs, test_posterior_states = monolithic_model.get_beliefs_and_states(test_posteriors)
                test_preds = monolithic_model.predict(test_priors, test_posteriors, test_observations.shape)

                # HACK: the last frame is masked out, so let's not count it.
                for t in range(args.max_episode_length-1):
                    test_bsize, c, h, w = test_observations[t].shape#[0]
                    frame = torch.cat([
                        test_observations[t],  # (B, C, H, W)
                        test_preds.pbd[t],  # (B, C, H, W)
                        # test_preds.cbd[t].permute((1,0,2,3,4)).reshape(test_bsize*args.num_slots, c, h, w),  # (B, C, H, W)
                        test_preds.cbd[t].permute((1,0,2,3,4)).reshape(test_bsize*args.num_slots, c, h, w),  # (B, C, H, W)
                        ], dim=0)
                    video_frames.append(make_grid(frame, nrow=args.test_episodes).cpu().numpy())  # Decentre

                gridbd = vis.visualize_slots_grid_dynamic(
                    ground_truth=test_observations[:args.max_episode_length-1, 0],  # (T, C, H, W)
                    prediction=test_preds.pbd[:args.max_episode_length-1, 0],  # (T, C, H, W)
                    components=test_preds.cbd[:args.max_episode_length-1, 0])  # (T, K, C, H, W)
                fprefix = os.path.join(results_dir, 'lvm_debug{}'.format(episode))
                torchvision.utils.save_image(gridbd, '{}_before_dynamics.png'.format(fprefix))



                # print(test_observations.shape)
                # print(test_actions.shape)

                # assert False


                # test_observations = torch.stack(test_observations).to(args.device)
                # test_actions = torch.stack(test_actions).to(args.device)

                train_priors, train_posteriors = monolithic_model.transition_model.forward(itf.InteractiveBatchData(obs=observations, act=actions))
                # test_beliefs, test_posterior_states = monolithic_model.get_beliefs_and_states(test_posteriors)
                train_preds = monolithic_model.predict(train_priors, train_posteriors, observations.shape)

                # # HACK: the last frame is masked out, so let's not count it.
                # for t in range(args.max_episode_length-1):
                #     train_bsize, c, h, w = observations[t].shape#[0]
                #     frame = torch.cat([
                #         test_observations[t],  # (B, C, H, W)
                #         test_preds.pbd[t],  # (B, C, H, W)
                #         # test_preds.cbd[t].permute((1,0,2,3,4)).reshape(test_bsize*args.num_slots, c, h, w),  # (B, C, H, W)
                #         test_preds.cbd[t].permute((1,0,2,3,4)).reshape(test_bsize*args.num_slots, c, h, w),  # (B, C, H, W)
                #         ], dim=0)
                #     video_frames.append(make_grid(frame, nrow=args.test_episodes).cpu().numpy())  # Decentre

                gridbd = vis.visualize_slots_grid_dynamic(
                    ground_truth=observations[:args.chunk_size, 0],  # (T, C, H, W)
                    prediction=train_preds.pbd[:args.chunk_size, 0],  # (T, C, H, W)
                    components=train_preds.cbd[:args.chunk_size, 0])  # (T, K, C, H, W)
                fprefix = os.path.join(results_dir, 'lvm_debug{}'.format(episode))
                torchvision.utils.save_image(gridbd, '{}_before_dynamics_train.png'.format(fprefix))








            else:

                #########################
                observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes, )), []
                belief, posterior_state, action = torch.zeros(args.test_episodes, args.belief_size, device=args.device), torch.zeros(args.test_episodes, args.state_size, device=args.device), torch.zeros(args.test_episodes, env.action_size, device=args.device)
                pbar = tqdm(range(args.max_episode_length // args.action_repeat))
                for t in pbar:
                    belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(args, test_envs, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device))
                    total_rewards += reward.numpy()

                    if not args.symbolic_env:  # Collect real vs. predicted frames for video
                        if args.slots:
                            frame = torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3)  # decenter

                            vis_b = belief.shape[0]
                            vis_belief = belief.reshape(vis_b, args.num_slots, -1)
                            vis_state = posterior_state.reshape(vis_b, args.num_slots, -1)
                            vis_x = torch.cat([vis_belief, vis_state], dim=-1)

                            masked_rgbs, pred = observation_model.decode(vis_x)
                            masked_rgbs, pred = masked_rgbs.cpu(), pred.cpu()

                            c, h, w = observation.shape[-3:]
                            frame = torch.cat([
                                observation,  # (B, C, H, W)
                                pred,  # (B, C, H, W)
                                masked_rgbs.permute((1,0,2,3,4)).reshape(vis_b*args.num_slots, c, h, w)  # (B*K, C, H, W)
                                ], dim=0)  # (B*(1+1+K), C, H, W)

                            # video_frames.append(make_grid(frame+0.5, nrow=args.test_episodes).numpy())  # Decentre

                            if isinstance(env, SimpleEntityEnv):
                                video_frames.append(make_grid(frame, nrow=args.test_episodes).numpy())  # Decentre
                            else:
                                video_frames.append(make_grid(frame+0.5, nrow=args.test_episodes).numpy())  # Decentre

                        else:
                            if isinstance(env, SimpleEntityEnv):
                                video_frames.append(make_grid(torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3), nrow=5).numpy())  # Decentre
                            else:
                                video_frames.append(make_grid(torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre


                    observation = next_observation
                    if done.sum().item() == args.test_episodes:
                        pbar.close()
                        break

                #########################
        
        # Update and plot reward metrics (and write video if applicable) and save metrics
        metrics['test_episodes'].append(episode)
        metrics['test_rewards'].append(total_rewards.tolist())
        lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
        lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step')
        if not args.symbolic_env:
            episode_str = str(episode).zfill(len(str(args.episodes)))
            write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
            save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Set models to train mode
        transition_model.train()
        observation_model.train()
        reward_model.train()
        encoder.train()
        actor_model.train()
        value_model.train()
        # Close test environments
        test_envs.close()

    writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
    writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1], metrics['steps'][-1]*args.action_repeat)
    writer.add_scalar("observation_loss", metrics['observation_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("reward_loss", metrics['reward_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("kl_loss", metrics['kl_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("actor_loss", metrics['actor_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("value_loss", metrics['value_loss'][0][-1], metrics['steps'][-1])  
    print("episodes: {}, total_steps: {}, train_reward: {} ".format(metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]))

    # Checkpoint models
    if episode % args.checkpoint_interval == 0:
        torch.save({'transition_model': transition_model.state_dict(),
                                'observation_model': observation_model.state_dict(),
                                'reward_model': reward_model.state_dict(),
                                'encoder': encoder.state_dict(),
                                'actor_model': actor_model.state_dict(),
                                'value_model': value_model.state_dict(),
                                'model_optimizer': model_optimizer.state_dict(),
                                'actor_optimizer': actor_optimizer.state_dict(),
                                'value_optimizer': value_optimizer.state_dict()
                                }, os.path.join(results_dir, 'models_%d.pth' % episode))
        if args.checkpoint_experience:
            torch.save(D, os.path.join(results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes


# Close training environment
env.close()
