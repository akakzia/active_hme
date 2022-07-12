import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import  random
from mpi4py import MPI
from arguments import get_args
from utils import get_eval_goals, generate_stacks_dict
import os 
import pickle as pkl

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

if __name__ == '__main__':
    directory_name = 'no_att_q=200_beta=20'
    epoch = 70

    n_eval = 50

    path = f'/home/ahmed/Documents/Amaterasu/hachibi/active_hme/results/internalization_study/{directory_name}/1/models/'
    model_path = path + f'model_{epoch}.pt'

    args = get_args()

    args.env_name = 'FetchManipulate5Objects-v0'

    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    args.seed = np.random.randint(1e6)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)

    # create the sac agent to interact with the environment
    policy = RLAgent(args, env.compute_reward, goal_sampler)
    policy.load(model_path, args)
    goal_sampler.setup_policy(policy)

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, args)

    eval_goals = []
    instructions = ['stack_4', 'stack_5'] * n_eval
    for instruction in instructions:
        eval_goal = get_eval_goals(instruction, n=args.n_blocks)
        eval_goals.append(eval_goal.squeeze(0))
    eval_goals = np.array(eval_goals)

    all_results = []

    episodes = rollout_worker.generate_rollout(eval_goals, true_eval=True, animated=False)
    results = np.array([e['rewards'][-1] == args.n_blocks for e in episodes])
    all_results.append(results)

    results = np.array(all_results)
    print('Av Success Rate: {}'.format(results.mean()))