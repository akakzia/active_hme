import torch
import numpy as np
from mpi4py import MPI
import env
import gym
import os
from arguments import get_args
from rl_modules.rl_agent import RLAgent
import random
from rollout import RolloutWorker
from goal_sampler import GoalSampler
from utils import init_storage, get_eval_goals
import time
from mpi_utils import logger

def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

def launch(args):
    # Set cuda arguments to True
    args.cuda = torch.cuda.is_available()

    rank = MPI.COMM_WORLD.Get_rank()

    t_total_init = time.time()

    # Algo verification
    assert args.algo == 'semantic', 'Only semantic algorithm is implemented'

    # Make the environment
    args.env_name = 'FetchManipulate{}Objects-v0'.format(args.n_blocks)
    env = gym.make(args.env_name)

    # set random seeds for reproducibility
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get saving paths
    if rank == 0:
        logdir, model_path = init_storage(args)
        logger.configure(dir=logdir)
        logger.info(vars(args))

    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)


    ### GENERATE BUCKET METHOD TESTING ###

    goal_sampler.discovered_goals = [3,25,21,2,255,22]

    goals_values = [0.3,0.1,0.5,0.8,0.9,0.2]

    print('Goal values: ', goals_values)

    print('Buckets without equal repartition: ', goal_sampler.generate_buckets(goals_values, 3))
    print('Buckets with equal repartition: ', goal_sampler.generate_buckets(goals_values, 3, equal_goal_repartition=True))

    goals_values = [0.3,0.1,0.05,0.08,0.09,0.2]

    print('Goal values: ', goals_values)

    print('Buckets without equal repartition: ', goal_sampler.generate_buckets(goals_values, 3))
    print('Buckets with equal repartition: ', goal_sampler.generate_buckets(goals_values, 3, equal_goal_repartition=True))

    ### GENERATE BUCKET METHOD COMPUTATION TIME ###

    goal_sampler.discovered_goals = np.zeros(70000)

    goals_values = np.random.rand(70000)

    print('Goal values: ', goals_values)

    time_0 = time.time()
    print('Buckets without equal repartition: ', goal_sampler.generate_buckets(goals_values, 3))
    print('Computation time: ', time.time() - time_0)

    time_0 = time.time()
    print('Buckets with equal repartition: ', goal_sampler.generate_buckets(goals_values, 3, equal_goal_repartition=True))
    print('Computation time: ', time.time() - time_0)




if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    launch(args)