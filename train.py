import torch
import numpy as np
from bidict import bidict
from typing import DefaultDict
from mpi4py import MPI
import env
import gym
import os
from arguments import get_args
from rl_modules.rl_agent import RLAgent
import random
from rollout import HMERolloutWorker
from goal_sampler import GoalSampler
from utils import get_env_params, init_storage, get_eval_goals
import networkit as nk
from graph.semantic_graph import SemanticGraph
from graph.agent_graph import AgentGraph
import time
import pickle as pkl
from mpi_utils import logger

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
    logdir = None
    if rank == 0:
        logdir, model_path, bucket_path = init_storage(args)
        logger.configure(dir=logdir)
        logger.info(vars(args))
    
    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)

    # Initialize RL Agent
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
        # policy.load('/home/ahmed/Documents/Amaterasu/hachibi/active_hme/results/value_network/agent_0/1/models/model_210.pt', args)
    else:
        raise NotImplementedError

    # Initialize Rollout Worker
    rollout_worker = HMERolloutWorker(env, policy, goal_sampler, args)

    # Sets the goal_evaluator estimator inside the goal sampler
    if args.goal_evaluator_method == 1:
        goal_sampler.setup_policy(policy)
    else:
        raise NotImplementedError('Only method 1 is implemented, please make sure you want to run method 2')

    # Load oracle graph
    nk_graph = nk.Graph(0,weighted=True, directed=True)
    semantic_graph = SemanticGraph(bidict(),nk_graph,args.n_blocks,True,args=args)
    agent_network = AgentGraph(semantic_graph,logdir,args)
    
    # # Temporary load discovered goals 
    # with open('/home/ahmed/Documents/Amaterasu/hachibi/active_hme/results/value_network/agent_0/1' + f'/buckets/discovered_g_ep_210.pkl', 'rb') as f:
    #         data_discovered = pkl.load(file=f)
    
    # # Add them to graph
    # for g in data_discovered:
    #     agent_network.semantic_graph.create_node(tuple(g))
    # agent_network.teacher.compute_frontier(agent_network.semantic_graph)
    # print(f'Frontier {len(agent_network.teacher.agent_frontier)}')
    # print(f'SS {len(agent_network.teacher.agent_stepping_stones)}')
    # frontier_ag = [agent_network.semantic_graph.getConfig(i) for i in agent_network.teacher.agent_frontier]
    # explore_goal = next(iter(agent_network.sample_from_frontier(frontier_ag[-1], 1)), None) 

    # Main interaction loop
    episode_count = 0
    for epoch in range(args.n_epochs):
        t_init = time.time()

        # setup time_tracking
        time_dict = DefaultDict(int)

        # log current epoch
        if rank == 0: logger.info('\n\nEpoch #{}'.format(epoch))

        # Cycles loop
        for _ in range(args.n_cycles):

            # Sample goals
            # t_i = time.time()
            # goals = goal_sampler.sample_goal(n_goals=args.num_rollouts_per_mpi, evaluation=False)
            # time_dict['goal_sampler'] += time.time() - t_i

            # Environment interactions
            t_i = time.time()
            # episodes = rollout_worker.generate_rollout(goals=goals,  # list of goal configurations
            #                                            true_eval=False,  # these are not offline evaluation episodes
            #                                           )
            episodes = rollout_worker.train_rollout(agent_network= agent_network,
                                                    t=epoch,
                                                    time_dict=time_dict)
            time_dict['rollout'] += time.time() - t_i

            # Goal Sampler updates
            t_i = time.time()
            if args.algo == 'semantic':
                episodes = goal_sampler.update(episodes, episode_count)
            time_dict['gs_update'] += time.time() - t_i

            # Storing episodes
            t_i = time.time()
            policy.store(episodes)
            time_dict['store'] += time.time() - t_i

            # Agent Network Update : 
            t_i = time.time()
            agent_network.update(episodes)
            time_dict['update_graph'] += time.time() - t_i

            # Updating observation normalization
            t_i = time.time()
            for e in episodes:
                policy._update_normalizer(e)
            time_dict['norm_update'] += time.time() - t_i

            # Policy updates
            t_i = time.time()
            for _ in range(args.n_batches):
                policy.train()
            time_dict['policy_train'] += time.time() - t_i
            episode_count += args.num_rollouts_per_mpi * args.num_workers

        time_dict['epoch'] += time.time() -t_init
        time_dict['total'] = time.time() - t_total_init

        if args.evaluations:
            if rank==0: logger.info('\tRunning eval ..')
            # Performing evaluations
            t_i = time.time()
            eval_goals = []
            if args.n_blocks == 3:
                instructions = ['close_1', 'close_2', 'close_3', 'stack_2', 'pyramid_3', 'stack_3']
            elif args.n_blocks == 5:
                instructions = ['close_1', 'close_2', 'close_3', 'stack_2', 'stack_3', '2stacks_2_2', '2stacks_2_3', 'pyramid_3',
                                'mixed_2_3', 'stack_4', 'stack_5']
            else:
                raise NotImplementedError
            for instruction in instructions:
                eval_goal = get_eval_goals(instruction, n=args.n_blocks)
                eval_goals.append(eval_goal.squeeze(0))
            eval_goals = np.array(eval_goals)
            episodes = rollout_worker.generate_rollout(goals=eval_goals,
                                                       true_eval=True,  # this is offline evaluations
                                                       )


            results = np.array([e['success'][-1].astype(np.float32) for e in episodes])
            rewards = np.array([e['rewards'][-1] for e in episodes])
            all_results = MPI.COMM_WORLD.gather(results, root=0)
            all_rewards = MPI.COMM_WORLD.gather(rewards, root=0)
            time_dict['eval'] += time.time() - t_i

            # Logs
            if rank == 0:
                assert len(all_results) == args.num_workers  # MPI test
                av_res = np.array(all_results).mean(axis=0)
                av_rewards = np.array(all_rewards).mean(axis=0)
                global_sr = np.mean(av_res)
                if goal_sampler.active_buckets_ids is not None:
                    logger.record_tabular('_nb_buckets', len(goal_sampler.active_buckets_ids))
                    for i, b in enumerate(goal_sampler.active_buckets_ids):
                        logger.record_tabular(f'_size_bucket_{i}', len(goal_sampler.buckets[b]))
                        logger.record_tabular(f'_lp_{i}', goal_sampler.lp[i])
                        logger.record_tabular(f'_p_{i}', goal_sampler.p[i])
                log_and_save(goal_sampler, epoch, episode_count, av_res, av_rewards, global_sr, time_dict)
                # Saving policy models
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                    goal_sampler.save_bucket_contents(bucket_path, epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


def log_and_save( goal_sampler, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
    goal_sampler.save(epoch, episode_count, av_res, av_rew, global_sr, time_dict)
    for k, l in goal_sampler.stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()


if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    launch(args)
