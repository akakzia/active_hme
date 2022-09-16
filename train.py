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
from mpi_utils import logger

def launch(args):
    # Set cuda arguments to True
    args.cuda = torch.cuda.is_available()

    rank = MPI.COMM_WORLD.Get_rank()

    t_total_init = time.time()

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
        logdir, model_path = init_storage(args)
        logger.configure(dir=logdir)
        logger.info(vars(args))
    
    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)

    # Initialize RL Agent
    policy = RLAgent(args, env.compute_reward, goal_sampler)

    # Initialize Rollout Worker
    rollout_worker = HMERolloutWorker(env, policy, goal_sampler, args)

    # Sets the goal_evaluator estimator inside the goal sampler
    goal_sampler.setup_policy(policy)

    # Load oracle graph
    nk_graph = nk.Graph(0,weighted=True, directed=True)
    semantic_graph = SemanticGraph(bidict(),nk_graph,args.n_blocks,True,args=args)
    agent_network = AgentGraph(semantic_graph,logdir,args)

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
            # Environment interactions
            t_i = time.time()
            episodes, episodes_type = rollout_worker.train_rollout(agent_network=agent_network,
                                                                   epoch=epoch,
                                                                   time_dict=time_dict)
            time_dict['rollout'] += time.time() - t_i

            # Goal Sampler updates
            t_i = time.time()
            episodes = goal_sampler.update(episodes)
            time_dict['gs_update'] += time.time() - t_i

            # Storing episodes
            t_i = time.time()
            policy.store(episodes, episodes_type)
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

            # Update query proba based on frequency 
            if episode_count // args.num_workers % args.query_proba_update_freq == 0:
                goal_sampler.update_query_proba()

        time_dict['epoch'] += time.time() -t_init
        time_dict['total'] = time.time() - t_total_init

        if args.evaluations:
            if rank==0: logger.info('\tRunning eval ..')
            # Performing evaluations
            t_i = time.time()
            eval_goals = goal_sampler.sample_goals(evaluation=True)
            episodes = rollout_worker.generate_rollout(goals=eval_goals,
                                                       true_eval=True,  # this is offline evaluations
                                                       )


            results = np.array([e['success'][-1].astype(np.float32) for e in episodes])
            rewards = np.array([e['rewards'][-1] for e in episodes])
            all_results = MPI.COMM_WORLD.gather(results, root=0)
            all_rewards = MPI.COMM_WORLD.gather(rewards, root=0)
            time_dict['eval'] += time.time() - t_i

            # synchronize goals count per class in teacher
            synchronized_stats, sync_nb_ss, sync_nb_beyond = sync(agent_network.teacher.stats, agent_network.teacher.ss_interventions,
                                                                  agent_network.teacher.beyond_interventions)

            # Logs
            if rank == 0:
                assert len(all_results) == args.num_workers  # MPI test
                av_res = np.array(all_results).mean(axis=0)
                av_rewards = np.array(all_rewards).mean(axis=0)
                global_sr = np.mean(av_res)

                agent_network.log(logger)
                log_and_save(goal_sampler, epoch, episode_count, av_res, av_rewards, global_sr, agent_network.stats, synchronized_stats, sync_nb_ss,
                 sync_nb_beyond, time_dict)
                # Saving policy models
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


def log_and_save( goal_sampler, epoch, episode_count, av_res, av_rew, global_sr, agent_stats, teacher_stats, proposed_ss, proposed_beyond, time_dict):
    goal_sampler.save(epoch, episode_count, av_res, av_rew, global_sr, agent_stats, teacher_stats, proposed_ss, proposed_beyond, time_dict)
    for k, l in goal_sampler.stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()

def sync(x, a, b):
    """ x: dictionary of counts for every class_goal proposed by the teacher
        return the synchronized dictionary among all cpus """
    res = x.copy()
    for k in x.keys():
        res[k] = MPI.COMM_WORLD.allreduce(x[k], op=MPI.SUM)
    sync_a = MPI.COMM_WORLD.allreduce(a, op=MPI.SUM)
    sync_b = MPI.COMM_WORLD.allreduce(b, op=MPI.SUM)
    return res, sync_a, sync_b
    
if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    # Align arguments according to agent
    if args.agent in ['F1andRandom', 'F2andRandom', 'F3andRandom', 'UniformandRandom']:
        args.fixed_queries = True
        args.fixed_query_proba = 0.8 # probability of following the goal chaining curriculum (with p=0.2, randomly target goal and pursue it)

    if args.agent in ['LPAgent', 'VDSAgent']:
        args.fixed_queries = True
        args.fixed_query_proba = 0
        args.eps_uniform_goal = 0.2
    launch(args)
