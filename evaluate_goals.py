import torch
import numpy as np
from mpi4py import MPI
import env
import gym
import os
import pickle as pkl
from arguments import get_args
from rl_modules.rl_agent import RLAgent
from rollout import RolloutWorker
from goal_sampler import GoalSampler
from utils import get_env_params, apply_on_table_config
from graph.sp_graph import SpGraph


def launch(args):
    rank = MPI.COMM_WORLD.Get_rank()
    agent_names = ['oracle_block_beta=0', 'main_beta=20', 'main_beta=50', 'main_beta=100', 'main_beta=200', 'main_beta=500']

    for agent_name in agent_names:
        if rank == 0:
            print(f'=-=-=-=-=-=-=-=-= {agent_name} =-=-=-=-=-=-=-=-=')
        path = f'{agent_name}/'
        epoch = 90 if agent_name == 'oracle_block_beta=0' else 130
        all_assignement_goals = []
        if rank == 0:
            all_assignement_goals = launch_eval_coverage(args, path, epoch)
        
        # Synchronize
        all_assignement_goals = MPI.COMM_WORLD.bcast(all_assignement_goals, root=0)

        goals_list = [all_assignement_goals]

        for goals in goals_list:
            launch_eval_sr(args, path, epoch, goals)


def launch_eval_sr(args, path, ep, goals):
    rank = MPI.COMM_WORLD.Get_rank()
    list_runs = sorted(os.listdir(path))

    res = []

    for i, run in enumerate(list_runs):
        model_path = path + run + f'/models/model_{ep}.pt'
        # Make the environment
        env = gym.make('FetchManipulate5Objects-v0')
        env.seed(args.seed)

        args.env_params = get_env_params(env)


        # Initialize agent
        goal_sampler = GoalSampler(args)

        # create the sac agent to interact with the environment
        policy = RLAgent(args, env.compute_reward, goal_sampler)
        policy.load(model_path, args)
        goal_sampler.setup_policy(policy)

        # def rollout worker
        rollout_worker = RolloutWorker(env, policy, args)

        # Dispatch all goals among workers
        goals_per_worker = len(goals) // args.num_workers
        eval_goals = np.array(goals[rank * goals_per_worker: (rank + 1) * goals_per_worker])

        episodes = rollout_worker.generate_rollout(eval_goals, true_eval=True, animated=False)
        results = np.array([e['rewards'][-1] == args.n_blocks for e in episodes])
        all_results = MPI.COMM_WORLD.gather(results, root=0)
        all_results = np.array(all_results).flatten()
        if rank == 0:
            mean = all_results.mean()
            # print('Av Success Rate: {}'.format(mean))
            res.append(mean)

    if rank == 0:
        res = np.array(res)
        av_res = np.array(res).mean(axis=0)
        std_res = np.array(res).std(axis=0)
        print(f'Av Success Rate: {av_res:.3f}±{std_res:.3f}')
    # all_results = MPI.COMM_WORLD.gather(results, root=0)
    # if rank == 0:
    #     assert len(all_results) == args.num_workers  # MPI test
    #     av_res = np.array(all_results).mean(axis=0)
    #     std_res = np.array(all_results).std(axis=0)
    #     print(f'Av Success Rate: {av_res:.3f}±{std_res:.3f}')
    # stop = 1


def launch_eval_coverage(args, path, ep):
    # Load oracle graph
    print(f'Loading Assignment Graph from {args.oracle_path + args.oracle_name} ...')
    sp_graph = SpGraph(args=args)

    # Retrieving Assignment Graph goals
    # sp_goals = [apply_on_table_config(g) for g in sp_graph.oracle_graph.configs.keys()]
    # print(f'Size of the assignment graph: {len(sp_goals)}')
    # Retrieving Oracle Graph Goals
    sp_goals = [apply_on_table_config(g) for g in sp_graph.oracle_graph.configs.keys()]
    print(f'Size of the oracle graph: {len(sp_goals)}')
    
    list_runs = sorted(os.listdir(path))
    ratios_assign = np.empty([len(list_runs)])
    ratios_assign.fill(np.nan)
    # ratios_oracle = np.empty([len(list_runs)])
    # ratios_oracle.fill(np.nan)
    ratios_new = np.empty([len(list_runs)])
    ratios_new.fill(np.nan)

    for i, run in enumerate(list_runs):
        run_path = path + run
        # Load Agent discovered goals
        with open(run_path + f'/buckets/discovered_g_ep_{ep}.pkl', 'rb') as f:
            agent_goals = pkl.load(file=f)

        agent_goals = [tuple(g) for g in agent_goals]    
        r_discovered_assigned, r_new = evaluate_coverage(agent_goals, sp_goals)
        ratios_assign[i] = r_discovered_assigned
        ratios_new[i] = r_new

    mean_ratio_assign = np.mean(ratios_assign)
    std_ratio_assign = np.std(ratios_assign)

    mean_ratio_new = np.mean(ratios_new)
    std_ratio_new = np.std(ratios_new)
    
    print(f'The ratio of discovered assignement goals: {mean_ratio_assign:.3f}±{std_ratio_assign:.3f}')
    print(f'The ratio of new goals: {mean_ratio_new:.3f}±{std_ratio_new:.3f}')

    return sp_goals


def evaluate_coverage(discovered_goals, assignement_goals):
    """ Given an array of the discovered goals by the agent, an array of goals assigned by SP and oracle goals
    Returns
    1/ the ratio of discovered assignement goals
    2/ the ratio of discovered oracle goals 
    3/ The ratio of discovered goals that are not within oracle """
    inter_assign = set(discovered_goals).intersection(set(assignement_goals))
    n_commun_assign = len(inter_assign)
    ratio_discovered_assignement = n_commun_assign / len(assignement_goals)

    ratio_new_goals = (len(discovered_goals) - n_commun_assign) / len(discovered_goals)

    return ratio_discovered_assignement, ratio_new_goals

if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    args = get_args()
    args.cuda = torch.cuda.is_available()
    launch(args)
