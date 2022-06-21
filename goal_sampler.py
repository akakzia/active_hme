from dis import dis
import numpy as np
from utils import generate_stacks_to_class, get_eval_goals
from utils import INSTRUCTIONS
from mpi4py import MPI
from goal_evaluator import GoalEvaluator
import pickle


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.n_blocks = args.n_blocks
        self.goal_dim = args.env_params['goal']

        # Keep track of the number of discovered goals
        self.nb_discovered_goals = 0

        # Define lists to store discovered goals as arrays and as strings
        self.discovered_goals = []
        self.discovered_goals_str = []
        self.discovered_goals_oracle_ids = []

        # Initialize value estimations list
        self.values_goals = []

        # Query arguments
        self.query_proba = 0.
        self.min_queue_length = args.min_queue_length 
        self.max_queue_length = args.max_queue_length
        self.beta = args.beta
        self.progress_function = args.progress_function

        # Initialize goal_evaluator
        self.goal_evaluator = GoalEvaluator(args)

        # Cycle counter
        self.n_cycles = 0

        self.stacks_to_class = generate_stacks_to_class()
        self.discovered_goals_per_stacks = {e:0 for e in set(self.stacks_to_class.values())}
        self.discovered_goals_per_stacks['others'] = 0 # for goals that are not in the stack classes

        self.use_stability_condition = args.use_stability_condition

        self.init_stats()
    
    def setup_policy(self, policy):
        """ Sets up the policy """
        self.goal_evaluator.setup_policy(policy)

    def sample_goals(self, n_goals=1, evaluation=False):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation:
            goals = []
            for instruction in INSTRUCTIONS:
                goal = get_eval_goals(instruction, n=self.n_blocks)
                goals.append(goal.squeeze(0))
            goals = np.array(goals)
        else:
            if len(self.discovered_goals) == 0:
                goals = np.random.choice([-1., 1.], size=(n_goals, self.goal_dim))
            else:
                # sample uniformly from discovered goals
                goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                goals = np.array(self.discovered_goals)[goal_ids]
        return goals
    
    def generate_intermediate_goals(self, goals):
        """ Given an array of goals, uses goal evaluator to generate intermediate goals that maximize the value """
        res = []
        for eval_goal in goals:
            repeat_goal = np.repeat(np.expand_dims(eval_goal, axis=0), repeats=len(self.discovered_goals), axis=0)
            norm_goals = self.goal_evaluator.estimate_goal_value(goals=repeat_goal, ag=self.discovered_goals)
            ind = np.argpartition(norm_goals, -2)[-2:]
            adjacent_goal = self.discovered_goals[ind[0]] if str(self.discovered_goals[ind[0]]) != str(eval_goal) else self.discovered_goals[ind[1]]
            res.append(adjacent_goal)
        
        res = np.array(res)

        return res

    def update(self, episodes):
        """
        Update discovered goals list from episodes
        Update list of successes and failures for LP curriculum
        Label each episode with the last ag (for buffer storage)
        """
        # Update the goal memory
        self.update_goal_memory(episodes)

        if self.rank == 0.:
            # Compute goal values
            norm_values = self.goal_evaluator.estimate_goal_value(goals=np.array(self.discovered_goals))
            self.values_goals.append(norm_values)

            # Compute Query Probabilities
            if len(self.values_goals) > self.min_queue_length:
                delta_value_goals = abs(self.values_goals[0] - self.values_goals[-1][:len(self.values_goals[0])])
                if self.progress_function == 'mean':
                    progress = np.mean(delta_value_goals) 
                elif self.progress_function == 'max':
                    progress = np.max(delta_value_goals)
                
                self.query_proba = np.exp(- self.beta * progress)
            
        self.sync_queries()

        return episodes

    def update_goal_memory(self, episodes):
        """ Given a batch of episodes, gathered from all workers, updates:
        1. the list of discovered goals (arrays and strings)
        2. the list of discovered goals' ids
        3. the number of discovered goals
        4. the bidict oracle id <-> goal str """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes for e in eps]

            for e in all_episode_list:
                # Retrive last achieved goal
                last_ag = e['ag'][-1]
                if self.use_stability_condition:
                    # Compute boolean conditions to determine the discovered goal stability 
                    # 1: the goal is stable for the last 10 steps
                    condition_stability = np.sum([str(last_ag) == str(el) for el in e['ag'][-10:]]) == 10.
                    # 2: Gripper is far from all objects
                    last_obs = e['obs'][-1]
                    pos_gripper = last_obs[:3]
                    pos_objects = [last_obs[10 + 15 * i: 13 + 15 * i] for i in range(5)]
                    condition_far = np.sum([np.linalg.norm(pos_gripper - pos_ob) >= 0.09 for pos_ob in pos_objects]) == 5.
                else:
                    # Always true
                    condition_stability = True
                    condition_far = True
                # Add last achieved goal to memory if first time encountered
                if str(last_ag) not in self.discovered_goals_str and condition_stability and condition_far:
                    self.discovered_goals.append(last_ag.copy())
                    self.discovered_goals_str.append(str(last_ag))
                    self.discovered_goals_oracle_ids.append(self.nb_discovered_goals)

                    # Check to which stack class corresponds the discovered goal
                    above_predicates = last_ag[10:30]
                    try:
                        c = self.stacks_to_class[str(above_predicates)]
                        self.discovered_goals_per_stacks[c] += 1
                    except KeyError:
                        self.discovered_goals_per_stacks['others'] += 1

                    # Increment number of discovered goals (to increment the id !)
                    self.nb_discovered_goals += 1
        
        self.sync()

    def sync(self):
        """ Synchronize the goal sampler's attributes between all workers """
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)
        self.discovered_goals_oracle_ids = MPI.COMM_WORLD.bcast(self.discovered_goals_oracle_ids, root=0)
        self.nb_discovered_goals = MPI.COMM_WORLD.bcast(self.nb_discovered_goals, root=0)
    
    def sync_queries(self):
        """ Synchronize the query's attributes between all workers """
        self.values_goals = MPI.COMM_WORLD.bcast(self.values_goals, root=0)
        self.values_goals = self.values_goals[-self.max_queue_length:]
        self.query_proba = MPI.COMM_WORLD.bcast(self.query_proba, root=0)

    def init_stats(self):
        self.stats = dict()
        # Number of classes of eval
        n = len(INSTRUCTIONS)
        for i in np.arange(1, n+1):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['Av_Rew_{}'.format(i)] = []
            self.stats['# class_teacher {}'.format(i)] = []
            self.stats['# class_agent {}'.format(i)] = []
        
        # Init for each stack class
        stack_classes = set(self.stacks_to_class.values())
        for c in stack_classes:
            self.stats[f'discovered_{c}'] = []
        # Add class that contains all goals that do not correspond to the stack_classes
        self.stats['discovered_others'] = []

        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        self.stats['nb_discovered'] = []
        self.stats['nb_internalized_pairs'] = []
        self.stats['proposed_ss'] = []
        self.stats['proposed_beyond'] = []
        self.stats['query_proba'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update', 'update_graph', 
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr,  agent_stats, goals_per_class, proposed_ss, proposed_beyond, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_discovered'].append(len(self.discovered_goals))
        for g_id in np.arange(1, len(av_res) + 1):
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id-1])
            self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id-1])

        for k, v in self.discovered_goals_per_stacks.items():
            self.stats[f'discovered_{k}'].append(v)
        self.stats['proposed_ss'].append(proposed_ss)
        self.stats['proposed_beyond'].append(proposed_beyond)
        for k in goals_per_class.keys():
            self.stats['# class_teacher {}'.format(k)].append(goals_per_class[k])
            self.stats['# class_agent {}'.format(k)].append(agent_stats[k])
    
    def save_discovered_goals(self, bucket_path, epoch):
        # save list of discovered goals
        with open(bucket_path + '/discovered_g_ep_{}.pkl'.format(epoch), 'wb') as f:
            pickle.dump(self.discovered_goals, f)
