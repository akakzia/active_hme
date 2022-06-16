from dis import dis
import numpy as np
from utils import generate_stacks_to_class
from mpi4py import MPI
from goal_evaluator import GoalEvaluator
from bidict import bidict
import pickle


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.goal_dim = args.env_params['goal']

        self.start_curriculum_k = args.start_curriculum_k
        self.bucket_generation_freq = args.bucket_generation_freq # frequency in term of episodes (taking workers into account)
        self.bucket_evaluation_freq = args.bucket_evaluation_freq # frequency in term of episodes (taking workers into account)

        # Keep track of the number of discovered goals
        self.nb_discovered_goals = 0

        # Define lists to store discovered goals as arrays and as strings
        self.discovered_goals = []
        self.discovered_goals_str = []
        self.discovered_goals_oracle_ids = []

        # Define bidict to map goals str to ids
        self.goal_str_to_oracle_id = bidict()

        # Define goal_buckets attributes
        self.buckets = None
        self.active_buckets_ids = None
        self.goal_buckets = None
        self.granularity = args.granularity

        # Define curriculum attributres
        self.values_buckets = None
        self.lp = None
        self.p = None
        self.epsilon = args.epsilon_curriculum

        # Initialize value estimations list
        self.values_goals = []

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

    def sample_goal(self, n_goals, evaluation):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation and len(self.discovered_goals) > 0:
            goals = np.random.choice(self.discovered_goals, size=self.num_rollouts_per_mpi)
        else:
            if len(self.discovered_goals) == 0:
                goals = np.random.choice([-1., 1.], size=(n_goals, self.goal_dim))
            else:
                # sample uniformly from discovered goals
                goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                goals = np.array(self.discovered_goals)[goal_ids]
        return goals

    def update(self, episodes, t):
        """
        Update discovered goals list from episodes
        Update list of successes and failures for LP curriculum
        Label each episode with the last ag (for buffer storage)
        """
        # Update the goal memory
        self.update_goal_memory(episodes)

        # do_curriculum = (self.n_cycles > self.start_curriculum_k)
        # do_bucket_generation = (self.n_cycles % self.bucket_generation_freq == 0)
        # do_bucket_evaluation = (self.n_cycles % self.bucket_evaluation_freq == 0)
        # if do_curriculum:
        #     assert len(self.discovered_goals) > 0, 'Attempting to perform curriculum while nothing is discovered yet !'
        #     if do_bucket_generation:
        #         # First estimate and normalize goal values for all discovered goals
        #         norm_values = self.goal_evaluator.estimate_goal_value(goals=np.array(self.discovered_goals))
        #         self.values_goals = [norm_values]

        #         # Then, generate buckets based on the normalized goal values
        #         self.buckets = self.generate_buckets(normalized_goal_values=norm_values, granularity=self.granularity, equal_goal_repartition=False)

        #         # Compute values per bucket
        #         self.values_buckets = [self.evaluate_buckets()]

        #         # Compute LP
        #         self.update_lp()

        #     elif do_bucket_evaluation and self.goal_buckets is not None:
        #         n_goals_in_buckets = len(self.goal_buckets)
        #         value_estimations = self.goal_evaluator.estimate_goal_value(goals = np.array(self.discovered_goals[:n_goals_in_buckets])) 
        #         self.values_goals.append(value_estimations)
                
        #         # Computes values per bucket
        #         self.values_buckets.append(self.evaluate_buckets())

        #         # Compute LP
        #         self.update_lp()
            
        #     self.sync_curriculum()

        self.n_cycles += 1

        return episodes
    
    def update_lp(self):
        """ Updates the learning progress  """
        # If only one evaluation is conducted, then initialize sampling probabilities uniformly
        assert len(self.values_buckets) > 0, 'Cannot perform LP updates unless goal values are estimated'
        nb_buckets = len(self.values_buckets[0])
        if len(self.values_buckets) == 1:
            self.lp = np.zeros(nb_buckets)
            self.p = np.ones(nb_buckets) / nb_buckets
        else:
            self.lp = np.abs(self.values_buckets[-1] - self.values_buckets[-2])
            self.p = (1 - self.values_buckets[-1]) * self.lp / np.sum((1 - self.values_buckets[-1]) * self.lp)
        
        if self.p.sum() > 1:
                self.p[np.argmax(self.p)] -= self.p.sum() - 1
        elif self.p.sum() < 1:
            self.p[-1] = 1 - self.p[:-1].sum()

        stop = 1

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

                    # Fill bidict
                    # self.goal_str_to_oracle_id[str(last_ag)] = self.nb_discovered_goals

                    # Increment number of discovered goals (to increment the id !)
                    self.nb_discovered_goals += 1
        
        self.sync()

        # Label each episode by its last achieved goal
        # for e in episodes:
        #     last_ag = e['ag_binary'][-1]
        #     oracle_id = self.goal_str_to_oracle_id[str(last_ag)]
        #     e['last_ag_oracle_id'] = oracle_id

    def sync(self):
        """ Synchronize the goal sampler's attributes between all workers """
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)
        self.discovered_goals_oracle_ids = MPI.COMM_WORLD.bcast(self.discovered_goals_oracle_ids, root=0)
        self.goal_str_to_oracle_id = MPI.COMM_WORLD.bcast(self.goal_str_to_oracle_id, root=0)
        self.nb_discovered_goals = MPI.COMM_WORLD.bcast(self.nb_discovered_goals, root=0)
    
    def sync_curriculum(self):
        """ Synchronize values, LP and sampling probabilities across workers"""
        self.goal_buckets = MPI.COMM_WORLD.bcast(self.goal_buckets, root=0)
        self.buckets = MPI.COMM_WORLD.bcast(self.buckets, root=0)
        self.active_buckets_ids = MPI.COMM_WORLD.bcast(self.active_buckets_ids, root=0)
        self.values_goals = MPI.COMM_WORLD.bcast(self.values_goals, root=0)
        self.values_buckets = MPI.COMM_WORLD.bcast(self.values_buckets, root=0)
        self.lp = MPI.COMM_WORLD.bcast(self.lp, root=0)
        self.p = MPI.COMM_WORLD.bcast(self.p, root=0)

    def build_batch(self, batch_size):
        buckets = np.random.choice(self.active_buckets_ids, p=self.p, size=batch_size)
        goal_ids = []
        for b in buckets:
            goal_ids.append(np.random.choice(self.buckets[b]))
        assert len(goal_ids) == batch_size

        # Check whether or not to perform prioritized replay using curriculum learning 
        # If true, this will ignore the generated goal ids
        prioritized_replay = np.random.uniform() > self.epsilon
        return goal_ids, prioritized_replay

    def init_stats(self):
        self.stats = dict()
        # Number of classes of eval
        if self.goal_dim == 35:
            n = 11
        else:
            n = 6
        for i in np.arange(1, n+1):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['Av_Rew_{}'.format(i)] = []
        
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
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update', 'update_graph', 
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
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

    def generate_buckets(self, normalized_goal_values, granularity, equal_goal_repartition=False):

        #verify that we have as many goals in the goals values list as discovered goals
        assert len(normalized_goal_values) == len(self.discovered_goals)

        #verify that all values sum to 1
        assert (np.array(normalized_goal_values) <= 1.).any()

        intervals = np.linspace(0., 1., num=granularity+1)[1:] # remove the first item as the 0 is not useful

        if equal_goal_repartition:
            argsort = np.argsort(normalized_goal_values) # indexes of sorted array of goal values
            equal_split = np.array_split(argsort, granularity) # split it into equal length arrays
            self.goal_buckets = np.zeros(len(self.discovered_goals), dtype=int)
            # retrieve bucket index from split
            for bucket_id, bucket in enumerate(equal_split):
                for goal in bucket:
                    self.goal_buckets[goal] = bucket_id

        else:
            # split goals values in the intervals
            self.goal_buckets = np.searchsorted(intervals, normalized_goal_values)

        self.active_buckets_ids = np.unique(self.goal_buckets)
        buckets_dict = dict()

        for i, b in enumerate(self.goal_buckets):
            try: 
                buckets_dict[b].append(i)
            except KeyError:
                buckets_dict[b] = [i]
        return buckets_dict
    
    def evaluate_buckets(self):

        self.buckets_mean_value = []

        # retrieve goals from buckets, evaluate and average
        for bucket in self.active_buckets_ids:
            goal_ids = np.where(self.goal_buckets == bucket)[0]
            bucket_mean_value = np.mean(self.values_goals[-1][goal_ids])
            self.buckets_mean_value.append(bucket_mean_value)


        return np.array(self.buckets_mean_value)
    
    def save_bucket_contents(self, bucket_path, epoch):
        # save the contents of buckets
        with open(bucket_path + '/bucket_ep_{}.pkl'.format(epoch), 'wb') as f:
            pickle.dump(self.buckets, f)
        with open(bucket_path + '/discovered_g_ep_{}.pkl'.format(epoch), 'wb') as f:
            pickle.dump(self.discovered_goals, f)
