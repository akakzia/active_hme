import random
import numpy as np
from mpi4py import MPI
from graph.agent_graph import AgentGraph
import time

def is_success(ag, g):
    return (ag == g).all()

def at_least_one_fallen(observation, n):
    """ Given a observation, returns true if at least one object has fallen """
    dim_body = 10
    dim_object = 15
    obs_objects = np.array([observation[dim_body + dim_object * i: dim_body + dim_object * (i + 1)] for i in range(n)])
    obs_z = obs_objects[:, 2]

    return (obs_z < 0.4).any()



class RolloutWorker:
    def __init__(self, env, policy, args):

        self.env = env
        self.policy = policy
        self.env_params = args.env_params
        self.args = args
        self.goal_dim = args.env_params['goal']

    def generate_rollout(self, goals, true_eval, animated=False):

        episodes = []
        # Reset only once for all the goals in cycle if not performing evaluation
        if not true_eval:
            observation = self.env.unwrapped.reset_goal(goal=np.array(goals[0]))
        for i in range(goals.shape[0]):
            if true_eval:
                observation = self.env.unwrapped.reset_goal(goal=np.array(goals[i]))
            obs = observation['observation']
            ag = observation['achieved_goal']
            ag_bin = observation['achieved_goal_binary']
            g = observation['desired_goal']
            g_bin = observation['desired_goal_binary']

            ep_obs, ep_ag, ep_ag_bin, ep_g, ep_g_bin, ep_actions, ep_success, ep_rewards = [], [], [], [], [], [], [], []

            # Start to collect samples
            for t in range(self.env_params['max_timesteps']):
                # Run policy for one step
                no_noise = true_eval  # do not use exploration noise if running self-evaluations or offline evaluations
                # feed both the observation and mask to the policy module
                action = self.policy.act(obs.copy(), ag.copy(), g.copy(), no_noise)

                # feed the actions into the environment
                if animated:
                    self.env.render()

                observation_new, r, _, _ = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                ag_new_bin = observation_new['achieved_goal_binary']

                # Append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_ag_bin.append(ag_bin.copy())
                ep_g.append(g.copy())
                ep_g_bin.append(g_bin.copy())
                ep_actions.append(action.copy())
                ep_rewards.append(r)
                ep_success.append(is_success(ag_new, g))

                # Re-assign the observation
                obs = obs_new
                ag = ag_new
                ag_bin = ag_new_bin

            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_ag_bin.append(ag_bin.copy())

            # Gather everything
            episode = dict(obs=np.array(ep_obs).copy(),
                           act=np.array(ep_actions).copy(),
                           g=np.array(ep_g).copy(),
                           ag=np.array(ep_ag).copy(),
                           success=np.array(ep_success).copy(),
                           g_binary=np.array(ep_g_bin).copy(),
                           ag_binary=np.array(ep_ag_bin).copy(),
                           rewards=np.array(ep_rewards).copy())


            episodes.append(episode)

            # if not eval, make sure that no block has fallen 
            fallen = at_least_one_fallen(obs, self.args.n_blocks)
            if not true_eval and fallen:
                observation = self.env.unwrapped.reset_goal(goal=np.array(goals[i]))

        return episodes

class HMERolloutWorker(RolloutWorker):
    def __init__(self, env, policy, goal_sampler, args):
        super().__init__(env, policy, args) 
        # Agent memory to internalize SP intervention
        self.stepping_stones_beyond_pairs_list = []

        self.max_episodes = args.num_rollouts_per_mpi
        self.episode_duration = 100
        self.strategy = args.strategy

        # Define goal sampler
        self.goal_sampler = goal_sampler

        # Variable declaration
        self.last_obs = None
        self.long_term_goal = None
        self.current_goal_id = None
        self.last_episode = None
        self.dijkstra_to_goal = None
        self.state = None
        self.config_path = None

        # Resetting rollout worker
        self.reset()

        self.exploration_noise_prob = 0.1
    
    @property
    def current_config(self):
        return tuple(self.last_obs['achieved_goal'])

    def reset(self):
        self.long_term_goal = None
        self.config_path = None
        self.current_goal_id = None
        self.last_episode = None
        self.last_obs = self.env.unwrapped.reset_goal(goal=np.array([None]))
        self.dijkstra_to_goal = None
        # Internalization
        if len(self.stepping_stones_beyond_pairs_list) > 0:
            (self.internalized_ss, self.internalized_beyond) = random.choices(self.stepping_stones_beyond_pairs_list, k=1)[0]
        else:
            self.internalized_ss = None
            self.internalized_beyond = None
        if self.strategy == 3:
            self.state = 'Explore'
        else:
            self.state ='GoToFrontier'
    
    def test_rollout(self,goals,agent_network:AgentGraph,episode_duration, animated=False):
        end_episodes = []
        for goal in goals : 
            self.reset()
            _,last_episode = self.guided_rollout(goal,True, agent_network, episode_duration, animated=animated)
            end_episodes.append(last_episode)
        self.reset()
        return end_episodes

    def test_social_rollouts(self,goals,agent_network:AgentGraph,episode_duration, animated=False):
        end_episodes = []
        for goal in goals :
            self.reset()
            path, _, _ = agent_network.teacher.oracle_graph.sample_shortest_path(self.last_obs['achieved_goal'], goal,
                                                                                 algorithm=self.args.evaluation_algorithm)
            try:
                intermediate_goal = path[-2]
                episodes,_ = self.guided_rollout(intermediate_goal,True, agent_network, episode_duration, animated=animated)
                if len(episodes) < self.args.max_path_len:
                    last_episode = self.generate_one_rollout(goal, True, episode_duration, animated=animated)
                else:
                    last_episode = episodes[-1]
                    last_episode['success'] = False
            except:
                last_episode = self.generate_one_rollout(goal, True, episode_duration, animated=animated)
            end_episodes.append(last_episode)
        self.reset()
        return end_episodes

    def guided_rollout(self,goal,evaluation,agent_network:AgentGraph,episode_duration,episode_budget=None, animated=False):
        episodes = []
        goal = tuple(goal)

        if self.current_goal_id is None:
            self.plan(agent_network,goal,evaluation)

        if len(self.config_path) > self.args.max_path_len:
            self.config_path = [self.config_path[0]] + self.config_path[-self.args.max_path_len+1:]

        while True:
            current_goal,goal_dist = self.get_next_goal(agent_network,goal,evaluation)
                
            episode = self.generate_one_rollout(current_goal, evaluation, episode_duration, animated=animated)
            episodes.append(episode)
            self.current_goal_id+=1
            
            success = episodes[-1]['success'][-1]

            if animated:
                print(f'success ',success  )

            if self.current_goal_id == len(self.config_path):
                break
            if episode_budget is not None and len(episodes) >= episode_budget:
                break
            if not success:
                break 

        return episodes,self.last_episode

    def generate_one_rollout(self, goal,evaluation, episode_duration, animated=False):
        g = np.array(goal)
        self.env.unwrapped.target_goal = np.array(goal)
        self.env.unwrapped.binary_goal = np.array(goal)
        obs = self.last_obs['observation']
        ag = self.last_obs['achieved_goal']

        ep_obs, ep_ag, ep_g, ep_actions, ep_success, ep_rewards = [], [], [], [], [], []
        # Start to collect samples
        for _ in range(episode_duration):
            # Run policy for one step
            no_noise = evaluation  # do not use exploration noise if running self-evaluations or offline evaluations
            # feed both the observation and mask to the policy module
            action = self.policy.act(obs.copy(), ag.copy(), g.copy(), no_noise)

            # feed the actions into the environment
            if animated:
                self.env.render()

            observation_new, r, _, _ = self.env.step(action)
            obs_new = observation_new['observation']
            ag_new = observation_new['achieved_goal']

            # Append rollouts
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_g.append(g.copy())
            ep_actions.append(action.copy())
            ep_rewards.append(r)
            ep_success.append((ag_new == g).all())

            # Re-assign the observation
            obs = obs_new
            ag = ag_new

        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())

        # Gather everything
        episode = dict(obs=np.array(ep_obs).copy(),
                        act=np.array(ep_actions).copy(),
                        g=np.array(ep_g).copy(),
                        ag=np.array(ep_ag).copy(),
                        success=np.array(ep_success).copy(),
                        rewards=np.array(ep_rewards).copy(),
                        self_eval=evaluation)

        self.last_obs = observation_new
        self.last_episode = episode

        return episode

    def plan(self,agent_network,goal,evaluation):

        if evaluation : 
            self.current_goal_id = 1
            self.config_path,_,_ = agent_network.get_path(self.current_config,goal,algorithm = self.args.evaluation_algorithm)
            if not self.config_path:
                self.config_path = [self.current_config,goal]
        elif self.args.rollout_exploration =='sr_and_k_distance':
            self.current_goal_id = 1
            if  np.random.rand()< self.args.rollout_distance_ratio:
                k_best_paths,_ = agent_network.semantic_graph.k_shortest_path(self.current_config,goal,
                                                                                        self.args.rollout_exploration_k,
                                                                                        use_weights = False,
                                                                                        unordered_bias = self.args.unordered_edge)
                self.config_path = random.choices(k_best_paths,k=1)[0] if k_best_paths else None
            else:
                self.config_path,_,_ = agent_network.get_path(self.current_config,goal)
            if not self.config_path:
                self.config_path = [self.current_config,goal]
        elif self.args.rollout_exploration =='sr_and_best_distance':
            self.current_goal_id = 1
            if np.random.rand() < self.args.rollout_distance_ratio:
                self.config_path,_,_ = agent_network.get_path(self.current_config,goal,algorithm='dijkstra')
            else : 
                self.config_path,_,_ = agent_network.get_path(self.current_config,goal,algorithm='bfs')
            if not self.config_path:
                self.config_path = [self.current_config,goal]
        elif self.args.rollout_exploration=='sample_sr' :
            self.current_goal_id = 1
            self.dijkstra_to_goal = agent_network.semantic_graph.get_sssp_to_goal(goal)
        else : raise Exception('unknown exploration method',self.args.rollout_exploration) 
    
    def get_next_goal(self,agent_network,goal,evaluation):
        if evaluation : 
            current_goal = self.config_path[self.current_goal_id]
        elif self.args.rollout_exploration =='sr_and_k_distance':
            current_goal = self.config_path[self.current_goal_id]
        elif self.args.rollout_exploration =='sr_and_best_distance':
            self.plan(agent_network,goal,evaluation)
            current_goal = self.config_path[self.current_goal_id]
        elif self.args.rollout_exploration=='sample_sr' :
            current_goal = None
            if self.dijkstra_to_goal: 
                current_goal = agent_network.sample_neighbour_based_on_SR_to_goal(self.current_config,self.dijkstra_to_goal,goal=goal)
            if current_goal is None: # if no path to goal, try to reach directly
                current_goal = goal
        else : raise Exception('unknown exploration method') 

        return current_goal, self.current_goal_id

    def perform_social_episodes(self, agent_network, time_dict):
        """ Inputs: agent_network and time_dict
        Return a list of episode rollouts by the agent using social goals"""
        all_episodes = []
        for _ in range(self.args.num_rollouts_per_mpi):
            self.reset()
            current_episodes = []
            while len(current_episodes) < self.max_episodes:
                if self.state == 'GoToFrontier':
                    if self.long_term_goal is None:
                        t_i = time.time()
                        frontier_ag = [agent_network.semantic_graph.getConfig(i) for i in agent_network.teacher.agent_frontier]
                        self.long_term_goal = random.choices(frontier_ag)[0]
                        # self.long_term_goal = next(iter(agent_network.sample_goal_in_frontier(self.current_config, 1)), None)  # first element or None
                        if time_dict:
                            time_dict['goal_sampler'] += time.time() - t_i
                        # if can't find frontier goal, explore directly
                        if self.long_term_goal is None or (self.long_term_goal == self.current_config and self.strategy == 2):
                            self.state = 'Explore'
                            continue
                    no_noise = np.random.uniform() > self.exploration_noise_prob
                    # episodes, _ = self.guided_rollout(self.long_term_goal, no_noise, agent_network, self.episode_duration,
                    #                                   episode_budget=self.max_episodes - len(all_episodes))
                    episode = self.generate_one_rollout(self.long_term_goal, no_noise, self.episode_duration)
                    current_episodes.append(episode)

                    success = episode['success'][-1]
                    if success and self.current_config == self.long_term_goal and self.strategy == 2:
                        self.state = 'Explore'
                    else:
                        # Add stepping stone to agent's memory for internalization
                        # self.update_ss_list(self.long_term_goal, agent_network.semantic_graph.semantic_operation)
                        self.reset()

                elif self.state == 'Explore':
                    t_i = time.time()
                    # if strategy is Beyond, first sample goal in frontier than sample a goal beyond
                    # only propose the beyond goal
                    if self.strategy == 3:
                        last_ag = next(iter(agent_network.sample_goal_in_frontier(self.current_config, 1)), None)
                        if last_ag is None:
                            last_ag = tuple(self.last_obs['achieved_goal'])
                    else:
                        last_ag = tuple(self.last_obs['achieved_goal'])
                    explore_goal = next(iter(agent_network.sample_from_frontier(last_ag, 1)), None)  # first element or None
                    if time_dict is not None:
                        time_dict['goal_sampler'] += time.time() - t_i
                    if explore_goal:
                        episode = self.generate_one_rollout(explore_goal, False, self.episode_duration)
                        current_episodes.append(episode)
                        success = episode['success'][-1]
                        if not success and self.long_term_goal:
                            # Add pair to agent's memory
                            self.stepping_stones_beyond_pairs_list.append((self.long_term_goal, explore_goal))
                    if explore_goal is None or (not success and self.strategy !=3):
                        self.reset()
                        continue
                    # if strategy is Beyond and goal not reached, then keep performing rollout until budget ends
                    elif self.strategy == 3 and not success:
                        while not success and len(current_episodes) < self.max_episodes:
                            episode = self.generate_one_rollout(explore_goal, False, self.episode_duration)
                            current_episodes.append(episode)
                            success = episode['success'][-1]
                else:
                    raise Exception(f"unknown state : {self.state}")
            
            all_episodes.append(current_episodes)
        return all_episodes

    def internalize_social_episodes(self, agent_network, time_dict):
        """ Inputs: agent_network and time_dict
        Return a list of episode rollouts by the agent using memory of SP interventions"""
        all_episodes = []
        for _ in range(self.args.num_rollouts_per_mpi):
            self.reset()
            current_episodes = []
            while len(current_episodes) < self.max_episodes:
                if self.state == 'GoToFrontier':
                    no_noise = np.random.uniform() > self.exploration_noise_prob
                    # episodes, _ = self.guided_rollout(self.internalized_ss, no_noise, agent_network, self.episode_duration,
                    #                                 episode_budget=self.max_episodes - len(all_episodes))
                    # current_episodes += episodes
                    episode = self.generate_one_rollout(self.internalized_ss, no_noise, self.episode_duration)
                    current_episodes.append(episode)

                    success = episode['success'][-1]
                    if success and self.current_config == self.internalized_ss and self.strategy == 2:
                        self.state = 'Explore'
                    else:
                        self.reset()

                elif self.state == 'Explore':
                    t_i = time.time()
                    if time_dict is not None:
                        time_dict['goal_sampler'] += time.time() - t_i
                    episode = self.generate_one_rollout(self.internalized_beyond, False, self.episode_duration)
                    current_episodes.append(episode)
                else:
                    raise Exception(f"unknown state : {self.state}")
            all_episodes.append(current_episodes)

        return all_episodes

    def sync(self):
        """ Synchronize the list of pairs (stepping stone, Beyond) between all workers"""
        # Transformed to set to avoid duplicates
        self.stepping_stones_beyond_pairs_list = list(set(MPI.COMM_WORLD.allreduce(self.stepping_stones_beyond_pairs_list)))


    def train_rollout(self, agent_network, t, time_dict=None):
        if t > 10:
            # SP intervenes
            generated_episodes = self.perform_social_episodes(agent_network, time_dict)

            # Concatenate mini-episodes and perform data augmentation
            updated_episodes = []
            for episode in generated_episodes:
                merged_mini_episodes = {k: np.concatenate([v[:100], episode[1][k]]) for k, v in episode[0].items() if k!= 'self_eval'}
                updated_episodes.append(merged_mini_episodes)
            
            all_episodes = updated_episodes
            # Augment episodes by relabeling using the last goal
            if self.args.data_augmentation:
                relabeled_episodes = updated_episodes.copy()
                for i in range(len(relabeled_episodes)):
                    relabeled_episodes[i]['g'][:] = relabeled_episodes[i]['g'][-1]
                    relabeled_episodes[i]['her'] = False
                
                all_episodes = updated_episodes + relabeled_episodes
        else:
            # Autotelic phase
            # if t > 10 and len(self.stepping_stones_beyond_pairs_list) > 0:
            #     # internalize SP intervention
            #     generated_episodes = self.perform_social_episodes(agent_network, time_dict)

            #     # Concatenate mini-episodes and perform data augmentation
            #     updated_episodes = []
            #     for episode in generated_episodes:
            #         merged_mini_episodes = {k: np.concatenate([v[:100], episode[1][k]]) for k, v in episode[0].items() if k!= 'self_eval'}
            #         updated_episodes.append(merged_mini_episodes)
                
            #     all_episodes = updated_episodes
            #     # Augment episodes by relabeling using the last goal
            #     if self.args.data_augmentation:
            #         relabeled_episodes = updated_episodes.copy()
            #         for i in range(len(relabeled_episodes)):
            #             relabeled_episodes[i]['g'][:] = relabeled_episodes[i]['g'][-1]
            #             relabeled_episodes[i]['her'] = False
                    
            #         all_episodes = updated_episodes + relabeled_episodes
            # else:
            t_i = time.time()
            goals = self.goal_sampler.sample_goal(n_goals=self.args.num_rollouts_per_mpi, evaluation=False)
            time_dict['goal_sampler'] += time.time() - t_i
            all_episodes = self.generate_rollout(goals=goals,  # list of goal configurations
                                                    true_eval=False,  # these are not offline evaluation episodes
                                                )
        self.sync()
        return all_episodes

