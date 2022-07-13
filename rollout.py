import random
import numpy as np
from mpi4py import MPI
from graph.agent_graph import AgentGraph
from utils import apply_on_table_config, merge_mini_episodes_and_relabel
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

                if true_eval and r == self.args.n_blocks:
                    # When performing offline evaluations, stop episode once the goal is reached
                    break

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
        self.beyond_list = [] # used in autotelic planning

        # List from which to remove when internalization is succeeded
        # self.to_remove_internalization = []

        self.nb_internalized_pairs = 0

        self.max_episodes = args.num_rollouts_per_mpi
        self.episode_duration = 100

        self.autotelic_planning_proba = args.autotelic_planning_proba

        self.internalization_strategy = args.internalization_strategy


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
        # if len(self.stepping_stones_beyond_pairs_list) > 0:
        #     (self.internalized_ss, self.internalized_beyond) = random.choices(self.stepping_stones_beyond_pairs_list, k=1)[0]
        # else:
        #     self.internalized_ss = None
        #     self.internalized_beyond = None
        self.state ='GoToFrontier'

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
                        rewards=np.array(ep_rewards).copy())

        self.last_obs = observation_new
        self.last_episode = episode

        return episode 

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
                        # self.long_term_goal = next(iter(agent_network.sample_goal_in_frontier(self.current_config[:30], 1)), None)
                        if self.long_term_goal is not None:
                            agent_network.teacher.ss_interventions += 1
                            self.long_term_goal = apply_on_table_config(self.long_term_goal)
                        if time_dict:
                            time_dict['goal_sampler'] += time.time() - t_i
                        # if can't find frontier goal, explore directly
                        if self.long_term_goal is None or (self.long_term_goal == self.current_config):
                            self.state = 'Explore'
                            continue
                    no_noise = np.random.uniform() > self.exploration_noise_prob
                    episode = self.generate_one_rollout(self.long_term_goal, no_noise, self.episode_duration)
                    current_episodes.append(episode)

                    success = episode['success'][-1]
                    if success and self.current_config == self.long_term_goal:
                        self.state = 'Explore'
                    else:
                        self.reset()

                elif self.state == 'Explore':
                    t_i = time.time()
                    last_ag = tuple(self.last_obs['achieved_goal'][:30])
                    explore_goal = next(iter(agent_network.sample_from_frontier(last_ag, 1)), None)  # first element or None
                    if time_dict is not None:
                        time_dict['goal_sampler'] += time.time() - t_i
                    if explore_goal:
                        explore_goal = apply_on_table_config(explore_goal)
                        episode = self.generate_one_rollout(explore_goal, False, self.episode_duration)
                        current_episodes.append(episode)
                        success = episode['success'][-1]
                        if not success and self.long_term_goal:
                            # Add pair to agent's memory
                            self.stepping_stones_beyond_pairs_list.append((self.long_term_goal, explore_goal))
                            self.beyond_list.append(explore_goal)
                    if explore_goal is None or not success:
                        self.reset()
                        continue
                else:
                    raise Exception(f"unknown state : {self.state}")
            
            all_episodes.append(current_episodes)
        return all_episodes

    def internalize_social_episodes(self, time_dict):
        """ Inputs: agent_network and time_dict
        Return a list of episode rollouts by the agent using memory of SP interventions"""
        all_episodes = []
        for _ in range(self.args.num_rollouts_per_mpi):
            self.reset()
            current_episodes = []
            while len(current_episodes) < self.max_episodes:
                if self.state == 'GoToFrontier':
                    no_noise = np.random.uniform() > self.exploration_noise_prob
                    episode = self.generate_one_rollout(self.internalized_ss, no_noise, self.episode_duration)
                    current_episodes.append(episode)

                    success = episode['success'][-1]
                    if success and self.current_config == self.internalized_ss:
                        self.state = 'Explore'
                    else:
                        self.reset()

                elif self.state == 'Explore':
                    t_i = time.time()
                    if time_dict is not None:
                        time_dict['goal_sampler'] += time.time() - t_i
                    episode = self.generate_one_rollout(self.internalized_beyond, False, self.episode_duration)
                    current_episodes.append(episode)
                    success = episode['success'][-1]
                    # if success:
                    #     self.to_remove_internalization.append((self.internalized_ss, self.internalized_beyond))
                else:
                    raise Exception(f"unknown state : {self.state}")
            all_episodes.append(current_episodes)

        return all_episodes
    
    def autotelic_planning(self, intermediate_goals, final_goals, evaluation=False, animated=False):
        """ The agent uses the generated intermediate goals to pursue the sampled final goals """
        all_episodes = []
        for i_g, f_g in zip(intermediate_goals, final_goals):
            self.reset()
            current_episodes = []
            while len(current_episodes) < 2:
                if self.state == 'GoToFrontier':
                    episode = self.generate_one_rollout(i_g, True, self.episode_duration, animated=animated)
                    current_episodes.append(episode)

                    success = episode['success'][-1]
                    if success and self.current_config == tuple(i_g):
                        self.state = 'Explore'
                    else:
                        self.reset()

                elif self.state == 'Explore':
                    episode = self.generate_one_rollout(f_g, evaluation, self.episode_duration, animated=animated)
                    current_episodes.append(episode)
                    success = episode['success'][-1]
                else:
                    raise Exception(f"unknown state : {self.state}")
            all_episodes.append(current_episodes)

        # Concatenate mini-episodes and perform data augmentation
        updated_episodes = []
        for episode in all_episodes:
            merged_mini_episodes = {k: np.concatenate([v[:100], episode[1][k]]) for k, v in episode[0].items()}
            # Relabel mini episodes according to final goal
            merged_mini_episodes['g'][:] = merged_mini_episodes['g'][-1]
            updated_episodes.append(merged_mini_episodes)

        return updated_episodes

    def launch_social_phase(self, agent_network, time_dict):
        """ Launch the social episodes phase: 
        1/ If there are some remaining (stepping stones, beyond) from internalization, than the agent selects to rehearse
        2/ If not than ask social partner """
        # Check the list of internalized pairs
        # If list is not empty, than rehearse social interventions
        # Else, ask social partner
        # if len(self.stepping_stones_beyond_pairs_list) > 0 and self.args.beta > 0 and self.attention_to_internalized_pairs:
        #     # internalize SP pairs
        #     generated_episodes = self.internalize_social_episodes(time_dict)
        # else:
        #     # SP intervenes
        generated_episodes = self.perform_social_episodes(agent_network, time_dict)

        all_episodes = merge_mini_episodes_and_relabel(generated_episodes)
        return all_episodes

    def launch_autotelic_phase(self, time_dict):
        """ Launch the autotelic episodes phase
        First sample goals. Than, compute the values of the goals. If value less than a threshold, than perform planning by 
        rehearsing the frontier/beyond procedure """
        # if np.random.uniform() < self.autotelic_planning_proba and len(self.beyond_list) > 0:
        #     t_i = time.time()
        #     norm_goals = self.goal_sampler.goal_evaluator.estimate_goal_value(goals=np.array(self.beyond_list))
        #     # Take indexes of goals with least value
        #     ind = np.argsort(norm_goals)[:self.args.num_rollouts_per_mpi]
        #     if len(ind) == 1:
        #         ind = np.repeat(ind, repeats=self.args.num_rollouts_per_mpi, axis=0)
        #     goals = np.array([self.beyond_list[i] for i in ind])
        #     time_dict['goal_sampler'] += time.time() - t_i
        #     # generate intermediate goals 
        #     intermediate_goals = self.goal_sampler.generate_intermediate_goals(goals)
        #     all_episodes = self.autotelic_planning(intermediate_goals, goals)
        # else:
        
        # Decide whether to do internalization
        do_internalization, goal_ids, proba_query_intern= self.goal_sampler.do_internalization()
        if do_internalization:
            # perform internalization
            if self.internalization_strategy == 2:
                self.internalized_ss, self.internalized_beyond = self.stepping_stones_beyond_pairs_list[goal_ids[0]]
                generated_episodes = self.internalize_social_episodes(time_dict)

                all_episodes = merge_mini_episodes_and_relabel(generated_episodes)
            elif self.internalization_strategy == 3:
                goals = np.array(self.beyond_list)[goal_ids]
                all_episodes = self.generate_rollout(goals=goals,  # list of goal configurations
                                                true_eval=False,  # these are not offline evaluation episodes
                                                )
        else:
            # Perform uniform autotelic episodes
            t_i = time.time()
            goals = self.goal_sampler.sample_goals(n_goals=self.args.num_rollouts_per_mpi, evaluation=False)
            time_dict['goal_sampler'] += time.time() - t_i
            all_episodes = self.generate_rollout(goals=goals,  # list of goal configurations
                                                true_eval=False,  # these are not offline evaluation episodes
                                                )
        return all_episodes, proba_query_intern

    def sync(self, p_query_intern):
        """ Synchronize the list of pairs (stepping stone, Beyond) between all workers"""
        # Transformed to set to avoid duplicates
        if self.args.beta > 0:
            self.stepping_stones_beyond_pairs_list = list(set(MPI.COMM_WORLD.allreduce(self.stepping_stones_beyond_pairs_list)))
            self.beyond_list = list(set(MPI.COMM_WORLD.allreduce(self.beyond_list)))
            # self.to_remove_internalization = list(set(MPI.COMM_WORLD.allreduce(self.to_remove_internalization)))

            # Remove elements that were successfully internalized
            # self.stepping_stones_beyond_pairs_list = [e for e in self.stepping_stones_beyond_pairs_list if e not in self.to_remove_internalization]
            # self.to_remove_internalization = []

            # Syncronize counts
            self.nb_internalized_pairs = len(self.stepping_stones_beyond_pairs_list)

            # Send internalized goals to goal sampler
            self.goal_sampler.update_internalization(ss_b_pairs=self.stepping_stones_beyond_pairs_list, b=self.beyond_list)
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.goal_sampler.stats['nb_internalized_pairs'].append(self.nb_internalized_pairs)
            self.goal_sampler.stats['query_proba'].append(self.goal_sampler.query_proba)
            self.goal_sampler.stats['query_proba_intern'].append(p_query_intern)


    def train_rollout(self, agent_network, epoch, time_dict=None):
        """ Run one rollout pass of self.args.num_rollouts_per_mpi episodes """

        # First, select episode type based on 1) the query proba and 2) whether or not there are remaining stepping stones to propose. 
        episodes_type = 'social' if np.random.uniform() < self.goal_sampler.query_proba and len(agent_network.teacher.agent_stepping_stones) > 0 else 'autotelic'
        
        if episodes_type == 'social' and epoch > self.args.n_freeplay_epochs: 
            all_episodes = self.launch_social_phase(agent_network, time_dict)
        else:
            all_episodes, proba_query_intern = self.launch_autotelic_phase(time_dict)
        
        self.sync(proba_query_intern)

        return all_episodes, episodes_type
