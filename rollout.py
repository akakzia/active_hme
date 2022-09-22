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

        self.agent = args.agent
        
        if self.agent == 'HME' or self.agent == 'F1andRandom': 
            self.frontier_type = 'f1'
        elif self.agent == 'F2andRandom':
            self.frontier_type = 'f2'
        elif self.agent == 'F3andRandom':
            self.frontier_type = 'f3'
        elif self.agent == 'UniformandRandom':
            self.frontier_type = 'f4'

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

        self.nb_internalized_pairs = 0

        self.max_episodes = args.num_rollouts_per_mpi
        self.episode_duration = 100

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
        self.state ='GoToFrontier'

    def generate_one_rollout(self, goal,evaluation, episode_duration, animated=False, random=False):
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
            if random:
                action = self.env.action_space.sample()
            else:
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
    
    def sample_goal_in_frontier(self, network, frontier='f1'):
        """ Return a goal sampled in one of the following frontiers:
        f1: frontier with reference to SP
        f2: frontier with ref to the number of visits
        f3: frontier with reference to competence
        """
        if self.long_term_goal is None:
            if frontier == 'f1':
                # Sample goal in the frontier of agent's exploration with reference to SP's model of the goal space
                t_i = time.time()
                frontier_ag = [network.semantic_graph.getConfig(i) for i in network.teacher.agent_frontier]
                self.long_term_goal = random.choices(frontier_ag)[0]
                if self.long_term_goal is not None:
                    network.teacher.ss_interventions += 1
                    self.long_term_goal = apply_on_table_config(self.long_term_goal)
                time_sample = time.time() - t_i
            elif frontier == 'f2':
                # Sample goal in regions that are sparsely visited 
                t_i = time.time()
                self.long_term_goal = self.goal_sampler.sample_rare_goal()
                time_sample = time.time() - t_i
            elif frontier == 'f3':
                # Sample goal that is neither too hard nor too simple (frontier of competence)
                t_i = time.time()
                self.long_term_goal = self.goal_sampler.sample_intermediate_complexity_goal()
                time_sample = time.time() - t_i
            elif frontier == 'f4':
                # Sample goal uniformly from the set of discovered goals then continue with random actions
                t_i = time.time()
                self.long_term_goal = self.goal_sampler.sample_uniform_goal()
                time_sample = time.time() - t_i
            else:
                raise NotImplementedError


            return self.long_term_goal, time_sample
            
    def perform_social_episodes(self, agent_network, time_dict):
        """ Inputs: agent_network and time_dict
        Return a list of episode rollouts by the agent using social goals"""
        all_episodes = []
        for _ in range(self.args.num_rollouts_per_mpi):
            self.reset()
            current_episodes = []
            while len(current_episodes) < self.max_episodes:
                if self.state == 'GoToFrontier':
                    self.long_term_goal, time_sample = self.sample_goal_in_frontier(agent_network, frontier=self.frontier_type)
                    if time_dict:
                        time_dict['goal_sampler'] += time_sample
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
                    if self.agent == 'HME':
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
                        # Perform random actions, no goal needed here
                        episode = self.generate_one_rollout(self.current_config, False, self.episode_duration)
                        current_episodes.append(episode)
                else:
                    raise Exception(f"unknown state : {self.state}")
            
            all_episodes.append(current_episodes)
        return all_episodes

    def launch_social_phase(self, agent_network, time_dict):
        """ Launch the social episodes phase: 
        First, produce rollouts. Then, concatenate obtained rollouts and relabel based on the beyond goal"""
        if self.agent == 'Beyond':
            # Sample beyond goal 
            if len(agent_network.teacher.agent_frontier) > 0:
                all_frontier = [agent_network.semantic_graph.getConfig(i) for i in agent_network.teacher.agent_frontier]
                frontier_goals = random.choices(all_frontier, k=self.args.num_rollouts_per_mpi)
                goals = np.array([next(iter(agent_network.sample_from_frontier(g, 1)), None) for g in frontier_goals])
                goals = np.array([apply_on_table_config(g) if g else apply_on_table_config(fg) for g, fg in zip(goals, frontier_goals)])
            else:
                goals = self.goal_sampler.sample_goals(n_goals=self.args.num_rollouts_per_mpi, evaluation=False)
            all_episodes = self.generate_rollout(goals=goals,  # list of goal configurations
                                            true_eval=False,  # these are not offline evaluation episodes
                                            )
        else:
            generated_episodes = self.perform_social_episodes(agent_network, time_dict)

            all_episodes = merge_mini_episodes_and_relabel(generated_episodes)
        return all_episodes

    def launch_autotelic_phase(self, time_dict):
        """ Launch the autotelic episodes phase
        First sample goals. Than, compute the values of the goals. If value less than a threshold, than perform planning by 
        rehearsing the frontier/beyond procedure """
        # Perform uniform autotelic episodes
        t_i = time.time()
        if np.random.uniform() < self.args.eps_uniform_goal or len(self.goal_sampler.discovered_goals) == 0:
            goals = self.goal_sampler.sample_goals(n_goals=self.args.num_rollouts_per_mpi, evaluation=False)
        else:
            if self.agent == 'LPAgent':
                goals = self.goal_sampler.sample_lp_goals(n_goals=self.args.num_rollouts_per_mpi)
            elif self.agent == 'VDSAgent':
                goals = self.goal_sampler.sample_vds_goals(initial_obs=self.last_obs ,n_goals=self.args.num_rollouts_per_mpi)
        time_dict['goal_sampler'] += time.time() - t_i
        all_episodes = self.generate_rollout(goals=goals,  # list of goal configurations
                                            true_eval=False,  # these are not offline evaluation episodes
                                            )
        return all_episodes

    def sync(self):
        """ Synchronize the list of pairs (stepping stone, Beyond) between all workers"""
        # Transformed to set to avoid duplicates
        if self.args.beta > 0:
            self.stepping_stones_beyond_pairs_list = list(set(MPI.COMM_WORLD.allreduce(self.stepping_stones_beyond_pairs_list)))
            self.beyond_list = list(set(MPI.COMM_WORLD.allreduce(self.beyond_list)))

            # Syncronize counts
            self.nb_internalized_pairs = len(self.stepping_stones_beyond_pairs_list)

            # Send internalized goals to goal sampler
            self.goal_sampler.update_internalization(ss_b_pairs=self.stepping_stones_beyond_pairs_list, b=self.beyond_list)
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.goal_sampler.stats['nb_internalized_pairs'].append(self.nb_internalized_pairs)
            self.goal_sampler.stats['query_proba'].append(self.goal_sampler.query_proba)


    def train_rollout(self, agent_network, epoch, time_dict=None):
        """ Run one rollout pass of self.args.num_rollouts_per_mpi episodes """

        # First, select episode type based on 1) the query proba and 2) whether or not there are remaining stepping stones to propose. 
        episodes_type = 'social' if np.random.uniform() < self.goal_sampler.query_proba and len(agent_network.teacher.agent_stepping_stones) > 0 else 'autotelic'

        if episodes_type == 'social' and epoch > self.args.n_freeplay_epochs: 
            all_episodes = self.launch_social_phase(agent_network, time_dict)
        else:
            all_episodes = self.launch_autotelic_phase(time_dict)
        
        self.sync()

        return all_episodes, episodes_type
