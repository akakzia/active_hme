from cv2 import norm
import torch
import numpy as np
from mpi4py import MPI


class GoalEvaluator():
    def __init__(self, args, policy=None, rollout_worker=None):
        assert args.goal_evaluator_method in [1, 2], 'Please select a valid evaluation method, only 1 and 2 are implemented !'
        assert args.normalization_technique in ['linear_fixed', 'linear_moving', 'mixed'], \
        'Please select a valid normalization technique from [linear_fixed, linear_moving, mixed]'

        self.method = args.goal_evaluator_method
        self.cuda = args.cuda
        self.normalization_technique = args.normalization_technique

        if self.method == 1:
            # Define the policy to 1) normalize; 2) evaluate goals.
            self.policy = policy

        elif self.method == 2:

            # add rollout_worker to estimate goal success rate
            self.rollout_worker = rollout_worker
            self.rank = MPI.COMM_WORLD.Get_rank()

    def setup_policy(self, policy):
        """ Sets up the policy """
        self.policy = policy

    def setup_rollout_worker(self, rollout_worker):
        """ Sets up the rollout worker """
        self.rollout_worker = rollout_worker

    def estimate_goal_value(self, goals):

        if self.method == 1:
            # Use value neural estimator to get goal values
            goal_values = self.forward_goal_values(goals)

        if self.method == 2:

            # evaluate policy on goals several time and average
            goal_values = []
            episodes = self.rollout_worker.generate_rollout(goals=np.array(goals), true_eval=True)
            goal_values_per_worker = np.array([e['success'][-1].astype(np.float32) for e in episodes])

            all_goal_values = MPI.COMM_WORLD.gather(goal_values_per_worker, root=0)

            if self.rank == 0:
                goal_values = np.mean(all_goal_values, axis=0)

            goal_values = MPI.COMM_WORLD.bcast(goal_values, root=0)


        # normalize goal values
        norm_g_values = self.normalize_goal_values(goal_values)

        return norm_g_values
    
    def forward_goal_values(self, goals):
        """ Normalize, tensorize and forward goals through the goal value estimator """
        n_goals = goals.shape[0]

        g_norm = self.policy.g_norm.normalize(goals)
        g_norm_tensor = torch.tensor(g_norm, dtype=torch.float32)
        if self.cuda:
            g_norm_tensor = g_norm_tensor.cuda()
        
        with torch.no_grad():
            self.policy.model.value_forward_pass(g_norm_tensor)
        if self.cuda:
            values = self.policy.model.value.cpu().numpy()
        else:
            values = self.policy.model.value.numpy()
        
        return values.squeeze()
    
    def normalize_goal_values(self, goal_values):
        """ Use the selected normalization technique to normalize goals """

        if self.normalization_technique == 'linear_fixed':
            if self.method == 1:
                max_value = 250
                min_value = 0
                norm_goals = (goal_values - min_value)/(max_value - min_value)
            elif self.method == 2:
                norm_goals = goal_values
        elif self.normalization_technique == 'linear_moving':
            max_value = np.max(goal_values, axis=0)
            min_value = np.min(goal_values, axis=0)
            norm_goals = (goal_values - min_value)/(max_value - min_value)
        elif self.normalization_technique == 'mixed':
            # Compute z-score
            mean_value = np.mean(goal_values, axis=0)
            std_value = np.std(goal_values, axis=0)
            z_scores = (goal_values - mean_value) / std_value

            # Compute linear moving normalization
            max_value = np.max(z_scores, axis=0)
            min_value = np.min(z_scores, axis=0)
            norm_goals = (z_scores - min_value) / (max_value - min_value)
        
        return norm_goals


