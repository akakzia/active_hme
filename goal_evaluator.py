from cv2 import norm
import torch
import numpy as np


class GoalEvaluator():
    def __init__(self, args, policy=None, rollout_worker=None):
        assert args.normalization_technique in ['linear_fixed', 'linear_moving', 'mixed'], \
        'Please select a valid normalization technique from [linear_fixed, linear_moving, mixed]'

        self.cuda = args.cuda
        self.normalization_technique = args.normalization_technique

        # Define the policy to 1) normalize; 2) evaluate goals.
        self.policy = policy

    def setup_policy(self, policy):
        """ Sets up the policy """
        self.policy = policy

    def estimate_goal_value(self, goals, ag=None):
        # If no initial goals are given, then estimate value starting from all coplanar
        ag = - np.ones(goals.shape).astype(np.float) if ag is None else ag
        # Use value neural estimator to get goal values
        goal_values = self.forward_goal_values(ag, goals)

        # normalize goal values
        # normalized_goal_values = goal_values/np.max(goal_values)
        norm_g_values = self.normalize_goal_values(goal_values)

        return norm_g_values
    
    def forward_goal_values(self, ag, goals):
        """ Normalize, tensorize and forward goals through the goal value estimator """
        ag_tensor = torch.tensor(ag, dtype=torch.float32)
        g_tensor = torch.tensor(goals, dtype=torch.float32)
        if self.cuda:
            ag_tensor = ag_tensor.cuda()
            g_tensor = g_tensor.cuda()
        
        with torch.no_grad():
            self.policy.model.value_forward_pass(ag_tensor, g_tensor)
        if self.cuda:
            values = self.policy.model.value.cpu().numpy()
        else:
            values = self.policy.model.value.numpy()
        
        return values.squeeze()
    
    def normalize_goal_values(self, goal_values):
        """ Use the selected normalization technique to normalize goals """

        if self.normalization_technique == 'linear_fixed':
            max_value = 250
            min_value = 0
            norm_goals = np.clip((goal_values - min_value)/(max_value - min_value), a_min=0., a_max=1.)
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


