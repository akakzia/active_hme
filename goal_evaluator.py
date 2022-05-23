import numpy as np


class GoalEvaluator():
    def __init__(self, method, goal_value_estimator=None, rollout_worker=None):

        self.method = method

        if self.method == 1:

            # add neural network to estimate goal value
            self.goal_value_estimator = goal_value_estimator

        elif self.method == 2:

            # add rollout_worker to estimate goal success rate
            self.rollout_worker = rollout_worker

    def estimate_goal_value(self, goals):

        if self.method == 1:

            # add forward pass to estimate goal values

            # temporary
            goal_values = np.random.rand(len(goals))

        if self.method == 2:

            # add rollout on goals to estimate success rate

            # temporary
            goal_values = np.random.rand(len(goals))

        # normalize goal values
        normalized_goal_values = goal_values/np.max(goal_values)

        return normalized_goal_values
