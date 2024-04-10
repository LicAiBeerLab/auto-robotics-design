import numpy as np

from auto_robot_design.optimization.rewards.reward_base import Reward

class HeavyLiftingReward(Reward):
    """Calculate the mass that can be held still using up to 70% of the motor capacity

    Args:
        Reward (float): mass capacity
    """
    def __init__(self, manipulability_key, mass_key, trajectory_key, error_key, max_effort_coef = 0.7) -> None:
        self.max_effort_coefficient = max_effort_coef
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.error_key = error_key
        self.mass_key = mass_key

    def calculate(self, point_criteria, trajectory_criteria, trajectory_results, **kwargs) -> float:
        """_summary_

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Raises:
            KeyError: this function requires motor description

        Returns:
            float: value of the reward
        """
        if "Actuator" in kwargs:
            pick_effort = kwargs["Actuator"].peak_effort
        else:
            raise KeyError("Lifting criterion requires the Actuator")

        manipulability_matrices: list[np.array] = point_criteria[self.manip_key]
        trajectory_points = trajectory_results[self.trajectory_key]
        errors = trajectory_results[self.error_key]
        mass = trajectory_criteria[self.mass_key]
        n_steps = len(trajectory_points)
        result = float('inf')
        for i in range(n_steps):
            if errors[i] > 1e-6:
                return 0 # if at least one point is not reachable the total reward is 0

            force_matrix = np.transpose(manipulability_matrices[i]) # maps forces to torques
            zrr = np.abs(force_matrix@np.array([0,1]))
            force_z = pick_effort*self.max_effort_coefficient/max(zrr)
            additional_force = abs(force_z) - 10*mass
            if additional_force < 0:
                return 0
            if additional_force < result:
                result = additional_force

        return result/10