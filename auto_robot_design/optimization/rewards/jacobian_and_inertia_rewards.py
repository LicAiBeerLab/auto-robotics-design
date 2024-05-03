import numpy as np

from auto_robot_design.optimization.rewards.reward_base import Reward

GRAVITY = 9.81

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
        reward_vector = []
        for i in range(n_steps):
            if errors[i] > 1e-6:
                return 0,  [] # if at least one point is not reachable the total reward is 0

            force_matrix = np.transpose(manipulability_matrices[i]) # maps forces to torques
            zrr = np.abs(force_matrix@np.array([0,1]))
            force_z = pick_effort*self.max_effort_coefficient/max(zrr)
            additional_force = abs(force_z) - GRAVITY*mass
            reward_vector.append(additional_force/GRAVITY)
            if additional_force < 0:
                return 0, []
            if additional_force < result:
                result = additional_force

        return result/GRAVITY, reward_vector


class AccelerationCapability(Reward):
    """Calculate the reward that combine effective inertia and force capability
    """
    def __init__(self, manipulability_key, trajectory_key, error_key, actuated_mass_key, max_effort_coef = 0.7) -> None:
        self.max_effort_coefficient = max_effort_coef
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.error_key = error_key
        self.actuated_mass_key = actuated_mass_key
    
    def calculate(self, point_criteria, trajectory_criteria, trajectory_results, **kwargs) -> float:
        """_summary_

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            float: value of the reward
        """
        if "Actuator" in kwargs:
            pick_effort = kwargs["Actuator"].peak_effort
        else:
            raise KeyError("Lifting criterion requires the Actuator")

        # get the manipulability for each point at the trajectory
        manipulability_matrices: list[np.array] = point_criteria[self.manip_key]
        effective_mass_matrices:list[np.array] = point_criteria[self.actuated_mass_key]
        trajectory_points = trajectory_results[self.trajectory_key]
        errors = trajectory_results[self.error_key]
        diff_vector = np.diff(trajectory_points, axis=0)[:, [0, 2]]
        n_steps = len(trajectory_points)
        result = 0
        reward_vector = []
        for i in range(n_steps-1):
            # the reward is none zero only if the point is reached
            if errors[i] > 1e-6:
                return 0, []
            # get the direction of the trajectory
            # trajectory_shift = np.array([trajectory_points[i+1][0]-trajectory_points[i][0], trajectory_points[i+1][2]-trajectory_points[i][2]])
            trajectory_shift = diff_vector[i]
            trajectory_direction = trajectory_shift / \
                np.linalg.norm(trajectory_shift)

            # get the manipulability matrix for the current point
            manipulability_matrix: np.array = manipulability_matrices[i]
            effective_mass_matrix: np.array = effective_mass_matrices[i]
            acc_2_torque = effective_mass_matrix@np.linalg.inv(manipulability_matrix)
            unit_acc_torque = np.abs(acc_2_torque@trajectory_direction)
            acc= pick_effort*self.max_effort_coefficient/max(unit_acc_torque)
            result+=acc
            reward_vector.append(acc)

        if errors[n_steps-1] > 1e-6:
             return 0, []

        return result/n_steps, reward_vector
