from typing import Tuple

import numpy as np

from auto_robot_design.optimization.rewards.reward_base import Reward
from auto_robot_design.pinokla.calc_criterion import DataDict

GRAVITY = 9.81


class HeavyLiftingReward(Reward):
    """Calculate the mass that can be held still using up to 70% of the motor capacity

    Args:
        Reward (float): mass capacity
    """

    def __init__(self, manipulability_key, mass_key: str, trajectory_key: str, error_key: str, max_effort_coef=0.7) -> None:
        super().__init__()
        self.max_effort_coefficient = max_effort_coef
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.error_key = error_key
        self.mass_key = mass_key

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
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

        errors = trajectory_results[self.error_key]
        is_trajectory_reachable = self.check_reachability(errors)
        # the reward is none zero only if the point is reached
        if not is_trajectory_reachable:
            return 0, []

        manipulability_matrices: list[np.array] = point_criteria[self.manip_key]
        trajectory_points = trajectory_results[self.trajectory_key]
        mass = trajectory_criteria[self.mass_key]
        n_steps = len(trajectory_points)
        result = float('inf')
        reward_vector = [0]*n_steps
        for i in range(n_steps):
            # the force matrix is the transpose of Jacobian, it transforms forces into torques
            force_matrix = np.transpose(manipulability_matrices[i])
            # calculate torque vector that is required to get unit force in the z direction.
            # it also declares the ratio of torques that provides z-directed force
            z_unit_force_torques = np.abs(force_matrix@np.array([0, 1]))
            # calculate the factor that max out the higher torque
            achievable_force_z = pick_effort * \
                self.max_effort_coefficient/max(z_unit_force_torques)
            # calculate extra force that can be applied to the payload
            additional_force = abs(achievable_force_z) - GRAVITY*mass
            reward_vector[i] = additional_force/GRAVITY
            # if at some point the force is not enough to even list leg itself, return 0
            if additional_force < 0:
                return 0, []
            # get the minimum force that can be applied to the payload
            if additional_force < result:
                result = additional_force

        return result/GRAVITY, reward_vector


class AccelerationCapability(Reward):
    """Calculate the reward that combine effective inertia and force capability
    """

    def __init__(self, manipulability_key: str, trajectory_key: str, error_key: str, actuated_mass_key: str, max_effort_coef=0.7) -> None:
        super().__init__()
        self.max_effort_coefficient = max_effort_coef
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.error_key = error_key
        self.actuated_mass_key = actuated_mass_key

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
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

        errors = trajectory_results[self.error_key]
        is_trajectory_reachable = self.check_reachability(errors)
        # the reward is none zero only if the point is reached
        if not is_trajectory_reachable:
            return 0, []

        # get the manipulability for each point at the trajectory
        manipulability_matrices: list[np.array] = point_criteria[self.manip_key]
        effective_mass_matrices: list[np.array] = point_criteria[self.actuated_mass_key]
        trajectory_points = trajectory_results[self.trajectory_key]
        diff_vector = np.diff(trajectory_points, axis=0)[:, [0, 2]]
        n_steps = len(trajectory_points)
        reward_vector = [0]*(n_steps-1)
        for i in range(n_steps-1):
            # get the direction of the trajectory
            trajectory_shift = diff_vector[i]
            trajectory_direction = trajectory_shift / \
                np.linalg.norm(trajectory_shift)

            # get the manipulability matrix and mass matrix for the current point
            manipulability_matrix: np.array = manipulability_matrices[i]
            effective_mass_matrix: np.array = effective_mass_matrices[i]
            # calculate the matrix that transforms quasi-static acceleration to required torque
            acc_2_torque = effective_mass_matrix@np.linalg.inv(
                manipulability_matrix)
            # calculate tthe torque vector that provides the unit acceleration in the direction of the trajectory
            unit_acc_torque = np.abs(acc_2_torque@trajectory_direction)
            # calculate the factor that max out the higher torque
            acc = pick_effort*self.max_effort_coefficient/max(unit_acc_torque)
            reward_vector[i] = acc

        return np.mean(np.array(reward_vector)), reward_vector


class MeanHeavyLiftingReward(Reward):
    """Calculate the mass that can be held still using up to 70% of the motor capacity

    Args:
        Reward (float): mass capacity
    """

    def __init__(self, manipulability_key, mass_key: str, trajectory_key: str, error_key: str, max_effort_coef=0.7) -> None:
        super().__init__()
        self.max_effort_coefficient = max_effort_coef
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.error_key = error_key
        self.mass_key = mass_key

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
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

        errors = trajectory_results[self.error_key]
        is_trajectory_reachable = self.check_reachability(errors)
        # the reward is none zero only if the point is reached
        if not is_trajectory_reachable:
            return 0, []

        manipulability_matrices: list[np.array] = point_criteria[self.manip_key]
        trajectory_points = trajectory_results[self.trajectory_key]
        mass = trajectory_criteria[self.mass_key]
        n_steps = len(trajectory_points)
        reward_vector = [0]*n_steps
        for i in range(n_steps):
            # the force matrix is the transpose of Jacobian, it transforms forces into torques
            force_matrix = np.transpose(manipulability_matrices[i])
            # calculate torque vector that is required to get unit force in the z direction.
            # it also declares the ratio of torques that provides z-directed force
            z_unit_force_torques = np.abs(force_matrix@np.array([0, 1]))
            # calculate the factor that max out the higher torque
            achievable_force_z = pick_effort * \
                self.max_effort_coefficient/max(z_unit_force_torques)
            # calculate extra force that can be applied to the payload
            additional_force = abs(achievable_force_z) - GRAVITY*mass
            reward_vector[i] = additional_force/GRAVITY

        return np.mean(np.array(reward_vector)), reward_vector


class MinAccelerationCapability(Reward):
    """Calculate the reward that combine effective inertia and force capability
    """

    def __init__(self, manipulability_key: str, trajectory_key: str, error_key: str, actuated_mass_key: str, max_effort_coef=0.7) -> None:
        super().__init__()
        self.max_effort_coefficient = max_effort_coef
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.error_key = error_key
        self.actuated_mass_key = actuated_mass_key

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
        """_summary_

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            float: value of the reward
        """
        # if "Actuator" in kwargs:
        #     pick_effort = kwargs["Actuator"].peak_effort
        # else:
        #     raise KeyError("Lifting criterion requires the Actuator")

        errors = trajectory_results[self.error_key]
        is_trajectory_reachable = self.check_reachability(errors)
        # the reward is none zero only if the point is reached
        if not is_trajectory_reachable:
            return 0, []

        # get the manipulability for each point at the trajectory
        manipulability_matrices: list[np.array] = point_criteria[self.manip_key]
        effective_mass_matrices: list[np.array] = point_criteria[self.actuated_mass_key]

        n_steps = len(errors)
        reward_vector = [0]*(n_steps)
        for i in range(n_steps):
            # get the manipulability matrix and mass matrix for the current point
            manipulability_matrix: np.array = manipulability_matrices[i]
            effective_mass_matrix: np.array = effective_mass_matrices[i]
            # calculate the matrix that transforms quasi-static acceleration to required torque

            torque_2_acc = manipulability_matrix@np.linalg.inv(
                effective_mass_matrix)
            step_result = np.min(abs(np.linalg.eigvals(torque_2_acc)))
            reward_vector[i] = step_result

        return np.mean(np.array(reward_vector)), reward_vector
