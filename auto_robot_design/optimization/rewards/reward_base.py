import numpy as np
from typing import Tuple
from auto_robot_design.pinokla.calc_criterion import DataDict
import os

from auto_robot_design.pinokla.criterion_math import convert_full_J_to_planar_xz


class Reward():
    """Interface for the optimization criteria"""

    def __init__(self) -> None:
        self.point_precision = 1e-6

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
        """Calculate the value of the criterion from the data"""

        raise NotImplementedError("A reward must implement calculate method!")

    def check_reachability(self, errors):
        if np.max(errors) > self.point_precision:
            raise ValueError(
                f"All points should be reachable to calculate a reward {max(errors)}")

        return True


class PositioningReward():
    """Mean position error for the trajectory"""

    def __init__(self,  pos_error_key: str) -> None:
        """Set the dictionary keys for the data

        Args:
            pos_error_key (str): key for mean position error
        """
        self.pos_error_key = pos_error_key

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
        """Just get the value for the mean positioning error

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            float: value of the reward
        """
        # get the manipulability for each point at the trajectory
        mean_error = trajectory_criteria[self.pos_error_key]
        # the empty list is for the consistency with the other rewards
        return -mean_error, []


class PositioningErrorCalculator():

    def __init__(self, error_key):
        self.error_key = error_key
        self.point_threshold = 1e-6

    def calculate(self, trajectory_results: DataDict):
        errors = trajectory_results[self.error_key]
        jacobians = trajectory_results["J_closed"]
        get_svd_s = lambda x: np.linalg.svd(x)[1]
        eig_val = []
        for jacob in jacobians:
            planar_J = convert_full_J_to_planar_xz(jacob)
            trans_planar_J = planar_J[:2, :2]
            U, S, Vh = np.linalg.svd(trans_planar_J)

            max_eig_val = np.max(S)
            min_eig_val = np.min(S)
            isotropic = max_eig_val / min_eig_val
            eig_val.append(isotropic)
        return eig_val


class PositioningConstrain():

    def __init__(self, error_calculator, points=None) -> None:
        self.points = points
        self.calculator = error_calculator

    def add_points_set(self, points_set):
        if self.points is None:
            self.points = [points_set]
        else:
            self.points.append(points_set)

    def calculate_constrain_error(self, criterion_aggregator, fixed_robot,
                                  free_robot):
        total_error = 0
        total_error = []
        total_error_pos = []
        results = []
        for point_set in self.points:
            tmp = criterion_aggregator.get_criteria_data(
                fixed_robot, free_robot, point_set)
            results.append(tmp)
            total_error.append(self.calculator.calculate(tmp[2]))
            total_error_pos = tmp[2]["error"]
        return total_error, results, total_error_pos


class RewardManager():
    """Manager class to aggregate trajectories and corresponding rewards

        User should add trajectories and then add rewards that are calculated for these trajectories.
    """
    def __init__(self, crag) -> None:
        self.trajectories = {}
        self.rewards = {}
        self.crag = crag
        self.precalculated_trajectories = None

    def add_trajectory(self, trajectory, idx):
        if not (idx in self.trajectories):
            self.trajectories[idx] = trajectory
            self.rewards[idx] = []
        else:
            raise KeyError(
                'Attempt to add trajectory id that already exist in RewardManager')

    def add_reward(self, reward, trajectory_id, weight):
        if trajectory_id in self.trajectories:
            self.rewards[trajectory_id].append((reward, weight))
        else:
            raise KeyError('Trajectory id not in the trajectories dict')

    def calculate_total(self, fixed_robot, free_robot, motor):
        trajectory_rewards = []
        partial_rewards = []
        for trajectory_id, trajectory in self.trajectories.items():
            rewards = self.rewards[trajectory_id]
            if self.precalculated_trajectories and (trajectory_id in self.precalculated_trajectories):
                point_criteria_vector, trajectory_criteria, res_dict_fixed = self.precalculated_trajectories[trajectory_id]
            else:
                point_criteria_vector, trajectory_criteria, res_dict_fixed = self.crag.get_criteria_data(
                    fixed_robot, free_robot, trajectory)

            # point_criteria_vector, trajectory_criteria, res_dict_fixed = self.crag.get_criteria_data(
            #     fixed_robot, free_robot, trajectory)
            reward_at_trajectory = 0
            partial_reward = [trajectory_id]
            for reward, weight in rewards:
                reward_value = reward.calculate(
                    point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator=motor)[0]
                partial_reward.append(reward_value)
                reward_at_trajectory += weight * reward_value
            # update reward lists
            trajectory_rewards.append((trajectory_id, reward_at_trajectory))
            partial_rewards.append(partial_reward)

        # calculate the total reward
        total_reward = -sum([reward for _, reward in trajectory_rewards])
        return total_reward, partial_rewards

    def dummy_partial(self):
        """Create partial reward with zeros to add for robots that failed constrains"""
        partial_rewards = []
        for trajectory_id, _ in self.trajectories.items():
            rewards = self.rewards[trajectory_id]
            partial_reward = [trajectory_id]
            for _, _ in rewards:
                partial_reward.append(0)
            partial_rewards.append(partial_reward)
        return partial_rewards

    def check_constrain_trajectory(self, trajectory, results):
        """Checks if a trajectory that was used in constrain calculation is also one of reward trajectories.
        
            If a trajectory is a reward trajectory save its results and use them to avoid recalculation 
        """
        temp_dict = {}
        for trajectory_id, in_trajectory in self.trajectories.items():
            if np.array_equal(trajectory, in_trajectory):
                temp_dict[trajectory_id] = results

        self.precalculated_trajectories = temp_dict
