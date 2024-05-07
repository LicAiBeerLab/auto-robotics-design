from typing import Tuple
import numpy as np

from auto_robot_design.optimization.rewards.reward_base import Reward

class VelocityReward(Reward):
    """Reward the mech for the value of the manipulability along the trajectory
    """

    def __init__(self, manipulability_key, trajectory_key, error_key) -> None:
        """Set the dictionary keys for the data

        Args:
            manipulability_key (str): key for the manipulability matrix
            trajectory_key (str): key for the trajectory points
            error_key (str): key for the pose errors 
        """
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.error_key = error_key

    def calculate(self, point_criteria, trajectory_criteria, trajectory_results, **kwargs) -> Tuple[float, list[float]]:
        """Calculate the length of the line from zero to the cross of the manipulability ellipsoid and trajectory direction

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            Tuple[float, list[float]]: value of the reward and the reward vector
        """
        errors = trajectory_results[self.error_key]
        is_trajectory_reachable = self.check_reachability(errors)
        # the reward is none zero only if the point is reached
        if not is_trajectory_reachable:
            return 0, []
        
        # get the manipulability for each point at the trajectory
        manipulability_matrices: list[np.array] = point_criteria[self.manip_key]
        trajectory_points = trajectory_results[self.trajectory_key]
        diff_vector = np.diff(trajectory_points, axis=0)[:, [0, 2]]
        n_steps = len(trajectory_points)
        result = 0
        reward_vector = [0]*(n_steps-1)
        for i in range(n_steps-1):
            # get the direction of the trajectory
            trajectory_shift = diff_vector[i]
            trajectory_direction = trajectory_shift / \
                np.linalg.norm(trajectory_shift)

            # get the manipulability matrix for the current point
            manipulability_matrix: np.array = manipulability_matrices[i]
            # find alpha from A@x = alpha*y, with ||x|| = 1 and y = trajectory_direction
            # get inverse of the manipulability matrix
            manipulability_matrix_inv = np.linalg.inv(manipulability_matrix)
            temp_vec = manipulability_matrix_inv@trajectory_direction
            result += 1/np.linalg.norm(temp_vec)
            reward_vector[i]=1/np.linalg.norm(temp_vec)

        return result/(n_steps-1), reward_vector

class ManipulabilityReward(Reward):
    """Calculate determinant of the manipulability matrix"""

    def __init__(self, manipulability_key, trajectory_key, error_key):
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.error_key = error_key
    
    def calculate(self, point_criteria, trajectory_criteria, trajectory_results, **kwargs) -> Tuple[float, list[float]]:
        """Get manipulability for each point in the trajectory and return the mean value

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            Tuple[float, list[float]]: value of the reward and the reward vector
        """

        errors = trajectory_results[self.error_key]
        is_trajectory_reachable = self.check_reachability(errors)
        # the reward is none zero only if the point is reached
        if not is_trajectory_reachable:
            return 0, []

        # get the manipulability for each point at the trajectory
        manipulability: list[np.array] = point_criteria[self.manip_key]
        result=np.mean(manipulability)
        reward_vector=list(manipulability)

        return result, reward_vector

class ForceEllipsoidReward(Reward):
    """Force capability along the trajectory"""

    def __init__(self, manipulability_key, trajectory_key, error_key) -> None:
        """Set the dictionary keys for the data

        Args:
            manipulability_key (str): key for the manipulability matrix
            error_key (str): key for the pose errors 
        """
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.error_key = error_key

    def calculate(self, point_criteria, trajectory_criteria, trajectory_results, **kwargs) -> Tuple[float, list[float]]:
        """Calculate reduction ratio along the trajectory for each point and return the mean value

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            Tuple[float, list[float]]: value of the reward and the reward vector
        """
        errors = trajectory_results[self.error_key]
        is_trajectory_reachable = self.check_reachability(errors)
        # the reward is none zero only if the point is reached
        if not is_trajectory_reachable:
            return 0, []

        manipulability_matrices: list[np.array] = point_criteria[self.manip_key]
        trajectory_points = trajectory_results[self.trajectory_key]
        diff_vector = np.diff(trajectory_points, axis=0)[:, [0, 2]]
        n_steps = len(trajectory_points)
        result = 0
        reward_vector = [0]*(n_steps-1)
        for i in range(n_steps-1):
            # get the direction of the trajectory
            trajectory_shift = diff_vector[i]
            trajectory_direction = trajectory_shift / \
                np.linalg.norm(trajectory_shift)

            manipulability_matrix: np.array = manipulability_matrices[i]
            force_matrix = np.transpose(manipulability_matrix)
            # inverse of the reduction ratio in the trajectory direction
            step_result = 1/np.linalg.norm(force_matrix@trajectory_direction)
            result += step_result
            reward_vector[i] = step_result

        return result/(n_steps-1), reward_vector

class EndPointZRRReward(Reward):
    """Reduction ratio along the vertical (z) axis in the edge points of the trajectory (stance poses)"""

    def __init__(self, manipulability_key, trajectory_key, error_key) -> None:
        """Set the dictionary keys for the data

        Args:
            manipulability_key (str): key for the manipulability matrix
            trajectory_key (str): key for the trajectory points
            error_key (str): key for the pose errors 
        """
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.error_key = error_key

    def calculate(self, point_criteria, trajectory_criteria, trajectory_results, **kwargs) -> Tuple[float, list[float]]:
        """Calculates the sum of ZRR in starting and end points

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            float: value of the reward
        """
        manipulability_matrices: list[np.array] = point_criteria[self.manip_key]
        errors = trajectory_results[self.error_key]
        # the reward is none zero only if the point is reached
        if errors[0] > 1e-6:
            starting_result = 0
        else:
            starting_pose_matrix = np.transpose(manipulability_matrices[0])
            starting_result = 1 / \
                np.linalg.norm(starting_pose_matrix@np.array([0, 1]))
        # the reward is none zero only if the point is reached
        if errors[-1] > 1e-6:
            end_result = 0
        else:
            end_pose_matrix = np.transpose(manipulability_matrices[-1])
            end_result = 1/np.linalg.norm(end_pose_matrix@np.array([0, 1]))

        return (starting_result + end_result)/2, [starting_result, end_result]
    