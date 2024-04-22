import numpy as np

from auto_robot_design.optimization.rewards.reward_base import Reward

class EndPointIMFReward(Reward):
    """IMF in the trajectory edge points"""

    def __init__(self, imf_key, trajectory_key, error_key) -> None:
        """Set the dictionary keys for the data

        Args:
            imf_key (str): key for the value of the IMF
            trajectory_key (str): key for the trajectory points
            error_key (str): key for the pose errors 
        """
        self.imf_key = imf_key
        self.trajectory_key = trajectory_key
        self.error_key = error_key

    def calculate(self, point_criteria, trajectory_criteria, trajectory_results, **kwargs) -> float:
        """Calculate the sum of IMF in starting and end points

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            float: value of the reward
        """
        IMF: list[np.array] = point_criteria[self.imf_key]
        errors = trajectory_results[self.error_key]

        if errors[0] > 1e-6:
            starting_result = 0
        else:
            starting_result = IMF[0]

        if errors[-1] > 1e-6:
            end_result = 0
        else:
            end_result = IMF[-1]

        return (starting_result + end_result)/2, [starting_result, end_result]

class MassReward():
    """Mass of the robot"""

    def __init__(self, mass_key) -> None:
        """Set the dictionary keys for the data

        Args:
            mass_key (str): key for the mech mass
        """
        self.mass_key = mass_key

    def calculate(self, point_criteria, trajectory_criteria, trajectory_results, **kwargs) -> float:
        """Just get the total mass from the data dictionaries

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            float: value of the reward
        """
        # get the manipulability for each point at the trajectory
        mass = trajectory_criteria[self.mass_key]
        return -mass, []