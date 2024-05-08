from typing import Tuple
from auto_robot_design.pinokla.calc_criterion import DataDict


class Reward():
    """Interface for the optimization criteria"""

    def __init__(self) -> None:
        self.point_precision = 1e-6

    def calculate(self, point_criteria: DataDict, trajectory_criteria: DataDict, trajectory_results: DataDict, **kwargs) -> Tuple[float, list[float]]:
        """Calculate the value of the criterion from the data"""

        raise NotImplementedError("A reward must implement calculate method!")

    def check_reachability(self, errors):
        if max(errors) > self.point_precision:
            return False

        return True


class PositioningReward():
    """Mean position error for the trajectory"""

    def __init__(self,  pos_error_key:str) -> None:
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
