
class Reward():
    """Interface for the optimization criteria"""

    def __init__(self) -> None:
        pass

    def calculate(self, point_criteria, trajectory_criteria, trajectory_results,**kwargs) -> float:
        """Calculate the value of the criterion from the data"""
        pass

class PositioningReward():
    """Mean position error for the trajectory"""

    def __init__(self,  pos_error_key) -> None:
        """Set the dictionary keys for the data

        Args:
            pos_error_key (str): key for mean position error
        """
        self.pos_error_key = pos_error_key

    def calculate(self, point_criteria, trajectory_criteria, trajectory_results, **kwargs) -> float:
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
        return -mean_error
