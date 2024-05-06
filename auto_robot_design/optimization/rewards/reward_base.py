import numpy as np
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
        return -mean_error, []
    
class SingularityPenalty(Reward):
    """Calculate the penalty for the manipulability matrix close to singular"""

    def __init__(self, manipulability_key, trajectory_key) -> None:
        """Set the dictionary keys for the data

        Args:
            manipulability_key (str): key for the manipulability matrix
            trajectory_key (str): key for the trajectory points
            error_key (str): key for the pose errors
            threshold (float): threshold for the manipulability matrix determinant
        """
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        
    def calculate(self, point_criteria, trajectory_criteria, trajectory_results, **kwargs) -> float:
        """Calculate the penalty for the manipulability matrix close to singular

        Args:
            point_criteria (DataDict): all data of the characteristics assigned to each point
            trajectory_criteria (DataDict): all data of the trajectory characteristics 
            trajectory_results (DataDict): data of trajectory and trajectory following

        Returns:
            float: value of the reward
        """
        # get the manipulability for each point at the trajectory
        manipulability_matrices: list[np.ndarray] = point_criteria[self.manip_key]
        trajectory_points = trajectory_results[self.trajectory_key]
        tangent_to_points
        n_steps = len(trajectory_points)
        
        trajectory_points
        
        result = 0
        reward_vector = []
        for i in range(n_steps):
            # the reward is none zero only if the point is reached
            if errors[i] > 1e-6:
                return 0, []
            # get the manipulability matrix for the current point
            manipulability_matrix: np.ndarray = manipulability_matrices[i]
            # find the determinant of the manipulability matrix
            
            if det < self.threshold:
                return 0, []
            result += det
            reward_vector.append(det)
        return result, reward_vector