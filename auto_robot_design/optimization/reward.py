import numpy as np


class Reward():
    def __init__(self) -> None:
        pass
    def calculate(self, point_criteria, trajectory_criteria, trajectory_results) -> float:
        pass


class VelocityReward(Reward):
    def __init__(self, manipulability_key, trajectory_key, poses_key) -> None:
        self.manip_key = manipulability_key
        self.trajectory_key = trajectory_key
        self.poses_key = poses_key

    def calculate(self, point_criteria, trajectory_criteria, trajectory_results) -> float:
        
        # get the manipulability for each point at the trajectory
        manipulability_matrices:List[np.array] = point_criteria[self.manip_key]
        trajectory_points = trajectory_results[self.trajectory_key]
        poses = trajectory_results[self.poses_key]
        n_steps = len(trajectory_points)
        result = 0
        for i in range(n_steps-1):
            if poses[i]!=trajectory_points[i]:
                continue
            # get the direction of the trajectory
            trajectory_shift = np.array([trajectory_points[i+1][0]-trajectory_points[i][0], trajectory_points[i+1][2]-trajectory_points[i][2]])
            trajectory_direction = trajectory_shift/np.linalg.norm(trajectory_shift)

            # get the manipulability matrix for the current point
            manipulability_matrix:np.array = manipulability_matrices[i]
            # get inverse of the manipulability matrix
            manipulability_matrix_inv = np.linalg.inv(manipulability_matrix)
            temp_vec = manipulability_matrix_inv@trajectory_direction
            result += 1/np.linalg.norm(temp_vec)
        return result
            

