import numpy as np


class Reward():
    def __init__(self) -> None:
        pass
    def calculate(self, point_criteria, trajectory_criteria) -> float:
        pass


class VelocityReward(Reward):
    def __init__(self, traj_6d: np.ndarray) -> None:
        self.traj_6d = traj_6d
        self. 
    def calculate(self, point_criteria, trajectory_criteria) -> float:
        pass
