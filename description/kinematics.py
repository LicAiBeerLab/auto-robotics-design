from dataclasses import dataclass,field

import numpy as np
import networkx as nx

from dm_control import mjcf

@dataclass
class JointPoint:
    """Describe a point in global frame where a joint is attached"""
    pos: np.ndarray = field(default_factory=np.zeros(3))
    weld: bool = False
    active: bool = False
    attach_ground: bool = False
    attach_endeffector: bool = False
    instance_counter: int = 0
    
    def __post_init__(self):
        JointPoint.instance_counter += 1
        self.__instance_counter = JointPoint.instance_counter
    
    def __hash__(self) -> int:
        return hash((self.pos[0], self.pos[1], self.pos[2], self.attach_ground, self.attach_endeffector, self.active, self.weld, self.__instance_counter))
    
    def __eq__(self, __value: object) -> bool:
        return hash(self) == hash(__value)

@dataclass
class LinkAttrib:
    variable: bool = False
    active: bool = False

class KinematicStructure(nx.Graph):
    pass

if __name__ == "__main__":
    print("Kinematic description of the mechanism")
    # Define the joint points
    joint_points = [
        JointPoint(pos=np.array([0, 0, 0]), weld=True, attach_ground=True),
        JointPoint(pos=np.array([1, 0, 0]), weld=True),
        JointPoint(pos=np.array([0, 1, 0])),
        JointPoint(pos=np.array([0, 0, 1]), weld=True),
        JointPoint(pos=np.array([1, 1, 0])),
        JointPoint(pos=np.array([1, 0, 1]), weld=True),
        JointPoint(pos=np.array([0, 1, 1])),
        JointPoint(pos=np.array([1, 1, 1]), weld=True, attach_endeffector=True),
    ]
    print(joint_points[0] == joint_points[1])
    print(joint_points[0] == joint_points[0])