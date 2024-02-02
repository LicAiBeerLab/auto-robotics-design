from dataclasses import dataclass,field

import numpy as np
import networkx as nx


@dataclass
class JointPoint:
    """Describe a point in global frame where a joint is attached"""
    r: np.ndarray = field(default_factory=np.zeros(3))
    w: np.ndarray = field(default_factory=np.zeros(3))
    active: bool = False
    attach_ground: bool = False
    attach_endeffector: bool = False
    name: str = ""  # noqa: F811
    instance_counter: int = 0
    
    def __post_init__(self):
        JointPoint.instance_counter += 1
        self.__instance_counter = JointPoint.instance_counter
        if self.name=="":
            self.name = "J" + str(self.__instance_counter)

    
    def reser_id_counter(self):
        JointPoint.instance_counter = 0
    
    def __hash__(self) -> int:
        return hash((self.r[0], self.r[1], self.r[2], self.w[0], self.w[1], self.w[2], self.attach_ground, self.attach_endeffector, self.active, self.weld, self.__instance_counter))
    
    def __eq__(self, __value: object) -> bool:
        return hash(self) == hash(__value)

@dataclass
class Link:
    name: str = ""
    joints: set[JointPoint] = field(default_factory=set)
    instance_counter: int = 0

    def __post_init__(self):
        Link.instance_counter += 1
        self.instance_counter = Link.instance_counter
        if self.name == "":
            self.name = "L" + str(self.instance_counter)

    def __hash__(self) -> int:
        return hash((self.name, *self.joints))

    def __eq__(self, __value: object) -> bool:
        self.joints == __value.joints

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