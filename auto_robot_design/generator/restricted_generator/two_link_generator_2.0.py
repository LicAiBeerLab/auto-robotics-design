import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List
import networkx as nx
from copy import deepcopy
import networkx

from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch
from auto_robot_design.description.utils import draw_joint_point
import itertools
from auto_robot_design.generator.restricted_generator.utilities import set_circle_points


class TwoLinkGenerator():
    """Generates all possible graphs with two links in main branch"""

    def __init__(self) -> None:
        self.variants = list(range(7))
        self.constrain_dict = {}  # should be updated after creating each joint
        self.current_main_branch = []
        self.graph = nx.Graph()
        self.ground_x_movement = (-0.4, 0.4)
        self.ground_z_movement = (-0.01,0)
        self.free_x_movement = (-0.3, 0.3)
        self.free_z_movement = (-0.3, 0.3)
        self.bound_x_movement = (-0.2, 0.2)
        self.bound_z_movement = (-0.2, 0.2)

    def reset(self):
        """Reset the graph builder."""
        self.constrain_dict = {}  # should be updated after creating each joint
        self.current_main_branch = []
        self.graph = nx.Graph()

    def build_standard_two_linker(self, l1:float, l2:float, q1:float, q2:float):
        """Create graph for standard two-link branch

        Args:
            l1 (float): upper link length
            l2 (float): lower link length
            q1 (float): hip angle
            q2 (float): knee angle
        """
        # ground joint of the main branch
        ground_joint = JointPoint(
            r=np.zeros(3),
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=True,
            name="Main_ground"
        )
        self.constrain_dict[ground_joint.name] = {'optim': False,
                                                  'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        self.current_main_branch.append(ground_joint)
        knee_x = -l1*np.sin(q1)
        knee_z = -l2*np.cos(q1)
        knee_joint_pos = np.array([knee_x, 0, knee_z])
        knee_joint = JointPoint(
            r=knee_joint_pos, w=np.array([0, 1, 0]), name="Main_knee")
        self.constrain_dict[knee_joint.name] = {
            'optim': True, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        self.current_main_branch.append(knee_joint)
        ee_x = knee_x-l2*np.sin(q2+q1)
        ee_z = knee_z-l2*np.cos(q2+q1)
        ee = JointPoint(
            r=np.array([ee_x, 0, ee_z]),
            w=np.array([0, 1, 0]),
            attach_endeffector=True,
            name="Main_ee"
        )
        self.current_main_branch.append(ee)
        self.constrain_dict[ee.name] = {
            'optim': False, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        add_branch(self.graph, self.current_main_branch)