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

class GraphBuilder():
    def __init__(self):
        self.graph = nx.Graph()
        self.constrain_dict = {}
        self.default_constrain = (-0.1, 0.1)
    def reset(self):
        """Reset the graph builder."""
        self.constrain_dict = {}  # should be updated after creating each joint
        self.current_main_branch = []
        self.graph = nx.Graph()
    def main_branch(self, length_list, constrain_list, actuation_mask, constrain_mask, normilize_value=None):
        ground_joint = JointPoint(
            r=np.zeros(3),
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=actuation_mask[0],
            name="Main_ground"
        )
        current_main_branch = []
        current_main_branch.append(ground_joint)

        if normilize_value is not None:
            norilizer = normilize_value/sum(length_list)
            length_list = [length/norilizer for length in length_list]

        for i in range(1,len(length_list)):
            joint = JointPoint(
                r=np.array([0, 0, sum(length_list[:i])]),
                w=np.array([0, 1, 0]),
                attach_ground=False,
                active=actuation_mask[i],
                attach_endeffector=False,
                name=f"main_joint_{i}"
            )
            self.constrain_dict[joint.name] = {'optim': constrain_mask[i], 'x_range': (-0.1, 0.1), 'z_range': (-0.1, 0.1)}
            current_main_branch.append(joint)

        joint = JointPoint(
            r=np.array([0, 0, sum(length_list)]),
            w=np.array([0, 1, 0]),
            attach_ground=False,
            active=False,
            attach_endeffector=True,
            name=f"main_ee"
        )
        current_main_branch.append(joint)