# %%
from collections import deque
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import modern_robotics as mr

import networkx as nx

from description.kinematics import JointPoint

# %%
graph = nx.Graph()

abs_ground = np.array([0, 0, 0])
J1 = np.array([1.087, 0, 1.435])
J2 = np.array([5.459, 0, 2.501])
J3 = np.array([4.2, 0, 0])
end = np.array([5.5, 0, 2.6])

ground_joint = JointPoint(r=abs_ground, w=np.array([0,1,0]), 
                        attach_ground=True, 
                        active=True,
                        name="J0")
joint_1 = JointPoint(r=J1, w=np.array([0,1,0]),
                     attach_endeffector=False,
                     name="J1")
joint_2 = JointPoint(J2, w=np.array([0,1,0]), 
                    attach_endeffector=False,
                    name="J2")
joint_3 = JointPoint(r=J3, w=np.array([0,1,0]),
                    attach_ground=True,
                    name="J3")
end_effector = JointPoint(r=end, w=np.array([0,1,0]),
                    attach_endeffector=True,
                    name="EE")

joints = [ground_joint, joint_1, joint_2, joint_3, end_effector]

# for j in joints:
#     graph.add_node(j)
main_branch = [ground_joint, joint_1, end_effector]
add_branch_1 = [joint_3, joint_2, [joint_1, end_effector]]


# %%
from description.builder import add_branch

add_branch(graph, [main_branch, add_branch_1])

# %%
from description.utils import draw_joint_point

draw_joint_point(graph) 

# %%
from description.mechanism import JointPoint2KinematicGraph
from description.utils import draw_links

kinematic_graph = JointPoint2KinematicGraph(graph)
draw_links(kinematic_graph, graph)

# %%
from description.utils import draw_kinematic_graph

draw_kinematic_graph(kinematic_graph)

# %%
from description.mechanism import get_span_tree_n_main_branch

kinematic_tree, main_branch = get_span_tree_n_main_branch(kinematic_graph)

# %%
from description.mechanism import define_link_frames

define_link_frames(kinematic_graph, kinematic_tree, main_branch=main_branch, all_joints=graph.nodes())

from description.builder import create_urdf

create_urdf(kinematic_graph)
