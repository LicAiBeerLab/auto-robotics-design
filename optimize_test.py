from matplotlib.scale import scale_factory
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

import networkx as nx
from auto_robot_design.description.actuators import TMotor_AK60_6, TMotor_AK80_9

from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch

graph = nx.Graph()

# %%  # Cassie

abs_ground = np.array([0.065, 0, -0.047])
pos_toeA_joint = np.array([0.060, 0, -0.052]) - abs_ground
pos_toeA_tarus_joint = np.array([-0.273, 0, -0.350]) - abs_ground
pos_shin_joint = np.array([0.021, 0, -0.159]) - abs_ground
pos_knee_spring = np.array([0.011, 0, -0.219]) - abs_ground
pos_tarus_joint = np.array([-0.237, 0, -0.464]) - abs_ground
pos_foot_joint = np.array([-0.080, 0, -0.753]) - abs_ground
pos_molet_joint = np.array([-0.207, 0, -0.552]) - abs_ground
pos_toeB_joint = np.array([-0.257, 0, -0.579]) - abs_ground
pos_toeB_foot_joint = np.array([-0.118, 0, -0.776]) - abs_ground

ground_joint = JointPoint(
    r=np.zeros(3), w=np.array([0, 1, 0]), attach_ground=True, active=True
)
shin_joint = JointPoint(r=pos_shin_joint, w=np.array([0, 1, 0]), active=True)
tarus_joint = JointPoint(r=pos_tarus_joint, w=np.array([0, 1, 0]))
foot_joint = JointPoint(
    r=pos_foot_joint, w=np.array([0, 1, 0]), attach_endeffector=True
)

toeA_joint = JointPoint(r=pos_toeA_joint, w=np.array([0, 1, 0]))
connect_toeA_tarus_joint = JointPoint(
    r=pos_toeA_tarus_joint, w=np.array([0, 1, 0])
)

jts = [
    ground_joint,
    shin_joint,
    tarus_joint,
    foot_joint,
    toeA_joint,
    connect_toeA_tarus_joint,
]

main_branch = [ground_joint, shin_joint, tarus_joint, foot_joint]
add_branch_1 = [
    [ground_joint, shin_joint],
    toeA_joint,
    connect_toeA_tarus_joint,
    [tarus_joint, foot_joint],
]

add_branch(graph, [main_branch, add_branch_1])

# %% 

from auto_robot_design.optimization.optimizer import Optimizer, create_dict_jp_limit

joints = graph.nodes() - set([ground_joint])

limits = {j:()}

opt_eng = Optimizer(graph, )

