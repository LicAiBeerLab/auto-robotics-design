from matplotlib.scale import scale_factory
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

import networkx as nx
from auto_robot_design.description.actuators import TMotor_AK60_6, TMotor_AK80_9

from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator


gen = TwoLinkGenerator()
graph, constrain_dict = gen.get_standard_set()[4]


from auto_robot_design.description.utils import draw_joint_frames, draw_joint_point, draw_link_frames

draw_joint_point(graph) 
plt.show()
# %% 

from auto_robot_design.optimization.optimizer import Optimizer, create_dict_jp_limit

optimizing_joints = dict(filter(lambda x: x[1]["optim"], constrain_dict.items()))
name2jp = dict(map(lambda x: (x.name, x), graph.nodes()))
optimizing_joints = dict(map(lambda x: (name2jp[x[0]], (x[1]["x_range"][0], x[1].get("z_range", [0,0])[0], x[1]["x_range"][1], x[1].get("z_range", [0,0])[1])), optimizing_joints.items()))
# limits = {j:()}

opt_eng = Optimizer(graph, optimizing_joints, np.ones(1)*1, "test_two_links")

print(opt_eng.get_cost(opt_eng.initial_xopt))