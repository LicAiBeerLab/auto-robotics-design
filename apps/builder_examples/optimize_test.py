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

from auto_robot_design.optimization.optimizer import Optimizer, PSOOptmizer
from auto_robot_design.description.builder import jps_graph2urdf

optimizing_joints = dict(filter(lambda x: x[1]["optim"], constrain_dict.items()))
name2jp = dict(map(lambda x: (x.name, x), graph.nodes()))
optimizing_joints = dict(map(lambda x: (name2jp[x[0]], (x[1]["x_range"][0], x[1].get("z_range", [-0.01,0.01])[0], x[1]["x_range"][1], x[1].get("z_range", [0,0])[1])), optimizing_joints.items()))
# limits = {j:()}
optimizer = {}

# opt_eng = Optimizer(graph, optimizing_joints, np.array([1, 0.4]), "two_links", **optimizer)
opt_eng = PSOOptmizer(graph, optimizing_joints, np.array([1, 0.4]), "two_links", **optimizer)
cost = opt_eng.get_cost(opt_eng.initial_xopt)
print(cost, opt_eng.calc_fval(cost))

opt_eng.run(50)

histor_x = sorted(opt_eng.history, key=lambda x: x[2])

print([x[2] for x in histor_x])

best_x = histor_x[0]
print(best_x[1], best_x[2])
opt_eng.mutate_JP_by_xopt(best_x[0])
draw_joint_point(opt_eng.graph)
plt.show()
urdf, __, __ = jps_graph2urdf(opt_eng.graph)
with open("test.urdf", "w") as f:
    f.write(urdf)

print("success")