# %%
from matplotlib.scale import scale_factory
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

import networkx as nx
from description.actuators import TMotor_AK60_6

from description.kinematics import JointPoint

from description.utils import draw_joint_point
from generator_functions import generate_graph

for i in range(10):
    body_counter = 0
    joint_counter = 0
    graph = generate_graph()

if graph:
    draw_joint_point(graph)
else:
    print("Fail!") 

max_node = sorted(list(graph.nodes()), key=lambda x: la.norm(x.r), reverse=True)[0]
print(max_node.r)
# %%
from description.utils import draw_joint_frames, draw_joint_point, draw_link_frames

draw_joint_point(graph) 
plt.show()

# %%
from description.mechanism import JointPoint2KinematicGraph
from description.utils import draw_links

kinematic_graph = JointPoint2KinematicGraph(graph)
draw_links(kinematic_graph, graph)
plt.show()

# %%
from description.utils import draw_kinematic_graph

draw_kinematic_graph(kinematic_graph)
plt.show()

# %%

kinematic_graph.define_main_branch()
kinematic_tree = kinematic_graph.define_span_tree()
print([l.name for l in kinematic_graph.main_branch])
draw_kinematic_graph(kinematic_graph.main_branch)
plt.show()
draw_kinematic_graph(kinematic_tree)
plt.show()
# %%

thickness = 0.06
density = 2700

for n in kinematic_graph.nodes():
    n.thickness = thickness
    n.density = density

for j in kinematic_graph.joint_graph.nodes():
    j.pos_limits = (-np.pi, np.pi)
    if j.jp.active:
        j.actuator = TMotor_AK60_6()
    j.damphing_friction = (0.05, 0)

kinematic_graph.define_link_frames()

draw_link_frames(kinematic_graph)
draw_links(kinematic_graph, graph)
plt.show()

draw_joint_frames(kinematic_graph)
draw_links(kinematic_graph, graph)
plt.show()

# %%

from description.builder import Builder, URDFLinkCreater, DetalizedURDFCreater

builder = Builder(DetalizedURDFCreater)

robot, ative_joints, constraints = builder.create_kinematic_graph(kinematic_graph)

with open("robot.urdf", "w") as f:
    f.write(robot.urdf())

if len(constraints) == 0:
    print("sasambs")
print(f"Active joints: {ative_joints}")
print(f"Constraints: {constraints}")