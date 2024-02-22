from matplotlib.scale import scale_factory
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

import networkx as nx
from auto_robot_design.description.actuators import TMotor_AK60_6, TMotor_AK80_9

from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import Builder, DetalizedURDFCreater, add_branch
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.description.utils import draw_joint_frames, draw_joint_point, draw_link_frames
 

gen = TwoLinkGenerator()
graphs_and_cons = gen.get_standard_set()

builder = Builder(DetalizedURDFCreater)
graph_list = []
for graph_i, constarin_i in graphs_and_cons:
    kinematic_graph = JointPoint2KinematicGraph(graph_i)
    robot, ative_joints, constraints = builder.create_kinematic_graph(kinematic_graph)



 
plt.show()