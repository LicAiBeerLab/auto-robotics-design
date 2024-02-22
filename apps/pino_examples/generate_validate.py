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
from auto_robot_design.pino_adapter import pino_adapter
from auto_robot_design.pinokla.criterion_agregator import ComputeConfg, calc_criterion_on_workspace_simple_input

gen = TwoLinkGenerator()
graphs_and_cons = gen.get_standard_set()

builder = Builder(DetalizedURDFCreater)
urdf_motors_cons_list = []
for graph_i, constarin_i in graphs_and_cons:
    kinematic_graph = JointPoint2KinematicGraph(graph_i)
    robot, ative_joints, constraints = builder.create_kinematic_graph(
        kinematic_graph)
    pino_j_des, pino_cons_des = pino_adapter.get_pino_description(
        ative_joints, constraints)
    urdf_motors_cons_tuple = (robot.urdf(), pino_j_des, pino_cons_des)
    urdf_motors_cons_list.append(urdf_motors_cons_tuple)
    # draw_joint_point(graph_i)
    # plt.show()

for urdf, mot, cons in urdf_motors_cons_list:
    robo_dict, res_dict = calc_criterion_on_workspace_simple_input(urdf, mot, cons, "G", "EE", 20, cmp_cfg=ComputeConfg(False, False, False, False))
    pass
