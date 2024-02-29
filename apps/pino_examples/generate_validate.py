from matplotlib.scale import scale_factory
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

import networkx as nx

from auto_robot_design.description.builder import Builder, DetalizedURDFCreater, add_branch, jps_graph2urdf
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.description.utils import draw_joint_frames, draw_joint_point, draw_link_frames
from auto_robot_design.optimization.optimizer import jps_graph2urdf
from auto_robot_design.pino_adapter import pino_adapter
from auto_robot_design.pinokla.criterion_agregator import ComputeConfg, calc_criterion_on_workspace_simple_input, save_criterion_traj

gen = TwoLinkGenerator()
graphs_and_cons = gen.get_standard_set()
DIR_NAME = "generated_1"
builder = Builder(DetalizedURDFCreater)
urdf_motors_cons_list = []
for graph_i, constarin_i in graphs_and_cons:
    robot, ative_joints, constraints = jps_graph2urdf(graph_i)
    urdf_motors_cons_tuple = (robot, ative_joints, constraints)
    draw_joint_point(graph_i)
    plt.show()
    urdf_motors_cons_list.append(urdf_motors_cons_tuple)
    # draw_joint_point(graph_i)
    # plt.show()

# for k, pack in enumerate(urdf_motors_cons_list):
#     urdf, mot, cons = pack
#     robo_dict, res_dict = calc_criterion_on_workspace_simple_input(urdf, mot, cons, "G", "EE", 100)
#     save_criterion_traj(robo_dict["urdf"], DIR_NAME, robo_dict["loop_des"],
#                     robo_dict["joint_des"], res_dict)
