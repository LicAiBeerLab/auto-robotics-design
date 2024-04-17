from matplotlib.scale import scale_factory
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

import networkx as nx
from auto_robot_design.description.actuators import TMotor_AK60_6, TMotor_AK80_9

from auto_robot_design.description.utils import draw_kinematic_graph
from auto_robot_design.description.kinematics import JointPoint

from auto_robot_design.description.utils import draw_joint_point
# from auto_robot_design.generator_functions import generate_graph
from auto_robot_design.generator.respawn_algorithm import generate_graph
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph
from auto_robot_design.description.utils import draw_links
from auto_robot_design.description.utils import draw_joint_frames, draw_joint_point, draw_link_frames
from auto_robot_design.description.builder import Builder, URDFLinkCreater
from auto_robot_design.pino_adapter.pino_adapter import get_pino_description


def generate_strange_graph():
    graph = generate_graph()

    max_node = sorted(list(graph.nodes()),
                      key=lambda x: la.norm(x.r),
                      reverse=True)[0]
    is_not_active = lambda x: not x.active
    not_active_nodes = list(filter(is_not_active, graph.nodes()))
    high_node = sorted(list(not_active_nodes), key=lambda x: x.r[2],
                       reverse=True)[0]
    #high_node.active = True
    mot2_name = high_node.name
    # # print(max_node.r)

    kinematic_graph = JointPoint2KinematicGraph(graph)

    kinematic_graph.define_main_branch()
    kinematic_tree = kinematic_graph.define_span_tree()
    # # print([l.name for l in kinematic_graph.main_branch])

    thickness = 0.04
    # # print(scale_factor)
    density = 2700 / 2.8

    for n in kinematic_graph.nodes():
        n.thickness = thickness
        n.density = density

    for j in kinematic_graph.joint_graph.nodes():
        j.pos_limits = (-np.pi, np.pi)
        if j.jp.active:
            j.actuator = TMotor_AK80_9()
        j.damphing_friction = (0.05, 0)

    kinematic_graph.define_link_frames()

    builder = Builder(URDFLinkCreater)

    robot, ative_joints, constraints = builder.create_kinematic_graph(
        kinematic_graph)
    ative_joints.append(mot2_name)
    joint_mot_description, loop_description = get_pino_description(ative_joints, constraints)
    return robot.urdf(), joint_mot_description, loop_description