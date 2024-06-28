import time
from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount

from auto_robot_design.pinokla.calc_criterion import search_workspace
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

import pinocchio as pin
from auto_robot_design.pinokla.closed_loop_kinematics import ForwardK
from auto_robot_design.pinokla.closed_loop_jacobian import jacobian_constraint

from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator

from auto_robot_design.description.actuators import TMotor_AK80_9

from auto_robot_design.description.mechanism import JointPoint2KinematicGraph
from auto_robot_design.description.utils import draw_links, draw_joint_point, draw_link_frames

from auto_robot_design.description.utils import draw_kinematic_graph
from auto_robot_design.description.kinematics import Link
from auto_robot_design.description.mechanism import KinematicGraph

import dill
import os

if __name__ == "__main__":
    # save_path = os.path.join('.', )
    gen = TwoLinkGenerator()
    builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE)
    graphs_and_cons = gen.get_standard_set()
    np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

    graph_jp, __ = graphs_and_cons[6]
    # robo, robo_free = jps_graph2pinocchio_robot(graph_jp, builder)

    # q0 = closedLoopProximalMount(
    #     robo.model,
    #     robo.data,
    #     robo.constraint_models,
    #     robo.constraint_data,
    #     max_it=100,
    # )





    kinematic_graph = JointPoint2KinematicGraph(graph_jp)
    # draw_links(kinematic_graph, graph_jp)
    # draw_joint_point(graph_jp)
    # plt.show()

    # draw_kinematic_graph(kinematic_graph)
    main_branch = kinematic_graph.define_main_branch()
    # draw_kinematic_graph(main_branch)
    kin_tree = kinematic_graph.define_span_tree()

    thickness = 0.04
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

    # draw_link_frames(kinematic_graph)
    # plt.show()


    # for n in kinematic_graph.nodes():
    #     print(n, n.name)
    # with open('saved_g.pkl', 'wb') as f:
    #     pickle.dump(kinematic_graph, f, pickle.HIGHEST_PROTOCOL)

    # KinematicGraph.__module__ = "mechanism"
    with open('./saved_g.pkl', 'wb') as f:
        dill.dump(graph_jp, f)

    with open('./saved_g.pkl', 'rb') as f:
        graph_jp_loaded = dill.load(f)




    kinematic_graph_restored = JointPoint2KinematicGraph(graph_jp_loaded)
    # draw_links(kinematic_graph_restored, graph_jp_loaded)
    # draw_joint_point(graph_jp)
    # plt.show()

    # draw_kinematic_graph_restored(kinematic_graph_restored)
    main_branch = kinematic_graph_restored.define_main_branch()
    # draw_kinematic_graph_restored(main_branch)
    kin_tree = kinematic_graph_restored.define_span_tree()

    thickness = 0.04
    density = 2700 / 2.8

    for n in kinematic_graph_restored.nodes():
        n.thickness = thickness
        n.density = density

    for j in kinematic_graph_restored.joint_graph.nodes():
        j.pos_limits = (-np.pi, np.pi)
        if j.jp.active:
            j.actuator = TMotor_AK80_9()
        j.damphing_friction = (0.05, 0)
        
    kinematic_graph_restored.define_link_frames()

    draw_joint_point(graph_jp_loaded)
    # draw_link_frames(kinematic_graph_restored)
    plt.show()
    