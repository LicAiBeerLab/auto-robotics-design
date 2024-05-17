from matplotlib import pyplot as plt
import numpy as np

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.mechanism import JointPoint2KinematicGraph
from auto_robot_design.description.utils import all_combinations_active_joints_n_actuator, draw_kinematic_graph, draw_joint_point, draw_links
from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions

green_color = [39/255, 245/255, 159/255, 0.9]
orange_color = [245/255, 151/255, 39/255, 0.9]
blue_color = [39/255, 125/255, 245/255, 0.9]
red_color = [245/255, 39/255, 112/255, 0.9]
thickness = 0.04

density = 2700 / 2.8

builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE,
                              density={"default": density, "G":10000},
                              thickness={"default": thickness, "EE":0.06},
                              size_ground=np.array([thickness*3, thickness*3, thickness*3]),
)

gen = TwoLinkGenerator()
all_graphs = gen.get_standard_set()
for id, (graph, __) in enumerate(all_graphs):
    kinematic_graph = JointPoint2KinematicGraph(graph)
    kinematic_graph.define_main_branch()
    kinematic_graph.define_span_tree()
    for link in kinematic_graph.nodes():
        link.geometry.color = red_color
        if link in kinematic_graph.main_branch.nodes():
            # print("yes")
            link.geometry.color = orange_color
    # for link in kinematic_graph.nodes():
        # print(link.name, link.geometry.color)
    kinematic_graph.define_link_frames()
    plt.figure()
    draw_kinematic_graph(kinematic_graph)
    plt.savefig("./results/kin_graphs/k_graph_" + str(id+1) + ".svg")
    plt.figure()
    draw_links(kinematic_graph, graph)
    plt.savefig("./results/links_graph/l_graph_" + str(id+1) + ".svg")
    # robo_urdf, joint_description, loop_description = builder.create_kinematic_graph(kinematic_graph, "mech" +str(id+1))
    # with open("mech_" +str(id+1) + ".urdf", "w") as f:
    #     f.write(robo_urdf.urdf())

