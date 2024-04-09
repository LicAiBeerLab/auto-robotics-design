from distutils.command import build
from matplotlib.scale import scale_factory
import numpy as np
import numpy.linalg as la

from auto_robot_design.description.actuators import TMotor_AK60_6, TMotor_AK80_64, t_motor_actuators

# from auto_robot_design.generator_functions import generate_graph
from auto_robot_design.description.utils import all_combinations_active_joints_n_actuator
from auto_robot_design.generator.respawn_algorithm import generate_graph
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph
from auto_robot_design.description.builder import DetalizedURDFCreaterFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pino_adapter.pino_adapter import get_pino_description


gen = TwoLinkGenerator()
graph, constrain_dict = gen.get_standard_set()[4]


pairs = all_combinations_active_joints_n_actuator(graph, t_motor_actuators)
# max_node = sorted(list(graph.nodes()), key=lambda x: la.norm(x.r), reverse=True)[0]
# is_not_active = lambda x: not x.active
# not_active_nodes = list(filter(is_not_active, graph.nodes()))
# high_node = sorted(list(graph.nodes()), key=lambda x: x.r[2], reverse=True)[0]
# #high_node.active = True
# mot2_name = high_node.name
# print(max_node.r)


thickness = 0.04
# # print(scale_factor)
density = 2700 / 2.8

print(pairs[0])
builder = ParametrizedBuilder(DetalizedURDFCreaterFixedEE,
                              density=density,
                              thickness={"default": thickness, "EE":0.08},
                              actuator=dict(pairs[0]),
                              size_ground=np.array([thickness*5, thickness*10, thickness*2]),
)

robo_urdf, __, __ = jps_graph2urdf_by_bulder(graph, builder)


with open("parametrized_builder_test.urdf", "w") as f:
    f.write(robo_urdf)

