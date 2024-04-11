import numpy as np

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.utils import all_combinations_active_joints_n_actuator
from auto_robot_design.description.builder import DetalizedURDFCreaterFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions


gen = TwoLinkGenerator()
graph, constrain_dict = gen.get_standard_set()[4]


pairs = all_combinations_active_joints_n_actuator(graph, t_motor_actuators)

thickness = 0.04

density = 2700 / 2.8

print(pairs[0])
builder = ParametrizedBuilder(DetalizedURDFCreaterFixedEE,
                              density={"default": density, "G":10000},
                              thickness={"default": thickness, "EE":0.08},
                              actuator=dict(pairs[0]),
                              size_ground=np.array([thickness*5, thickness*10, thickness*2]),
)

robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(graph, builder)


with open("parametrized_builder_test.urdf", "w") as f:
    f.write(robo_urdf)

