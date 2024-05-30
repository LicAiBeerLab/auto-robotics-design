import numpy as np

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.utils import all_combinations_active_joints_n_actuator
from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder, MIT_CHEETAH_PARAMS_DICT
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions


gen = TwoLinkGenerator()
graph, constrain_dict = gen.get_standard_set()[1]



thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]

density = MIT_CHEETAH_PARAMS_DICT["density"]
body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]


builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE,
                              density={"default": density, "G":body_density},
                              thickness={"default": thickness, "EE":0.033},
                              actuator={"default": MIT_CHEETAH_PARAMS_DICT["actuator"]},
                              size_ground=np.array(MIT_CHEETAH_PARAMS_DICT["size_ground"]),
                              offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"]
)

robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(graph, builder)


with open("parametrized_builder_test.urdf", "w") as f:
    f.write(robo_urdf)

