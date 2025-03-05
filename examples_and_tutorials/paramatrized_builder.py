import time
import numpy as np

import pinocchio as pin
from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.mesh_builder.mesh_builder import jps_graph2pinocchio_meshes_robot
from auto_robot_design.description.utils import all_combinations_active_joints_n_actuator
from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder, MIT_CHEETAH_PARAMS_DICT, jps_graph2pinocchio_robot_3d_constraints
from auto_robot_design.generator.topologies.graph_manager_2l import get_preset_by_index
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
from auto_robot_design.vizualization.meshcat_utils import create_meshcat_vizualizer
from auto_robot_design.utils.configs import get_standard_builder, get_mesh_builder, get_standard_crag, get_standard_rewards
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph, KinematicGraph
from auto_robot_design.description.builder import BLUE_COLOR, DEFAULT_PARAMS_DICT, GREEN_COLOR, RED_COLOR
from auto_robot_design.pino_adapter.pino_adapter import get_pino_description_3d_constraints
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
from auto_robot_design.description.actuators import TMotor_AK80_9, MIT_Actuator, TMotor_AK60_6_small

thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]

density = MIT_CHEETAH_PARAMS_DICT["density"]
body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]

params = {"main_thickness":thickness*1.3, "sub_thickness":thickness, "density":density, "topologies":0}
gm = get_preset_by_index(params["topologies"])

graph = gm.get_graph(gm.generate_central_from_mutation_range())



builder = get_mesh_builder(manipulation=True)
# builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE,
#                               density={"default": density, "G":body_density},
#                               thickness={"default": thickness, "EE":0.033},
#                               actuator={"default": MIT_CHEETAH_PARAMS_DICT["actuator"]},
#                             #   size_ground=np.array(MIT_CHEETAH_PARAMS_DICT["size_ground"]),
#                               offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"]
# )

# builder = get_mesh_builder(True)

# robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(graph, builder)
kinematic_graph = JointPoint2KinematicGraph(graph)
kinematic_graph.define_main_branch()
kinematic_graph.define_span_tree()

del builder.mesh_creator.predefind_mesh["EE"]
builder.thickness.update({"default":params["main_thickness"]})
builder.density.update({"default":params["density"]})
builder.actuator.update({name: TMotor_AK60_6_small() for name in ['Main_knee', 'Main_ee', 'Main_connection_1', 'branch_0', 'Ground_connection', 'branch_1', 'Main_connection_2', 'branch_2']})
# print(builder.actuator)
subbranch_thickness = {}
for link in kinematic_graph.nodes():
  if link not in kinematic_graph.main_branch:
    subbranch_thickness[link.name] = params["sub_thickness"]

builder.thickness.update(subbranch_thickness)

i = 1
k = 1
name_link_in_aux_branch = []
for link in kinematic_graph.nodes():
    if link.name == "G":
        link.geometry.color = RED_COLOR[0,:].tolist()
    elif link in kinematic_graph.main_branch.nodes():
        # print("yes")
        link.geometry.color = BLUE_COLOR[i,:].tolist()
        i = (i + 1) % 6
    else:
        link.geometry.color = GREEN_COLOR[k,:].tolist()
        name_link_in_aux_branch.append(link.name)
        k = (k + 1) % 5


kinematic_graph.define_link_frames()

robot, active_joints, constraints = builder.create_kinematic_graph(kinematic_graph)

# with open("robot.urdf", "w") as f:
#     f.write(robot.urdf())

act_description, constraints_descriptions = get_pino_description_3d_constraints(
    active_joints, constraints
)
robot = build_model_with_extensions(robot.urdf(),
                            joint_description=act_description,
                            loop_description=constraints_descriptions,
                            actuator_context=kinematic_graph,
                            fixed=True)


# robot,__ = jps_graph2pinocchio_meshes_robot(graph, builder)

viz = create_meshcat_vizualizer(robot)
time.sleep(1)
viz.display(np.zeros(robot.model.nq))

print(params)
print(pin.computeTotalMass(robot.model, robot.data))
# with open("parametrized_builder_test.urdf", "w") as f:
#     f.write(robo_urdf)

