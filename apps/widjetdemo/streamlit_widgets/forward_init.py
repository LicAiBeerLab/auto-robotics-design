from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import meshcat
import numpy as np
import pinocchio as pin
import streamlit as st

from auto_robot_design.description.builder import ParametrizedBuilder, URDFLinkCreater3DConstraints, MIT_CHEETAH_PARAMS_DICT
from auto_robot_design.description.mesh_builder.mesh_builder import MeshBuilder
from auto_robot_design.description.mesh_builder.urdf_creater import MeshCreator, URDFMeshCreator
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
from auto_robot_design.generator.topologies.graph_manager_2l import GraphManager2L
from auto_robot_design.motion_planning.trajectory_ik_manager import TrajectoryIKManager
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import ZRRReward
from auto_robot_design.optimization.rewards.reward_base import PositioningConstrain, RewardManager
from auto_robot_design.pinokla.calc_criterion import (ActuatedMass,
                                                      EffectiveInertiaCompute,
                                                      ManipJacobian,
                                                      MovmentSurface,
                                                      NeutralPoseMass)
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.default_traj import (
    add_auxilary_points_to_trajectory, convert_x_y_to_6d_traj_xz,
    create_simple_step_trajectory, get_vertical_trajectory,
    get_workspace_trajectory)

pin.seed(1)
from auto_robot_design.utils.configs import get_standard_builder, get_mesh_builder, get_standard_crag, get_standard_rewards
@dataclass
class OptimizationData:
    graph_manager: GraphManager2L
    optimization_builder: ParametrizedBuilder
    crag: CriteriaAggregator
    reward_manager: RewardManager
    soft_constraint: PositioningConstrain
    actuator: str


@st.cache_resource
def build_constant_objects():
    optimization_builder = get_standard_builder()
    visualization_builder = get_mesh_builder()
    crag = get_standard_crag()
    graph_managers = {f"Топология_{i}": get_preset_by_index_with_bounds(i) for i in range(9)}
    reward_dict = get_standard_rewards()
    return graph_managers, optimization_builder, visualization_builder, crag, reward_dict

def add_trajectory_to_vis(pin_vis, trajectory):
    material = meshcat.geometry.MeshPhongMaterial()
    material.color = int(0xFF00FF)
    material.color = int(0x00FFFF)
    material.color = int(0xFFFF00)
    material.opacity = 0.3
    for idx, point in enumerate(trajectory):
        if idx <150 and idx%2==0:
            ballID = "world/ball" + str(idx)
            pin_vis.viewer[ballID].set_object(meshcat.geometry.Sphere(0.01), material)
            T = np.r_[np.c_[np.eye(3), point[:3]+np.array([0,-0.04,0])], np.array([[0, 0, 0, 1]])]
            pin_vis.viewer[ballID].set_transform(T)

# @st.cache_resource
# def create_visualizer():
#     gm = get_preset_by_index_with_bounds(-1)
#     fixed_robot, free_robot = jps_graph2pinocchio_meshes_robot(gm.graph, builder)
#     visualizer = MeshcatVisualizer(
#         fixed_robot.model, fixed_robot.visual_model, fixed_robot.visual_model
#     )
# with fake_output:
#     visualizer.viewer = meshcat.Visualizer()
# visualizer.viewer["/Background"].set_property("visible", False)
# visualizer.viewer["/Grid"].set_property("visible", False)
# visualizer.viewer["/Axes"].set_property("visible", False)
# visualizer.viewer["/Cameras/default/rotated/<object>"].set_property(
#     "position", [0, 0.0, 0.8]
# )
# visualizer.clean()
# visualizer.loadViewerModel()
# visualizer.display(pin.neutral(fixed_robot.model))

from auto_robot_design.optimization.rewards.inertia_rewards import (
    ActuatedMassReward, MassReward, TrajectoryIMFReward)
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import (
    AccelerationCapability, HeavyLiftingReward, MeanHeavyLiftingReward,
    MinAccelerationCapability)
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import (
    DexterityIndexReward, ManipulabilityReward, MinForceReward,
    MinManipulabilityReward, VelocityReward, ZRRReward)
from auto_robot_design.optimization.rewards.reward_base import (
    PositioningConstrain, PositioningErrorCalculator, RewardManager)
from auto_robot_design.pinokla.calc_criterion import (ActuatedMass,
                                                      EffectiveInertiaCompute,
                                                      ImfCompute, ManipCompute,
                                                      ManipJacobian,
                                                      MovmentSurface,
                                                      NeutralPoseMass,
                                                      TranslationErrorMSE)
from auto_robot_design.pinokla.criterion_math import ImfProjections


@st.cache_resource 
def set_criteria_and_crag():
    # we should create a crag that calculates all possible data
    dict_trajectory_criteria = {
        "MASS": NeutralPoseMass(),
        "POS_ERR": TranslationErrorMSE()  # MSE of deviation from the trajectory
    }
    # criteria calculated for each point on the trajectory
    dict_point_criteria = {
        # Impact mitigation factor along the axis
        "IMF": ImfCompute(ImfProjections.Z),
        "MANIP": ManipCompute(MovmentSurface.XZ),
        "Effective_Inertia": EffectiveInertiaCompute(),
        "Actuated_Mass": ActuatedMass(),
        "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ)
    }
    # special object that calculates the criteria for a robot and a trajectory
    crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)

    #reward_list=[]
    # reward_list.append(EndPointIMFReward(imf_key='IMF', trajectory_key="traj_6d", error_key="error"))
    # reward_list.append(ActuatedMassReward(mass_key='Actuated_Mass'))
    reward_dict={}
    reward_dict['mass'] = MassReward(mass_key='MASS')
    reward_dict['actuated_inertia_matrix']  = ActuatedMassReward(mass_key='Actuated_Mass', reachability_key="is_reach")
    reward_dict['z_imf'] = TrajectoryIMFReward(imf_key='IMF',trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['trajectory_manipulability'] = VelocityReward(manipulability_key='Manip_Jacobian', trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['manipulability'] = ManipulabilityReward(manipulability_key='MANIP',
                                                    trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['min_manipulability'] = MinManipulabilityReward(manipulability_key='Manip_Jacobian',
                                                        trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['min_force'] = MinForceReward(manipulability_key='Manip_Jacobian',
                                                    trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['trajectory_zrr'] = ZRRReward(manipulability_key='Manip_Jacobian',
                                                        trajectory_key="traj_6d",  reachability_key="is_reach")
    reward_dict['dexterity'] = DexterityIndexReward(manipulability_key='Manip_Jacobian',
                                                    trajectory_key="traj_6d", reachability_key="is_reach")
    reward_dict['trajectory_acceleration'] = AccelerationCapability(manipulability_key='Manip_Jacobian',
                                                    trajectory_key="traj_6d", reachability_key="is_reach", actuated_mass_key="Actuated_Mass")
    reward_dict['min_acceleration'] = MinAccelerationCapability(manipulability_key='Manip_Jacobian', trajectory_key="traj_6d", reachability_key="is_reach", actuated_mass_key="Actuated_Mass")
    reward_dict['mean_heavy_lifting']  = MeanHeavyLiftingReward(manipulability_key='Manip_Jacobian', reachability_key="is_reach", mass_key="MASS")
    reward_dict['min_heavy_lifting']  = HeavyLiftingReward(manipulability_key='Manip_Jacobian',mass_key='MASS', reachability_key="is_reach")
    reward_list = list(reward_dict.values())
    # reward_list.append(TrajectoryIMFReward(imf_key='IMF',trajectory_key="traj_6d", error_key="error"))
    # reward_list.append(VelocityReward(manipulability_key='Manip_Jacobian', trajectory_key="traj_6d", error_key="error"))
    # reward_list.append(ManipulabilityReward(manipulability_key='MANIP',
    #                                                 trajectory_key="traj_6d", error_key="error"))
    # reward_list.append(MinManipulabilityReward(manipulability_key='Manip_Jacobian',
    #                                                 trajectory_key="traj_6d", error_key="error"))
    # reward_list.append(ForceEllipsoidReward(manipulability_key='Manip_Jacobian',
    #                                                 trajectory_key="traj_6d", error_key="error"))
    # reward_list.append(ZRRReward(manipulability_key='Manip_Jacobian',
    #                                                 trajectory_key="traj_6d", error_key="error"))
    # # reward_list.append(EndPointZRRReward(manipulability_key='Manip_Jacobian',
    # #                                                 trajectory_key="traj_6d", error_key="error"))
    # reward_list.append(MinForceReward(manipulability_key='Manip_Jacobian',
    #                                                 trajectory_key="traj_6d", error_key="error"))
    # reward_list.append(DexterityIndexReward(manipulability_key='Manip_Jacobian',
    #                                                 trajectory_key="traj_6d", error_key="error"))
    # reward_list.append(AccelerationCapability(manipulability_key='Manip_Jacobian',
    #                                                     trajectory_key="traj_6d", error_key="is_reach", actuated_mass_key="Actuated_Mass"))
    # reward_list.append(MinAccelerationCapability(manipulability_key='Manip_Jacobian',
    #                                                     trajectory_key="traj_6d", error_key="is_reach", actuated_mass_key="Actuated_Mass"))
    # reward_list.append(HeavyLiftingReward(manipulability_key='Manip_Jacobian',mass_key='MASS',
    #                                                     trajectory_key="traj_6d", error_key="is_reach"))
    # reward_list.append(MeanHeavyLiftingReward(manipulability_key='Manip_Jacobian',mass_key='MASS',
    #                                                     trajectory_key="traj_6d", error_key="is_reach"))
    motor = MIT_CHEETAH_PARAMS_DICT["actuator"]
    return crag, reward_list,  motor