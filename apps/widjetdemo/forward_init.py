from auto_robot_design.optimization.saver import (
    load_checkpoint,
)
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import os
import pinocchio as pin
import meshcat
import time
import streamlit as st
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.decomposition.asf import ASF
from pymoo.algorithms.soo.nonconvex.pso import PSO
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.optimization.problems import SingleCriterionProblem
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import (
    ActuatedMass,
    EffectiveInertiaCompute,
    MovmentSurface,
    NeutralPoseMass,
    ManipJacobian,
)
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.default_traj import (
    add_auxilary_points_to_trajectory,
    convert_x_y_to_6d_traj_xz,
    get_vertical_trajectory,
    create_simple_step_trajectory,
    get_workspace_trajectory,
)
from auto_robot_design.optimization.rewards.reward_base import (
    PositioningConstrain,
    PositioningErrorCalculator,
    RewardManager,
)
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import ZRRReward
from auto_robot_design.description.builder import (
    ParametrizedBuilder,
    DetailedURDFCreatorFixedEE,
    URDFLinkCreater3DConstraints,
    jps_graph2pinocchio_robot,
    MIT_CHEETAH_PARAMS_DICT,
)
from auto_robot_design.description.mesh_builder.mesh_builder import (
    MeshBuilder,
    jps_graph2pinocchio_meshes_robot,
)
from auto_robot_design.description.mesh_builder.urdf_creater import (
    URDFMeshCreator,
    MeshCreator,
)
from auto_robot_design.generator.topologies.graph_manager_2l import (
    GraphManager2L,
    get_preset_by_index,
    MutationType,
)
from auto_robot_design.generator.topologies.bounds_preset import (
    get_preset_by_index_with_bounds,
)
from auto_robot_design.optimization.saver import ProblemSaver
from auto_robot_design.motion_planning.trajectory_ik_manager import TrajectoryIKManager
from pinocchio.visualize import MeshcatVisualizer


from auto_robot_design.motion_planning.trajectory_ik_manager import TrajectoryIKManager
from auto_robot_design.simulation.trajectory_movments import TrajectoryMovements

pin.seed(1)


@st.cache_resource
def build_constant_objects():
    thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]
    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
    density = MIT_CHEETAH_PARAMS_DICT["density"]
    body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]

    optimization_builder = ParametrizedBuilder(
        URDFLinkCreater3DConstraints,
        density={"default": density, "G": body_density},
        thickness={"default": thickness, "EE": 0.12},
        actuator={"default": actuator},
        size_ground=np.array(MIT_CHEETAH_PARAMS_DICT["size_ground"]),
        offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"],
    )

    predined_mesh = {"G": "./mesh/body.stl", "EE": "./mesh/wheel_small.stl"}

    mesh_creator = MeshCreator(predined_mesh)
    urdf_creator = URDFMeshCreator()
    visualization_builder = MeshBuilder(
        urdf_creator,
        mesh_creator,
        density={"default": density, "G": body_density},
        thickness={"default": thickness, "EE": 0.12},
        actuator={"default": actuator},
        size_ground=np.array(MIT_CHEETAH_PARAMS_DICT["size_ground"]),
        offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"],
    )

    # trajectories
    workspace_trajectory = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(
            get_workspace_trajectory([-0.15, -0.35], 0.14, 0.3, 30, 60)
        )
    )
    ground_symmetric_step1 = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(
            create_simple_step_trajectory(
                starting_point=[-0.14, -0.34],
                step_height=0.12,
                step_width=0.28,
                n_points=100,
            )
        )
    )
    ground_symmetric_step2 = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(
            create_simple_step_trajectory(
                starting_point=[-0.14 + 0.015, -0.34],
                step_height=0.10,
                step_width=-2 * (-0.14 + 0.015),
                n_points=100,
            )
        )
    )
    ground_symmetric_step3 = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(
            create_simple_step_trajectory(
                starting_point=[-0.14 + 0.025, -0.34],
                step_height=0.08,
                step_width=-2 * (-0.14 + 0.025),
                n_points=100,
            )
        )
    )
    central_vertical = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(get_vertical_trajectory(-0.34, 0.12, 0, 100)))
    left_vertical = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(get_vertical_trajectory(-0.34, 0.12, -0.12, 100)))
    right_vertical = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory(get_vertical_trajectory(-0.34, 0.12, 0.12, 100)))
    dict_trajectory_criteria = {
        "MASS": NeutralPoseMass(),
    }
    dict_point_criteria = {
        "Effective_Inertia": EffectiveInertiaCompute(),
        "Actuated_Mass": ActuatedMass(),
        "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ),
    }
    crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)
    graph_managers = {f"Topology_{i}": get_preset_by_index_with_bounds(i) for i in range(9)}

    return graph_managers, optimization_builder, visualization_builder, crag, workspace_trajectory, ground_symmetric_step1, ground_symmetric_step2, ground_symmetric_step3, central_vertical, left_vertical, right_vertical

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

from auto_robot_design.optimization.rewards.reward_base import PositioningConstrain, PositioningErrorCalculator, RewardManager
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability, MeanHeavyLiftingReward, MinAccelerationCapability
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import EndPointZRRReward, VelocityReward, ManipulabilityReward, ForceEllipsoidReward, ZRRReward, MinForceReward, MinManipulabilityReward,DexterityIndexReward
from auto_robot_design.optimization.rewards.inertia_rewards import MassReward, EndPointIMFReward,ActuatedMassReward, TrajectoryIMFReward
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE, ManipJacobian
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

    reward_list=[]
    reward_list.append(EndPointIMFReward(imf_key='IMF', trajectory_key="traj_6d", error_key="error"))
    reward_list.append(ActuatedMassReward(mass_key='Actuated_Mass'))
    reward_list.append(TrajectoryIMFReward(imf_key='IMF',trajectory_key="traj_6d", error_key="error"))
    reward_list.append(VelocityReward(manipulability_key='Manip_Jacobian', trajectory_key="traj_6d", error_key="error"))
    reward_list.append(ManipulabilityReward(manipulability_key='MANIP',
                                                    trajectory_key="traj_6d", error_key="error"))
    reward_list.append(MinManipulabilityReward(manipulability_key='Manip_Jacobian',
                                                    trajectory_key="traj_6d", error_key="error"))
    reward_list.append(ForceEllipsoidReward(manipulability_key='Manip_Jacobian',
                                                    trajectory_key="traj_6d", error_key="error"))
    reward_list.append(MinForceReward(manipulability_key='Manip_Jacobian',
                                                    trajectory_key="traj_6d", error_key="error"))
    motor = MIT_CHEETAH_PARAMS_DICT["actuator"]
    return crag, reward_list,  motor