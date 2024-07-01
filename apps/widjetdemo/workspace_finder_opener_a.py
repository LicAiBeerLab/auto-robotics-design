from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import hppfcl
import meshcat
from apps.widjetdemo.robot_viz import RobotVisualizer
from auto_robot_design.description.builder import MIT_CHEETAH_PARAMS_DICT, jps_graph2pinocchio_robot
from auto_robot_design.optimization.dataset.dataset_sort import RewardCalculator
from auto_robot_design.optimization.problems import CalculateMultiCriteriaProblem, get_optimizing_joints
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import MinForceReward
from auto_robot_design.pinokla.calc_criterion import ManipJacobian, MovmentSurface, folow_traj_by_proximal_inv_k
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_workspace_trajectory
from apps.widjetdemo import traj_graph_setup
from apps.widjetdemo import create_reward_manager
from auto_robot_design.pinokla.loader_tools import Robot, make_Robot_copy
from auto_robot_design.user_interface.check_in_ellips import Ellipse, check_points_in_ellips
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import meshcat.geometry as mg


def get_indices_by_point(mask: np.ndarray, reach_array: np.ndarray):
    mask_true_sum = np.sum(mask)
    reachability_sums = reach_array @ mask
    target_indices = np.where(reachability_sums == mask_true_sum)
    return target_indices[0]


graph, optimizing_joints, constrain_dict, builder, workspace_trajectory = traj_graph_setup.get_graph_and_traj(
    8)
reward_manager, crag, soft_constrain = create_reward_manager.get_manager_mock(
    workspace_trajectory)

actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
# Problem needs only for mutate graph by x
problem = CalculateMultiCriteriaProblem(
    graph,
    builder=builder,
    jp2limits=optimizing_joints,
    crag=[],
    soft_constrain=soft_constrain,
    rewards_and_trajectories=reward_manager,
    Actuator=actuator)

data = np.load("WORKSPACE_TOP8.npz")
reach_arrays = data["reach_array"]
q_arrays = data["q_array"]


optimizing_joints = get_optimizing_joints(graph, constrain_dict)

dict_trajectory_criteria = {}
# criteria calculated for each point on the trajectory
dict_point_criteria = {
    "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ)
}
# special object that calculates the criteria for a robot and a trajectory
crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)
trajectory = convert_x_y_to_6d_traj_xz(*(np.array([-0.017]), np.array([-0.2167])))
reward = MinForceReward(manipulability_key="Manip_Jacobian", error_key='error')
reward.point_precision = 0.001

mask = np.ones(100, dtype=np.bool_)
target_indices = get_indices_by_point(mask, reach_arrays)

dataset_x = data['x_vec']
rewards = RewardCalculator(graph, builder, crag, optimizing_joints).calculate_rewards(dataset_x[0:700], reward, trajectory)
print(dataset_x[np.argmax(rewards)])
