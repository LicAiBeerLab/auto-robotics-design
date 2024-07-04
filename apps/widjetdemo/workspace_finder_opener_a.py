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


def get_sample_x(x_vec: np.ndarray, size: int):
    #np.random.seed(420)
    index = np.random.choice(x_vec.shape[0], size)
    sample_x = x_vec[index]
    return sample_x


def get_best_meshs(topology_number, mask_ell: Ellipse, reach_arrays, x_vecs, x_power, y_power, n_top=3):
    points_x_viz, points_y_viz = get_workspace_trajectory(
        [-0.15, -0.35], 0.2, 0.3, 50, 50)
    points_viz = np.vstack([points_x_viz.flatten(), points_y_viz.flatten()])
    mask_viz = check_points_in_ellips(points_viz, mask_ell)

    traj_3d_viz = np.zeros((len(points_x_viz[mask_viz]), 3), dtype=np.float64)
    traj_3d_viz[:, 0] = points_x_viz[mask_viz].flatten()
    traj_3d_viz[:, 2] = points_y_viz[mask_viz].flatten()

    SAMPLE_X_LEN = 1500
    graph, optimizing_joints, constrain_dict, builder, workspace_trajectory = traj_graph_setup.get_graph_and_traj(
        topology_number)
    reward_manager, mock_crag, soft_constrain = create_reward_manager.get_manager_mock(
        workspace_trajectory)
    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]

    dict_trajectory_criteria = {}
    # criteria calculated for each point on the trajectory
    dict_point_criteria = {
        "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ)
    }
    crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)
    # Problem needs only for mutate graph by x
    problem = CalculateMultiCriteriaProblem(
        graph,
        builder=builder,
        jp2limits=optimizing_joints,
        crag=[],
        soft_constrain=soft_constrain,
        rewards_and_trajectories=reward_manager,
        Actuator=actuator)

    trajectory_power_point = convert_x_y_to_6d_traj_xz(
        *(np.array([x_power]), np.array([y_power])))

    points_x = workspace_trajectory[:, 0]
    points_y = workspace_trajectory[:, 2]

    points = np.vstack([points_x.flatten(), points_y.flatten()])

    mask = check_points_in_ellips(points, mask_ell)

    target_indices = get_indices_by_point(mask, reach_arrays)

    x_vecs_reached = x_vecs[target_indices]
    sample_x = get_sample_x(x_vecs_reached, SAMPLE_X_LEN)

    reward = MinForceReward(
        manipulability_key="Manip_Jacobian", error_key='error')
    reward.point_precision = 0.001
    rewards = RewardCalculator(graph, builder, crag, optimizing_joints).calculate_rewards(
        sample_x, reward, trajectory_power_point)

    sorted_indx = np.argsort(rewards)
    best_indx = sorted_indx[-n_top:]
    best_x = sample_x[best_indx]
    best_rewards = rewards[best_indx]

    best_graphs = []
    best_robot = []
    for x_i in best_x:
        problem.mutate_JP_by_xopt(x_i)
        tested_gr = deepcopy(problem.graph)
        best_graphs.append(tested_gr)
        fixed_robot, free_robot = jps_graph2pinocchio_robot(tested_gr, builder)
        best_robot.append(fixed_robot)
    return best_graphs, best_robot, best_rewards, traj_3d_viz

def play_robot_animation(power_point_x, power_point_y, ellipse, traj_3d_viz, viz_robo):
    rb_viz = RobotVisualizer(viz_robo)
    rb_viz.add_ellips_to_viz(
    np.array([ellipse.p_center[0], 0, ellipse.p_center[1]]), -ellipse.angle, ellipse.axis[0], ellipse.axis[1])
    rb_viz.add_ellips_to_viz(np.array([power_point_x, 0, power_point_y]), np.deg2rad(0), 0.01, 0.01, is_red=True)
    poses, q_array, constraint_errors = folow_traj_by_proximal_inv_k(
    viz_robo.model, viz_robo.data, viz_robo.constraint_models, viz_robo.constraint_data, "EE", traj_3d_viz)
    rb_viz.play_animation(q_array)
 
data = np.load("WORKSPACE_TOP82000.npz")
reach_arrays = data["reach_array"]
q_arrays = data["q_array"]
x_vecs = data['x_vec']

power_point_x = 0.0
power_point_y = -0.25
ellipse = Ellipse(np.array([0, -0.25]), np.deg2rad(30), np.array([0.05, 0.1]))

point_ellipse = ellipse.get_points()
plt.plot(point_ellipse[0, :], point_ellipse[1, :], "g", linewidth=3)
plt.scatter([power_point_x], [power_point_y], marker="*")
plt.xlim([-0.15, 0.15])
plt.ylim([-0.35, -0.15])
plt.show()

best_graphs, best_robot, best_rewards, traj_3d_viz = get_best_meshs(
    8, ellipse, reach_arrays, x_vecs, power_point_x, power_point_y)

print(f"Max power values: 1){best_rewards.round(3)[0]},  2){best_rewards.round(3)[1]},  3){best_rewards.round(3)[2]} ")
viz_robo = best_robot[0]
for viz_robo_i in best_robot:
    play_robot_animation(power_point_x, power_point_y, ellipse, traj_3d_viz, viz_robo_i)

# optimizing_joints = get_optimizing_joints(graph, constrain_dict)

# dict_trajectory_criteria = {}
# # criteria calculated for each point on the trajectory
# dict_point_criteria = {
#     "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ)
# }
# # special object that calculates the criteria for a robot and a trajectory
# crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)
# x_target = workspace_trajectory[25][0]
# y_target = workspace_trajectory[25][2]
# trajectory = convert_x_y_to_6d_traj_xz(
#     *(np.array([x_target]), np.array([y_target])))
# reward = MinForceReward(manipulability_key="Manip_Jacobian", error_key='error')
# reward.point_precision = 0.001

# mask = np.ones(100, dtype=np.bool_)
# target_indices = get_indices_by_point(mask, reach_arrays)

# dataset_x = data['x_vec'][target_indices]
# poses_masked = data['poses'][target_indices]
# reach_masked = reach_arrays[target_indices]

# index = np.random.choice(dataset_x.shape[0], 1000)
# sample_x = dataset_x[index]
# rewards = RewardCalculator(graph, builder, crag, optimizing_joints).calculate_rewards(
#     sample_x, reward, trajectory)
# sorted_indx = np.argsort(rewards)
# best_indx = sorted_indx[-3:]

# best_x = sample_x[best_indx]
# rewards2 = RewardCalculator(graph, builder, crag, optimizing_joints).calculate_rewards(
#     best_x, reward, trajectory)
# print("coc")
