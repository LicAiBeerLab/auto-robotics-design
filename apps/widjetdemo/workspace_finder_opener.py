from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import hppfcl
import meshcat
from apps.widjetdemo.robot_viz import RobotVisualizer
from auto_robot_design.description.builder import MIT_CHEETAH_PARAMS_DICT, jps_graph2pinocchio_robot
from auto_robot_design.optimization.problems import CalculateMultiCriteriaProblem
from auto_robot_design.pinokla.calc_criterion import folow_traj_by_proximal_inv_k
from auto_robot_design.pinokla.default_traj import get_workspace_trajectory
from apps.widjetdemo import traj_graph_setup
from apps.widjetdemo import create_reward_manager
from auto_robot_design.pinokla.loader_tools import Robot, make_Robot_copy
from auto_robot_design.user_interface.check_in_ellips import Ellipse, check_points_in_ellips
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import meshcat.geometry as mg

# model = pin.Model()
# collision_model = pin.GeometryModel()

# ell_clear = hppfcl.Ellipsoid(1, 0.5, 0.5)
# octree_object = pin.GeometryObject("octree", 0, pin.SE3.Identity(), ell_clear)
# octree_object.meshColor[3] = 0.3
# octree_object.meshColor[0] = 0.1
# octree_object.meshColor[0] = 0.1
# collision_model.addGeometryObject(octree_object)


def get_indices_by_point(mask: np.ndarray, reach_array: np.ndarray):
    mask_true_sum = np.sum(mask)
    reachability_sums = reach_array @ mask
    target_indices = np.where(reachability_sums == mask_true_sum)
    return target_indices[0]


graph, optimizing_joints, constrain_dict, builder, workspace_trajectory = traj_graph_setup.get_graph_and_traj(
    0)
reward_manager, crag, soft_constrain = create_reward_manager.get_manager_mock(
    workspace_trajectory)

actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
# Problem needs only for mutate graph by x
problem = CalculateMultiCriteriaProblem(graph, builder=builder,
                                        jp2limits=optimizing_joints,
                                        crag=[],
                                        soft_constrain=soft_constrain,
                                        rewards_and_trajectories=reward_manager,
                                        Actuator=actuator)


data = np.load("test_workspace_BF_RES_0.npz")
reach_arrays = data["reach_array"]
q_arrays = data["q_array"]


TOPOLGY_NAME = 0
points_x, points_y = get_workspace_trajectory([-0.15, -0.35], 0.2, 0.3, 10, 10)
points_x_1, points_y_1 = get_workspace_trajectory(
    [-0.15, -0.35], 0.2, 0.3, 50, 50)
ellipse = Ellipse(np.array([0, -0.25]), np.deg2rad(30), np.array([0.05, 0.1]))
point_ellipse = ellipse.get_points()


points = np.vstack([points_x.flatten(), points_y.flatten()])
points_1 = np.vstack([points_x_1.flatten(), points_y_1.flatten()])
mask = check_points_in_ellips(points, ellipse)
mask_1 = check_points_in_ellips(points_1, ellipse)
points_1_m = np.vstack([points_x_1[mask_1].flatten(),
                       points_y_1[mask_1].flatten()])

traj_3d = np.zeros((len(points_x_1[mask_1]), 3), dtype=np.float64)
traj_3d[:, 0] = points_x_1[mask_1].flatten()
traj_3d[:, 2] = points_y_1[mask_1].flatten()

rev_mask = np.array(1-mask, dtype="bool")


target_indices = get_indices_by_point(mask, reach_arrays)

x_0 = data["x_vec"][0]

problem.mutate_JP_by_xopt(x_0)
tested_gr = deepcopy(problem.graph)

fixed_robot, free_robot = jps_graph2pinocchio_robot(tested_gr, builder)
# fixed_robot.visual_model.addGeometryObject(octree_object)

rb_viz = RobotVisualizer(fixed_robot)
rb_viz.add_ellips_to_viz(np.array([0, 0, -0.25]), np.deg2rad(-30), 0.01, 0.01, is_red= True)
rb_viz.add_ellips_to_viz(np.array([0, 0, -0.25]), np.deg2rad(-30), 0.05, 0.1)


poses, q_array, constraint_errors = folow_traj_by_proximal_inv_k(
    fixed_robot.model, fixed_robot.data, fixed_robot.constraint_models, fixed_robot.constraint_data, "EE", traj_3d)
rb_viz.play_animation(q_array)


plt.figure(figsize=(10, 10))
plt.plot(point_ellipse[0, :], point_ellipse[1, :], "g", linewidth=3)
plt.scatter(points[:, rev_mask][0], points[:, rev_mask][1])
plt.scatter(points[:, mask][0], points[:, mask][1])

plt.show()
