from copy import deepcopy
from dataclasses import dataclass
import time
import odio_urdf
from auto_robot_design.pinokla.closed_loop_kinematics import ForwardK, ForwardK1, closedLoopProximalMount
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions, Robot, completeRobotLoader, completeRobotLoaderFromStr
from auto_robot_design.pinokla.calc_criterion import calc_IMF_along_traj, calc_force_ell_along_trj_trans, kinematic_simulation, search_workspace, set_end_effector
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import os
import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from itertools import product
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull
 
from efim_zoo import (
    efim_robot_str3,
    efim_robot3_loop_description,
    efim_robot3_joint_description,
    efim_robot_str2,
    efim_robot2_joint_description,
    efim_robot2_loop_description

)
robo = build_model_with_extensions(efim_robot_str2,
                                   joint_description=efim_robot2_joint_description,
                                   loop_description=efim_robot2_loop_description,
                                   fixed=True)


robo_free = build_model_with_extensions(
    efim_robot_str2,
    joint_description=efim_robot2_joint_description,
    loop_description=efim_robot2_loop_description,
    fixed=False
)

q0 = closedLoopProximalMount(
    robo.model,
    robo.data,
    robo.constraint_models,
    robo.constraint_data,
    robo.actuation_model,
    max_it=100,
)


viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
viz.viewer = meshcat.Visualizer().open()
viz.clean()
viz.loadViewerModel()
viz.display(q0)


EFFECTOR_NAME = "EE"
BASE_FRAME = "G"
q_space_mot_1 = np.linspace(-np.pi, np.pi, 5)
q_space_mot_2 = np.linspace(-np.pi, np.pi, 5)
q_mot_double_space = list(product(q_space_mot_1, q_space_mot_2))

workspace_xyz, available_q = search_workspace(robo.model, robo.data, EFFECTOR_NAME, BASE_FRAME, np.array(
    q_mot_double_space), robo.actuation_model, robo.constraint_models)

traj_M, traj_J_closed, traj_dq = kinematic_simulation(
    robo.model, robo.data, robo.actuation_model, robo.constraint_models, robo.constraint_data, EFFECTOR_NAME, BASE_FRAME, available_q)

normal_pose = np.array([0, 0, 0, 0, 0, 0, 1])
free_body_q = np.repeat(normal_pose[np.newaxis, :], len(available_q), axis=0)
free_available_q = np.concatenate((free_body_q, available_q), axis=1)

free_traj_M, free_traj_J_closed, free_traj_dq = kinematic_simulation(
    robo_free.model, robo_free.data, robo_free.actuation_model, robo_free.constraint_models, robo_free.constraint_data, EFFECTOR_NAME, BASE_FRAME, free_available_q, False)

traj_force_cap = calc_force_ell_along_trj_trans(traj_J_closed)
work_space = ConvexHull(np.column_stack(
    (workspace_xyz[:, 0], workspace_xyz[:, 2])))

calc_IMF_along_traj(free_traj_M, free_traj_dq, free_traj_J_closed)
calc_IMF_along_traj(free_traj_M, free_traj_dq, free_traj_J_closed)
# plt.scatter(list([x[0]for x in workspace_xyz]),  list([x[2]for x in workspace_xyz]))
# plt.show()


pass
