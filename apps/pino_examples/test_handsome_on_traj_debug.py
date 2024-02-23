from copy import deepcopy
from dataclasses import dataclass
import time
import odio_urdf
from auto_robot_design.pinokla.closed_loop_kinematics import ForwardK, ForwardK1, closedLoopProximalMount
from auto_robot_design.pinokla.criterion_agregator import ComputeConfg, calc_criterion_along_traj, calc_reward_wrapper, calc_traj_error, compute_along_q_space, load_criterion_traj, save_criterion_traj
from hashlib import sha256
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions, Robot, completeRobotLoader, completeRobotLoaderFromStr
from auto_robot_design.pinokla.calc_criterion import calc_IMF_along_traj, calc_force_ell_along_trj_trans, folow_traj_by_proximal_inv_k, kinematic_simulation, search_workspace, set_end_effector
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
from pathlib import Path

DIR_NAME_FOR_LOAD = "apps\data\handsome"
file_list = os.listdir(DIR_NAME_FOR_LOAD)

new_list = [Path(DIR_NAME_FOR_LOAD + "/" + str(item)) for item in file_list]

for num, path_i in enumerate(new_list):
    res_i = load_criterion_traj(path_i)
    urdf_i = str(res_i["urdf"])
    workspace_xyz = res_i["workspace_xyz"]
    joint_description_i = res_i["mot_description"].item()
    loop_description_i = res_i["loop_description"].item()
    x = list([x[0] for x in workspace_xyz])
    y = list([x[2] for x in workspace_xyz])
 

    
    # plt.figure()
    # plt.scatter(x, y)
    # plt.title("Manip")
    # plt.xlabel("X")
    # plt.xlabel("Y")
    # plt.show()

    cmp_cfg = ComputeConfg(False, False, False, False)
    x_traj, y_traj = get_simple_spline()
    traj_6d = convert_x_y_to_6d_traj_xz(x_traj, y_traj)

    robo = build_model_with_extensions(urdf_i, joint_description_i, loop_description_i)
    free_robo = build_model_with_extensions(urdf_i, joint_description_i, loop_description_i,
                                            False)
    viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
    viz.viewer = meshcat.Visualizer().open()
    viz.clean()
    viz.loadViewerModel()
    
    poses_3d, q_array, constraint_errors = folow_traj_by_proximal_inv_k(
        robo.model, robo.data, robo.constraint_models, robo.constraint_data, "EE", traj_6d, viz)
    pos_errors = traj_6d[:, :3] - poses_3d
    plt.figure()
    plt.scatter(poses_3d[:, 0], poses_3d[:,2],   marker = "d")
    plt.scatter(traj_6d[:, 0], traj_6d[:,2],   marker = ".")
    plt.title("Traj")
 
    plt.show()
    res = np.linalg.norm(sum(pos_errors))
    print(res)
