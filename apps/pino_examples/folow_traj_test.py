from copy import deepcopy
from dataclasses import dataclass
import time
import odio_urdf
from auto_robot_design.pinokla.closed_loop_kinematics import ForwardK, ForwardK1, closedLoopProximalMount
from auto_robot_design.pinokla.criterion_agregator import calc_criterion_along_traj, compute_along_q_space, load_criterion_traj, save_criterion_traj
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
from scipy.interpolate import CubicSpline


def get_spline():
    # Sample data points
    x = np.array([-0.5, 0, 0.25])
    y = np.array([-0.4, -0.1, -0.4])
    # y = y + 0.15
    # x = x + 0.4
    # Create the cubic spline interpolator
    cs = CubicSpline(x, y)

    # Create a dense set of points where we evaluate the spline
    x_traj_spline = np.linspace(x.min(), x.max(), 20)
    y_traj_spline = cs(x_traj_spline)

    # Plot the original data points
    # plt.plot(x, y, 'o', label='data points')

    # Plot the spline interpolation
    # plt.plot(x_traj_spline, y_traj_spline, label='cubic spline')

    # plt.legend()
    # plt.show()
    return (x_traj_spline, y_traj_spline)


DIR_NAME_FOR_LOAD = "generated_1_select"
file_list = os.listdir(DIR_NAME_FOR_LOAD)
handsome_guys = []
new_list = [Path(DIR_NAME_FOR_LOAD + "/" + str(item)) for item in file_list]

for path_i in new_list:
    res_i = load_criterion_traj(path_i)
    handsome_guys.append(res_i)
for select_robot in handsome_guys:
 
    robo = build_model_with_extensions(
        str(select_robot["urdf"]),
        joint_description=select_robot["mot_description"].item(),
        loop_description=select_robot["loop_description"].item(),
        fixed=True)
    free_robo = build_model_with_extensions(
        str(select_robot["urdf"]),
        joint_description=select_robot["mot_description"].item(),
        loop_description=select_robot["loop_description"].item(),
        fixed=False)

    # viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
    # viz.viewer = meshcat.Visualizer().open()
    # viz.clean()
    # viz.loadViewerModel()

    x_traj, y_traj = get_spline()
    traj_6d = convert_x_y_to_6d_traj_xz(x_traj, y_traj)


    EFFECTOR_NAME = "EE"
    poses, q_array, constraint_errors = folow_traj_by_proximal_inv_k(
        robo.model, robo.data, robo.constraint_models, robo.constraint_data, EFFECTOR_NAME, traj_6d)
    pos_errors, q_array2, traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF = calc_criterion_along_traj(
        robo, free_robo, "G", "EE", traj_6d)

    poses = np.array(poses)

    plt.figure()
    plt.plot(traj_force_cap, marker="d")
 
    plt.title("Force cap")


    # plt.figure()
    # plt.plot(traj_foot_inertia, marker="d")
    
    # plt.title("Foot inertia")


    # plt.figure()
    # plt.plot(traj_manipulability, marker="d")
    
    # plt.title("Manip")


    # plt.figure()
    # plt.scatter(poses[:, 0],  poses[:, 2], c=traj_IMF, marker="d")
    # plt.colorbar()
    # plt.title("IFM")
plt.show()
