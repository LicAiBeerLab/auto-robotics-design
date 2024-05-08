from copy import deepcopy
from dataclasses import dataclass
import re
import time
import odio_urdf
from auto_robot_design.pinokla.closed_loop_kinematics import (
    ForwardK,
    ForwardK1,
    closedLoopProximalMount,
)
from auto_robot_design.pinokla.criterion_agregator import (
    calc_criterion_along_traj,
    calc_traj_error_with_visualization,
    compute_along_q_space,
    load_criterion_traj,
    save_criterion_traj,
    calc_jacob_svd_along_traj
)
from hashlib import sha256
from auto_robot_design.pinokla.default_traj import (
    convert_x_y_to_6d_traj_xz,
    get_simple_spline,
)
from auto_robot_design.pinokla.loader_tools import (
    build_model_with_extensions,
    Robot,
    completeRobotLoader,
    completeRobotLoaderFromStr,
)
from auto_robot_design.pinokla.calc_criterion import (
    calc_IMF_along_traj,
    calc_force_ell_along_trj_trans,
    folow_traj_by_proximal_inv_k,
    kinematic_simulation,
    search_workspace,
    set_end_effector,
)
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import os
import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from itertools import product
import matplotlib.pyplot as plt
import os
import scipy as sp
from scipy.spatial import ConvexHull
from pathlib import Path
from scipy.interpolate import CubicSpline


def get_spline():
    # Sample data points
    x = np.array([-0.5, 0, 0.25])
    y = np.array([-0.4, -0.1, -0.4])
    y = y - 0.3
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


def get_circle(R, ps=10):
    ang = np.linspace(-np.pi, np.pi, ps)
    return np.array([R*np.cos(ang), R*np.sin(ang)]).T

if __name__=="__main__":

    DIR_NAME_FOR_LOAD = "generated_1_select"
    file_list = os.listdir(DIR_NAME_FOR_LOAD)
    handsome_guys = []
    new_list = [Path(DIR_NAME_FOR_LOAD + "/" + str(item)) for item in file_list]

    for path_i in new_list:
        res_i = load_criterion_traj(path_i)
        handsome_guys.append(res_i)
    for select_robot in handsome_guys[1:2]:

        robo = build_model_with_extensions(
            str(select_robot["urdf"]),
            joint_description=select_robot["mot_description"].item(),
            loop_description=select_robot["loop_description"].item(),
            fixed=True,
        )
        free_robo = build_model_with_extensions(
            str(select_robot["urdf"]),
            joint_description=select_robot["mot_description"].item(),
            loop_description=select_robot["loop_description"].item(),
            fixed=False,
        )

        # viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
        # viz.viewer = meshcat.Visualizer().open()
        # viz.clean()
        # viz.loadViewerModel()

        x_traj, y_traj = get_spline()
        traj_6d = convert_x_y_to_6d_traj_xz(x_traj, y_traj)
        # time.sleep(5)
        EFFECTOR_NAME = "EE"
        poses, q_array, constraint_errors = folow_traj_by_proximal_inv_k(
            robo.model,
            robo.data,
            robo.constraint_models,
            robo.constraint_data,
            EFFECTOR_NAME,
            traj_6d,
            # viz
        )
        pos_errors, q_array, svd_J = calc_jacob_svd_along_traj(
            robo,
            "G",
            EFFECTOR_NAME,
            traj_6d,
        )
        poses = np.array(poses)
        
        fig, (ax1, ax2) = plt.subplots(1,2)
        eig_J = np.zeros((poses.shape[0], 3))
        ez = np.array([0, 1])
        d_poses = np.diff(poses, axis=0)
        d_poses = np.vstack([d_poses, [0,0,0]])
        length_vectors = 0.04
        for id, pos, d_pos, svd, in zip(range(eig_J.shape[0]), poses, d_poses, svd_J):
            xy_circle = get_circle(0.05, 100)
            
            d_xy = d_pos[np.array([0,2])]
            J_svd = np.linalg.svd(svd)
            eig_J[id, :2] =  J_svd[1]
            eig_J[id, 2] = np.prod(J_svd[1])
            
            u1 = 1/J_svd[1][0] * J_svd[0][0,:]
            u2 = 1/J_svd[1][1] * J_svd[0][1,:]
            print(f'{"".join(str(np.round(pos, 4).tolist())):=^30}')
            print(f"Velocity: S_close: {eig_J[id, :2]}, Det CK:  {eig_J[id, 2]}")
            print(f"Force:  S_close: {eig_J[id, :2] ** (-1)}, Det CK: {1/ eig_J[id, 2]}")
            print(f"Projection: <d_traj|S1> {np.dot(d_xy, u1)}, <d_traj|S2> {np.dot(d_xy, u2)}")
            print(f"Projection: <ez|S1> {np.dot(ez, u1)}, <ez|S2> {np.dot(ez, u2)}")
            print()
            # print(f"InnerProd: {}")
            # x = sp.linalg.sqrtm((svd @ svd.T)) @ xy_circle.T
            # ax1.plot(x[0,:] + pos[0], x[1,:]+ pos[2])
            # x = sp.linalg.sqrtm(np.linalg.inv((svd @ svd.T))) @ xy_circle.T
            # ax2.plot(x[0,:] + pos[0], x[1,:]+ pos[2])
            ax2.arrow(pos[0], pos[2], d_pos[0] , d_pos[2], color="k")
            ax2.arrow(pos[0], pos[2],  1/J_svd[1][0] * J_svd[0][0,0] * length_vectors, 1/J_svd[1][0] *  J_svd[0][0,1] * length_vectors, color="r")
            ax2.arrow(pos[0], pos[2],  1/J_svd[1][1] * J_svd[0][1,0] * length_vectors,  1/J_svd[1][1] *  J_svd[0][1,1] * length_vectors, color="b")
            # ax2.arrow(pos[0], pos[2],  J_svd[0][0,0] * length_vectors,  J_svd[0][0,1] * length_vectors, color="r")
            # ax2.arrow(pos[0], pos[2],   J_svd[0][1,0] * length_vectors,    J_svd[0][1,1] * length_vectors, color="b")

        ax1.plot(x_traj,y_traj)
        ax2.plot(x_traj,y_traj)
        plt.axis("equal")
        plt.show()
        
        # plt.plot(np.arange(eig_J.shape[0]),eig_J[:,0])
        # plt.plot(np.arange(eig_J.shape[0]),eig_J[:,1])
        # plt.show()
        # plt.title("Manip")

        # plt.figure()
        # plt.scatter(poses[:, 0], poses[:, 2], c=traj_IMF, marker="d")
        # plt.colorbar()
        # plt.title("IFM")
        # plt.show()
