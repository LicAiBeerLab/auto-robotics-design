from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import time
from typing import NamedTuple
import odio_urdf
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj, convert_x_y_to_6d_traj_xz, get_simple_spline
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions, Robot, completeRobotLoader, completeRobotLoaderFromStr
from auto_robot_design.pinokla.calc_criterion import calc_IMF_along_traj, calc_foot_inertia_along_traj, calc_force_ell_along_trj_trans, calc_manipulability_along_trj, calc_manipulability_along_trj_trans, folow_traj_by_proximal_inv_k, kinematic_simulation, search_workspace, set_end_effector
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

def calc_traj_error_with_visualization():
    pass
@dataclass
class ComputeConfg:
    IMF: bool = True
    ForcreCapability: bool = True
    Manipulability: bool = True
    ApparentInertia: bool = True


def compute_along_q_space(rob: Robot, rob_free: Robot, base_frame_name: str, ee_frame_name: str, q_space: np.ndarray, cmp_cfg: ComputeConfg = ComputeConfg()):

    normal_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    free_body_q = np.repeat(normal_pose[np.newaxis, :], len(q_space), axis=0)
    free_space_q = np.concatenate((free_body_q, q_space), axis=1)

    free_traj_M, free_traj_J_closed, free_traj_dq = kinematic_simulation(
        rob_free.model, rob_free.data, rob_free.actuation_model, rob_free.constraint_models, rob_free.constraint_data, ee_frame_name, base_frame_name, free_space_q, False)

    traj_M, traj_J_closed, traj_dq = kinematic_simulation(
        rob.model, rob.data, rob.actuation_model, rob.constraint_models, rob.constraint_data, ee_frame_name, base_frame_name, q_space)
    if cmp_cfg.ForcreCapability:
        traj_force_cap = calc_force_ell_along_trj_trans(traj_J_closed)
    else:
        traj_force_cap = None

    if cmp_cfg.ApparentInertia:
        traj_foot_inertia = calc_foot_inertia_along_traj(
            free_traj_M, free_traj_dq, free_traj_J_closed)
    else:
        traj_foot_inertia = None

    if cmp_cfg.Manipulability:
        traj_manipulability = calc_manipulability_along_trj_trans(
            traj_J_closed)
    else:
        traj_manipulability = None

    if cmp_cfg.IMF:
        traj_IMF = calc_IMF_along_traj(
            free_traj_M, free_traj_dq, free_traj_J_closed)
    else:
        traj_IMF = None

    return (traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF)


def calc_criterion_on_workspace(robo: Robot, robo_free: Robot, base_frame_name: str, ee_frame_name: str, lin_space_num: int, cmp_cfg: ComputeConfg = ComputeConfg()):

    q_space_mot_1 = np.linspace(-np.pi, np.pi, lin_space_num)
    q_space_mot_2 = np.linspace(-np.pi, np.pi, lin_space_num)
    q_mot_double_space = list(product(q_space_mot_1, q_space_mot_2))

    workspace_xyz, available_q = search_workspace(robo.model, robo.data, ee_frame_name, base_frame_name, np.array(
        q_mot_double_space), robo.actuation_model, robo.constraint_models)

    try:
        traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF = compute_along_q_space(
            robo, robo_free, base_frame_name, ee_frame_name, available_q, cmp_cfg)
    except:
        traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF = None, None, None, None

    return workspace_xyz, available_q, traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF


def calc_criterion_along_traj(robo: Robot, robo_free: Robot, base_frame_name: str, ee_frame_name: str, traj_6d: np.ndarray, cmp_cfg: ComputeConfg = ComputeConfg()):

    poses_3d, q_array, constraint_errors = folow_traj_by_proximal_inv_k(
        robo.model, robo.data, robo.constraint_models, robo.constraint_data, ee_frame_name, traj_6d)
    pos_errors = traj_6d[:, :3] - poses_3d
    # pos_error_sum = sum(map(np.linalg.norm, pos_errors))
    try:
        traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF = compute_along_q_space(
            robo, robo_free, base_frame_name, ee_frame_name, q_array, cmp_cfg)
    except:
        traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF = None, None, None, None

    return pos_errors, q_array, traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF


def calc_reward_wrapper(urdf_str: str,
                        joint_des: dict,
                        loop_des: dict,
                        traj_6d: np.ndarray,
                        cmp_cfg: ComputeConfg = ComputeConfg(),
                        base_frame_name="G",
                        ee_frame_name="EE"):
    robo = build_model_with_extensions(urdf_str, joint_des, loop_des)
    free_robo = build_model_with_extensions(urdf_str, joint_des, loop_des,
                                            False)

    pos_errors, q_array, traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF = calc_criterion_along_traj(
        robo, free_robo, base_frame_name, ee_frame_name, traj_6d, cmp_cfg)

    return pos_errors, q_array, traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF


def calc_traj_error(urdf_str: str,
                    joint_des: dict,
                    loop_des: dict):
    cmp_cfg = ComputeConfg(False, False, False, False)
    x_traj, y_traj = get_simple_spline()
    traj_6d = convert_x_y_to_6d_traj_xz(x_traj, y_traj)
    pos_errors, q_array, traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF = calc_reward_wrapper(
        urdf_str, joint_des, loop_des, traj_6d)
    res = sum(np.linalg.norm(pos_errors, axis=1)) / traj_6d.shape[0]
    return res

def calc_traj_error_with_visualization(urdf_str: str,
                                       joint_des: dict,
                                        loop_des: dict):
    cmp_cfg = ComputeConfg(False, False, False, False)
    x_traj, y_traj = get_simple_spline()
    traj_6d = convert_x_y_to_6d_traj_xz(x_traj, y_traj)

    robo = build_model_with_extensions(urdf_str, joint_des, loop_des)
    viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
    viz.viewer = meshcat.Visualizer().open()
    viz.clean()
    viz.loadViewerModel()
    plt.show()
    poses_3d, q_array, constraint_errors = folow_traj_by_proximal_inv_k(
        robo.model, robo.data, robo.constraint_models, robo.constraint_data, "EE", traj_6d, viz)
    pos_errors = traj_6d[:, :3] - poses_3d
    plt.figure()
    plt.scatter(poses_3d[:, 0], poses_3d[:,2],   marker = "d")
    plt.scatter(traj_6d[:, 0], traj_6d[:,2],   marker = ".")
    plt.title("Traj")
 
    plt.show()
    res = sum(np.linalg.norm(pos_errors, axis=1)) / traj_6d.shape[0]
    return res

def calc_criterion_on_workspace_simple_input(
        urdf_str: str,
        joint_des: dict,
        loop_des: dict,
        base_frame_name: str,
        ee_frame_name: str,
        lin_space_num: int,
        cmp_cfg: ComputeConfg = ComputeConfg()):
    try:
        robo = build_model_with_extensions(urdf_str, joint_des, loop_des)
        free_robo = build_model_with_extensions(urdf_str, joint_des, loop_des,
                                                False)
        workspace_xyz, available_q, traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF = calc_criterion_on_workspace(
            robo, free_robo, base_frame_name, ee_frame_name, lin_space_num,
            cmp_cfg)
        robo_dict = {
            "urdf": urdf_str,
            "joint_des": joint_des,
            "loop_des": loop_des
        }
        res_dict = {
            "workspace_xyz": workspace_xyz,
            "available_q": available_q,
            "traj_force_cap": traj_force_cap,
            "traj_foot_inertia": traj_foot_inertia,
            "traj_manipulability": traj_manipulability,
            "traj_IMF": traj_IMF
        }

        coverage = len(available_q) / (lin_space_num * lin_space_num)
        if coverage > 0.5:
            print("Greate mech heare!")
    except:
        print("Validate is fail")
        robo_dict = {}
        res_dict = {}
    return robo_dict, res_dict


def save_criterion_traj(urdf: str, directory: str, loop_description: dict, mot_description: dict, data_dict: dict):

    graph_name = sha256(urdf.encode()).hexdigest()
    path_with_name = Path(directory) / graph_name
    savable_dict = {"urdf": urdf, "loop_description": loop_description,
                    "mot_description": mot_description}

    savable_dict.update(data_dict)
    os.makedirs(Path(directory), exist_ok=True)
    np.savez(path_with_name, **savable_dict)


def load_criterion_traj(name: str):
    path = Path(name)
    load_data = np.load(path, allow_pickle=True)
    return dict(load_data)
