from calendar import c
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import time
from typing import NamedTuple
import odio_urdf
from auto_robot_design.pinokla.calc_criterion2 import DataDict, along_criteria_calc, iterate_over_q_space, moment_criteria_calc
from auto_robot_design.pinokla.default_traj import (
    convert_x_y_to_6d_traj,
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
    calc_foot_inertia_along_traj,
    calc_force_ell_along_trj_trans,
    calc_force_ell_projection_along_trj,
    calc_manipulability_along_trj,
    calc_manipulability_along_trj_trans,
    folow_traj_by_proximal_inv_k,
    kinematic_simulation,
    search_workspace,
    set_end_effector,
    calc_svd_j_along_trj_trans,
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
from scipy.spatial import ConvexHull
from pathlib import Path


def calculate_quasi_static_simdata(free_robot: Robot,
                                   fixed_robot: Robot,
                                   ee_frame_name: str,
                                   traj_6d: np.ndarray,
                                   viz=None) -> tuple[DataDict, DataDict]:
    poses, q_fixed, constraint_errors = folow_traj_by_proximal_inv_k(
        fixed_robot.model, fixed_robot.data, fixed_robot.constraint_models,
        fixed_robot.constraint_data, ee_frame_name, traj_6d, viz)

    normal_pose = np.array([0, 0, 0, 0, 0, 0, 1])
    free_body_q = np.repeat(normal_pose[np.newaxis, :], len(q_fixed), axis=0)
    free_space_q = np.concatenate((free_body_q, q_fixed), axis=1)

    res_dict_free = iterate_over_q_space(free_robot, free_space_q,
                                         ee_frame_name)
    res_dict_fixed = iterate_over_q_space(fixed_robot, q_fixed, ee_frame_name)

    res_dict_fixed["traj_6d_ee"] = poses
    res_dict_free["traj_6d_ee"] = poses

    res_dict_fixed["traj_6d"] = traj_6d
    res_dict_free["traj_6d"] = traj_6d

    return res_dict_free, res_dict_fixed


class CriteriaAggregator:

    def __init__(self) -> None:
        self.dict_moment_criteria = {}
        self.dict_along_criteria = {}
        self.traj_6d = {}

    def get_criteria_data(self, urdf_str: str, mot_des: dict, loop_des: dict):
        fixed_robot = build_model_with_extensions(urdf_str, mot_des, loop_des,
                                                  True)
        free_robot = build_model_with_extensions(urdf_str, mot_des, loop_des)

        res_dict_free, res_dict_fixed = calculate_quasi_static_simdata(
            free_robot, fixed_robot, self.traj_6d)

        moment_critria_trj = moment_criteria_calc(self.dict_moment_criteria,
                                                  res_dict_free)
        along_critria_trj = along_criteria_calc(self.dict_along_criteria,
                                                res_dict_fixed, fixed_robot)

        return moment_critria_trj, along_critria_trj, res_dict_fixed
