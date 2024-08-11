import time
from collections import UserDict
from enum import IntFlag, auto
from typing import NamedTuple, Optional

import numpy as np
import pinocchio as pin
from numpy.linalg import norm

from auto_robot_design.pinokla.closed_loop_jacobian import (
    closedLoopInverseKinematicsProximal, dq_dqmot,
    inverseConstraintKinematicsSpeed)
from auto_robot_design.pinokla.closed_loop_kinematics import (
    ForwardK, closedLoopProximalMount, closed_loop_ik_grad, closed_loop_ik_pseudo_inverse)
from auto_robot_design.pinokla.criterion_math import (calc_manipulability,
                                                      ImfProjections, calc_actuated_mass, calc_effective_inertia,
                                                      calc_force_ell_projection_along_trj, calc_IMF, calculate_mass,
                                                      convert_full_J_to_planar_xz)
from auto_robot_design.pinokla.loader_tools import Robot


def bfs_workspace_search(
    model,
    data,
    constraint_models,
    constraint_data,
    end_effector_frame: str,
    grid_6d: np.ndarray,
    viz=None,
    q_start: np.ndarray = None,
):

    q = pin.neutral(model)
    ee_frame_id = model.getFrameId(end_effector_frame)
    starting_index = ()

    poses = np.zeros((len(traj_6d), 3))
    reach_array = np.zeros(len(traj_6d))
    q_array = np.zeros((len(traj_6d), len(q)))
    constraint_errors = np.zeros((len(traj_6d), 1))
