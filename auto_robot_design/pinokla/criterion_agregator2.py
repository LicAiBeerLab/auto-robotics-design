from calendar import c
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import time
from typing import NamedTuple
import odio_urdf
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

