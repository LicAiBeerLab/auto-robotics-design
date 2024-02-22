from copy import deepcopy
from dataclasses import dataclass
import time
import odio_urdf
from auto_robot_design.pinokla.closed_loop_kinematics import ForwardK, ForwardK1, closedLoopProximalMount
from auto_robot_design.pinokla.criterion_agregator import compute_along_q_space, load_criterion_traj, save_criterion_traj
from hashlib import sha256
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
from pathlib import Path

DIR_NAME_FOR_LOAD = "handsome"
file_list = os.listdir(DIR_NAME_FOR_LOAD)
handsome_guys = []
new_list = [Path(DIR_NAME_FOR_LOAD + "/" + str(item)) for item in file_list]
 


for num, res_i in enumerate(handsome_guys):

    select_meh = res_i
    workspace_xyz = select_meh["workspace_xyz"]
    traj_manipulability = select_meh["traj_manipulability"]
    available_q = select_meh["available_q"]
    traj_force_cap = select_meh["traj_force_cap"]
    traj_IMF = select_meh["traj_IMF"]

    x = list([x[0] for x in workspace_xyz])
    y = list([x[2] for x in workspace_xyz])
    try:
        work_space_convex_hull = ConvexHull(
            np.column_stack((workspace_xyz[:, 0], workspace_xyz[:, 2])))
        work_area = work_space_convex_hull.area
    except:
        work_area = 0

    print("Coverage q " + str(len(available_q) / (100 * 100)))
    print("Space " + str(work_area))

    plt.figure()
    plt.scatter(x, y, c=traj_manipulability, cmap='viridis', vmax=1)
    plt.title("Manip")
    plt.xlabel("X")
    plt.xlabel("Y")
    plt.colorbar()

    plt.figure()
    plt.scatter(x, y, c=traj_force_cap, cmap='viridis')
    plt.title("Force cap")
    plt.xlabel("X")
    plt.xlabel("Y")
    plt.colorbar()

    plt.figure()
    plt.scatter(x, y, c=traj_IMF, cmap='viridis')
    plt.title("IMF")
    plt.xlabel("X")
    plt.xlabel("Y")
    plt.colorbar()

    plt.show()
    plt.close()
