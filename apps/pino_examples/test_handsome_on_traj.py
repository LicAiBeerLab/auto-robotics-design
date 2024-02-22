from copy import deepcopy
from dataclasses import dataclass
import time
import odio_urdf
from auto_robot_design.pinokla.closed_loop_kinematics import ForwardK, ForwardK1, closedLoopProximalMount
from auto_robot_design.pinokla.criterion_agregator import calc_traj_error, compute_along_q_space, load_criterion_traj, save_criterion_traj
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
 
new_list = [Path(DIR_NAME_FOR_LOAD + "/" + str(item)) for item in file_list]

for num, path_i in enumerate(new_list):
    res_i = load_criterion_traj(path_i)
    urdf_i = str(res_i["urdf"])
    joint_description_i =res_i["mot_description"].item()
    loop_description_i = res_i["loop_description"].item()
    
    res = calc_traj_error(urdf_i, joint_description_i, loop_description_i)
    print(res)
 