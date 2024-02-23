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

from scipy.spatial import ConvexHull
from efim_zoo import (
    efim_robot_str4,
    efim_robot4_loop_description,
    efim_robot4_joint_description,
    efim_robot_str7,
    efim_robot7_joint_description,
    efim_robot7_loop_description,
    efim_robot_str6,
    efim_robot6_joint_description,
    efim_robot6_loop_description,
    efim_robot_str5,
    efim_robot5_joint_description,
    efim_robot5_loop_description,
    efim_robot_str8,
    efim_robot8_joint_description,
    efim_robot8_loop_description,)

robo = build_model_with_extensions(efim_robot_str6,
                                   joint_description=efim_robot6_joint_description,
                                   loop_description=efim_robot6_loop_description,
                                   fixed=True)


robo_free = build_model_with_extensions(
    efim_robot_str6,
    joint_description=efim_robot6_joint_description,
    loop_description=efim_robot6_loop_description,
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

q_space_mot_1 = np.linspace(-np.pi, np.pi, 100)
q_space_mot_2 = np.linspace(-np.pi, np.pi, 100)
q_mot_double_space = list(product(q_space_mot_1, q_space_mot_2))

workspace_xyz, available_q = search_workspace(robo.model, robo.data, EFFECTOR_NAME, BASE_FRAME, np.array(
    q_mot_double_space), robo.actuation_model, robo.constraint_models, viz)


traj_force_cap, traj_foot_inertia, traj_manipulability, traj_IMF = compute_along_q_space(
    robo, robo_free, BASE_FRAME, EFFECTOR_NAME, available_q)


x = list([x[0]for x in workspace_xyz])
y = list([x[2]for x in workspace_xyz])
colors = traj_manipulability

work_space_convex_hull = ConvexHull(np.column_stack(
    (workspace_xyz[:, 0], workspace_xyz[:, 2])))
work_area = work_space_convex_hull.area
dict_to_save = {"imf": traj_IMF}






print("Coverage q "  + str(len(available_q)/(100*100)))
print("Space "  +  str(work_area))

plt.figure()
plt.scatter(x,  y, c=traj_manipulability, cmap='viridis')
plt.title("Manip")
plt.xlabel("X")
plt.xlabel("Y")
plt.colorbar()

plt.figure()
plt.scatter(x,  y, c=traj_force_cap, cmap='viridis'  )
plt.title("Force cap")
plt.xlabel("X")
plt.xlabel("Y")
plt.colorbar()


plt.figure()
plt.scatter(x,  y, c=traj_IMF, cmap='viridis'  )
plt.title("IMF")
plt.xlabel("X")
plt.xlabel("Y")
plt.colorbar()

plt.show()
