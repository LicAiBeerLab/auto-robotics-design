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

DIR_NAME_FOR_LOAD = "result"
file_list = os.listdir(DIR_NAME_FOR_LOAD)
file_list_2 = [
    "08519f8b599986a9db89bab16f00917e6bc514863b819eabde3b022e1a36fd2b",
    "0a85fb9a3ebd7eb91e10ac608e457f74a035016d9a0a61c0cfd7f845f30c5ef2",
    "0c504a6a9b9a0a37341a5adcca5f847b78182f9c3f4d261475e4c07b7541c73f",
    '119f10e0b2d175be37338bd3fead634f6768cec40694209d036e49d5c4ee43d8',
    '1c58a625ebe73f0136144cfb15b4ed828679ce816b9efb204de7c1a183c0afcb',
    '21a622e05d15323020afb28b21be14fb53473a9854a73965797f8e8a741e304e',
    '275e0a78420b18d4812d8536c7799a7f9bcd36a4bb09c767c9224fe57e891d9e',
    '30e2caef0013dfc63d313003af243afbf89d4198772ff82f6e8c17975ab2f990',
    '42e4ff10bdf20d8c9d7d923e83626cafcbecf4fa579876926cc184445d17effc',
    '44b5b025e25113be75438dd85767cac5c7cb0b99a87d7296dcdcc16b5745ac4f',
    '479fa900fbb2fe8a192d961070264731fb1becde359a86b6a60c19b5307575d1',
    '4e22fa3ec5a64da50ceb80f8e81667e92c2ac362ca5cf3e00278cc7392a188b8',
    '5e6723d8371f856d4d7975f18523a37934d6a8725dd281e9f818ca15e0dc71bb',
    '602189459e20663006d0228d8a848f3a29df9053798306c41041aa400d9886c6',
    '620471c945fdc82dc044fb4b58dbc68d60b21ac6ecc14aefe102d33df6938532',
    '63e288d237c15ade1b8724809886e435d6be15de57571cfe53110b248b4b693f',
    '7570b8eccaa01fdbe698047d604253465e52281eec79d060551b0a3d4fec4d74'
]
handsome_guys = []
new_list = [
    Path(DIR_NAME_FOR_LOAD + "/" + str(item) + ".npz") for item in file_list_2
]
for path_i in new_list:
    res_i = load_criterion_traj(path_i)
    robo = build_model_with_extensions(
        str(res_i["urdf"]),
        joint_description=res_i["mot_description"].item(),
        loop_description=res_i["loop_description"].item(),
        fixed=True)
    lin_space_num = 20
    q_space_mot_1 = np.linspace(-np.pi, np.pi, lin_space_num)
    q_space_mot_2 = np.linspace(-np.pi, np.pi, lin_space_num)
    q_mot_double_space = list(product(q_space_mot_1, q_space_mot_2))

    viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
    viz.viewer = meshcat.Visualizer().open()
    viz.clean()
    viz.loadViewerModel()
    

 

    transform = pin.SE3(pin.Quaternion(0.707, 0, 0, 0.707), np.array([1.5, 0,
                                                                      0.5]))
 
    viz.viewer.set_transform(transform.np)
 
    search_workspace(robo.model, robo.data, "EE", "G",
                     np.array(q_mot_double_space), robo.actuation_model,
                     robo.constraint_models, viz)
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

    # plt.figure()
    # plt.scatter(x, y, c=traj_manipulability, cmap='viridis', vmax=1)
    # plt.title("Manip")
    # plt.xlabel("X")
    # plt.xlabel("Y")
    # plt.colorbar()

    # plt.figure()
    # plt.scatter(x, y, c=traj_force_cap, cmap='viridis')
    # plt.title("Force cap")
    # plt.xlabel("X")
    # plt.xlabel("Y")
    # plt.colorbar()

    # plt.figure()
    # plt.scatter(x, y, c=traj_IMF, cmap='viridis')
    # plt.title("IMF")
    # plt.xlabel("X")
    # plt.xlabel("Y")
    # plt.colorbar()

    # plt.show()
    # plt.close()
