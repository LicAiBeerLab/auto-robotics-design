import time
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from auto_robot_design.control.trajectory_planning import trajectory_planning
from auto_robot_design.pinokla.closed_loop_jacobian import (
    inverseConstraintKinematicsSpeed,
    inverseConstraintPlaneKinematicsSpeed,
)
from auto_robot_design.pinokla.default_traj import (
    convert_x_y_to_6d_traj_xz,
    create_simple_step_trajectory,
    get_simple_spline,
)
from auto_robot_design.generator.topologies.bounds_preset import (
    get_preset_by_index_with_bounds,
)

from auto_robot_design.description.builder import (
    ParametrizedBuilder,
    URDFLinkCreator,
    URDFLinkCreater3DConstraints,
    DetailedURDFCreatorFixedEE,
    jps_graph2pinocchio_robot,
    jps_graph2pinocchio_robot_3d_constraints
)

import pinocchio as pin
import numpy as np
import meshcat
from pinocchio.visualize import MeshcatVisualizer
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R

import os
import sys
from auto_robot_design.pinokla.closed_loop_kinematics import (
    closedLoopInverseKinematicsProximal,
    openLoopInverseKinematicsProximal,
    closedLoopProximalMount,
)

from auto_robot_design.control.model_based import (
    TorqueComputedControl,
    OperationSpacePDControl,
)
from auto_robot_design.simulation.trajectory_movments import TrajectoryMovements

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, "../utils")
sys.path.append(mymodule_dir)


# Load robot
builder = ParametrizedBuilder(URDFLinkCreater3DConstraints)
np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

graph_manager = get_preset_by_index_with_bounds(0)

x_centre = graph_manager.generate_central_from_mutation_range()
graph_jp = graph_manager.get_graph(x_centre)

robo, __ = jps_graph2pinocchio_robot_3d_constraints(graph_jp, builder)


# Visualizer

viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
viz.viewer = meshcat.Visualizer().open()
viz.clean()
viz.loadViewerModel()

ee_id_ee = robo.model.getFrameId("EE")
q = np.zeros(robo.model.nq)
# viz.displayFrames(True)
# Trajectory by points

# x_point = np.array([-0.5, 0, 0.25]) * 0.5
# y_point = np.array([-0.4, -0.1, -0.4]) * 0.5
# y_point = y_point - 0.7
# traj_6d = convert_x_y_to_6d_traj_xz(x_point, y_point)
traj_6d = convert_x_y_to_6d_traj_xz(
    *create_simple_step_trajectory(
        starting_point=[-0.11, -0.2], step_height=0.07, step_width=0.22, n_points=20
    )
)

# Trajectory by points in joint space
q_des_points = []
time_start = time.time()
for num, i_pos in enumerate(traj_6d):
    q, min_feas, is_reach = closedLoopInverseKinematicsProximal(
        robo.model,
        robo.data,
        robo.constraint_models,
        robo.constraint_data,
        i_pos,
        ee_id_ee,
        onlytranslation=True,
        q_start=q,
        eps=1e-3,
        mu=1e-2,
    )
    ballID = "world/ball" + str(num)
    material = meshcat.geometry.MeshPhongMaterial()
    if not is_reach:
        q = closedLoopProximalMount(
            robo.model, robo.data, robo.constraint_models, robo.constraint_data, q
        )
        material.color = int(0xFF0000)
    else:
        material.color = int(0x00FF00)
    material.opacity = 0.3
    viz.viewer[ballID].set_object(meshcat.geometry.Sphere(0.001), material)
    T = np.r_[np.c_[np.eye(3), i_pos[:3]], np.array([[0, 0, 0, 1]])]
    viz.viewer[ballID].set_transform(T)
    q_des_points.append(q.copy())
    boxID = "world/box" + str(num)
    box_material = meshcat.geometry.MeshPhongMaterial()
    box_material.color = int(0xAFAF00)
    viz.viewer[boxID].set_object(
        meshcat.geometry.Box([0.001, 0.001, 0.001]), box_material
    )
    pin.framesForwardKinematics(robo.model, robo.data, q)
    real_ee_pos = robo.data.oMf[ee_id_ee].translation
    Tbox = np.r_[np.c_[np.eye(3), real_ee_pos[:3]], np.array([[0, 0, 0, 1]])]
    viz.viewer[boxID].set_transform(Tbox)
print("Time for IK: ", time.time() - time_start)
q = q_des_points[0]

# for i, q in enumerate(q_des_points):
#     viz.display(q)
#     time.sleep(0.5)

# q = q_des_points[0]
q = q_des_points[-10]
viz.display(q)

# final_time = 0.8
name_ee = "EE"
model = robo.model
data = robo.data

Kp = 0.1
Kd = np.sqrt(Kp) * 2
DT = 1e-5
id_ee = model.getFrameId(name_ee)

P_ini = traj_6d[-12]
pin.framesForwardKinematics(model, data, q)

pos = np.concatenate([data.oMf[id_ee].translation, pin.log(data.oMf[id_ee]).angular])
oldpos = pos.copy()
it = 0
Lxtest = []
old_vq = np.zeros(model.nv)
while norm(pos - P_ini) > 1e-6 and it < 10000:
    Lxtest.append(norm(pos - P_ini))
    vreel = pos - oldpos
    # va = Kp * (P_ini - pos)
    va = Kp * (P_ini - pos) / DT 
    # va = (P_ini - pos) / DT 
    va[4] = 0
    print(va)
    if np.isclose(np.sum(va), 0):
        print("zero")
    vq, Jf36_closed = inverseConstraintPlaneKinematicsSpeed(
        model,
        data,
        robo.constraint_models,
        robo.constraint_data,
        robo.actuation_model,
        q,
        id_ee,
        # va,
        data.oMf[id_ee].action @ va,
    )
    # vq[robo.actuation_model.idvfree] = np.zeros_like(robo.actuation_model.idvfree)
    # vq[0] = 0
    old_vq += vq
    print(vq * DT)
    q = pin.integrate(model, q, vq * DT)
    viz.display(q)
    pin.framesForwardKinematics(model, data, q)
    it = it + 1
    pos = np.concatenate(
        [data.oMf[id_ee].translation, pin.log(data.oMf[id_ee]).angular]
    )
    err = np.sum([norm(cd.contact_placement_error.np) for cd in robo.constraint_data])
    if err > 1e-1:
        break  # check the start position

# viz.viewer.close()
# # Init dynamics
# pin.initConstraintDynamics(robo.model, robo.data, robo.constraint_models)
# DT = 1e-3
# N_it = int(final_time / DT)
# tauq = np.zeros(robo.model.nv)
# id_mt1 = robo.actuation_model.idMotJoints[0]
# id_mt2 = robo.actuation_model.idqmot[1]
# tauq[id_mt1] = 0
# tauq[id_mt2] = 0
# vq = np.zeros(robo.model.nv)

# accuracy = 1e-8
# mu_sim = 1e-8
# max_it = 100
# dyn_set = pin.ProximalSettings(accuracy, mu_sim, max_it)

# tauq = np.zeros(robo.model.nv)
# id_mt1 = robo.actuation_model.idqmot[0]
# id_mt2 = robo.actuation_model.idqmot[1]
# id_vmt1 = robo.actuation_model.idvmot[0]
# id_vmt2 = robo.actuation_model.idvmot[1]

# vq = np.zeros(robo.model.nv)

# # Trajectory generation in joint space
# q_des_points = np.array(q_des_points)
# __, q_des_traj, dq_des_traj, ddq_des_traj = trajectory_planning(
#     q_des_points.T[[id_mt1, id_mt2], :], 0, 0, 0, N_it * DT, DT, True
# )

# name_ee = "EE"
# # Trajectory generation in operational space
# test = TrajectoryMovements(traj_6d[:,[0,2]], final_time, DT, name_ee)

# __, traj6d, traj_d6d = test.prepare_trajectory(robo)
# xz_ee_des_arr = traj6d[:, [0, 2]]
# x_point = traj_6d[:, 0]
# y_point = traj_6d[:, 2]
# d_xz_ee_des_arr = traj_d6d[:, [0, 2]]
# # Init control

# # Torque computed control in joint space
# # K = 500 * np.eye(2)
# # Kd = 50 * np.eye(2)
# # ctrl = TorqueComputedControl(robo, K, Kd)

# # Operation space PD control
# Kimp = np.eye(6) * 1000
# Kimp[3, 3] = 0
# Kimp[4, 4] = 0
# Kimp[5, 5] = 0
# Kdimp = np.eye(6) * 100
# Kdimp[3, 3] = 0
# Kdimp[4, 4] = 0
# Kdimp[5, 5] = 0
# ctrl = OperationSpacePDControl(robo, Kimp, Kdimp, ee_id_ee)

# nvmot = len(robo.actuation_model.idvmot)
# t_arr = np.zeros(N_it)
# q_arr = np.zeros((N_it, robo.model.nq))
# q_des_arr = np.zeros((N_it, nvmot))
# vq_arr = np.zeros((N_it, robo.model.nv))
# taua_arr = np.zeros((N_it, 2))
# x_body_arr = np.zeros((N_it, 3))
# x_body_des_arr = np.zeros((N_it, 3))

# for i in range(N_it):
#     # Forward dynamics
#     a = pin.constraintDynamics(
#         robo.model,
#         robo.data,
#         q,
#         vq,
#         tauq,
#         robo.constraint_models,
#         robo.constraint_data,
#         dyn_set,
#     )
#     vq += a * DT
#     q = pin.integrate(robo.model, q, vq * DT)

#     viz.display(q)

#     q_d = np.zeros(robo.model.nq)
#     vq_d = np.zeros(robo.model.nv)
#     qa_d = q_des_traj[i]
#     vqa_d = dq_des_traj[i]

#     q_d[[id_mt1, id_mt2]] = qa_d
#     vq_d[[id_vmt1, id_vmt2]] = vqa_d
#     pin.framesForwardKinematics(robo.model, robo.data, q)
#     x_body_curr = np.concatenate(
#         (
#             robo.data.oMf[ee_id_ee].translation,
#             R.from_matrix(robo.data.oMf[ee_id_ee].rotation).as_rotvec(),
#         )
#     )
#     # Desired traj in operational space
#     x_body_des = np.zeros(6)
#     x_body_des[0] = xz_ee_des_arr[i, 0]
#     x_body_des[2] = xz_ee_des_arr[i, 1]

#     # Current velocity ee in operational space
#     v_body = np.concatenate(
#         (
#             pin.getFrameVelocity(
#                 robo.model, robo.data, ee_id_ee, pin.LOCAL_WORLD_ALIGNED
#             ).linear,
#             pin.getFrameVelocity(
#                 robo.model, robo.data, ee_id_ee, pin.LOCAL_WORLD_ALIGNED
#             ).angular,
#         )
#     )
#     # Torque computed control in joint space
#     # tauq = ctrl.compute(q, vq, qa_d, vqa_d, ddq_des_traj[i])
#     # Operation space PD control
#     tauq = ctrl.compute(q, vq, traj6d[i],traj_d6d[i])#np.zeros(6))

#     t_arr[i] = i * DT
#     q_arr[i] = q
#     vq_arr[i] = vq
#     taua_arr[i] = tauq[[id_mt1, id_mt2]]
#     q_des_arr[i] = qa_d
#     x_body_arr[i] = x_body_curr[:3]
#     x_body_des_arr[i] = x_body_des[:3]


# # Ploting
# fig, ax = plt.subplots(3, 1)


# def name(id):
#     if id in set([id_mt1, id_mt2]):
#         return "q_a" + str(id)
#     return "q" + str(id)


# for i in range(robo.model.nq):
#     if i not in set([id_mt1, id_mt2]):
#         ax[0].plot(t_arr, q_arr[:, i], label=name(i))
# ax[0].set_xlabel("time, s")
# ax[0].set_ylabel("q, rad")
# ax[0].set_xlim((t_arr[0], t_arr[-1]))
# ax[0].grid()
# ax[0].legend()

# for i in range(nvmot):
#     ax[1].plot(
#         t_arr, q_des_arr[:, i], label="q_des" + str(i), linestyle="--", linewidth=2
#     )
# for i in range(robo.model.nq):
#     if i in set([id_mt1, id_mt2]):
#         ax[1].plot(t_arr, q_arr[:, i], label=name(i))
# ax[1].set_xlabel("time, s")
# ax[1].set_ylabel("q, rad")
# ax[1].set_xlim((t_arr[0], t_arr[-1]))
# ax[1].grid()
# ax[1].legend()


# def name(id):
#     if id in set([id_vmt1, id_vmt2]):
#         return "dq_a" + str(id)
#     return "dq" + str(id)


# def name(id):
#     return "taua" + str(id)


# for i in range(2):
#     ax[2].plot(t_arr, taua_arr[:, i], label=name(i))
# ax[2].set_xlabel("time, s")
# ax[2].set_ylabel("tau_a, Nm")
# ax[2].set_xlim((t_arr[0], t_arr[-1]))
# ax[2].grid()
# ax[2].legend()

# plt.show()


# xz_ee_arr = x_body_arr[:, [0, 2]]
# xz_points = np.array([x_point, y_point]).T
# fig, ax = plt.subplots(2, 1)
# name_x = ["x", "z"]
# for i, name in enumerate(name_x):
#     ax[i].plot(t_arr, xz_ee_des_arr[:, i], label="des", linestyle="--", linewidth=2)
#     ax[i].plot(t_arr, xz_ee_arr[:, i], label="curr")
#     ax[i].plot(
#         np.linspace(0, DT * N_it, xz_points.shape[0]),
#         xz_points[:, i],
#         "o",
#         label="points",
#     )
#     ax[i].set_xlabel("time, s")
#     ax[i].set_ylabel(name + ", m")
#     ax[i].set_xlim((t_arr[0], t_arr[-1]))
#     ax[i].grid()
#     ax[i].legend()
# plt.show()

# plt.figure()
# plt.plot(
#     xz_ee_des_arr[:, 0], xz_ee_des_arr[:, 1], label="des", linestyle="--", linewidth=2
# )
# plt.plot(
#     xz_ee_des_arr[:, 0], xz_ee_des_arr[:, 1], label="des", linestyle="--", linewidth=2
# )
# plt.plot(xz_ee_arr[:, 0], xz_ee_arr[:, 1], label="curr")
# plt.xlabel("x, m")
# plt.ylabel("z, m")
# plt.grid()
# plt.legend()
# plt.show()

# # Check constraints
# err = np.sum(
#     [
#         norm(pin.log(cd.c1Mc2).np[: cm.size()])
#         for (cd, cm) in zip(robo.constraint_data, robo.constraint_models)
#     ]
# )
# print(err)
# graph_jp
