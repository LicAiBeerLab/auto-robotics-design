from auto_robot_design.pinokla.closed_loop_jacobian import inverseConstraintKinematicsSpeed, constraint_jacobian_active_to_passive, jacobian_constraint
from auto_robot_design.pinokla.loader_tools import Robot, build_model_with_extensions, make_Robot_copy
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.description.builder import DetalizedURDFCreaterFixedEE
import pinocchio as pin
import numpy as np
import meshcat
from pinocchio.visualize import MeshcatVisualizer


from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator,
)
from auto_robot_design.description.builder import (
    DetalizedURDFCreaterFixedEE,
    ParametrizedBuilder,
    jps_graph2urdf_by_bulder
)
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
from auto_robot_design.pinokla.robot_utils import add_3d_constrain_current_q

def get_pino_models():
    pass
    gen = TwoLinkGenerator()
    graph, constrain_dict = gen.get_standard_set()[1]


    pairs = all_combinations_active_joints_n_actuator(graph, t_motor_actuators)

    thickness = 0.04

    density = 2000

    print(pairs[0])
    builder = ParametrizedBuilder(
        DetalizedURDFCreaterFixedEE,
        density=density,
        thickness={"default": thickness, "EE": 0.08},
        actuator=dict(pairs[0]),
        size_ground=np.array([thickness * 10, thickness * 10, thickness * 10]),
    )

    robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
        graph, builder
    )

    robo_planar = build_model_with_extensions(
        robo_urdf,
        joint_description=joint_description,
        loop_description=loop_description,
        actuator_context=None,
        fixed=False,
        root_joint_type=pin.JointModelPZ()

    )
    robo = build_model_with_extensions(
        robo_urdf,
        joint_description=joint_description,
        loop_description=loop_description,
        actuator_context=None,
        fixed=True
    )
    robo_free = build_model_with_extensions(
        robo_urdf,
        joint_description=joint_description,
        loop_description=loop_description,
        actuator_context=None,
        fixed=False
    )
    return robo, robo_planar, robo_free


def find_squat_q(translation_fix_robo: Robot, pos):
    pass

q_start_squat = np.array([-0.14596352, -0.34829299, 0.40971321, -0.53028812, 0.73886123, -0.27628751,
 -0.21486729, 0.42973463, 0.05512599 ])

robo, robo_planar, robo_free = get_pino_models()
 
q0 = closedLoopProximalMount(
    robo.model, robo.data, robo.constraint_models, robo.constraint_data)
q0_trans = np.concatenate([np.array([0]), q0])

robo_planar = add_3d_constrain_current_q(robo_planar, "EE", q0_trans)

# viz = MeshcatVisualizer(
#     robo_planar.model, robo_planar.visual_model, robo_planar.visual_model)
# viz.viewer = meshcat.Visualizer().open()
# viz.clean()
# viz.loadViewerModel()


pin.initConstraintDynamics(
    robo_planar.model, robo_planar.data, robo_planar.constraint_models)
DT = 1e-3
N_it = int(1e3)
tauq = np.zeros(robo_planar.model.nv)
id_mt1 = robo_planar.actuation_model.idqmot[0]
id_mt2 = robo_planar.actuation_model.idqmot[1]
id_vmt1 = robo_planar.actuation_model.idvmot[0]
id_vmt2 = robo_planar.actuation_model.idvmot[1]


vq = np.zeros(robo_planar.model.nv)

accuracy = 1e-8
mu_sim = 1e-8
max_it = 10000
dyn_set = pin.ProximalSettings(accuracy, mu_sim, max_it)
 
# tauq[id_mt1] = 10  # tau[0]
# tauq[id_mt2] = 10  # tau[1]


q = q0_trans
pin.computeGeneralizedGravity(robo_planar.model, robo_planar.data, q)

ee_id_g = robo_planar.model.getFrameId("G")
vq, J_closed = inverseConstraintKinematicsSpeed(
    robo_planar.model,
    robo_planar.data,
    robo_planar.constraint_models,
    robo_planar.constraint_data,
    robo_planar.actuation_model,
    q,
    ee_id_g,
    robo_planar.data.oMf[ee_id_g].action @ np.zeros(6),
)

Jda, E_tau = constraint_jacobian_active_to_passive(robo_planar.model,
    robo_planar.data,
    robo_planar.constraint_models,
    robo_planar.constraint_data,
    robo_planar.actuation_model,
    q)

nvmot = len(robo_planar.actuation_model.idvmot)

t_arr = np.zeros(N_it)
q_arr = np.zeros((N_it, robo_planar.model.nq))
q_des_arr = np.zeros((N_it, nvmot))
vq_arr = np.zeros((N_it, robo_planar.model.nv))
taua_arr = np.zeros((N_it, 2))

z_body_arr = np.zeros(N_it)

ad_arr = np.zeros((N_it, len(robo_planar.actuation_model.idvfree)))

prev_Jmot, prev_Jfree = jacobian_constraint(robo_planar.model, robo_planar.data, robo_planar.constraint_models, robo_planar.constraint_data,  robo_planar.actuation_model, q)

robo_planar2 = Robot(*make_Robot_copy(robo_planar))

for i in range(N_it):

    a = pin.constraintDynamics(robo_planar.model, robo_planar.data, q, vq,
                               tauq , robo_planar.constraint_models, robo_planar.constraint_data, dyn_set)
    vq += a*DT
    q = pin.integrate(robo_planar.model, q, vq*DT)

    id_body = robo_planar.model.getFrameId("G")
    vq_cstr, J_closed = inverseConstraintKinematicsSpeed(
    robo_planar.model,
    robo_planar.data,
    robo_planar.constraint_models,
    robo_planar.constraint_data,
    robo_planar.actuation_model,
    q,
    ee_id_g,
    robo_planar.data.oMf[ee_id_g].action @ np.zeros(6),
)   
    
    q_d = np.zeros(robo_planar.model.nq)
    vq_d = np.zeros(robo_planar.model.nv)
    qa_d = np.array([0.0, 0.0])
    vqa_d = np.array([0.0, 0.0])
    
    q_d[[id_mt1, id_mt2]] = qa_d
    vq_d[[id_vmt1, id_vmt2]] = vqa_d
    
    M = pin.crba(robo_planar.model, robo_planar.data, q)
    b = pin.rnea(robo_planar.model, robo_planar.data, q, vq, np.zeros(robo_planar.model.nv))

    z_body_arr[i] = robo_planar.data.oMf[id_body].translation[2]
    
    x_body_curr = np.concatenate((robo_planar.data.oMf[id_body].translation, R.from_matrix(robo_planar.data.oMf[id_body].rotation).as_rotvec()))
    
    Jmot, Jfree = jacobian_constraint(robo_planar.model, robo_planar.data, robo_planar.constraint_models, robo_planar.constraint_data, robo_planar.actuation_model, q)
    
    d_Jmot = ((Jmot - prev_Jmot)/DT).round(6)
    d_Jfree = ((Jfree - prev_Jfree)/DT).round(6)
    # pin.dDifference
    a_d = -np.linalg.pinv(Jfree) @ (d_Jmot @ vq[robo_planar.actuation_model.idvmot] + d_Jfree @ vq[robo_planar.actuation_model.idvfree])
    
    Jda, E_tau = constraint_jacobian_active_to_passive(robo_planar.model,
        robo_planar.data,
        robo_planar.constraint_models,
        robo_planar.constraint_data,
        robo_planar.actuation_model,
        q)

    v_body = np.concatenate((robo_planar.data.v[id_body].linear,robo_planar.data.v[id_body].angular))
    Ma = Jda.T @ E_tau.T @ M @ E_tau @ Jda
    b_a = Jda.T @ E_tau.T @ (b + M @ E_tau @ np.concatenate((np.zeros(nvmot),a_d)))
    q_a = q[[id_mt1, id_mt2]]
    vq_a = vq[[id_vmt1, id_vmt2]]
    a_a = a[[id_vmt1, id_vmt2]]
    K = 100
    Kd = 10
    
    Kimp = 5000
    Kdimp = 1000
    
    # q_d = np.array([0, 0.5*np.sin(2*np.pi * 2 * i*DT)])
    # tau_a = Ma @ (K * (qa_d - q_a) + Kd * (vqa_d - vq_a)) - b_a
    tau_a = Ma @ a_a - b_a
    cho = J_closed.T @ (Kimp * (np.array([0, 0, -0.1, 0, 0, 0]) - x_body_curr) + Kdimp * (np.zeros(6) - v_body))
    tauq[id_vmt1] = cho[0]# - tau_a[0]
    tauq[id_vmt2] = cho[1]# - tau_a[1]
    # viz.display(q)
    # print(f"q: {q.round(2)}")
    # print(f"tau: {tauq.round(2)}")
    # print(f"cho: {cho.round(2)}")
    
    t_arr[i] = i * DT
    q_arr[i] = q
    vq_arr[i] = vq
    taua_arr[i] = tau_a
    ad_arr[i] = a_d
    q_des_arr[i] = qa_d
    prev_Jmot = Jmot
    prev_Jfree = Jfree

print(q_a)

# err = np.sum([norm(pin.log(cd.c1Mc2).np[:cm.size()])
#              for (cd, cm) in zip(robo.constraint_data, robo.constraint_models)])
# print(err)

fig, ax = plt.subplots(3,1)

def name(id): 
    if id in set([id_mt1, id_mt2]):
        return "q_a" + str(id)
    return "q" + str(id)


for i in range(robo.model.nq):
    if i not in set([id_mt1, id_mt2]):
        ax[0].plot(t_arr, q_arr[:, i], label=name(i))
ax[0].set_xlabel("time, s")
ax[0].set_ylabel("q, rad")
ax[0].set_xlim((t_arr[0], t_arr[-1]))
ax[0].grid()
ax[0].legend()

for i in range(nvmot):
    ax[1].plot(t_arr, q_des_arr[:, i], label="q_des" + str(i), linestyle="--", linewidth=2)
for i in range(robo.model.nq):
    if i in set([id_mt1, id_mt2]):
        ax[1].plot(t_arr, q_arr[:, i], label=name(i))
ax[1].set_xlabel("time, s")
ax[1].set_ylabel("q, rad")
ax[1].set_xlim((t_arr[0], t_arr[-1]))
ax[1].grid()
ax[1].legend()

def name(id): 
    if id in set([id_vmt1, id_vmt2]):
        return "dq_a" + str(id)
    return "dq" + str(id)


# for i in range(robo.model.nq):
#     ax[1].plot(t_arr, vq_arr[:, i], label=name(i))
# ax[1].set_xlabel("time, s")
# ax[1].set_ylabel("dq, rad/s")
# ax[1].set_xlim((t_arr[0], t_arr[-1]))
# ax[1].grid()
# ax[1].legend()


def name(id): return "taua" + str(id)


for i in range(2):
    ax[2].plot(t_arr, taua_arr[:,i], label=name(i))
ax[2].set_xlabel("time, s")
ax[2].set_ylabel("tau_a, Nm")
ax[2].set_xlim((t_arr[0], t_arr[-1]))
ax[2].grid()
ax[2].legend()

plt.show()

plt.plot(t_arr, z_body_arr)
plt.grid()
plt.show()

plt.plot(t_arr, ad_arr, ".")
plt.grid()
plt.show()
# pin.computeGeneralizedGravity(robo.model, robo.data, q)
# total_mass = pin.computeTotalMass(robo.model)


# q_arr = np.zeros((N_it, robo.model.nq))
# vq_arr = np.zeros((N_it, robo.model.nq))
# tau_arr = np.zeros((N_it, tauq.size))
# t_arr = np.zeros(N_it)

# base_id = robo.model.getFrameId('G')
# ee_id = robo.model.getFrameId('EE')
# vq, J_closed = inverseConstraintKinematicsSpeed(
#     robo.model,
#     robo.data,
#     robo.constraint_models,
#     robo.constraint_data,
#     robo.actuation_model,
#     q,
#     ee_id,
#     robo.data.oMf[ee_id].action @ np.zeros(6),
# )
