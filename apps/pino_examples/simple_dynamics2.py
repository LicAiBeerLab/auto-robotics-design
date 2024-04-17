from copy import copy, deepcopy
from auto_robot_design.pinokla.closed_loop_jacobian import inverseConstraintKinematicsSpeed
from auto_robot_design.pinokla.loader_tools import Robot, build_model_with_extensions, make_Robot_copy
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.description.builder import DetalizedURDFCreaterFixedEE
import pinocchio as pin
import numpy as np
import meshcat
from pinocchio.visualize import MeshcatVisualizer

from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopInverseKinematicsProximal, closedLoopProximalMount
import numpy as np

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator, )
from auto_robot_design.description.builder import (DetalizedURDFCreaterFixedEE,
                                                   ParametrizedBuilder,
                                                   jps_graph2urdf_by_bulder)
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
from auto_robot_design.pinokla.robot_utils import add_3d_constrain_current_q

def add_root_joint(fixed_base_robo: Robot, joint_type = pin.JointModelFreeFlyer()):
    ROOT_JOINT_NAME =  "root_joint"
    new_robot = make_Robot_copy(fixed_base_robo)
    new_robot = Robot(*new_robot)
    new_robot_joint_names = list(new_robot.model.names)
    new_robot_joint_names.insert(1, ROOT_JOINT_NAME)
    
    
    int = new_robot.model.inertias[0]
    jid = new_robot.model.addJoint(0, joint_type, pin.SE3(), ROOT_JOINT_NAME)
    #new_robot.new_model.appendBodyToJoint(jid, iner, pin.SE3.Identity())
    return new_robot


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
        thickness={
            "default": thickness,
            "EE": 0.08
        },
        actuator=dict(pairs[0]),
        size_ground=np.array([thickness * 5, thickness * 5, thickness * 5]),
    )

    robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
        graph, builder)

    robo_planar = build_model_with_extensions(
        robo_urdf,
        joint_description=joint_description,
        loop_description=loop_description,
        actuator_context=None,
        fixed=False,
        root_joint_type=pin.JointModelPZ(),
        is_act_root_joint = False)
    
    robo = build_model_with_extensions(robo_urdf,
                                       joint_description=joint_description,
                                       loop_description=loop_description,
                                       actuator_context=None,
                                       fixed=True)
    
    robo_free = build_model_with_extensions(
        robo_urdf,
        joint_description=joint_description,
        loop_description=loop_description,
        actuator_context=None,
        fixed=False)
    return robo, robo_planar, robo_free


def find_squat_q(translation_fix_robo: Robot, pos, ee_name, q_s):
    ee_id_g = robo_planar.model.getFrameId(ee_name)
    q, min_feas, is_reach = closedLoopInverseKinematicsProximal(
        translation_fix_robo.model,
        translation_fix_robo.data,
        translation_fix_robo.constraint_models,
        translation_fix_robo.constraint_data,
        pos,
        ee_id_g,
        onlytranslation=True,
        q_start=q_s,
    )
    return q, is_reach

q_start_squat = np.array([
    -0.14596352, -0.34829299, 0.40971321, -0.53028812, 0.73886123, -0.27628751,
    -0.21486729, 0.42973463, 0.05512599
])

robo, robo_planar, robo_free = get_pino_models()

q0 = closedLoopProximalMount(robo.model, robo.data, robo.constraint_models,
                             robo.constraint_data)
q0_trans = np.concatenate([np.array([0]), q0])

robo_planar = add_3d_constrain_current_q(robo_planar, "EE", q0_trans)

viz = MeshcatVisualizer(robo_planar.model, robo_planar.visual_model,
                        robo_planar.visual_model)
viz.viewer = meshcat.Visualizer().open()
viz.clean()
viz.loadViewerModel()
pin.framesForwardKinematics(robo.model, robo.data, q0)
HIGHT = 0.3
ee_id_g = robo.model.getFrameId("G")
ee_id = robo.model.getFrameId("EE")
default_hight = robo.data.oMf[ee_id].translation
default_hight[2] = default_hight[2] + HIGHT
needed_q, min_feas, is_reach = closedLoopInverseKinematicsProximal(
        robo.model,
        robo.data,
        robo.constraint_models,
        robo.constraint_data,
        default_hight,
        ee_id,
        onlytranslation=True,
    )
koooooo  = add_root_joint(robo)
needed_q = np.concatenate([np.array([-HIGHT]), needed_q])
pin.framesForwardKinematics(robo_planar.model, robo_planar.data, needed_q)
viz.display(needed_q)
pass
pin.initConstraintDynamics(robo_planar.model, robo_planar.data,
                           robo_planar.constraint_models)
DT = 5e-4
N_it = int(3e3)
tauq = np.zeros(robo_planar.model.nv)
id_mt1 = robo_planar.actuation_model.idqmot[0]
id_mt2 = robo_planar.actuation_model.idqmot[1]
vq = np.zeros(robo_planar.model.nv)

accuracy = 1e-8
mu_sim = 1e-8
max_it = 10000
dyn_set = pin.ProximalSettings(accuracy, mu_sim, max_it)

# tauq[id_mt1] = 10  # tau[0]
# tauq[id_mt2] = 10  # tau[1]

q = needed_q
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

robo_planar2 = Robot(*make_Robot_copy(robo_planar))
o_g = robo_planar.data.g[0]
for i in range(N_it):

    a = pin.constraintDynamics(robo_planar.model, robo_planar.data, q, vq,
                               tauq, robo_planar.constraint_models,
                               robo_planar.constraint_data, dyn_set)
    vq += a * DT
    q = pin.integrate(robo_planar.model, q, vq * DT)

    # pin.computeJointJacobians(robo_planar2.model,robo_planar2.data,q)

    pin.getConstraintJacobian(robo_planar.model, robo_planar.data,
                              robo_planar.constraint_models[0],
                              robo_planar.constraint_data[0])
    vq2, J_closed = inverseConstraintKinematicsSpeed(
        robo_planar.model,
        robo_planar.data,
        robo_planar.constraint_models,
        robo_planar.constraint_data,
        robo_planar.actuation_model,
        q,
        ee_id_g,
        robo_planar.data.oMf[ee_id_g].action @ np.zeros(6),
    )

    cho = 1*J_closed.T @ np.array([0, 0, o_g, 0, 0, 0])

    tauq[id_mt1] = cho[0]
    tauq[id_mt2] = cho[1]
    viz.display(q)
    print(tauq.round(3))

# err = np.sum([norm(pin.log(cd.c1Mc2).np[:cm.size()])
#              for (cd, cm) in zip(robo.constraint_data, robo.constraint_models)])
# print(err)

# def name(id): return "q" + str(id)

# for i in range(robo.model.nq):
#     plt.plot(t_arr, q_arr[:, i], label=name(i))
# plt.xlabel("time, s")
# plt.ylabel("q, rad")
# plt.xlim((t_arr[0], t_arr[-1]))
# plt.grid()
# plt.legend()
# plt.show()

# def name(id): return "dq" + str(id)

# for i in range(robo.model.nq):
#     plt.plot(t_arr, vq_arr[:, i], label=name(i))
# plt.xlabel("time, s")
# plt.ylabel("dq, rad/s")
# plt.xlim((t_arr[0], t_arr[-1]))
# plt.grid()
# plt.legend()
# plt.show()

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
