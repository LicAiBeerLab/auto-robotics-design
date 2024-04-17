import numpy as np


import pinocchio as pin
from auto_robot_design.description.builder import Builder, URDFLinkCreater, jps_graph2urdf_parametrized
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator

from auto_robot_design.pinokla.calc_criterion import calc_IMF, calc_manipulability, convert_full_J_to_planar_xz
from auto_robot_design.pinokla.calc_criterion import iterate_over_q_space
from auto_robot_design.pinokla.closed_loop_jacobian import dq_dqmot, inverseConstraintKinematicsSpeed
from auto_robot_design.pinokla.loader_tools import Robot, build_model_with_extensions
from auto_robot_design.pinokla.robot_utils import freezeJointsWithoutVis


def inverseConstraintKinematicsSpeed_robo(robo: Robot, q: np.ndarray, frame_name_ee: str = "EE", des_ee_speed: np.ndarray = np.ones(6)):
    id_ee = robo.model.getFrameId(frame_name_ee)
    vq, J_closed = inverseConstraintKinematicsSpeed(
        robo.model, robo.data, robo.constraint_models, robo.constraint_data,
        robo.actuation_model, q, id_ee,
        robo.data.oMf[id_ee].action@des_ee_speed)
    return vq, J_closed

def freeze_joint_robo(robo: Robot, joint_f: str):
    joint_ee_id = robo.model.getJointId(joint_f)
    reduced_model, reduced_constraint_models, reduced_actuation_model = freezeJointsWithoutVis(
        robo.model, robo.constraint_models, robo.actuation_model, [joint_ee_id])

    constraint_data = [c.createData() for c in reduced_constraint_models]
    red_data = reduced_model.createData()
    robo_fixed_EE = Robot(reduced_model, reduced_constraint_models,
                          reduced_actuation_model, robo.visual_model, constraint_data, red_data)
    return robo_fixed_EE


gen = TwoLinkGenerator()
builder = Builder(URDFLinkCreater)
graphs_and_cons = gen.get_standard_set()
np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

graph_jp, constrain = graphs_and_cons[0]
robot_urdf, ative_joints, constraints = jps_graph2urdf_parametrized(graph_jp)

robo = build_model_with_extensions(robot_urdf, ative_joints, constraints)
free_robo = build_model_with_extensions(
    robot_urdf, ative_joints, constraints, fixed=False)
robo_fixed_EE = freeze_joint_robo(robo, "TL_ee")

robo.model.armature[robo.actuation_model.idvmot[:]] = np.array([0.1, 0.1])
free_robo.model.armature[robo.actuation_model.idvmot[:]] = np.array([0.1, 0.1])



q_0 = pin.neutral(robo.model)
q_0_ee = pin.neutral(robo_fixed_EE.model)
q0_free = np.concatenate([np.array([0, 0, 0, 0, 0, 0, 1]), q_0])
iterate_over_q_space(robo, [q_0], "EE")
M_ee = pin.crba(robo_fixed_EE.model, robo_fixed_EE.data, q_0_ee)
M = pin.crba(robo.model, robo.data, q_0)
LJ_free = []
for (cm, cd) in zip(free_robo.constraint_models, free_robo.constraint_data):
    Jc = pin.getConstraintJacobian(free_robo.model, free_robo.data, cm, cd)
    LJ_free.append(Jc)

LJ = []
for (cm, cd) in zip(robo.constraint_models, robo.constraint_data):
    Jc = pin.getConstraintJacobian(robo.model, robo.data, cm, cd)
    LJ.append(Jc)

LJ_ee = []
for (cm, cd) in zip(robo_fixed_EE.constraint_models, robo_fixed_EE.constraint_data):
    Jc = pin.getConstraintJacobian(
        robo_fixed_EE.model, robo_fixed_EE.data, cm, cd)
    LJ_ee.append(Jc)

id_ee_free = free_robo.model.getFrameId("EE")
id_EE_ee_free = robo_fixed_EE.model.getFrameId("EE")
id_ee = robo.model.getFrameId("EE")

pin.centerOfMass(robo.model, robo.data, q_0)

M_free = pin.crba(free_robo.model, free_robo.data, q0_free)
M = pin.crba(robo.model, robo.data, q_0)

vq_free, J_closed_free = inverseConstraintKinematicsSpeed_robo(
    free_robo, q0_free)
vq, Jf36_closed = inverseConstraintKinematicsSpeed_robo(robo, q_0)
vq_free2, J_closed_EE = inverseConstraintKinematicsSpeed_robo(
    robo_fixed_EE, q_0_ee)

dq_free = dq_dqmot(free_robo.model, free_robo.actuation_model, LJ_free)
dq = dq_dqmot(robo.model, robo.actuation_model, LJ)
dq_ee = dq_dqmot(robo_fixed_EE.model, robo_fixed_EE.actuation_model, LJ_ee)

planar_J = convert_full_J_to_planar_xz(Jf36_closed)

manip = calc_manipulability(planar_J[:2, :2])


planar_J = convert_full_J_to_planar_xz(Jf36_closed)
# Lambda = calc_foot_inertia(M, dq, planar_J)
# Lambda_ee = calc_foot_inertia(M_ee, dq_ee, J_closed_EE)
# Lambda_free = calc_foot_inertia(M_free, dq_free, J_closed_free)
ddd = np.linalg.svd(Jf36_closed@Jf36_closed.T)
IMF = calc_IMF(M_free, dq_free, J_closed_free)
pass

# draw_joint_point(graph_i)
# plt.show()

# for k, pack in enumerate(urdf_motors_cons_list):
#     urdf, mot, cons = pack
#     robo_dict, res_dict = calc_criterion_on_workspace_simple_input(urdf, mot, cons, "G", "EE", 100)
#     save_criterion_traj(robo_dict["urdf"], DIR_NAME, robo_dict["loop_des"],
#                     robo_dict["joint_des"], res_dict)
