from copy import copy, deepcopy
from auto_robot_design.pinokla.closed_loop_jacobian import inverseConstraintKinematicsSpeed
from auto_robot_design.pinokla.loader_tools import Robot, build_model_with_extensions, make_Robot_copy
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.description.builder import DetalizedURDFCreaterFixedEE
import pinocchio as pin
import numpy as np
import meshcat
from pinocchio.visualize import MeshcatVisualizer


from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount
import numpy as np

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

viz = MeshcatVisualizer(
    robo_planar.model, robo_planar.visual_model, robo_planar.visual_model)
viz.viewer = meshcat.Visualizer().open()
viz.clean()
viz.loadViewerModel()


pin.initConstraintDynamics(
    robo_planar.model, robo_planar.data, robo_planar.constraint_models)
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

robo_planar2 = Robot(*make_Robot_copy(robo_planar))
for i in range(N_it):
    
    
    
    a = pin.constraintDynamics(robo_planar.model, robo_planar.data, q, vq,
                               tauq , robo_planar.constraint_models, robo_planar.constraint_data, dyn_set)
    vq += a*DT
    q = pin.integrate(robo_planar.model, q, vq*DT)
    
    vq, J_closed = inverseConstraintKinematicsSpeed(
    robo_planar2.model,
    robo_planar2.data,
    robo_planar2.constraint_models,
    robo_planar2.constraint_data,
    robo_planar2.actuation_model,
    q,
    ee_id_g,
    robo_planar2.data.oMf[ee_id_g].action @ np.zeros(6),
)   

    # cho = J_closed.T @ np.array([0, 0, 10, 0, 0, 0])
    # tauq[id_mt1] = cho[0]
    # tauq[id_mt2] = cho[1]
    viz.display(q)
    print(q)

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
