import numpy as np

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.utils import all_combinations_active_joints_n_actuator
from auto_robot_design.description.builder import DetalizedURDFCreaterFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions

import pinocchio as pin

gen = TwoLinkGenerator()
graph, constrain_dict = gen.get_standard_set()[4]


pairs = all_combinations_active_joints_n_actuator(graph, t_motor_actuators)

thickness = 0.04

density = 2700 / 2.8

print(pairs[0])
builder = ParametrizedBuilder(DetalizedURDFCreaterFixedEE,
                              density=density,
                              thickness={"default": thickness, "EE":0.08},
                              actuator=dict(pairs[0]),
                              size_ground=np.array([thickness*5, thickness*10, thickness*2]),
)

robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(graph, builder)

robo = build_model_with_extensions(robo_urdf,
                                joint_description=joint_description,
                                loop_description=loop_description,
                                actuator_context=None,
                                fixed=True)
    
q = pin.neutral(robo.model)
q = closedLoopProximalMount(robo.model, robo.data, robo.constraint_models, robo.constraint_data, q)
nominal_M = pin.crba(robo.model, robo.data, q)
nominal_names = np.array(robo.model.names)[1:]
for pair_j_act in pairs:
    robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(graph, builder)

    robo = build_model_with_extensions(robo_urdf,
                                    joint_description=joint_description,
                                    loop_description=loop_description,
                                    actuator_context=pair_j_act,
                                    fixed=True)
    q = pin.neutral(robo.model)
    q = closedLoopProximalMount(robo.model, robo.data, robo.constraint_models, robo.constraint_data, q)
    M = pin.crba(robo.model, robo.data, q)
    ids = np.array([robo.model.joints[robo.model.getJointId(name)].idx_v for name in nominal_names])
    E = np.zeros_like(M)
    E[np.arange(E.shape[0]), ids] = 1
    M = E @ M @ E.T
    print("Inertia Matrix in initial configuration with armature and without")
    print(M.round(3))
    print(nominal_M.round(3))
    print(f"Joint Ordere: {nominal_names}")
    print(f"Reflected Inertia of Acts: {[(j_a[0], np.round(j_a[1].reduction_ratio ** -2 * j_a[1].inertia,5)) for j_a in pair_j_act]}")
    print("Difference Inertia Matrix in initial configuration with armature and without")
    print((M - nominal_M).round(4))
    print()