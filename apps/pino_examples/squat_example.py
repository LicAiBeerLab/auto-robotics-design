

from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
import numpy as np

import matplotlib.pyplot as plt

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator, )

from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.simulation_compare import max_toque_ratio, сomparison_ratio_positive_torques
from auto_robot_design.pinokla.squat import SquatHopParameters, SimulateSquatHop
gen = TwoLinkGenerator()
graph, constrain_dict = gen.get_standard_set()[6]
graph2, constrain_dict2 = gen.get_standard_set()[2]

pairs = all_combinations_active_joints_n_actuator(graph, t_motor_actuators)
pairs2 = all_combinations_active_joints_n_actuator(graph2, t_motor_actuators)

thickness = 0.04

density = 1000

print(pairs[0])
builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE,
                              density={"default": density, "G":5000},
                              thickness={"default": thickness, "EE":0.08},
                              actuator=dict(pairs[0]),
                              size_ground=np.array([thickness*5, thickness*5, thickness*2]),
)
robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
    graph, builder)

builder.actuator = dict(pairs[0])

robo_urdf2, joint_description2, loop_description2 = jps_graph2urdf_by_bulder(
    graph2, builder)

sqh_p = SquatHopParameters(hop_flight_hight=0.3,
                           squatting_up_hight=0,
                           squatting_down_hight=-0.3,
                           total_time=0.55)
hoppa = SimulateSquatHop(sqh_p)


q_act, vq_act, acc_act, tau = hoppa.simulate(
    robo_urdf, joint_description, loop_description, is_vis=False)

q_act, vq_act, acc_act, tau = hoppa.simulate(
    robo_urdf, joint_description, loop_description, is_vis=False)

trj_f = hoppa.create_traj_equation()
t = np.linspace(0, sqh_p.total_time,  len(q_act))
list__234 = np.array(list(map(trj_f, t)))

trg1 = np.sin(80*0.001*t)
trg2 = np.sin(100*0.001*t)
double_sin = np.column_stack((trg1, trg2))
сomparison_ratio_positive_torques(tau, double_sin)
max_toque_ratio(tau, double_sin)
pass
print("FFF")


# plt.figure()
# plt.plot(list__234[:, 2])
# plt.title("Desired acceleration")
# plt.xlabel("Time")
# plt.ylabel("Z-acc")
# plt.grid(True)

# plt.figure()
# plt.plot(tau[:, 0])
# plt.plot(tau[:, 1])
# plt.title("Actual torques")
# plt.xlabel("Time")
# plt.ylabel("Torques")
# plt.grid(True)


# plt.figure()

# plt.plot(vq_act[:, 0])
# plt.plot(list__234[:, 1])
# plt.title("Velocities")
# plt.xlabel("Time")
# plt.ylabel("Z-vel")
# plt.legend(["actual vel", "desired vel"])
# plt.grid(True)

# plt.show()
pass
