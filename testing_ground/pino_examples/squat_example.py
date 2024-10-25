from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
import numpy as np

import matplotlib.pyplot as plt

from auto_robot_design.description.actuators import t_motor_actuators
from auto_robot_design.generator.topologies.graph_manager_2l import GraphManager2L

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator, draw_joint_point)

from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.squat import SquatHopParameters, SimulateSquatHop

gm = GraphManager2L()
gm.reset()
gm.build_main(0.4, fully_actuated=True)
gm.set_mutation_ranges()
print(gm.mutation_ranges)
center = gm.generate_central_from_mutation_range()
center[0] = -0.2
print(center)
graph = gm.get_graph(center)
draw_joint_point(graph)
plt.show()

pairs = all_combinations_active_joints_n_actuator(graph, t_motor_actuators)

thickness = 0.04

density = 1000

print(pairs[0])
builder = ParametrizedBuilder(
    DetailedURDFCreatorFixedEE,
    density={
        "default": density,
        "G": 5000
    },
    thickness={
        "default": thickness,
        "EE": 0.08
    },
    actuator=dict(pairs[0]),
    size_ground=np.array([thickness * 5, thickness * 5, thickness * 2]),
)
robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
    graph, builder)
sqh_p = SquatHopParameters(hop_flight_hight=0.3,
                           squatting_up_hight=0,
                           squatting_down_hight=-0.1,
                           total_time=0.2)
hoppa = SimulateSquatHop(sqh_p)

q_act, vq_act, acc_act, tau = hoppa.simulate(robo_urdf,
                                             joint_description,
                                             loop_description,
                                             is_vis=True)

trj_f = hoppa.create_traj_equation()
t = np.linspace(0, sqh_p.total_time, len(q_act))
list__234 = np.array(list(map(trj_f, t)))

plt.figure()
plt.plot(list__234[:, 2])
plt.title("Desired acceleration")
plt.xlabel("Time")
plt.ylabel("Z-acc")
plt.grid(True)

plt.figure()
plt.plot(tau[:, 0])
plt.plot(tau[:, 1])
plt.title("Actual torques")
plt.xlabel("Time")
plt.ylabel("Torques")
plt.grid(True)

plt.figure()

plt.plot(vq_act[:, 0])
plt.plot(list__234[:, 1])
plt.title("Velocities")
plt.xlabel("Time")
plt.ylabel("Z-vel")
plt.legend(["actual vel", "desired vel"])
plt.grid(True)

plt.show()
pass
