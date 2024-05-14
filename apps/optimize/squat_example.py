

from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
import numpy as np

import matplotlib.pyplot as plt

from auto_robot_design.description.actuators import MyActuator_RMD_MT_RH_17_100_N, t_motor_actuators

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator, )

from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths
from auto_robot_design.optimization.saver import load_checkpoint
from auto_robot_design.pinokla.squat import SquatHopParameters, SimulateSquatHop
import dill
import os
 

path = "results\\th_1909_num1_2024-05-08_19-14-25"



problem = CalculateCriteriaProblemByWeigths.load(
    path
)  # **{"elementwise_runner":runner})
checklpoint = load_checkpoint(path)

optimizer = PymooOptimizer(problem, checklpoint)
optimizer.load_history(path)

hist_flat = np.array(optimizer.history["F"]).flatten()
not_super_best_id = np.argsort(hist_flat)[0]
sorted_reward = np.sort(hist_flat)
sorted_reward_big1 = sorted_reward[np.where(sorted_reward < -0.5)]
# plt.figure()
# plt.hist(sorted_reward_big1, 10)
# plt.show()
best_id = np.argsort(hist_flat)[0]
best_rew = optimizer.history["F"][best_id] 
not_super_best_rew = optimizer.history["F"][not_super_best_id] 
print(f"Best rew: {best_rew}")
print(f"Tested rew: {not_super_best_rew}")
problem.mutate_JP_by_xopt(optimizer.history["X"][not_super_best_id])
graph = problem.graph
 

#actuator = TMotor_AK10_9()
actuator = MyActuator_RMD_MT_RH_17_100_N()
thickness = 0.04
builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE, size_ground=np.array(
    [thickness*5, thickness*10, thickness*2]), actuator=actuator,thickness=thickness)
robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
    graph, builder)


sqh_p = SquatHopParameters(hop_flight_hight=0.25,
                           squatting_up_hight=0.0,
                           squatting_down_hight=-0.38,
                           total_time=0.8)
hoppa = SimulateSquatHop(sqh_p)


q_act, vq_act, acc_act, tau = hoppa.simulate(
    robo_urdf, joint_description, loop_description, is_vis=False)

trj_f = hoppa.create_traj_equation()
t = np.linspace(0, sqh_p.total_time,  len(q_act))
list__234 = np.array(list(map(trj_f, t)))


plt.figure()
plt.plot(acc_act[:, 0])
plt.plot(list__234[:, 2])
plt.title("Desired acceleration")
plt.xlabel("Time")
plt.ylabel("Z-acc")
plt.legend(["actual acc", "desired acc"])
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


plt.figure()

plt.plot(q_act[:, 0])
plt.plot(list__234[:, 0])
plt.title("Position")
plt.xlabel("Time")
plt.ylabel("Z-Pos")
plt.legend(["actual vel", "desired vel"])
plt.grid(True)

plt.show()
pass
