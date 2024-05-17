from functools import partial
from typing import Callable
from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
import numpy as np

import matplotlib.pyplot as plt

from auto_robot_design.description.actuators import MyActuator_RMD_MT_RH_17_100_N, TMotor_AK10_9, t_motor_actuators

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator, )

from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths
from auto_robot_design.optimization.saver import load_checkpoint
from auto_robot_design.pinokla.squat import SquatHopParameters, SimulateSquatHop
import dill
import os



def reward_with_context(sim_hopp: SimulateSquatHop, robo_urdf: str,
                        joint_description: dict, loop_description: dict,
                        x: float):
    NUMBER_LAST_VALUE = 150
    TRQ_DEVIDER = 1000

    q_act, vq_act, acc_act, tau = sim_hopp.simulate(robo_urdf,
                                                    joint_description,
                                                    loop_description, control_coefficient=float(x))
    trj_f = sim_hopp.create_traj_equation()
    t = np.linspace(0, sim_hopp.squat_hop_parameters.total_time, len(q_act))
    q_vq_acc_des = np.array(list(map(trj_f, t)))
    vq_des = q_vq_acc_des[:, 1]
    tail_vq_des = vq_des[-NUMBER_LAST_VALUE:]
    tail_vq_act = vq_act[-NUMBER_LAST_VALUE:]
    tail_vq_act = tail_vq_act[:, 0]
    vq_erorr = np.mean(np.abs(tail_vq_des - tail_vq_act))
    return vq_erorr


def min_error_control_brute_force(min_fun: Callable[[float], float]):
    x_vec = np.linspace(0.65, 0.9, 10)
    errors = []
    for x in x_vec:
        try:
            res = min_fun(x)
        except:
            res = 1
        errors.append(res)
    x_and_err = zip(x_vec, errors)
    key_fun = lambda tup: tup[1]
    min_x_and_error = min(x_and_err, key=key_fun)
    return min_x_and_error


def get_history_and_problem(path):
    problem = CalculateCriteriaProblemByWeigths.load(
        path)  
    checklpoint = load_checkpoint(path)

    optimizer = PymooOptimizer(problem, checklpoint)
    optimizer.load_history(path)
  
    return optimizer.history, problem

def get_sorted_history(history : dict):
    rewards =  np.array(history["F"]).flatten()
    x_value = np.array(history["X"]).flatten()
    ids_sorted = np.argsort(rewards)
    sorted_reward = rewards[ids_sorted]
    sorted_x_values = x_value[ids_sorted]
    return sorted_reward, sorted_x_values

def get_histogram_data(rewards):
    NUMBER_BINS = 10
    _, bins_edg = np.histogram(rewards, NUMBER_BINS)
    bin_indices = np.digitize(rewards, bins_edg, right=True)
    bins_set_id = [np.where(bin_indices == i)[0] for i in range(1, len(bins_edg))]
    return bins_set_id

def get_tested_reward_and_x(sorted_reward :np.ndarray, sorted_x_values: np.ndarray):
    bins_set_id = get_histogram_data(sorted_reward)
    best_in_bins = [i[0] for i in bins_set_id]
    return sorted_reward[best_in_bins], sorted_x_values[best_in_bins]

  


path = "results\\vertical_acceleration_heavy_generator_0_2024-05-16_14-50-48"

problem = CalculateCriteriaProblemByWeigths.load(
    path)  # **{"elementwise_runner":runner})
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
actuator = TMotor_AK10_9()
thickness = 0.04
builder = ParametrizedBuilder(
    DetailedURDFCreatorFixedEE,
    size_ground=np.array([thickness * 5, thickness * 10, thickness * 2]),
    actuator=actuator,
    thickness=thickness)
robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
    graph, builder)

sqh_p = SquatHopParameters(hop_flight_hight=0.2,
                           squatting_up_hight=0.0,
                           squatting_down_hight=-0.28,
                           total_time=0.6)
hoppa = SimulateSquatHop(sqh_p)

opti = partial(reward_with_context, hoppa, robo_urdf, joint_description,
               loop_description)

res = min_error_control_brute_force(opti)
x_vec = np.linspace(0.65, 0.9, 10)


q_act, vq_act, acc_act, tau = hoppa.simulate(robo_urdf,
                                             joint_description,
                                             loop_description,
                                             control_coefficient=res[0],
                                             is_vis=True)

trj_f = hoppa.create_traj_equation()
t = np.linspace(0, sqh_p.total_time, len(q_act))
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
