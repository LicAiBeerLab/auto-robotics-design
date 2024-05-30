from math import tau
import os
import re
from turtle import pos

import matplotlib.pyplot as plt
import numpy as np
import dill

from auto_robot_design.optimization.rewards.reward_base import RewardManager
import auto_robot_design.simulation.evaluation as eval
from auto_robot_design.description.builder import jps_graph2pinocchio_robot
from auto_robot_design.simulation.trajectory_movments import (
    ControlOptProblem,
    TrajectoryMovements,
)


from auto_robot_design.optimization.analyze import (
    get_optimizer_and_problem,
    get_pareto_sample_linspace,
    get_pareto_sample_histogram,
    get_design_from_pareto_front,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Computer Modern Serif",
    }
)

IS_VIS = False
SAVE = False
SHOW = True

PATH = r"results/topology_0_2024-05-29_18-48-58"
FOLDER_NAME_PLOTS = "plots"

path_plots = os.path.join(PATH, FOLDER_NAME_PLOTS)

with open(os.path.join(PATH, "coeffs_OpPD_step.pkl"), "rb") as f:
    coeffs = dill.load(f)

optimizer, problem, res = get_optimizer_and_problem(PATH)

id2coeffs = {}
for x_var, Kp, Kd, input_for_task, fun in coeffs:
    design_index = int(np.argwhere(np.all(res.X == x_var, axis=1))[0].squeeze())
    id2coeffs[design_index] = (x_var, Kp, Kd, input_for_task, fun)
    
id_design2simout = {}


for id_design, coeffs in id2coeffs.items():
    problem.mutate_JP_by_xopt(res.X[id_design])
    robo, __ = jps_graph2pinocchio_robot(problem.graph, problem.builder)
    test = TrajectoryMovements(*coeffs[3])
    time_arr, des_traj_6d, __ = test.prepare_trajectory(robo)
    test.Kp = coeffs[1]
    test.Kd = coeffs[2]

    id_design2simout[id_design] = test.simulate(robo, IS_VIS, coeffs[4])
    

def get_reward_name_by_idF(reward_manager: RewardManager):
    
    reward_names = []
    for agg in reward_manager.agg_list:
        traj_id = agg[0][0]
        r_name = reward_manager.rewards[traj_id][0][0].__class__.__name__
        reward_names.append(r_name)
    return reward_names

reward_names = get_reward_name_by_idF(problem.rewards_and_trajectories)

power_arrs = []
pos_ee_arrs = []
tau_arrs = []
labels = []
for id_design, simout in id_design2simout.items():
    power_arrs.append(simout.power)
    pos_ee_arrs.append(simout.pos_ee_frame)
    tau_arrs.append(simout.tau)

    label = f"ID: {id_design}; "
    for i, r_name in enumerate(reward_names):
        label += f"{re.sub('[^A-Z]', '', r_name)}: {res.F[id_design, i]:.2f}; "
    labels.append(label)
    

if SAVE:
    mean_powers, sum_power, abs_powers =  eval.compare_power_quality(time_arr, power_arrs, True, path_plots, names=labels)
    mean_errors = eval.compare_movments_in_xz_plane(time_arr, pos_ee_arrs, des_traj_6d[:,:3], True, path_plots, names=labels)
    max_torques, mean_torques = eval.compare_torque_evaluation(time_arr, tau_arrs, True, path_plots, names=labels)
if SHOW:
    mean_powers, sum_power, abs_powers =  eval.compare_power_quality(time_arr, power_arrs, True, names=labels)
    mean_errors = eval.compare_movments_in_xz_plane(time_arr, pos_ee_arrs, des_traj_6d[:,:3], True, names=labels)
    max_torques, mean_torques = eval.compare_torque_evaluation(time_arr, tau_arrs, True, names=labels)


