import os
import re

import matplotlib.pyplot as plt
import numpy as np
import dill

from auto_robot_design.description.builder import jps_graph2pinocchio_robot
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.optimization.rewards.reward_base import RewardManager
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

#### !!! BIG WARNING !!! ####
#### !!! CHANE THESE PARAMETERS FOR YOUR CASE !!! ####
MAGIC_TRAJECTORY_INDEX = 1

SAVE = True
SHOW = False

PATH = r"results/topology_0_2024-05-29_18-48-58"
FOLDER_NAME_PLOTS = "plots"

NUM_DESIGNS = 5

NAME_EE = "EE"

path_plots = os.path.join(PATH, FOLDER_NAME_PLOTS)
os.makedirs(path_plots, exist_ok=True)

optimizer, problem, res = get_optimizer_and_problem(PATH)


# Set of weights for ASF decomposition
SET_WEIGHTS = np.zeros((NUM_DESIGNS, res.F.shape[1]))
SET_WEIGHTS[:, 0] = np.linspace(0.1, 0.9, NUM_DESIGNS)
SET_WEIGHTS[:, 1] = 1 - SET_WEIGHTS[:, 0]

design_idx = get_design_from_pareto_front(res.F, SET_WEIGHTS)

approx_ideal = res.F.min(axis=0)
approx_nadir = res.F.max(axis=0)
plt.figure(figsize=(7, 5))
plt.scatter(res.F[:, 0], res.F[:, 1], s=30, facecolors="none", edgecolors="blue")
plt.scatter(
    approx_ideal[0],
    approx_ideal[1],
    facecolors="none",
    edgecolors="red",
    marker="*",
    s=100,
    label="Ideal Point (Approx)",
)
plt.scatter(
    approx_nadir[0],
    approx_nadir[1],
    facecolors="none",
    edgecolors="black",
    marker="p",
    s=100,
    label="Nadir Point (Approx)",
)
for i, idx in enumerate(design_idx):
    plt.scatter(res.F[idx, 0], res.F[idx, 1], marker="x", color="red", s=200)
    plt.text(
        res.F[idx, 0] + 0.04 * np.abs(res.F[idx, 0]),
        res.F[idx, 1] + 0.04 * np.abs(res.F[idx, 1]),
        str(idx),
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
plt.title("Objective Space")
plt.legend()

if SAVE:
    plt.savefig(os.path.join(path_plots, "design_in_objective_space.svg"))
if SHOW:
    plt.show()


arr_X_test = res.X[design_idx]


opt_task_traj = problem.rewards_and_trajectories.trajectories[MAGIC_TRAJECTORY_INDEX]

plt.figure(figsize=(7, 5))
for traj_id, trajectory in problem.rewards_and_trajectories.trajectories.items():
    if traj_id == MAGIC_TRAJECTORY_INDEX:
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 2],
            "o-",
            linewidth=3,
            label="Optim Task Traj",
        )
    else:
        plt.plot(trajectory[:, 0], trajectory[:, 2])
plt.title("Trajectory")
plt.grid()
plt.legend()
if SAVE:
    plt.savefig(os.path.join(path_plots, "trajectory.svg"))
if SHOW:
    plt.show()

def get_reward_name_by_idF(reward_manager: RewardManager):
    
    reward_names = []
    for agg in reward_manager.agg_list:
        traj_id = agg[0][0]
        r_name = reward_manager.rewards[traj_id][0][0].__class__.__name__
        reward_names.append(r_name)
    return reward_names


rewards_names = get_reward_name_by_idF(problem.rewards_and_trajectories)
plt.figure(figsize=(NUM_DESIGNS*6, 8))
for design_id in design_idx:
    plt.subplot(1, len(design_idx), design_idx.tolist().index(design_id) + 1)
    problem.mutate_JP_by_xopt(res.X[design_id])
    draw_joint_point(problem.graph, False, False)
    title = f"ID: {design_id}; \n"
    for i, r_name in enumerate(rewards_names):
        title += f"{re.sub('[^A-Z]', '', r_name)}: {res.F[design_id, i]:.2f}; "
    plt.title(title)
    
if SAVE:
    plt.savefig(os.path.join(path_plots, "designs.svg"))
if SHOW:
    plt.show()

optimal_coeffs_traj = []
for i, x_opt in enumerate(arr_X_test):
    problem.mutate_JP_by_xopt(x_opt)
    robo, __ = jps_graph2pinocchio_robot(problem.graph, problem.builder)
    input_for_task = (opt_task_traj[:, [0, 2]], 0.4, 0.001, NAME_EE)
    test = TrajectoryMovements(*input_for_task)
    time_arr, des_traj_6d, __ = test.prepare_trajectory(robo)

    Kp, Kd, fun = test.optimize_control(robo)
    test.Kp = Kp
    test.Kd = Kd
    optimal_coeffs_traj.append((x_opt, Kp, Kd, input_for_task, fun))

# if os.path.exists(os.path.join(PATH, "coeffs_OpPD_step.pkl")):
#     with open(os.path.join(PATH, "coeffs_OpPD_step.pkl"), "rb") as f:
#         optimal_coeffs_traj_old = dill.load(f)

#     for i,  x_opt in enumerate(arr_X_test):
#         old_coeffs = list(filter(lambda x: np.all(x[0] == x_opt), optimal_coeffs_traj_old))
#         if old_coeffs > 0 and old_coeffs[0][4] < optimal_coeffs_traj[i][4]:
            
#             and optimal_coeffs_traj_old[i][3] == optimal_coeffs_traj[i][3]
#             and optimal_coeffs_traj_old[i][4] < optimal_coeffs_traj[i][4]
#         ):
#             del optimal_coeffs_traj_old[i]
#     optimal_coeffs_traj.update(optimal_coeffs_traj_old)
    

with open(os.path.join(PATH, "coeffs_OpPD_step.pkl"), "wb") as f:
    dill.dump(optimal_coeffs_traj, f)