
import os
import re

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

def get_reward_name_by_idF(reward_manager: RewardManager):
    
    reward_names = []
    for agg in reward_manager.agg_list:
        traj_id = agg[0][0]
        r_name = reward_manager.rewards[traj_id][0][0].__class__.__name__
        reward_names.append(r_name)
    return reward_names

PATH_TO_SAVE= r"results"

PATHS = [r"results/topology_0_2024-05-29_18-48-58",
         r"results/topology_1_2024-05-29_19-37-36",
         r"results/topology_3_2024-05-29_23-01-44",
         r"results/topology_4_2024-05-29_23-46-17",
         r"results/topology_5_2024-05-30_00-32-21",
         r"results/topology_7_2024-05-30_01-15-44",
         r"results/topology_8_2024-05-30_10-40-12"]

ids_topology = [0,1,3,4,5,7,8]

allF = []

plt.figure(figsize=(10,10))
for path, id in zip(PATHS, ids_topology):
    optimizer, problem, res = get_optimizer_and_problem(path)
    plt.scatter(-res.F[:, 0], -res.F[:, 1], label=f"Topology {id}", s=100)
    
    
    for desF in res.F:
        allF.append(-desF)



approx_ideal = np.array(allF).max(axis=0)
approx_nadir = np.array(allF).min(axis=0)
plt.title("Objective Space")
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

rewards_names = get_reward_name_by_idF(problem.rewards_and_trajectories)
plt.xlabel(rewards_names[0])
plt.ylabel(rewards_names[1])
plt.legend()
plt.savefig(os.path.join(PATH_TO_SAVE, "pareto_front_topology.svg"))

# plt.show()
