from json import load
import multiprocessing
import os
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

from pymoo.core.problem import StarmapParallelization

from auto_robot_design.optimization.saver import (
    load_checkpoint,
)


from auto_robot_design.optimization.visualizer import (
    prepare_data_to_visualize,
    draw_jps_cost_on_graph,
    prepare_data_to_visualize_separeate_jps,
    draw_jps_distribution,
    draw_costs,
    MARKERS,
)
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.test_criteria import calculate_mass
from auto_robot_design.description.utils import draw_joint_point


path = "results/test_2024-05-06_12-22-40"

# n_proccess = 2
# pool = multiprocessing.Pool(n_proccess)
# runner = StarmapParallelization(pool.starmap)

problem = CalculateCriteriaProblemByWeigths.load(
    path
)  # **{"elementwise_runner":runner})
checklpoint = load_checkpoint(path)

optimizer = PymooOptimizer(problem, checklpoint)
optimizer.load_history(path)

features, costs, total_cost = prepare_data_to_visualize_separeate_jps(
    optimizer.history, problem
)
# print(feature[0])
best_id = np.argmin(optimizer.history["F"])

problem.mutate_JP_by_xopt(optimizer.history["X"][best_id])
# for id, feat in enumerate(features):
#     plt.figure(figsize=(10, 10))
#     marker = MARKERS[id % len(MARKERS)]
#     # draw_jps_cost_on_graph(feat, costs[:, 0], problem, marker)
#     draw_jps_distribution(feat)
#     draw_joint_point(problem.graph)
#     plt.colorbar()
#     plt.axis("equal")
#     plt.show()

plt.figure(figsize=(10, 10))
draw_costs(costs[:, 0], costs[:, 1]*0.4)
# plt.axis("equal")
plt.show()
# res = optimizer.run(False, **{"seed":1, "termination":("n_gen", 5), "verbose":True})
