from json import load
import multiprocessing
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

from pymoo.core.problem import StarmapParallelization

from auto_robot_design.optimization.saver import (
    load_checkpoint,
)


from auto_robot_design.optimization.visualizer import prepare_data_to_visualize, draw_jps_cost_on_graph, prepare_data_to_visualize_separeate_jps, MARKERS
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.criterion_agregator import calc_traj_error
from auto_robot_design.optimization.test_criteria import calculate_mass
from auto_robot_design.description.utils import draw_joint_point


path = "results/test"

# n_proccess = 2
# pool = multiprocessing.Pool(n_proccess)
# runner = StarmapParallelization(pool.starmap)

problem = CalculateCriteriaProblemByWeigths.load(path) #**{"elementwise_runner":runner})
checklpoint = load_checkpoint(path)

optimizer = PymooOptimizer(problem, checklpoint)
optimizer.load_history(path)

features, costs, total_cost = prepare_data_to_visualize_separeate_jps(optimizer.history, problem)
# print(feature[0])

for id, feat in enumerate(features):
    plt.figure(figsize=(10, 10))
    draw_joint_point(problem.graph)
    marker = MARKERS[id % len(MARKERS)]
    draw_jps_cost_on_graph(feat, costs[:,0], problem, marker)
    plt.colorbar()
    plt.axis("equal")
    plt.show()
# res = optimizer.run(False, **{"seed":1, "termination":("n_gen", 5), "verbose":True})