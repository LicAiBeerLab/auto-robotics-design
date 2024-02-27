from json import load
import multiprocessing
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

from pymoo.core.problem import StarmapParallelization

from auto_robot_design.optimization.saver import (
    load_checkpoint,
)
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions


from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.criterion_agregator import calc_traj_error
from auto_robot_design.optimization.test_criteria import calculate_mass


path = "results/test"

n_proccess = 2
pool = multiprocessing.Pool(n_proccess)
runner = StarmapParallelization(pool.starmap)

problem = CalculateCriteriaProblemByWeigths.load(path, **{"elementwise_runner":runner})
checklpoint = load_checkpoint(path)

optimizer = PymooOptimizer(problem, checklpoint)
optimizer.load_history(path)
res = optimizer.run(False, **{"seed":1, "termination":("n_gen", 5), "verbose":True})