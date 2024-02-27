from json import load
import multiprocessing
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize


import pinocchio as pin

from auto_robot_design.description.builder import jps_graph2urdf
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.optimization.saver import (
    ProblemSaver,
)
from auto_robot_design.optimization.test_criteria import calculate_mass


from auto_robot_design.description.utils import (
    draw_joint_point,
)
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.criterion_agregator import calc_traj_error


gen = TwoLinkGenerator()
graph, constrain_dict = gen.get_standard_set()[4]


draw_joint_point(graph)
plt.show()
# %%


optimizing_joints = dict(filter(lambda x: x[1]["optim"], constrain_dict.items()))
name2jp = dict(map(lambda x: (x.name, x), graph.nodes()))
optimizing_joints = dict(
    map(
        lambda x: (
            name2jp[x[0]],
            (
                x[1]["x_range"][0],
                x[1].get("z_range", [-0.01, 0.01])[0],
                x[1]["x_range"][1],
                x[1].get("z_range", [0, 0])[1],
            ),
        ),
        optimizing_joints.items(),
    )
)


criteria = [calc_traj_error, calculate_mass]


n_proccess = 2
pool = multiprocessing.Pool(n_proccess)
runner = StarmapParallelization(pool.starmap)

problem = CalculateCriteriaProblemByWeigths(graph, optimizing_joints, criteria, np.array([1, 0.4]),
                                            elementwise_runner=runner)


saver = ProblemSaver(problem, "test", False)
saver.save_nonmutable()


algorithm = PSO(pop_size=10, save_history=True)

optimizer = PymooOptimizer(problem, algorithm, saver)

res = optimizer.run(True, **{"seed":1, "termination":("n_gen", 2), "verbose":True})

best_id = np.argmin(optimizer.history["F"])
best_x = optimizer.history["X"][best_id]

problem.mutate_JP_by_xopt(best_x)
draw_joint_point(problem.graph)
plt.show()
urdf, actuators, constainrs = jps_graph2urdf(problem.graph)
with open("test.urdf", "w") as f:
    f.write(urdf)

r1 = calc_traj_error(urdf, actuators, constainrs)
r2 = calculate_mass(urdf, actuators, constainrs)
print(r1 + 0.4 * r2)
print(optimizer.history["F"][best_id])