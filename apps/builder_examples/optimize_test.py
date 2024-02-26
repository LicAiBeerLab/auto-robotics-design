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
    CallbackSaver,
    ProblemSaver,
    load_nonmutable,
    load_checkpoint,
)
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions


from auto_robot_design.description.utils import (
    draw_joint_point,
)
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths
from auto_robot_design.pinokla.criterion_agregator import calc_traj_error


gen = TwoLinkGenerator()
graph, constrain_dict = gen.get_standard_set()[4]


def calculate_mass(urdf, joint_description, loop_description):
    free_robo = build_model_with_extensions(
        urdf, joint_description, loop_description, False
    )
    pin.computeAllTerms(
        free_robo.model,
        free_robo.data,
        np.zeros(free_robo.model.nq),
        np.zeros(free_robo.model.nv),
    )
    total_mass = pin.computeTotalMass(free_robo.model, free_robo.data)
    com_dist = la.norm(pin.centerOfMass(free_robo.model, free_robo.data))

    return total_mass * com_dist


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


criteria = [calculate_mass, calc_traj_error]


# n_proccess = 2
# pool = multiprocessing.Pool(n_proccess)
# runner = StarmapParallelization(pool.starmap)

# problem = CalculateCriteriaProblemByWeigths(graph, optimizing_joints, criteria, np.array([1, 0.4]),
#                                             elementwise_runner=runner)

problem = CalculateCriteriaProblemByWeigths(
    graph, optimizing_joints, criteria, np.array([1, 0.4])
)

saver = ProblemSaver(problem, "test", False)
saver.save_nonmutable()

callback = CallbackSaver(saver)

algorithm = PSO(pop_size=10, save_history=True, callback=callback)

res = minimize(problem, algorithm, seed=1, termination=("n_gen", 2), verbose=True)

problem = CalculateCriteriaProblemByWeigths(
    graph, optimizing_joints, criteria, np.array([1, 0.4])
)

load_nonmutable(problem, saver.path)
checkpoint = load_checkpoint(saver.path)

problem.mutate_JP_by_xopt(res.X)
draw_joint_point(problem.graph)
plt.show()
urdf, __, __ = jps_graph2urdf(problem.graph)
with open("test.urdf", "w") as f:
    f.write(urdf)

print("success")
