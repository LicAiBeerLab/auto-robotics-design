import multiprocessing
import numpy as np

import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import StarmapParallelization


from auto_robot_design.description.builder import jps_graph2urdf
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.optimization.saver import (
    ProblemSaver, )

from auto_robot_design.description.utils import (
    draw_joint_point, )
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import ForceEllProjections, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline

if __name__ == '__main__':
    gen = TwoLinkGenerator()
    graph, constrain_dict = gen.get_standard_set()[0]

    #draw_joint_point(graph)
    # plt.show()
    # plt.close()
    # %%

    optimizing_joints = dict(
        filter(lambda x: x[1]["optim"], constrain_dict.items()))
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
        ))

    dict_along_criteria = {
        "MASS": NeutralPoseMass(),
        "POS_ERR": TranslationErrorMSE(),
        "ELL_PRJ": ForceEllProjections()
    }
    dict_moment_criteria = {
        "IMF": ImfCompute(ImfProjections.Z),
        "MANIP": ManipCompute(MovmentSurface.XZ)
    }

    x_traj, y_traj = get_simple_spline()
    traj_6d = convert_x_y_to_6d_traj_xz(x_traj, y_traj)

    crag = CriteriaAggregator(dict_moment_criteria, dict_along_criteria,
                              traj_6d)

    n_proccess = 4
    pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)

    problem = CalculateCriteriaProblemByWeigths(graph,
                                                optimizing_joints,
                                                crag, [1, 1, 1],
                                                elementwise_runner=runner)

    saver = ProblemSaver(problem, "test", False)
    saver.save_nonmutable()

    algorithm = PSO(pop_size=25, save_history=True)

    optimizer = PymooOptimizer(problem, algorithm, saver)

    res = optimizer.run(
        True, **{
            "seed": 1,
            "termination": ("n_gen", 10),
            "verbose": True
        })

    best_id = np.argmin(optimizer.history["F"])
    best_x = optimizer.history["X"][best_id]

    problem.mutate_JP_by_xopt(best_x)
    draw_joint_point(problem.graph)
    plt.show()
    urdf, actuators, constainrs = jps_graph2urdf(problem.graph)
    with open("test.urdf", "w") as f:
        f.write(urdf)

    print(optimizer.history["F"][best_id])
