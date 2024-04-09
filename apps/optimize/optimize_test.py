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
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths, get_optimizing_joints
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import ForceCapabilityProjectionCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline
from auto_robot_design.optimization.reward import VelocityReward, EndPointZRRReward, EndPointIMFReward, PositioningReward, MassReward, ForceEllipsoidReward


if __name__ == '__main__':
    gen = TwoLinkGenerator()
    graph, constrain_dict = gen.get_standard_set()[0]

    draw_joint_point(graph)
    plt.show()
    plt.close()
    
    optimizing_joints = get_optimizing_joints(graph, constrain_dict)

    # set the criteria to be calculated for each mechanism using the dictionaries 
    # criteria that either calculated without any reference to points, or calculated through the aggregation of values from all points on trajectory
    dict_along_criteria = {
        "MASS": NeutralPoseMass(),
        "POS_ERR": TranslationErrorMSE(), # MSE of deviation from the trajectory
        "ELL_PRJ": ForceCapabilityProjectionCompute() 
    }
    # criteria calculated for each point on the trajectory
    dict_moment_criteria = {
        "IMF": ImfCompute(ImfProjections.Z), # Impact mitigation factor along the axis
        "MANIP": ManipCompute(MovmentSurface.XZ) 
    }

    # trajectory construction, each point is three coordinates and 
    x_traj, y_traj = get_simple_spline()
    traj_6d = convert_x_y_to_6d_traj_xz(x_traj, y_traj)

    # class that enables calculating of criteria along the trajectory for the urdf description of the mechanism
    crag = CriteriaAggregator(dict_moment_criteria, dict_along_criteria,
                              traj_6d)

    n_proccess = 4
    pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)

    rewards = [(VelocityReward(manipulability_key='MANIP', trajectory_key="traj_6d", error_key="error"), 1),
               (ForceEllipsoidReward(manipulability_key='MANIP', trajectory_key="traj_6d", error_key="error"),1),
               (EndPointIMFReward(imf_key='IMF', trajectory_key="traj_6d", error_key="error"), 1),
               (EndPointZRRReward(manipulability_key='MANIP', trajectory_key="traj_6d", error_key="error"),1),
               (PositioningReward(pos_error_key="POS_ERR"),1),(MassReward(mass_key="MASS"),1)
               ]

    problem = CalculateCriteriaProblemByWeigths(graph,
                                                optimizing_joints,
                                                crag, [1, 1, 1],rewards=rewards,
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
