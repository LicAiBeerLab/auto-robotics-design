import multiprocessing
import numpy as np

import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import StarmapParallelization
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator

from auto_robot_design.description.builder import jps_graph2urdf

from auto_robot_design.optimization.saver import (
    ProblemSaver, )

from auto_robot_design.description.utils import (
    draw_joint_point, )
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import ForceCapabilityProjectionCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline
from auto_robot_design.optimization.reward import VelocityReward, EndPointZRRReward, EndPointIMFReward, PositioningReward, MassReward, ForceEllipsoidReward


if __name__ == '__main__':
    # set the optimisation task
    # 1) trajectories
    x_traj, y_traj = get_simple_spline()
    traj_6d = convert_x_y_to_6d_traj_xz(x_traj, y_traj)
    # 2) characteristics to be calculated
    # criteria that either calculated without any reference to points, or calculated through the aggregation of values from all points on trajectory
    dict_trajectory_criteria = {
        "MASS": NeutralPoseMass(),
        "POS_ERR": TranslationErrorMSE(), # MSE of deviation from the trajectory
        "ELL_PRJ": ForceCapabilityProjectionCompute() 
    }
    # criteria calculated for each point on the trajectory
    dict_point_criteria = {
        "IMF": ImfCompute(ImfProjections.Z), # Impact mitigation factor along the axis
        "MANIP": ManipCompute(MovmentSurface.XZ) 
    }

    # class that enables calculating of criteria along the trajectory for the urdf description of the mechanism
    crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria, traj_6d)
    # set the rewards and weights for the optimization task 
    rewards = [(VelocityReward(manipulability_key='MANIP', trajectory_key="traj_6d", error_key="error"), 1),
                (ForceEllipsoidReward(manipulability_key='MANIP', trajectory_key="traj_6d", error_key="error"),1),
                (EndPointIMFReward(imf_key='IMF', trajectory_key="traj_6d", error_key="error"), 1),
                (EndPointZRRReward(manipulability_key='MANIP', trajectory_key="traj_6d", error_key="error"),1),
                (PositioningReward(pos_error_key="POS_ERR"),1),(MassReward(mass_key="MASS"),1)
                ]
    # activate multiprocessing
    n_proccess = 12
    pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)
    # optimisation algorithm is the same for all topologies
    

    # create the total list of topology graphs
    generator = TwoLinkGenerator()
    all_graphs = generator.get_standard_set()
    #set the list of graphs that should be tested
    topology_list = list(range(3))
    best_vector = []
    for i in topology_list:
        # get the graph from the generator
        graph, constrain_dict = all_graphs[i]
        # filter the joints to be optimized
        optimizing_joints = dict(
            filter(lambda x: x[1]["optim"], constrain_dict.items()))
        name2jp = dict(map(lambda x: (x.name, x), graph.nodes()))
        # the procedure below is rather unstable 
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
        # the result is the dict with key - joint_point, value - tuple of all possible coordinate moves
        
        # create the problem for the current optimization
        problem = CalculateCriteriaProblemByWeigths(graph,
                                                    optimizing_joints,
                                                    crag, [1, 1, 1],rewards=rewards,
                                                    elementwise_runner=runner)
        saver = ProblemSaver(problem, "test", True)
        saver.save_nonmutable()
        algorithm = PSO(pop_size=25, save_history=True)
        optimizer = PymooOptimizer(problem, algorithm, saver)

        res = optimizer.run(
            True, **{
                "seed": 1,
                "termination": ("n_gen", 5),
                "verbose": True
            })
        
        best_id = np.argmin(optimizer.history["F"])
        best_x = optimizer.history["X"][best_id]
        problem.mutate_JP_by_xopt(best_x)
        best_vector.append(problem.graph)

    draw_joint_point(best_vector[2])
    plt.show()
