import multiprocessing
import numpy as np

import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import StarmapParallelization
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator

from auto_robot_design.optimization.saver import (
    ProblemSaver, )

from auto_robot_design.description.utils import (
    draw_joint_point, )
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline, get_vertical_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningReward
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability
from auto_robot_design.description.actuators import TMotor_AK10_9, TMotor_AK60_6, TMotor_AK70_10, TMotor_AK80_64, TMotor_AK80_9
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE


if __name__ == '__main__':
    # set the optimization task
    # 1) trajectories
    x_traj, y_traj = get_simple_spline()
    traj_6d_step = convert_x_y_to_6d_traj_xz(x_traj, y_traj)
    x_traj, y_traj = get_vertical_trajectory(50)
    traj_6d_vertical = convert_x_y_to_6d_traj_xz(x_traj, y_traj)
    # 2) characteristics to be calculated
    # criteria that either calculated without any reference to points, or calculated through the aggregation of values from all points on trajectory
    dict_trajectory_criteria = {
        "MASS": NeutralPoseMass(),
        "POS_ERR": TranslationErrorMSE()  # MSE of deviation from the trajectory
    }
    # criteria calculated for each point on the trajectory
    dict_point_criteria = {
        # Impact mitigation factor along the axis
        "IMF": ImfCompute(ImfProjections.Z),
        "MANIP": ManipCompute(MovmentSurface.XZ),
        "Effective_Inertia": EffectiveInertiaCompute(),
        "Actuated_Mass": ActuatedMass()
    }

    # class that enables calculating of criteria along the trajectory for the urdf description of the mechanism
    crag_step = CriteriaAggregator(
        dict_point_criteria, dict_trajectory_criteria, traj_6d_step)
    # set the rewards and weights for the optimization task
    crag_vertical = CriteriaAggregator(
        dict_point_criteria, dict_trajectory_criteria, traj_6d_vertical)
    # set the rewards and weights for the optimization task
    rewards_step = [(PositioningReward(pos_error_key="POS_ERR"), 1)
           #(AccelerationCapability(manipulability_key='MANIP', trajectory_key="traj_6d", error_key="error", actuated_mass_key="Actuated_Mass"), 1)
           ]
    
    rewards_vertical = [(PositioningReward(pos_error_key="POS_ERR"), 1)
                        #(HeavyLiftingReward(manipulability_key='MANIP', trajectory_key="traj_6d", error_key="error", mass_key="MASS"),1)
                        ]
    # activate multiprocessing
    N_PROCESS = 4
    pool = multiprocessing.Pool(N_PROCESS)
    runner = StarmapParallelization(pool.starmap)

    # create the total list of topology graphs
    generator = TwoLinkGenerator()
    all_graphs = generator.get_standard_set()
    # set the list of graphs that should be tested
    topology_list = list(range(3))
    best_vector = []
    actuator_list = [TMotor_AK10_9(), TMotor_AK60_6(
    ), TMotor_AK70_10(), TMotor_AK80_64(), TMotor_AK80_9()]
    criteria_and_rewards = [(crag_step, rewards_step),(crag_vertical, rewards_vertical)]
    for i in topology_list:
        for j in actuator_list:
            # create builder
            thickness = 0.04
            builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE, size_ground=np.array(
                [thickness*5, thickness*10, thickness*2]), actuator=j,thickness=thickness)
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
            problem = CalculateCriteriaProblemByWeigths(graph,builder=builder,
                                                        jp2limits=optimizing_joints,
                                                        criteria_and_rewards=criteria_and_rewards,
                                                        elementwise_runner=runner, Actuator = j)
            saver = ProblemSaver(problem, "test", True)
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
            best_reward = optimizer.history["F"][best_id]
            problem.mutate_JP_by_xopt(best_x)
            best_vector.append((problem.graph, j, best_reward))

    draw_joint_point(best_vector[2][0])
    plt.show()
