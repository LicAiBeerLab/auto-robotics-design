"""Script for optimization of a single topology"""
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
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths, get_optimizing_joints
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE, ManipJacobian
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline, get_vertical_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningReward
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability
from auto_robot_design.description.actuators import TMotor_AK10_9, TMotor_AK60_6, TMotor_AK70_10, TMotor_AK80_64, TMotor_AK80_9
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE


### Parametrization
## mechanism configuration
generator = TwoLinkGenerator()
all_graphs = generator.get_standard_set()
graph = 


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
    "Actuated_Mass": ActuatedMass(),
    "Manip_Jacobian":ManipJacobian(MovmentSurface.XZ)
}

# set the rewards and weights for the optimization task
rewards_step = [(PositioningReward(pos_error_key="POS_ERR"), 1),
        (AccelerationCapability(manipulability_key='Manip_Jacobian', trajectory_key="traj_6d", error_key="error", actuated_mass_key="Actuated_Mass"), 1)
        ]

rewards_vertical = [(PositioningReward(pos_error_key="POS_ERR"), 1),
                    (HeavyLiftingReward(manipulability_key='Manip_Jacobian', trajectory_key="traj_6d", error_key="error", mass_key="MASS"),1)
                    ]
actuator = TMotor_AK70_10()


if __name__ == '__main__':
    # activate multiprocessing
    N_PROCESS = 4
    pool = multiprocessing.Pool(N_PROCESS)
    runner = StarmapParallelization(pool.starmap)

    # class that enables calculating of criteria along the trajectory for the urdf description of the mechanism
    crag_step = CriteriaAggregator(
        dict_point_criteria, dict_trajectory_criteria, traj_6d_step)
    # set the rewards and weights for the optimization task
    crag_vertical = CriteriaAggregator(
        dict_point_criteria, dict_trajectory_criteria, traj_6d_vertical)

    criteria_and_rewards = [(crag_step, rewards_step),(crag_vertical, rewards_vertical)]