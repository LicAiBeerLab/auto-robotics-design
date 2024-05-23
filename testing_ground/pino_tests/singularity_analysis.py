import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import StarmapParallelization
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator, visualize_constrains

from auto_robot_design.optimization.saver import (
    ProblemSaver, )
from auto_robot_design.description.builder import jps_graph2pinocchio_robot
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths, get_optimizing_joints
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE, ManipJacobian
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline, get_vertical_trajectory, create_simple_step_trajectory, get_workspace_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningReward, PositioningConstrain, PositioningErrorCalculator, RewardManager
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability, MeanHeavyLiftingReward, MinAccelerationCapability
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import EndPointZRRReward, VelocityReward, ForceEllipsoidReward
from auto_robot_design.optimization.rewards.inertia_rewards import MassReward
from auto_robot_design.description.actuators import TMotor_AK10_9, TMotor_AK60_6, TMotor_AK70_10, TMotor_AK80_64, TMotor_AK80_9
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot

generator = TwoLinkGenerator()
all_graphs = generator.get_standard_set(shift=0.3)
graph, constrain_dict = all_graphs[0]

actuator = TMotor_AK10_9()
thickness = 0.04
builder = ParametrizedBuilder(
    DetailedURDFCreatorFixedEE,
    size_ground=np.array([thickness * 5, thickness * 10, thickness * 2]),
    actuator=actuator,
    thickness=thickness)


dict_trajectory_criteria = {"MASS": NeutralPoseMass()}

dict_point_criteria = {
    "Effective_Inertia": EffectiveInertiaCompute(),
    "Actuated_Mass": ActuatedMass(),
    "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ)
}

crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)

optimizing_joints = get_optimizing_joints(graph, constrain_dict)
ground_symmetric_step = convert_x_y_to_6d_traj_xz(
    *create_simple_step_trajectory(starting_point=[-0.5, -1.0],
                                   step_height=0.3,
                                   step_width=0.7,
                                   n_points=150))

acceleration_capability = AccelerationCapability(
    manipulability_key='Manip_Jacobian',
    trajectory_key="traj_6d",
    error_key="error",
    actuated_mass_key="Actuated_Mass")


error_calculator = PositioningErrorCalculator(
    error_key='error', jacobian_key="Manip_Jacobian")
soft_constrain = PositioningConstrain(error_calculator=error_calculator,
                                      points=[ground_symmetric_step])
reward_manager = RewardManager(crag=crag)
reward_manager.add_trajectory(ground_symmetric_step, 0)
reward_manager.add_reward(acceleration_capability, 0, 1)

fixed_robot, free_robot = jps_graph2pinocchio_robot(graph, builder)

draw_joint_point(graph)
for _, trajectory in reward_manager.trajectories.items():
    plt.plot(trajectory[:, 0], trajectory[:, 2])

plt.figure()
for _, trajectory in reward_manager.trajectories.items():
    plt.plot(trajectory[:, 0], trajectory[:, 2])


total_error, results = soft_constrain.calculate_constrain_error(
    reward_manager.crag, fixed_robot, free_robot)


print(f"Constrain error {total_error}")
isotropic_values = error_calculator.calculate_isotropic_values(results[0][0])
 

plt.figure()
plt.title("Isotropic value")
plt.plot(isotropic_values)


plt.figure()
plt.title("Pos err")
plt.plot(results[0][2]["error"])

plt.show()
