"""Calculate rewards for main branch"""
import numpy as np
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.calc_criterion import (ActuatedMass, ImfCompute, ManipCompute,
                MovmentSurface, NeutralPoseMass, TranslationErrorMSE, EffectiveInertiaCompute)
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import (convert_x_y_to_6d_traj_xz, 
                                    get_simple_spline, get_vertical_trajectory)
from auto_robot_design.optimization.rewards.reward_base import PositioningReward
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability
from auto_robot_design.description.actuators import TMotor_AK10_9, TMotor_AK60_6, TMotor_AK70_10, TMotor_AK80_64, TMotor_AK80_9
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE
from auto_robot_design.description.builder import jps_graph2pinocchio_robot

# set the optimization task
# 1) trajectories
x_traj, y_traj = get_simple_spline()
x_traj, y_traj = get_vertical_trajectory(3)
traj_6d = convert_x_y_to_6d_traj_xz(x_traj, y_traj)
# 2) characteristics to be calculated
# criteria that either calculated without any reference to points, or calculated through
# the aggregation of values from all points on trajectory
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

# class that enables calculating of criteria along the trajectory
# for the urdf description of the mechanism
crag = CriteriaAggregator(
    dict_point_criteria, dict_trajectory_criteria, traj_6d)
# set the rewards and weights for the optimization task
rewards = [(PositioningReward(pos_error_key="POS_ERR"), 1),
           (HeavyLiftingReward(manipulability_key='MANIP', trajectory_key="traj_6d", error_key="error", mass_key="MASS"), 1),
           (AccelerationCapability(manipulability_key='MANIP', trajectory_key="traj_6d", error_key="error", actuated_mass_key="Actuated_Mass"), 1)]

# set the list of graphs that should be tested
topology_list = list(range(3))
best_vector = []
actuator_list = [TMotor_AK10_9(), TMotor_AK60_6(
), TMotor_AK70_10(), TMotor_AK80_64(), TMotor_AK80_9()]
generator = TwoLinkGenerator()
generator.build_standard_two_linker()
graph = generator.graph
for jp in graph.nodes:
    if not jp.attach_endeffector:
        jp.active = True


result_vector = []
for j in actuator_list:
    # create builder
    thickness = 0.04
    builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE, size_ground=np.array(
        [thickness*5, thickness*10, thickness*2]), actuator=j, thickness=thickness)

    # builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE, size_ground=np.array(
    #     [thickness*10, thickness*20, thickness*4]), actuator=j, thickness=thickness)

    fixed_robot, free_robot = jps_graph2pinocchio_robot(graph, builder)

    crag = CriteriaAggregator(
        dict_point_criteria, dict_trajectory_criteria, traj_6d)


    point_criteria_vector, trajectory_criteria, res_dict_fixed = crag.get_criteria_data(fixed_robot, free_robot)

    # all rewards are calculated and added to the result
    total_result = 0
    partial_results = []

    for reward, weight in rewards:
        partial_results.append(reward.calculate(
            point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator=j))
        total_result += weight*partial_results[-1]

    result_vector.append((total_result, partial_results))

print(result_vector)
