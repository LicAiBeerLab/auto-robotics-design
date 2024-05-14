# %%
"""Script for optimization of a single topology"""
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import StarmapParallelization
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator, visualize_constrains

from auto_robot_design.optimization.saver import (
    ProblemSaver, )
from auto_robot_design.description.builder import jps_graph2pinocchio_robot
from auto_robot_design.description.utils import (
    draw_joint_point, )
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths, get_optimizing_joints
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE, ManipJacobian
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline, get_vertical_trajectory, create_simple_step_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningReward
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import EndPointZRRReward, VelocityReward, ForceEllipsoidReward
from auto_robot_design.optimization.rewards.inertia_rewards import MassReward
from auto_robot_design.description.actuators import MyActuator_RMD_MT_RH_17_100_N, TMotor_AK10_9, TMotor_AK60_6, TMotor_AK70_10, TMotor_AK80_64, TMotor_AK80_9
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot

# %% [markdown]
# ### Parametrization

# %% [markdown]
# #### 1) mechanism configuration

# %%
generator = TwoLinkGenerator()
all_graphs = generator.get_standard_set(shift=0.3)
actuator = MyActuator_RMD_MT_RH_17_100_N()
 
graph, constrain_dict = all_graphs[1]


# %% [markdown]
# #### 2) set optimization task

# %%
# trajectories
ground_symmetric_step = convert_x_y_to_6d_traj_xz(*create_simple_step_trajectory(
    starting_point=[-0.5, -0.95], step_height=0.4, step_width=1, n_points=50))
left_shift_step = convert_x_y_to_6d_traj_xz(*create_simple_step_trajectory(
    starting_point=[-0.65, -0.95], step_height=0.4, step_width=1, n_points=50))
right_shift_step = convert_x_y_to_6d_traj_xz(*create_simple_step_trajectory(
    starting_point=[-0.35, -0.95], step_height=0.4, step_width=1, n_points=50))


central_vertical = convert_x_y_to_6d_traj_xz(
    *get_vertical_trajectory(-1.1, 0.5, 0, 50))
left_vertical = convert_x_y_to_6d_traj_xz(
    *get_vertical_trajectory(-1.1, 0.5, -0.15, 50))
right_vertical = convert_x_y_to_6d_traj_xz(
    *get_vertical_trajectory(-1.1, 0.5, 0.15, 50))

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
    "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ)
}
crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)
 
step_trajectories = [ground_symmetric_step]

heavy_lifting = HeavyLiftingReward(
    manipulability_key='Manip_Jacobian', trajectory_key="traj_6d", error_key="error", mass_key="MASS")
rewards_vertical = [(PositioningReward(pos_error_key="POS_ERR"), 1),
                    (heavy_lifting, 1)]
 
vertical_trajectories = [central_vertical]

# rewards_and_trajectories = [
#     (rewards_step, step_trajectories), (rewards_vertical, vertical_trajectories)]
rewards_and_trajectories = [(rewards_vertical, vertical_trajectories)]
  
# %% [markdown]
# #### Calculate rewards for initial graph.

# %%
# create builder
def run_optimize(actuator, graph, constrain_dict, dict_trajectory_criteria, dict_point_criteria, rewards_and_trajectories, folder_name):
    thickness = 0.04
    builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE, size_ground=np.array(
    [thickness*5, thickness*10, thickness*2]), actuator=actuator, thickness=thickness)

    fixed_robot, free_robot = jps_graph2pinocchio_robot(graph, builder)

    crag = CriteriaAggregator(
    dict_point_criteria, dict_trajectory_criteria)

# %%
# activate multiprocessing
    N_PROCESS = 16
    pool = multiprocessing.Pool(N_PROCESS)
    runner = StarmapParallelization(pool.starmap)

# the result is the dict with key - joint_point, value - tuple of all possible coordinate moves
    optimizing_joints = get_optimizing_joints(graph, constrain_dict)

# %%
    population_size = 64
    n_generations = 30

# create the problem for the current optimization
    problem = CalculateCriteriaProblemByWeigths(graph,builder=builder,
                                            jp2limits=optimizing_joints,
                                            crag = crag,
                                            rewards_and_trajectories=rewards_and_trajectories,
                                            elementwise_runner=runner, Actuator = actuator)
    
    saver = ProblemSaver(problem, folder_name, True)
    saver.save_nonmutable()
    algorithm = PSO(pop_size=population_size, save_history=True, c2=3)
    optimizer = PymooOptimizer(problem, algorithm, saver)

    res = optimizer.run(
    True, **{
        "seed": 1,
        "termination": ("n_gen", n_generations),
        "verbose": True
    })
    return optimizer


if __name__ == '__main__':
    for num, (graph_i, constrain_dict_i) in enumerate(all_graphs):
        folder_name = "th_1909_num" + str(num)
        optimizer = run_optimize(actuator, graph_i, constrain_dict_i, dict_trajectory_criteria, dict_point_criteria, rewards_and_trajectories, folder_name)
        best_id = np.argmin(optimizer.history["F"])
        best_x = optimizer.history["X"][best_id]
        best_reward = optimizer.history["F"][best_id]
        print("The minimum result in optimization task:", best_reward)

