
import multiprocessing
from matplotlib.ticker import LinearLocator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline, get_vertical_trajectory, create_simple_step_trajectory,get_workspace_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningReward, PositioningConstrain, PositioningErrorCalculator, RewardManager, PositioningReward
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability, MeanHeavyLiftingReward, MinAccelerationCapability
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import EndPointZRRReward, VelocityReward, ForceEllipsoidReward
from auto_robot_design.optimization.rewards.inertia_rewards import MassReward, TrajectoryIMFReward
from auto_robot_design.description.actuators import TMotor_AK10_9, TMotor_AK60_6, TMotor_AK70_10, TMotor_AK80_64, TMotor_AK80_9
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot, MIT_CHEETAH_PARAMS_DICT


generator = TwoLinkGenerator()
all_graphs = generator.get_standard_set(-0.2, shift=0.1)
graph, constrain_dict = all_graphs[0]

thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]
actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
density = MIT_CHEETAH_PARAMS_DICT["density"]
body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]


builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE,
                              density={"default": density, "G":body_density},
                              thickness={"default": thickness, "EE":0.033},
                              actuator={"default": actuator},
                              size_ground=np.array(MIT_CHEETAH_PARAMS_DICT["size_ground"]),
                              offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"]
)

# 2) characteristics to be calculated
# criteria that either calculated without any reference to points, or calculated through the aggregation of values from all points on trajectory
dict_trajectory_criteria = {
    "MASS": NeutralPoseMass(),
    "POS_ERR": TranslationErrorMSE()
}
# criteria calculated for each point on the trajectory
dict_point_criteria = {
    "Effective_Inertia": EffectiveInertiaCompute(),
    "Actuated_Mass": ActuatedMass(),
    "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ),
    "IMF": ImfCompute(ImfProjections.Z),
}
# special object that calculates the criteria for a robot and a trajectory
crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)
# the result is the dict with key - joint_point, value - tuple of all possible coordinate moves
constrain_dict["2L_ground"]["optim"] = True
constrain_dict["2L_bot"]["optim"] = False
constrain_dict["2L_knee"]["optim"] = False

optimizing_joints = get_optimizing_joints(graph, constrain_dict)

# central_vertical = convert_x_y_to_6d_traj_xz(
#     *get_vertical_trajectory(-1, 0.4, 0, 50))
constrain_dictstep = convert_x_y_to_6d_traj_xz(
    *create_simple_step_trajectory([-0.2,-1],0.1,0.4))

acceleration_capability = AccelerationCapability(manipulability_key='Manip_Jacobian',
                                                 trajectory_key="traj_6d", error_key="error", actuated_mass_key="Actuated_Mass")

# set up special classes for reward calculations
pos_error_reward = PositioningReward(pos_error_key='POS_ERR')
error_calculator = PositioningErrorCalculator(error_key='error', jacobian_key='Manip_Jacobian')
soft_constrain = PositioningConstrain(error_calculator=error_calculator, points = [constrain_dictstep])
reward_manager = RewardManager(crag=crag)
# reward_manager.add_trajectory(ground_symmetric_step, 0)
reward_manager.add_trajectory(constrain_dictstep, 0)
reward_manager.add_reward(acceleration_capability, 0, 1)

# JP, (x_min, z_min, x_max, z_max)
problem = CalculateCriteriaProblemByWeigths(graph,builder=builder,
                                            jp2limits=optimizing_joints,
                                            crag = crag,
                                            soft_constrain=soft_constrain,
                                            rewards_and_trajectories=reward_manager,
                                             Actuator = actuator)
                                            # elementwise_runner=runner,


nx = 20
nz = 20

# curr_j, bounds = list(optimizing_joints.items())[0]
curr_j, bounds = list(optimizing_joints.items())[0]
curr_j_2, bounds = list(optimizing_joints.items())[1]
x = np.linspace(-0.8, 0.8, nx)
z = np.linspace(-0.8, 0.8, nz)
# x = np.linspace(bounds[0], bounds[2], nx)
# z = np.linspace(bounds[1], bounds[3], nz)

X, Z = np.meshgrid(x, z)
res = np.zeros((nx, nz))
pos_error = np.zeros((nx, nz))

initial_pos_JP = curr_j.r
initial_pos_JP2 = curr_j_2.r
for i in range(nx):
    for j in range(nz):
        # additional_x = np.array([X[i,j], 0, Z[i,j]])
        # current_x = initial_pos_JP + additional_x
        # additional_x = np.array([X[i,j], 0, 0])
        # current_x = initial_pos_JP + additional_x
        
        # additional_x2 = np.array([Z[i,j], 0, 0])
        # current_x2 = initial_pos_JP2 + additional_x2
        
        additional_x = np.array([0, 0, X[i,j]])
        current_x = initial_pos_JP + additional_x
        
        additional_x2 = np.array([0, 0, Z[i,j]])
        current_x2 = initial_pos_JP2 + additional_x2
        # problem.mutate_JP_by_xopt(current_x[[0,2]])
        problem.mutate_JP_by_xopt([0, current_x[2], 0, current_x2[2]])
        fixed_robot, free_robot = jps_graph2pinocchio_robot(problem.graph, builder=builder)
        trajectory = constrain_dictstep
        point_criteria_vector, trajectory_criteria, res_dict_fixed = crag.get_criteria_data(fixed_robot, free_robot, trajectory)
        reward, reward_list = acceleration_capability.calculate(point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator = actuator)
        res[i,j] = reward
        
        err = pos_error_reward.calculate(point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator = actuator)
        pos_error[i,j] = err[0]
    print(f"i={i}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("summer")
ax.plot_trisurf(X.flatten(), Z.flatten(), res.flatten(), antialiased=True)
# ax.plot_surface(X, Z, res,  linewidth=0.2, antialiased=True)
# sct = ax.scatter(X.flatten(), Z.flatten(), res.flatten(), s=np.abs(pos_error.flatten())*5e3, c=np.abs(pos_error.flatten())*10, cmap=cmhot)
ax.set_zlim(res.min()-0.5, res.max()+0.5)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
# fig.colorbar(sct, shrink=0.5, aspect=5)
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Reward')
plt.show()
