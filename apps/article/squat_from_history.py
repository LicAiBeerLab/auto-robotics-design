import multiprocessing
from networkx import Graph
import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import StarmapParallelization
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator, visualize_constrains

from auto_robot_design.optimization.saver import (
    ProblemSaver, )
from auto_robot_design.description.builder import jps_graph2pinocchio_robot
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.optimization.problems import CalculateCriteriaProblemByWeigths, CalculateMultiCriteriaProblem, get_optimizing_joints
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE, ManipJacobian
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_horizontal_trajectory, get_simple_spline, get_vertical_trajectory, create_simple_step_trajectory, get_workspace_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningReward, PositioningConstrain, PositioningErrorCalculator, RewardManager
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability, MeanHeavyLiftingReward, MinAccelerationCapability
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import EndPointZRRReward, VelocityReward, ForceEllipsoidReward
from auto_robot_design.optimization.rewards.inertia_rewards import MassReward
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot, MIT_CHEETAH_PARAMS_DICT
from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
import numpy as np

import matplotlib.pyplot as plt

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator, )

from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.squat import SquatHopParameters, SimulateSquatHop

chavo = CalculateMultiCriteriaProblem.load(
    "results\\multi_opti_preset2\\topology_0_2024-05-29_18-48-58")
optimizer_stub = PymooOptimizer(chavo, 1)
optimizer_stub.load_history(
    "results\\multi_opti_preset2\\topology_0_2024-05-29_18-48-58")
print("COC")


# robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
#     graph, builder)
# sqh_p = SquatHopParameters(hop_flight_hight=0.3,
#                            squatting_up_hight=0,
#                            squatting_down_hight=-0.3,
#                            total_time=0.55)
# hoppa = SimulateSquatHop(sqh_p)


# q_act, vq_act, acc_act, tau = hoppa.simulate(
#     robo_urdf, joint_description, loop_description, is_vis=True)

# trj_f = hoppa.create_traj_equation()
# t = np.linspace(0, sqh_p.total_time,  len(q_act))
# list__234 = np.array(list(map(trj_f, t)))


# plt.figure()
# plt.plot(list__234[:, 2])
# plt.title("Desired acceleration")
# plt.xlabel("Time")
# plt.ylabel("Z-acc")
# plt.grid(True)

# plt.figure()
# plt.plot(tau[:, 0])
# plt.plot(tau[:, 1])
# plt.title("Actual torques")
# plt.xlabel("Time")
# plt.ylabel("Torques")
# plt.grid(True)


# plt.figure()

# plt.plot(vq_act[:, 0])
# plt.plot(list__234[:, 1])
# plt.title("Velocities")
# plt.xlabel("Time")
# plt.ylabel("Z-vel")
# plt.legend(["actual vel", "desired vel"])
# plt.grid(True)

# plt.show()
# pass
