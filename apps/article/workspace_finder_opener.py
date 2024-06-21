from functools import partial
import multiprocessing
import time
from joblib import Parallel, cpu_count, delayed
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
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE, ManipJacobian, folow_traj_by_proximal_inv_k_2
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline, get_vertical_trajectory, create_simple_step_trajectory, get_workspace_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningReward, PositioningConstrain, PositioningErrorCalculator, RewardManager
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, AccelerationCapability, MeanHeavyLiftingReward, MinAccelerationCapability
from auto_robot_design.optimization.rewards.pure_jacobian_rewards import EndPointZRRReward, VelocityReward, ForceEllipsoidReward
from auto_robot_design.optimization.rewards.inertia_rewards import MassReward
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot, MIT_CHEETAH_PARAMS_DICT
from apps.article import create_reward_manager
from apps.article import traj_graph_setup
from pymoo.algorithms.moo.age2 import AGEMOEA2
import os

def get_indices_by_point(mask: np.ndarray, reach_array: np.ndarray):
    mask_true_sum = np.sum(mask)
    reachability_sums = reach_array @ mask
    target_indices = np.where(reachability_sums == mask_true_sum)
    return target_indices[0]

data = np.load("WORKSPACE_TOP0_test.npz")
reach_arrays = data["reach_array"]
q_arrays = data["q_array"]


mask = np.ones(100, dtype=np.bool_)
mask[55] = False
mask[0] = False
mask[10] = False
start = time.time()
target_indices = get_indices_by_point(mask, reach_arrays)
curent_time = time.time()
ellap = curent_time - start
print(ellap)
print(len(target_indices))