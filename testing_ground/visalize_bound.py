import multiprocessing
from arrow import get
import numpy as np
import matplotlib.pyplot as plt
import os

from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.decomposition.asf import ASF
from auto_robot_design.optimization.saver import (ProblemSaver)
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.optimization.problems import MultiCriteriaProblem
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.pinokla.calc_criterion import ActuatedMass, EffectiveInertiaCompute, MovmentSurface, NeutralPoseMass, ManipJacobian
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_vertical_trajectory, create_simple_step_trajectory, get_workspace_trajectory
from auto_robot_design.optimization.rewards.reward_base import PositioningConstrain, PositioningErrorCalculator, RewardManager
from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import HeavyLiftingReward, MinAccelerationCapability
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot, MIT_CHEETAH_PARAMS_DICT
from auto_robot_design.generator.topologies.graph_manager_2l import GraphManager2L, plot_2d_bounds, MutationType
from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds

gm = get_preset_by_index_with_bounds(8)
center = gm.generate_central_from_mutation_range()
# center[0] = -0.3
# print(center)
graph = gm.get_graph(center)

for jp, gen_info in gm.generator_dict.items():
    if gen_info.mutation_type == MutationType.UNMOVABLE:
        continue
    print(f"{jp.name:-^50}")
    print(gen_info.mutation_type)
    if isinstance(gen_info.relative_to, list):
        for jp_rel in gen_info.relative_to:
            print(f"Relative : {jp_rel.name}")
    elif gen_info.relative_to is not None:
        print(f"Relative : {gen_info.relative_to.name}")
    print(f"Init: {gen_info.initial_coordinate} -- Range: {gen_info.mutation_range}")

draw_joint_point(graph)
plot_2d_bounds(gm)
plt.show()


