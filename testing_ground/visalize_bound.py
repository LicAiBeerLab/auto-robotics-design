import multiprocessing
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


def set_preset_bounds(graph_manager, bounds):
    nam2jp = {jp.name: jp for jp in graph_manager.generator_dict.keys()}
    
    for name, (init_coord, range) in bounds.items():
        jp = nam2jp[name]
        graph_manager.generator_dict[jp].mutation_range = range
        graph_manager.generator_dict[jp].initial_coordinate = init_coord

# 6n4p_symmetric

"Ground_connection"
"Main_knee"
"Main_connection_1"
"Main_connection_2"
"branch_0"
"branch_1"
"branch_2"

# 3n2p

bounds_preset_3n2p_02 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.07)]),
    "Main_knee": ([0,0,-0.2], [None, None, (-0.1, 0.1)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_1": (None, [(-0.05, 0.1), None, (-0.3, -0.1)])
}
bounds_preset_3n2p_12 = {
    "Main_knee": ([0,0,-0.2], [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (0.3, 0.6)]),
    "branch_1": (None, [(-0.05, 0.1), None, (-0.3, -0.1)])
}

bounds_preset_6n4p_s_012 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ([0,0,-0.2], [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_0": (None, [(-0.1, 0.05), None, (-0.25, -0.01)]),
    "branch_1": (None, [(-0.1, -0.02), None, (-0.1, 0.1)]),
    "branch_2": (None, [(-0.1, -0.02), None, (0.05, 0.15)])
}
bounds_preset_6n4p_a_012 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ([0,0,-0.2], [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_0": (None, [(-0.1, 0.05), None, (-0.15, -0.01)]),
    "branch_1": (None, [(-0.1, -0.02), None, (-0.15, 0.0)]),
    "branch_2": (None, [(-0.15, -0.02), None, (0.05, 0.15)])
}
bounds_preset_6n4p_a_120 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ([0,0,-0.2], [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_2": (None, [(-0.1, 0.05), None, (-0.15, -0.01)]),
    "branch_0": (None, [(-0.1, -0.02), None, (-0.15, 0.0)]),
    "branch_1": (None, [(-0.15, -0.02), None, (0.05, 0.15)])
}
bounds_preset_6n4p_a_102 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ([0,0,-0.2], [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_1": (None, [(-0.1, 0.05), None, (-0.25, -0.01)]),
    "branch_0": (None, [(-0.1, -0.02), None, (-0.1, 0.1)]),
    "branch_2": (None, [(-0.1, -0.02), None, (0.05, 0.15)])
}
bounds_preset_6n4p_a_210 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ([0,0,-0.2], [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_2": (None, [(-0.1, 0.05), None, (-0.15, 0.05)]),
    "branch_1": (None, [(-0.1, -0.02), None, (-0.1, 0.1)]),
    "branch_0": (None, [(-0.15, -0.02), None, (0.08, 0.2)])
}
bounds_preset_6n4p_a_201 = {
    "Ground_connection": ([0,0,0.001], [(-0.2, 0.0), None, (-0.03, 0.1)]),
    "Main_knee": ([0,0,-0.2], [None, None, (-0.1, 0.1)]),
    "Main_connection_1": (None, [(-0.2, 0.2), None, (-0.6, 0.4)]),
    "Main_connection_2": (None, [(-0.2, 0.2), None, (-0.3, 0.6)]),
    "branch_1": (None, [(-0.1, 0.05), None, (-0.15, 0.05)]),
    "branch_2": (None, [(-0.1, -0.02), None, (-0.1, 0.1)]),
    "branch_0": (None, [(-0.15, -0.02), None, (0.08, 0.2)])
}

# "Ground_connection"
# "Main_knee"
# "Main_connection_2"
# "branch_1"


gm = GraphManager2L()
gm.reset()
gm.build_main(0.4)
gm.build_6n4p_asymmetric([2, 0, 1])
# gm.build_6n4p_asymmetric([1, 2])
# gm.build_3n2p_branch([0, 2])
set_preset_bounds(gm, bounds_preset_6n4p_a_201)
gm.set_mutation_ranges()
# print(gm.mutation_ranges)
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


