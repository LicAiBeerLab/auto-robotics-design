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
def chunk_list(lst, chunk_size):
    """Yield successive chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def save_result_append(filename, new_data):
    if os.path.exists(filename):
        # Load existing data
        existing_data = np.load(filename)
        # Append new data
        combined_data = {key: np.stack(
            (existing_data[key], new_data[key])) for key in new_data.keys()}
    else:
        combined_data = new_data
    # Save combined data back to file
    np.savez(filename, **combined_data)


def get_n_dim_linspace(upper_bounds, lower_bounds):
    ranges = np.array([lower_bounds, upper_bounds]).T

    LINSPACE_N = 5
    linspaces = [np.linspace(start, stop, LINSPACE_N)
                 for start, stop in ranges]
    meshgrids = np.meshgrid(*linspaces)
    vec = np.array([dim_i.flatten() for dim_i in meshgrids]).T
    return vec


def test_graph(problem: CalculateMultiCriteriaProblem, workspace_trj: np.ndarray, x_vec: np.ndarray):
    problem.mutate_JP_by_xopt(x_vec)
    fixed_robot, free_robot = jps_graph2pinocchio_robot(
        problem.graph, problem.builder)
    poses, q_array, constraint_errors, reach_array = folow_traj_by_proximal_inv_k_2(fixed_robot.model, fixed_robot.data,
                                                                                    fixed_robot.constraint_models, fixed_robot.constraint_data, "EE", workspace_trj)
    return poses, q_array, constraint_errors, reach_array, x_vec


def convert_res_to_dict(poses, q_array, constraint_errors, reach_array, x_vec):
    return {"poses": poses, "q_array": q_array, "constraint_errors": constraint_errors, "reach_array": reach_array, "x_vec": x_vec}


def stack_dicts(dict_list, stack_func=np.stack):
    """
    Stacks dictionaries containing numpy arrays using the specified stacking function.

    Parameters:
    dict_list (list of dict): List of dictionaries to stack.
    stack_func (function): Numpy stacking function to use (e.g., np.vstack, np.hstack, np.concatenate).

    Returns:
    dict: A dictionary with stacked numpy arrays.
    """
    # Initialize an empty dictionary to hold the stacked arrays
    stacked_dict = {}

    # Iterate through the keys of the first dictionary (assuming all dicts have the same keys)
    for key in dict_list[0].keys():
        # Stack the arrays for the current key across all dictionaries
        stacked_dict[key] = stack_func([d[key] for d in dict_list])

    return stacked_dict


def test_chunk(problem: CalculateMultiCriteriaProblem, x_vecs: np.ndarray, workspace_trj: np.ndarray, file_name):
    grabbed_fun = partial(test_graph, problem, workspace_trj)
    parallel_results = []
    cpus = cpu_count(only_physical_cores=True)
    parallel_results = Parallel(cpus, backend="multiprocessing", verbose=100, timeout=60 * 1000)(delayed(grabbed_fun)(i)
                                                                                                 for i in x_vecs)
    list_dict = []
    for i in parallel_results:
        list_dict.append(convert_res_to_dict(*i))
    staced_res = stack_dicts(list_dict)
    save_result_append(file_name, staced_res)
    return staced_res


if __name__ == '__main__':
    start_time = time.time()
    TOPOLGY_NAME = 0
    FILE_NAME = "WORKSPACE_TOP" + str(TOPOLGY_NAME)
    graph, optimizing_joints, constrain_dict, builder, step_trajs, squat_trajs, workspace_trajectory = traj_graph_setup.get_graph_and_traj(
        TOPOLGY_NAME)
    reward_manager, crag, soft_constrain = create_reward_manager.get_manager_preset_2_stair_climber(
        graph, optimizing_joints, workspace_traj=workspace_trajectory, step_trajs=step_trajs, squat_trajs=squat_trajs)

    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
    problem = CalculateMultiCriteriaProblem(graph, builder=builder,
                                            jp2limits=optimizing_joints,
                                            crag=crag,
                                            soft_constrain=soft_constrain,
                                            rewards_and_trajectories=reward_manager,
                                            Actuator=actuator)

    x_opt, opt_joints, upper_bounds, lower_bounds = problem.convert_joints2x_opt()
    # optimizing_joints = get_optimizing_joints(graph, constrain_dict)

    vecs = get_n_dim_linspace(upper_bounds, lower_bounds)
    chunk_vec = list(chunk_list(vecs, 100))
    for num, i_vec in enumerate(chunk_vec):
        try:
            test_chunk(problem, i_vec, workspace_trajectory, FILE_NAME)
        except:
            print("FAILD")
        print(f"Tested chunk {num} / {len(chunk_vec)}")
        ellip = ( time.time() - start_time) / 60
        print(f"Remaining minute {ellip}")
 
 