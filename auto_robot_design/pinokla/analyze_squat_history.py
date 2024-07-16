from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Tuple, Union
from auto_robot_design.description.builder import jps_graph2urdf_by_bulder
import numpy as np

from auto_robot_design.optimization.analyze import get_optimizer_and_problem, get_pareto_sample_histogram
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.problems import CalculateMultiCriteriaProblem, MultiCriteriaProblem
from auto_robot_design.optimization.saver import load_checkpoint
from auto_robot_design.pinokla.criterion_agregator import save_criterion_traj
from auto_robot_design.pinokla.squat import SquatHopParameters, SimulateSquatHop


class SimulationSquatHelper:
    def __init__(self) -> None:
        pass

    def reward_vel_with_context(self, sim_hopp: SimulateSquatHop, robo_urdf: str,
                                joint_description: dict, loop_description: dict,
                                x: float) -> float:
        """
        Calculate velocity error from NUMBER_LAST_VALUE samples
        for given control coefficient.

        Args:
            sim_hopp: Squat hop simulation object.
            robo_urdf: URDF string.
            joint_description: Description of joints.
            loop_description: Description of loop.
            x: Control coefficient.

        Returns:
            Velocity error.
        """
        NUMBER_LAST_VALUE = 100

        try:
            q_act, vq_act, acc_act, tau = sim_hopp.simulate(robo_urdf,
                                                            joint_description,
                                                            loop_description, control_coefficient=float(x))
        except Exception as e:
            print(f"Simulation error: {e}")
            return 1.0

        trj_f = sim_hopp.create_traj_equation()
        t = np.linspace(
            0, sim_hopp.squat_hop_parameters.total_time, len(q_act))
        q_vq_acc_des = np.array(list(map(trj_f, t)))
        vq_des = q_vq_acc_des[:, 1]
        tail_vq_des = vq_des[-NUMBER_LAST_VALUE:]
        tail_vq_act = vq_act[-NUMBER_LAST_VALUE:, 0]
        vq_error = np.mean(np.abs(tail_vq_des - tail_vq_act))

        return vq_error

    def min_vel_error_control_brute_force(self, sim_hopp: SimulateSquatHop, robo_urdf: str,
                                          joint_description: dict, loop_description: dict) -> Tuple[float, float]:
        """
        Find minimum velocity error using brute force method.
        """
        x_vec = np.linspace(0.65, 0.9, 10)
        min_fun = self.make_loss_velocity_function(
            sim_hopp, robo_urdf, joint_description, loop_description)
        errors = [min_fun(x) for x in x_vec]
        min_x_and_error = min(zip(x_vec, errors), key=lambda tup: tup[1])
        return min_x_and_error

    def make_loss_velocity_function(self, sim_hopp: SimulateSquatHop, robo_urdf: str,
                                    joint_description: dict, loop_description: dict):

        loss_fun = partial(self.reward_vel_with_context, sim_hopp, robo_urdf, joint_description,
                           loop_description)
        return loss_fun


def get_urdf_from_problem(sample_X: np.ndarray, problem: Union[MultiCriteriaProblem, CalculateMultiCriteriaProblem]):
    graphs = []
    urdf_j_des_l_des = []
    for x_i in sample_X:
        if isinstance(problem, MultiCriteriaProblem):
            mutated_graph = problem.graph_manager.get_graph(x_i)
        elif isinstance(problem, CalculateMultiCriteriaProblem):
            mutated_graph = deepcopy(problem.mutate_JP_by_xopt(x_i))
        else:
            raise Exception("Problem type is not supported")

        robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(
            mutated_graph, problem.builder)
        graphs.append(mutated_graph)
        urdf_j_des_l_des.append(
            (robo_urdf, joint_description, loop_description))
    return graphs, urdf_j_des_l_des


def run_hop_simulations(urdf_j_des_l_des: tuple[list[str], list[dict], list[dict]],
                        hop_cfg: SquatHopParameters, is_vis=False) -> list[dict]:

    saved_dict_list = []
    for i, (robo_urdf_i, joint_description_i, loop_description_i) in enumerate(urdf_j_des_l_des):
        hop_simulator = SimulateSquatHop(hop_cfg)
        sim_help = SimulationSquatHelper()

        res = sim_help.min_vel_error_control_brute_force(hop_simulator, robo_urdf_i, joint_description_i,
                                                         loop_description_i)
        q_act, vq_act, acc_act, tau = hop_simulator.simulate(robo_urdf_i,
                                                             joint_description_i,
                                                             loop_description_i,
                                                             control_coefficient=res[0],
                                                             is_vis=is_vis)
        max1t = max(np.abs(tau[:, 0]))
        max2t = max(np.abs(tau[:, 1]))

        saved_dict = {}

        saved_dict["ControlConst"] = res[0]
        saved_dict["ControlError"] = res[1]

        saved_dict["pos_act"] = q_act[:, 0]
        saved_dict["v_act"] = vq_act[:, 0]
        saved_dict["acc_act"] = acc_act[:, 0]
        saved_dict["tau"] = tau
        saved_dict["HopParams"] = hop_cfg

        saved_dict_list.append(saved_dict)

        print(
            f"Max 1 act: {max1t}, Max 2 act: {max2t}, Error vel: {res[1]}")

    return saved_dict_list


def load_run_hop_sims(path, hop_cfg: SquatHopParameters, number_of_candidates=10, is_viz=False):
    optimizer, problem, res = get_optimizer_and_problem(path)

    all_x = res.X
    rewards_vec = res.F
    sample_X, sample_F = get_pareto_sample_histogram(
        rewards_vec, all_x, number_of_candidates)
    graphs, urdf_j_des_l_des = get_urdf_from_problem(sample_X, problem)

    saved_dict_list = run_hop_simulations(urdf_j_des_l_des, hop_cfg, is_viz)

    path_to_save_result = Path(path) / "squat_compare"

    for i, save_dict_i in enumerate(saved_dict_list):
        save_dict_i["X"] = sample_X[i]
        save_dict_i["Graph"] = graphs[i]
        save_dict_i["Reward"] = sample_F[i]
        save_criterion_traj(urdf_j_des_l_des[0][i], path_to_save_result,
                            urdf_j_des_l_des[2][i], urdf_j_des_l_des[1][i], save_dict_i)
    return saved_dict_list


# def get_sample_torque_traj_from_sample_multi(path, is_vis=False):

#     path_to_save_result = Path(path) / "squat_compare"
#     optimizer, problem, res = get_optimizer_and_problem(path)
#     sample_X, sample_F = get_pareto_sample_histogram(res, 10)
#     graphs, urdf_j_des_l_des = get_urdf_from_problem(sample_X, problem)
#     sqh_p = SquatHopParameters(hop_flight_hight=0.10,
#                                squatting_up_hight=0.0,
#                                squatting_down_hight=-0.04,
#                                total_time=0.2)
#     hoppa = SimulateSquatHop(sqh_p)

#     for i, (robo_urdf_i, joint_description_i, loop_description_i) in enumerate(urdf_j_des_l_des):

#         opti = SimulationSquatHelper.make_loss_velocity_function(hoppa, robo_urdf_i, joint_description_i,
#                                                                  loop_description_i)

#         res = SimulationSquatHelper.min_vel_error_control_brute_force(opti)
#         q_act, vq_act, acc_act, tau = hoppa.simulate(robo_urdf_i,
#                                                      joint_description_i,
#                                                      loop_description_i,
#                                                      control_coefficient=res[0],
#                                                      is_vis=is_vis)
#         max1t = max(np.abs(tau[:, 0]))
#         max2t = max(np.abs(tau[:, 1]))
#         trj_f = hoppa.create_traj_equation()
#         t = np.linspace(0, sqh_p.total_time, len(q_act))
#         list__234 = np.array(list(map(trj_f, t)))

#         saved_dict = {}
#         saved_dict["Graph"] = graphs[i]
#         saved_dict["ControlConst"] = res[0]
#         saved_dict["ControlError"] = res[1]
#         saved_dict["Reward"] = sample_F[i]
#         saved_dict["X"] = sample_X[i]
#         saved_dict["pos_act"] = q_act[:, 0]
#         saved_dict["v_act"] = vq_act[:, 0]
#         saved_dict["acc_act"] = acc_act[:, 0]
#         saved_dict["tau"] = tau
#         saved_dict["HopParams"] = sqh_p
#         print(
#             f"Max 1 act: {max1t}, Max 2 act: {max2t}, Reward:{sample_F[i]}, Error vel: {res[1]}")
#         save_criterion_traj(robo_urdf_i, path_to_save_result,
#                             loop_description_i, joint_description_i, saved_dict)
