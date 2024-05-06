import os
import dill
from typing import Union, Tuple
import numpy as np
from pyparsing import C

from auto_robot_design.description.builder import jps_graph2pinocchio_robot


from pymoo.core.problem import ElementwiseProblem

from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.optimization.rewards.reward_base import Reward 
def get_optimizing_joints(graph, constrain_dict):
    """
    Retrieves the optimizing joints from a graph based on the given constraint dictionary.
    Adapter constraints from generator to the optimization problem.

    Parameters:
    - graph (Graph): The graph containing the joints.
    - constrain_dict (dict): A dictionary containing the constraints for each joint.

    Returns:
    - optimizing_joints (dict): A dictionary containing the optimizing joints and their corresponding ranges.

    """
    name2jp = dict(map(lambda x: (x.name, x), graph.nodes()))
    # filter the joints to be optimized
    optimizing_joints = dict(
        filter(lambda x: x[1]["optim"] and x[0] in name2jp, constrain_dict.items()))
    # the procedure below is rather unstable
    optimizing_joints = dict(
        map(
            lambda x: (
                name2jp[x[0]],
                (
                    x[1]["x_range"][0],
                    x[1].get("z_range", [-0.01, 0.01])[0],
                    x[1]["x_range"][1],
                    x[1].get("z_range", [0, 0])[1],
                ),
            ),
            optimizing_joints.items(),
        ))
    return optimizing_joints

class CalculateCriteriaProblemByWeigths(ElementwiseProblem):
    def __init__(self, graph, builder, jp2limits, rewards_and_trajectories : list, crag, **kwargs):
        if "Actuator" in kwargs:
            self.motor = kwargs["Actuator"]
        self.graph = graph
        self.builder = builder
        self.jp2limits = jp2limits
        self.opt_joints = list(self.jp2limits.keys())
        self.crag = crag
        self.rewards_and_trajectories:list[Tuple[list[Reward],list[np.array]]] = rewards_and_trajectories
        self.initial_xopt, __, upper_bounds, lower_bounds = self.convert_joints2x_opt()
        super().__init__(
            n_var=len(self.initial_xopt),
            n_obj=1,
            xu=upper_bounds,
            xl=lower_bounds,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        self.mutate_JP_by_xopt(x)
        fixed_robot, free_robot = jps_graph2pinocchio_robot(self.graph, self.builder)
        # all rewards are calculated and added to the result
        total_reward=0
        print(total_reward)
        partial_rewards = []
        for rewards, trajectories in self.rewards_and_trajectories:
            max_reward = -float('inf')
            max_partial = None
            for trajectory_id, trajectory in enumerate(trajectories):
                point_criteria_vector, trajectory_criteria, res_dict_fixed = self.crag.get_criteria_data(fixed_robot, free_robot, trajectory)
                current_total = 0
                current_partial = []
                for reward, weight in rewards:
                    current_partial.append(reward.calculate(point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator=self.motor)[0])
                    current_total += weight*current_partial[-1]
                if current_total > max_reward:
                    max_reward = current_total
                    max_partial = current_partial
                    best_trajectory_id = trajectory_id
            total_reward+= max_reward
            max_partial = [best_trajectory_id]+max_partial
            partial_rewards.append(max_partial)

        # the form of the output required by the pymoo lib
        
        out["F"] = -total_reward
        out["Fs"] = partial_rewards


    def convert_joints2x_opt(self):
        x_opt = np.zeros(len(self.opt_joints) * 2)
        upper_bounds = np.zeros(len(x_opt))
        lower_bounds = np.zeros(len(x_opt))
        i = 0
        for jp in self.opt_joints:
            lims = self.jp2limits[jp]
            x_opt[i : i + 2] = np.array([jp.r[0], jp.r[2]])
            upper_bounds[i : i + 2] = np.array(lims[2:]) + x_opt[i : i + 2]
            lower_bounds[i : i + 2] = np.array(lims[:2]) + x_opt[i : i + 2]
            i += 2

        return x_opt, self.opt_joints, upper_bounds, lower_bounds

    def mutate_JP_by_xopt(self, x_opt):
        num_params_one_jp = len(x_opt) // len(self.opt_joints)

        for id, jp in zip(range(0, len(x_opt), num_params_one_jp), self.opt_joints):
            xz = x_opt[id : (id + num_params_one_jp)]
            list_nodes = list(self.graph.nodes())
            id = list_nodes.index(jp)
            list_nodes[id].r = np.array([xz[0], 0, xz[1]])

    @classmethod
    def load(cls, path, builder, criteria_agrregator, **kwargs):
        with open(os.path.join(path, "problem_data.pkl"), "rb") as f:
            graph = dill.load(f)
            opt_joints = dill.load(f)
            initial_xopt = dill.load(f)
            jp2limits = dill.load(f)
            rewards_and_trajectories = dill.load(f)
        istance = cls(graph, builder, jp2limits, rewards_and_trajectories, criteria_agrregator, **kwargs)
        istance.initial_xopt = initial_xopt
        return istance


class CalculateMultiCriteriaProblem(ElementwiseProblem):
    def __init__(self, graph, jp2limits, criteria, **kwargs):
        self.graph = graph
        self.jp2limits = jp2limits
        self.opt_joints = list(self.jp2limits.keys())
        self.criteria = criteria
        self.initial_xopt, __, upper_bounds, lower_bounds = self.convert_joints2x_opt()
        super().__init__(
            n_var=len(self.initial_xopt),
            n_obj=len(self.criteria),
            xu=upper_bounds,
            xl=lower_bounds,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        self.mutate_JP_by_xopt(x)
        urdf, joint_description, loop_description = jps_graph2urdf(self.graph)

        F = [
            criteria(urdf, joint_description, loop_description)
            for criteria in self.criteria
        ]
        out["F"] = F

    def convert_joints2x_opt(self):
        x_opt = np.zeros(len(self.opt_joints) * 2)
        upper_bounds = np.zeros(len(x_opt))
        lower_bounds = np.zeros(len(x_opt))
        i = 0
        for jp in self.opt_joints:
            lims = self.jp2limits[jp]
            x_opt[i : i + 2] = np.array([jp.r[0], jp.r[2]])
            upper_bounds[i : i + 2] = np.array(lims[2:]) + x_opt[i : i + 2]
            lower_bounds[i : i + 2] = np.array(lims[:2]) + x_opt[i : i + 2]
            i += 2

        return x_opt, self.opt_joints, upper_bounds, lower_bounds

    def mutate_JP_by_xopt(self, x_opt):
        num_params_one_jp = len(x_opt) // len(self.opt_joints)

        for id, jp in zip(range(0, len(x_opt), num_params_one_jp), self.opt_joints):
            xz = x_opt[id : (id + num_params_one_jp)]
            self.graph[jp].r = np.array([xz[0], 0, xz[1]])

class CalculateCriteriaMeanTraj(CalculateCriteriaProblemByWeigths):
    def __init__(self, graph, builder, jp2limits, rewards_and_trajectories : list, crag, **kwargs):
        super().__init__(graph, builder, jp2limits, rewards_and_trajectories, crag, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        self.mutate_JP_by_xopt(x)
        fixed_robot, free_robot = jps_graph2pinocchio_robot(self.graph, self.builder)
        # all rewards are calculated and added to the result
        partial_rewards = []
        total_reward = 0
        for rewards, trajectories in self.rewards_and_trajectories:
            mean_reward = np.zeros(len(trajectories))
            mean_partial = np.zeros((len(trajectories),len(rewards)))
            for trajectory_id, trajectory in enumerate(trajectories):
                point_criteria_vector, trajectory_criteria, res_dict_fixed = self.crag.get_criteria_data(fixed_robot, free_robot, trajectory)
                current_total = 0
                current_partial = []
                for reward, weight in rewards:
                    current_partial.append(reward.calculate(point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator=self.motor)[0])
                    current_total += weight*current_partial[-1]
                mean_reward[trajectory_id] = current_total
                mean_partial[trajectory_id] = current_partial
            # print(mean_reward)
            # print(mean_partial)
            total_reward += np.mean(mean_reward)
            partial_rewards.append([np.mean(mean_reward)]+list(np.mean(mean_partial, axis=0)))

        # the form of the output required by the pymoo lib
        
        out["F"] = -total_reward
        out["Fs"] = partial_rewards


class CalculateCriteriaMeanMultiTraj(CalculateCriteriaProblemByWeigths):
    def __init__(self, graph, builder, jp2limits, rewards_and_trajectories : list, crag, **kwargs):
        super().__init__(graph, builder, jp2limits, rewards_and_trajectories, crag, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        self.mutate_JP_by_xopt(x)
        fixed_robot, free_robot = jps_graph2pinocchio_robot(self.graph, self.builder)
        # all rewards are calculated and added to the result
        partial_rewards = []
        total_reward = np.zeros(len(self.rewards_and_trajectories))
        for id, (rewards, trajectories) in enumerate(self.rewards_and_trajectories):
            mean_reward = np.zeros(len(trajectories))
            mean_partial = np.zeros((len(trajectories),len(rewards)))
            for trajectory_id, trajectory in enumerate(trajectories):
                point_criteria_vector, trajectory_criteria, res_dict_fixed = self.crag.get_criteria_data(fixed_robot, free_robot, trajectory)
                current_total = 0
                current_partial = []
                for reward, weight in rewards:
                    current_partial.append(reward.calculate(point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator=self.motor)[0])
                    current_total += weight*current_partial[-1]
                mean_reward[trajectory_id] = current_total
                mean_partial[trajectory_id] = current_partial
            total_reward[id] = np.mean(mean_reward)
            partial_rewards.append([total_reward]+list(np.mean(mean_partial, axis=0)))

        # the form of the output required by the pymoo lib
        # print(total_reward)
        out["F"] = -total_reward
        out["Fs"] = partial_rewards