import os
import dill

import numpy as np

from auto_robot_design.description.builder import jps_graph2urdf


from pymoo.core.problem import ElementwiseProblem

from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator


class CalculateCriteriaProblemByWeigths(ElementwiseProblem):
    def __init__(self, graph, jp2limits, criteria : CriteriaAggregator, weights, rewards=[], **kwargs):
        self.graph = graph
        self.jp2limits = jp2limits
        self.opt_joints = list(self.jp2limits.keys())
        self.weights = weights
        self.criteria = criteria
        self.rewards = rewards
        self.initial_xopt, __, upper_bounds, lower_bounds = self.convert_joints2x_opt()
        super().__init__(
            n_var=len(self.initial_xopt),
            n_obj=1,
            xu=upper_bounds,
            xl=lower_bounds,
            **kwargs,
        )

    def add_reward(self, reward, weight=1.0):
        self.rewards.append((reward, weight))
        
    def _evaluate(self, x, out, *args, **kwargs):
        self.mutate_JP_by_xopt(x)
        urdf, joint_description, loop_description = jps_graph2urdf(self.graph)

        #calculates all characteristics declared in the CriteriaAggregator
        point_criteria_vector, trajectory_criteria, res_dict_fixed = self.criteria.get_criteria_data(urdf, joint_description, loop_description)

        # all rewards are calculated and added to the result
        total_result = 0
        partial_results = []
        for reward, weight in self.rewards:
            partial_results.append(reward.calculate(point_criteria_vector, trajectory_criteria, res_dict_fixed))
            total_result+= weight*partial_results[-1]
        
        # the form of the output required by the pymoo lib
        out["F"] = -total_result
        out["Fs"] = partial_results

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
    def load(cls, path, **kwargs):
        with open(os.path.join(path, "problem_data.pkl"), "rb") as f:
            graph = dill.load(f)
            opt_joints = dill.load(f)
            initial_xopt = dill.load(f)
            jp2limits = dill.load(f)
            criteria = dill.load(f)
        istance = cls(graph, jp2limits, criteria, np.ones(len(criteria)), **kwargs)
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



