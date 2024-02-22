import os
import time
import numpy as np
import networkx as nx

from cmaes import CMA

from auto_robot_design.pinokla.criterion_agregator import calc_traj_error, calc_traj_error_with_visualization

from auto_robot_design.pino_adapter.pino_adapter import get_pino_description
from auto_robot_design.description.actuators import TMotor_AK80_9
from auto_robot_design.description.builder import Builder, DetalizedURDFCreater
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph


def jps_graph2urdf(graph: nx.Graph):
    kinematic_graph = JointPoint2KinematicGraph(graph)
    kinematic_graph.define_main_branch()
    kinematic_graph.define_span_tree()
    thickness = 0.04
    # # print(scale_factor)
    density = 2700 / 2.8

    for n in kinematic_graph.nodes():
        n.thickness = thickness
        n.density = density

    for j in kinematic_graph.joint_graph.nodes():
        j.pos_limits = (-np.pi, np.pi)
        if j.jp.active:
            j.actuator = TMotor_AK80_9()
        j.damphing_friction = (0.05, 0)
    kinematic_graph.define_link_frames()
    builder = Builder(DetalizedURDFCreater)

    robot, ative_joints, constraints = builder.create_kinematic_graph(kinematic_graph)

    act_description, constraints_descriptions = get_pino_description(ative_joints, constraints)

    return robot.urdf(), act_description, constraints_descriptions


def create_dict_jp_limit(joints, limit):
    jp2limits = {}
    for jp, lim in zip(joints, limit):
        jp2limits[jp] = lim
    return jp2limits


class Optimizer:
    def __init__(
        self, graph: nx.Graph, jp2limits, weights, name, **params_optimizer
    ) -> None:
        self.graph = graph
        self.jp2limits = jp2limits
        self.params_optimizer = params_optimizer
        self.opt_joints = list(self.jp2limits.keys())
        self.initial_xopt, __, __, __ = self.convert_joints2x_opt()
        self.weights = weights
        self.history = {}
        self.cmaes_learning_data = []
        date = "_" + time.strftime("%Y-%m-%d_%H-%M-%S")
        self.name_experiment = name + date

    def convert_joints2x_opt(self):
        x_opt = []
        upper_bounds = []
        lower_bounds = []
        for jp in self.opt_joints:
            lims = self.jp2limits[jp]
            x_opt += [jp.r[0], jp.r[2]]
            upper_bounds += list(lims[2:])
            lower_bounds += list(lims[:2])

        return x_opt, self.opt_joints, upper_bounds, lower_bounds

    def mutate_JP_by_xopt(self, x_opt):
        num_params_one_jp = len(x_opt) // len(self.opt_joints)

        for id, jp in zip(range(0, len(x_opt), num_params_one_jp), self.opt_joints):
            xz = x_opt[id : (id + num_params_one_jp)]
            jp.r = np.array([xz[0], 0, xz[1]])

    def run(self):
        ___, opt_joints, upper_bounds, lower_bounds = self.convert_joints2x_opt()

        bounds = np.array(list(zip(lower_bounds, upper_bounds))).T

        optimizer_CMA = CMA(
            mean=self.initial_xopt, bounds=bounds, **self.params_optimizer
        )

        for generation in range(self.params_optimizer["generations"]):
            solutions = []
            for __ in range(self.params_optimizer["population_size"]):
                x = optimizer_CMA.ask()
                costs = self.get_cost(x)
                fval = self.calc_fval(costs)
                solutions.append((x, fval))
                self.history[x] = (costs, fval)
            optimizer_CMA.tell(solutions)
            self.cmaes_learning_data.append(optimizer_CMA.result)
            if optimizer_CMA.should_stop():
                popsize = optimizer_CMA.population_size * 2
                self.params_optimizer["population_size"] = popsize
                mean = np.array(lower_bounds) + (
                    np.random.rand(len(lower_bounds))
                    * (np.array(upper_bounds) - np.array(lower_bounds))
                )
                optimizer_CMA = CMA(mean=mean, bounds=bounds, **self.params_optimizer)

    def get_cost(self, x):
        self.mutate_JP_by_xopt(x)
        urdf, joint_description, loop_description = jps_graph2urdf(self.graph)
        # res = calc_traj_error(urdf, joint_description, loop_description)
        res = calc_traj_error_with_visualization(urdf, joint_description, loop_description)
        return res

    def calc_fval(self, costs):
        return np.sum(costs @ self.weights)

    def save(self, path="./results"):
        pass

    def _prepare_path(self, folder_name: str):
        """Create a folder for saving results.

        Args:
            folder_name (str): The name of the folder where the results will be saved.

        Returns:
            str: The path to the folder.
        """
        folders = ["results", self.name_experiment, folder_name]
        path = "./"
        for folder in folders:
            path = os.path.join(path, folder)
            if not os.path.exists(path):
                os.mkdir(path)
        path = os.path.abspath(path)
        print(f"Dara data will be in {path}")

        return path
