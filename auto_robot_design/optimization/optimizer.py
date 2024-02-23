import os
import pickle
import time
import numpy as np
import numpy.linalg as la
import networkx as nx

from cmaes import CMA

import pinocchio as pin

from pinokla.criterion_agregator import calc_traj_error, calc_traj_error_with_visualization
from pinokla.loader_tools import build_model_with_extensions

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
        self.history = []
        self.cmaes_learning_data = []
        date = "_" + time.strftime("%Y-%m-%d_%H-%M-%S")
        self.name_experiment = name + date

    def convert_joints2x_opt(self):
        x_opt = np.zeros(len(self.opt_joints) * 2)
        upper_bounds = np.zeros(len(x_opt))
        lower_bounds = np.zeros(len(x_opt))
        i = 0
        for jp in self.opt_joints:
            lims = self.jp2limits[jp]
            x_opt[i:i+2] = np.array([jp.r[0], jp.r[2]])
            upper_bounds[i:i+2] = np.array(lims[2:]) + x_opt[i:i+2]
            lower_bounds[i:i+2] = np.array(lims[:2]) + x_opt[i:i+2]
            i += 2

        return x_opt, self.opt_joints, upper_bounds, lower_bounds

    def mutate_JP_by_xopt(self, x_opt):
        num_params_one_jp = len(x_opt) // len(self.opt_joints)

        for id, jp in zip(range(0, len(x_opt), num_params_one_jp), self.opt_joints):
            xz = x_opt[id : (id + num_params_one_jp)]
            jp.r = np.array([xz[0], 0, xz[1]])

    def run(self, generations = 100):
        ___, opt_joints, upper_bounds, lower_bounds = self.convert_joints2x_opt()

        bounds = np.array(list(zip(lower_bounds, upper_bounds)))

        optimizer_CMA = CMA(
            self.initial_xopt, 0.5, bounds=bounds, **self.params_optimizer
        )

        for generation in range(generations):
            solutions = []
            for __ in range(optimizer_CMA.population_size):
                x = optimizer_CMA.ask()
                costs = self.get_cost(x)
                fval = self.calc_fval(costs)
                solutions.append((x, fval))
                self.history.append((x, costs, fval))
            optimizer_CMA.tell(solutions)
            print(f"Generation {generation} is done")
            # self.cmaes_learning_data.append(optimizer_CMA.result)
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
        res = calc_traj_error(urdf, joint_description, loop_description)
        # res = calc_traj_error_with_visualization(urdf, joint_description, loop_description)
        
        free_robo = build_model_with_extensions(urdf, joint_description, loop_description,
                                        False)
        total_mass = pin.computeTotalMass(free_robo.model, free_robo.data)
        com_dist = la.norm(pin.centerOfMass(free_robo.model, free_robo.data))
        return np.array([res, total_mass])

    def calc_fval(self, costs):
        return np.sum(costs @ self.weights)

    def save(self, folder_name: str):
        path = self._prepare_path(folder_name)
        with open(os.path.join(path, "history.txt"), "w") as f:
            save_history = []
            for item in self.history:
                save_history.append((item[0].tolist(), item[1].tolist(), item[2].tolist()))
            f.write(str(save_history))
        np.save(os.path.join(path, "cmaes_learning_data.npy"), self.cmaes_learning_data)
        np.save(os.path.join(path, "weights.npy"), self.weights)
        with open(os.path.join(path, "params_optimizer.txt"), "w") as f:
            f.write(str(self.params_optimizer))
        with open(os.path.join(path, "graph.pkl"), "wb") as f:
            pickle.dump(self.graph, f)
        with open(os.path.join(path, "opt_joints.pkl"), "wb") as f:
            pickle.dump(self.opt_joints, f)
        return path

    def load(self, folder_name: str):
        path = self._prepare_path(folder_name)
        with open(os.path.join(path, "params_optimizer.txt"), "r") as f:
            self.params_optimizer = eval(f.read())
        with open(os.path.join(path, "history.txt"), "r") as f:
            self.history = eval(f.read())
        self.cmaes_learning_data = np.load(os.path.join(path, "cmaes_learning_data.npy"))
        self.weights = np.load(os.path.join(path, "weights.npy"))
        with open(os.path.join(path, "params_optimizer.txt"), "r") as f:
            self.params_optimizer = eval(f.read())
        with open(os.path.join(path, "graph.pkl"), "rb") as f:
            self.graph = pickle.load(f)
        with open(os.path.join(path, "opt_joints.pkl"), "rb") as f:
            self.opt_joints = pickle.load(f)

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
