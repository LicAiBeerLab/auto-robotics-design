import os
import time
import numpy as np
import networkx as nx

from cmaes import CMA

def create_dict_jp_limit(joints, limit):
    jp2limits = {}
    for jp, lim in zip(joints, limit):
        jp2limits[jp] = lim
    return jp2limits

class Optimizer:
    def __init__(self, graph: nx.Graph, jp2limits, weights, name, **params_optimizer) -> None:
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
        num_params_one_jp = len(x_opt) / len(self.opt_joints)
        
        for id, jp in zip(range(0, len(x_opt), num_params_one_jp), self.opt_joints):
            jp.r = x_opt[id:(id+num_params_one_jp)]


    def run(self):
        ___, opt_joints, upper_bounds, lower_bounds = self.convert_joints2x_opt()
        
        bounds = np.array(list(zip(lower_bounds, upper_bounds))).T
        
        optimizer_CMA = CMA(mean=self.initial_xopt,bounds=bounds, **self.params_optimizer)
        
        for generation in range(self.params_optimizer['generations']):
            solutions = []
            for __ in range(self.params_optimizer['population_size']):
                x = optimizer_CMA.ask()
                costs = self.get_cost(x)
                fval = self.calc_fval(costs)
                solutions.append((x, fval))
                self.history[x] = (costs, fval)
            optimizer_CMA.tell(solutions)
            self.cmaes_learning_data.append(optimizer_CMA.result)
            if optimizer_CMA.should_stop():
                popsize = optimizer_CMA.population_size * 2
                self.params_optimizer['population_size'] = popsize
                mean = np.array(lower_bounds) + (np.random.rand(len(lower_bounds)) * (np.array(upper_bounds) - np.array(lower_bounds)))
                optimizer_CMA = CMA(mean=mean,bounds=bounds, **self.params_optimizer)
    
    def get_cost(self, x):
        self.mutate_JP_by_xopt(x)
        pass
    
    def calc_fval(self, costs):
        return np.sum(costs @ self.weights)
    
    
    def save(self, path = "./results"):
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