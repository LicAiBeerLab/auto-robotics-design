import pickle
import dill
import time
import os

from matplotlib import pyplot as plt
import numpy as np
from pymoo.core.problem import Problem

from auto_robot_design.description.utils import draw_joint_point
from pymoo.core.callback import Callback



def load_nonmutable(problem, path: str):
    with open(os.path.join(path, "problem_data.pkl"), "rb") as f:
        problem.graph = dill.load(f)
        problem.opt_joints = dill.load(f)
        problem.initial_xopt = dill.load(f)
        problem.jp2limits = dill.load(f)
        problem.criteria = dill.load(f)

def load_checkpoint(path: str):
    with open(os.path.join(path, "checkpoint.pkl"), "rb") as f:
        history = dill.load(f)
    return history

class ProblemSaver:
    def __init__(self, problem: Problem, folder_name: str, use_date: bool = True) -> None:
        
        self.problem = problem
        date = "_" + time.strftime("%Y-%m-%d_%H-%M-%S") if use_date else ""
        self.folder_name = folder_name + date
        self.use_date = use_date
        self.path = self._prepare_folder()
        
    def _prepare_folder(self):

        folders = ["results", self.folder_name]
        path = "./"
        for folder in folders:
            path = os.path.join(path, folder)
            if not os.path.exists(path):
                os.mkdir(path)
        path = os.path.abspath(path)
        
        return path
    

    def save_nonmutable(self):
        with open(os.path.join(self.path, "problem_data.pkl"), "wb") as f:
            dill.dump(self.problem.graph, f)
            dill.dump(self.problem.opt_joints, f)
            dill.dump(self.problem.initial_xopt, f)
            dill.dump(self.problem.jp2limits, f)
            dill.dump(self.problem.criteria, f)
        draw_joint_point(self.problem.graph)
        plt.savefig(os.path.join(self.path, "initial_mechanism.png"))

    def save_checkpoint(self):
        pass

class CallbackSaver(Callback):
    def __init__(self, problem_saver: ProblemSaver) -> None:
        super().__init__()
        self.problem_saver = problem_saver
        
    def notify(self, algorithm):
        self.problem_saver.save_checkpoint()
        with open(os.path.join(self.problem_saver.path, "checkpoint.pkl"), "wb") as f:
            dill.dump(algorithm, f)
        np.save(os.path.join(self.problem_saver.path, "best.npy"), algorithm.pop.get("F").min())