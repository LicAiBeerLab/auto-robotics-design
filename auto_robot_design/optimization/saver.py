import dill
import time
import os

from matplotlib import pyplot as plt
from pymoo.core.problem import Problem

from auto_robot_design.description.utils import draw_joint_point
from pymoo.core.callback import Callback


def load_checkpoint(path: str):
    with open(os.path.join(path, "checkpoint.pkl"), "rb") as f:
        algorithm = dill.load(f)
    return algorithm


class ProblemSaver:
    def __init__(
        self, problem: Problem, folder_name: str, use_date: bool = True
    ) -> None:

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
            dill.dump(self.problem, f)
        draw_joint_point(self.problem.graph)
        plt.savefig(os.path.join(self.path, "initial_mechanism.png"))
        plt.close()

    def save_history(self, history):
        with open(os.path.join(self.path, "history.pkl"), "wb") as f:
            dill.dump(history, f)


class CallbackSaver(Callback):
    def __init__(self, problem_saver: ProblemSaver) -> None:
        super().__init__()
        self.problem_saver = problem_saver

    def notify(self, algorithm):
        with open(os.path.join(self.problem_saver.path, "checkpoint.pkl"), "wb") as f:
            dill.dump(algorithm, f)
