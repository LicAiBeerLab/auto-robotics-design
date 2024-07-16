from copy import deepcopy
from auto_robot_design.description.builder import jps_graph2urdf_by_bulder
import numpy as np
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.problems import CalculateMultiCriteriaProblem
from auto_robot_design.optimization.saver import load_checkpoint


def get_optimizer_and_problem(path: str) -> tuple[PymooOptimizer, CalculateMultiCriteriaProblem]:
    """
    Load the optimizer and problem from a checkpoint file.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        tuple: A tuple containing:
            - PymooOptimizer: The optimizer object.
            - CalculateMultiCriteriaProblem: The multi-criteria problem object.
            - dict: The result after running the optimizer.
    """
    problem = CalculateMultiCriteriaProblem.load(path)
    checkpoint = load_checkpoint(path)
    optimizer = PymooOptimizer(problem, checkpoint)
    optimizer.load_history(path)
    res = optimizer.run()
    return optimizer, problem, res


def get_pareto_sample_linspace(res, sample_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Get Pareto samples using linspace.

    Args:
        res: The result object containing optimization results.
        sample_len (int): Number of samples to retrieve.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Sampled input parameters (X).
            - np.ndarray: Corresponding output results (F).
    """
    sample_indices = np.linspace(0, len(res.F) - 1, sample_len, dtype=int)
    sample_x = res.X[sample_indices]
    sample_F = res.F[sample_indices]
    return sample_x, sample_F


def get_pareto_sample_histogram(rewards_vector, x_vector, sample_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Get Pareto samples using histogram binning.

    Args:
        res: The result object containing optimization results.
        sample_len (int): Number of samples to retrieve.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Sampled input parameters (X).
            - np.ndarray: Corresponding output results (F).
    """

    _, bins_edg = np.histogram(rewards_vector[:, 0], sample_len)
    bin_indices = np.digitize(rewards_vector[:, 0], bins_edg, right=True)
    bins_set_id = [np.where(bin_indices == i)[0]
                   for i in range(1, len(bins_edg))]
    best_in_bins = [i[0] for i in bins_set_id if len(i) > 0]
    sample_F = rewards_vector[best_in_bins]
    sample_X = x_vector[best_in_bins]
    return sample_X, sample_F

 

def get_histogram_data(rewards: np.ndarray) -> list:
    """
    Get histogram bin data for rewards.
    Returns 2d list where each row is array 
    indexes of elements that includes into bins. 

    Args:
        rewards (np.ndarray): Array of reward values.

    Returns:
        list: A list of arrays where each array contains indices of rewards that fall into the corresponding bin.
    """
    NUMBER_BINS = 10
    _, bins_edg = np.histogram(rewards, NUMBER_BINS)
    bin_indices = np.digitize(rewards, bins_edg, right=True)
    bins_set_id = [np.where(bin_indices == i)[0]
                   for i in range(1, len(bins_edg))]

    return bins_set_id
