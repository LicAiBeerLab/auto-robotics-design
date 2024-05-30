import numpy as np
from pymoo.decomposition.asf import ASF


def get_design_from_pareto_front(pareto_front, set_weights: np.ndarray):
    """
    Returns the indexes of the designs from the Pareto front based on the given set of weights (ASF method).

    Args:
        pareto_front (np.ndarray): The Pareto front containing the designs.
        set_weights (np.ndarray): The set of weights used for optimization.

    Returns:
        np.ndarray: The indexes of the designs from the Pareto front based on the given set of weights.
    """

    approx_ideal = pareto_front.min(axis=0)
    approx_nadir = pareto_front.max(axis=0)

    nF = (pareto_front - approx_ideal) / (approx_nadir - approx_ideal)

    decomp = ASF()

    indexes = np.zeros(set_weights.shape[0], dtype=int)

    for i, weight in enumerate(set_weights):
        indexes[i] = decomp.do(nF, 1 / weight).argmin()

    return indexes


