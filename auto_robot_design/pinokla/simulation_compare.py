from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
import numpy as np

import matplotlib.pyplot as plt

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator, )

from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.squat import SquatHopParameters, SimulateSquatHop

from typing import overload


def сomparison_ratio_positive_torques(tau_traj_1: np.ndarray, tau_traj_2: np.ndarray):
    abs_1 = np.abs(tau_traj_1)
    abs_1_max = np.max(abs_1, axis=1)
    abs_2 = np.abs(tau_traj_2)
    abs_2_max = np.max(abs_2, axis=1)
    divided = np.divide(abs_1_max,
                        abs_2_max,
                        out=1000*np.ones_like(abs_2_max),
                        where=abs_2_max > 0.00001)

    return divided

def max_toque_ratio(tau_traj_1: np.ndarray, tau_traj_2: np.ndarray):
    abs_1 = np.abs(tau_traj_1)
    abs_2 = np.abs(tau_traj_2)
    TOP_PERCENT = 0.15
    top_samples_number = int(len(tau_traj_1)*TOP_PERCENT)
    
    abs_1_top = np.sort(abs_1, axis=0)[-top_samples_number:]
    abs_2_top = np.sort(abs_2, axis=0)[-top_samples_number:]
    abs_1_top_mean = np.mean(abs_1_top, axis=0)
    abs_2_top_mean = np.mean(abs_2_top, axis=0)

    return abs_1_top_mean, abs_2_top_mean 