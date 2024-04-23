from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder, jps_graph2urdf_by_bulder
import numpy as np

import matplotlib.pyplot as plt

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator, )

from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.squat import SquatHopParameters, SimulateSquatHop


from typing import overload


def compare_powers_torque(tau_taj_1, tau_traj_2):
    # Вычисляем сумму abs() по траекториям. Чтобы дальнейшее 
    # вычитание было иммунно к смене главного двигателя
    # Делим на ссумму чтобы узнать процент
    # Выводим ввиде графика и среденго значния улучшения
    pass
def max_toque(tau_taj_1, tau_traj_2):
    # Максимальный момент в абсолютных значениях
    pass
def calculate_cooperation(tau_taj_1, tau_traj_2):
    # Вычисляем разницу в моментах на движках
    pass
