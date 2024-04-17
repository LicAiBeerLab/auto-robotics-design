from copy import copy, deepcopy
from auto_robot_design.pinokla.closed_loop_jacobian import inverseConstraintKinematicsSpeed
from auto_robot_design.pinokla.loader_tools import Robot, build_model_with_extensions, make_Robot_copy
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.description.builder import DetalizedURDFCreaterFixedEE
import pinocchio as pin
import numpy as np
import meshcat
from pinocchio.visualize import MeshcatVisualizer

from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopInverseKinematicsProximal, closedLoopProximalMount
import numpy as np

from auto_robot_design.description.actuators import t_motor_actuators

from auto_robot_design.description.utils import (
    all_combinations_active_joints_n_actuator, )
from auto_robot_design.description.builder import (DetalizedURDFCreaterFixedEE,
                                                   ParametrizedBuilder,
                                                   jps_graph2urdf_by_bulder)
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
from auto_robot_design.pinokla.robot_utils import add_3d_constrain_current_q


def get_pino_models(robo_urdf, joint_description, loop_description, actuator_context):

    robo_translation_base = build_model_with_extensions(
        robo_urdf,
        joint_description=joint_description,
        loop_description=loop_description,
        actuator_context=actuator_context,
        fixed=False,
        root_joint_type=pin.JointModelPZ())

    robo_fixed_base = build_model_with_extensions(robo_urdf,
                                                  joint_description=joint_description,
                                                  loop_description=loop_description,
                                                  actuator_context=actuator_context,
                                                  fixed=True)
    robo_free_base = build_model_with_extensions(
        robo_urdf,
        joint_description=joint_description,
        loop_description=loop_description,
        actuator_context=actuator_context,
        fixed=False)
    return robo_fixed_base, robo_translation_base, robo_free_base


def quartic_func_free_acc(q0, qf, T, qd0=0, qdf=0):
    """
    Quartic scalar polynomial as a function.
    Final acceleration is unconstrained, start acceleration is zero.

    :param q0: initial value
    :type q0: float
    :param qf: final value
    :type qf: float
    :param T: trajectory time
    :type T: float
    :param qd0: initial velocity, defaults to 0
    :type q0: float, optional
    :param qdf: final velocity, defaults to 0
    :type q0: float, optional
    :return: polynomial function :math:`f: t \mapsto (q(t), \dot{q}(t), \ddot{q}(t))`
    :rtype: callable

    Returns a function which computes the specific quartic polynomial, and its
    derivatives, as described by the parameters.
 

    """

    # solve for the polynomial coefficients using least squares
    # fmt: off
    X = [
        [0.0,         0.0,        0.0,     0.0,  1.0],
        [T**4,        T**3,       T**2,    T,    1.0],
        [0.0,         0.0,        0.0,     1.0,  0.0],
        [4.0 * T**3,  3.0 * T**2, 2.0 * T, 1.0,  0.0],
        [0.0,         0.0,        2.0,     0.0,  0.0],
        
    ]
    # fmt: on
    coeffs, resid, rank, s = np.linalg.lstsq(
        X, np.r_[q0, qf, qd0, qdf, 0], rcond=None
    )

    # coefficients of derivatives
    coeffs_d = coeffs[0:4] * np.arange(4, 0, -1)
    coeffs_dd = coeffs_d[0:3] * np.arange(3, 0, -1)

    return lambda x: (
        np.polyval(coeffs, x),
        np.polyval(coeffs_d, x),
        np.polyval(coeffs_dd, x),
    )

def calculate_final_v(desired_hop_high: float):
    g = 9.81
    return np.sqrt(desired_hop_high * 2 * g)

def create_trajectory(fixed_robot: Robot, desired_hop_high: float, squatting_hight: float, hop_time: float):
    pass
    # 1) 

def create_start_q(fixed_robo: Robot) -> start_q:
    pass

def simulate(robo_on_stand: Robot, start_q, traj_fun, time):
    # init sim
    # prepare model
    # 
    pass
