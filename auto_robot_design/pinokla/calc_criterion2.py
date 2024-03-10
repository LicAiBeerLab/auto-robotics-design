from typing import NamedTuple, Optional
from auto_robot_design.pinokla.loader_tools import (
    build_model_with_extensions,
    Robot,
    completeRobotLoader,
    completeRobotLoaderFromStr,
)
import time
from matplotlib.pylab import LinAlgError
import pinocchio as pin
from numpy.linalg import norm
import numpy as np
from auto_robot_design.pinokla.closed_loop_jacobian import dq_dqmot, inverseConstraintKinematicsSpeed, closedLoopInverseKinematicsProximal
from auto_robot_design.pinokla.closed_loop_kinematics import ForwardK, closedLoopProximalMount
import numpy.typing as npt


class PsedoStepResault(NamedTuple):
    J_closed: Optional[np.ndarray] = None
    M: Optional[np.ndarray] = None
    dq: Optional[np.ndarray] = None


def psedo_static_step(robot: Robot, q_state : np.ndarray, ee_frame_name: str) -> PsedoStepResault:

    ee_frame_id = robot.model.getFrameId(ee_frame_name)
    pin.framesForwardKinematics(robot.model, robot.data, q_state)
    pin.computeJointJacobians(robot.model, robot.data, q_state)
    pin.centerOfMass(robot.model, robot.data, q_state)

    vq, J_closed = inverseConstraintKinematicsSpeed(
        robot.model, robot.data, robot.constraint_models, robot.constraint_data, robot.actuation_model,
        q_state, ee_frame_id, robot.data.oMf[ee_frame_id].action@np.zeros(6))
    LJ = []
    for (cm, cd) in zip(robot.constraint_models, robot.constraint_data):
        Jc = pin.getConstraintJacobian(robot.model, robot.data, cm, cd)
        LJ.append(Jc)

    M = pin.crba(robot.model, robot.data, q_state)
    dq = dq_dqmot(robot.model, robot.actuation_model, LJ)

    return PsedoStepResault(J_closed, M, dq)



def iterate_over_q_space(robot: Robot,  q_space: np.ndarray, ee_frame_name: str):
    zero_resault = psedo_static_step(robot, q_space[0], ee_frame_name)
    



