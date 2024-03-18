from copy import deepcopy
from enum import Enum, IntFlag, auto

from typing import NamedTuple, Optional
from auto_robot_design.pinokla.criterion_math import ImfProjections, calc_IMF, calc_force_ell_projection_along_trj, calc_force_ellips_space, calculate_mass, calc_manipulability, convert_full_J_to_planar_xz
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
from collections import UserDict


class MovmentSurface(IntFlag):
    XZ = auto()
    ZY = auto()
    YX = auto()


class PsedoStepResault(NamedTuple):
    J_closed: np.ndarray = None
    M: np.ndarray = None
    dq: np.ndarray = None


class DataDict(UserDict):

    def get_frame(self, index):
        extracted_elements = {}
        for key, array in self.items():
            extracted_elements[key] = array[index]
        return extracted_elements

    def get_data_len(self):
        return len(self[next(iter(self))])


def search_workspace(model,
                     data,
                     effector_frame_name: str,
                     base_frame_name: str,
                     q_space: np.ndarray,
                     actuation_model,
                     constraint_models,
                     viz=None):
    c = 0
    q_start = pin.neutral(model)
    workspace_xyz = np.empty((len(q_space), 3))
    available_q = np.empty((len(q_space), len(q_start)))
    for q_sample in q_space:

        q_dict_mot = zip(actuation_model.idqmot, q_sample)
        for key, value in q_dict_mot:
            q_start[key] = value
        q3, error = ForwardK(
            model,
            constraint_models,
            actuation_model,
            q_start,
            150,
        )

        if error < 1e-11:
            if viz:
                viz.display(q3)
                time.sleep(0.005)
            q_start = q3
            pin.framesForwardKinematics(model, data, q3)
            id_effector = model.getFrameId(effector_frame_name)
            id_base = model.getFrameId(base_frame_name)
            effector_pos = data.oMf[id_effector].translation
            base_pos = data.oMf[id_base].translation
            transformed_pos = effector_pos - base_pos

            workspace_xyz[c] = transformed_pos
            available_q[c] = q3
            c += 1
    return (workspace_xyz[0:c], available_q[0:c])


def folow_traj_by_proximal_inv_k(model,
                                 data,
                                 constraint_models,
                                 constraint_data,
                                 end_effector_frame: str,
                                 traj_6d: np.ndarray,
                                 viz=None,
                                 q_start: np.ndarray = None):
    if q_start:
        q = q_start
    else:
        q = pin.neutral(model)

    ee_frame_id = model.getFrameId(end_effector_frame)
    poses = np.zeros((len(traj_6d), 3))
    q_array = np.zeros((len(traj_6d), len(q)))
    constraint_errors = np.zeros((len(traj_6d), 1))

    for num, i_pos in enumerate(traj_6d):
        q, min_feas, is_reach = closedLoopInverseKinematicsProximal(
            model,
            data,
            constraint_models,
            constraint_data,
            i_pos,
            ee_frame_id,
            onlytranslation=True,
            q_start=q)
        if not is_reach:
            q = closedLoopProximalMount(model, data, constraint_models,
                                        constraint_data, q)
        if viz:
            viz.display(q)
            time.sleep(0.1)

        pin.framesForwardKinematics(model, data, q)
        poses[num] = data.oMf[ee_frame_id].translation
        q_array[num] = q
        constraint_errors[num] = min_feas

    return poses, q_array, constraint_errors


def psedo_static_step(robot: Robot, q_state: np.ndarray,
                      ee_frame_name: str) -> PsedoStepResault:

    ee_frame_id = robot.model.getFrameId(ee_frame_name)
    pin.framesForwardKinematics(robot.model, robot.data, q_state)
    pin.computeJointJacobians(robot.model, robot.data, q_state)
    pin.centerOfMass(robot.model, robot.data, q_state)

    vq, J_closed = inverseConstraintKinematicsSpeed(
        robot.model, robot.data, robot.constraint_models,
        robot.constraint_data, robot.actuation_model, q_state, ee_frame_id,
        robot.data.oMf[ee_frame_id].action @ np.zeros(6))
    LJ = []
    for (cm, cd) in zip(robot.constraint_models, robot.constraint_data):
        Jc = pin.getConstraintJacobian(robot.model, robot.data, cm, cd)
        LJ.append(Jc)

    M = pin.crba(robot.model, robot.data, q_state)
    dq = dq_dqmot(robot.model, robot.actuation_model, LJ)

    return PsedoStepResault(J_closed, M, dq)


def iterate_over_q_space(robot: Robot, q_space: np.ndarray,
                         ee_frame_name: str):
    zero_step = psedo_static_step(robot, q_space[0], ee_frame_name)

    res_dict = DataDict()
    for key, value in zero_step._asdict().items():
        alocate_array = np.zeros((len(q_space), *value.shape),
                                 dtype=np.float64)
        res_dict[key] = alocate_array

    for num, q_state in enumerate(q_space):
        one_step_res = psedo_static_step(robot, q_state, ee_frame_name)
        for key, value in one_step_res._asdict().items():
            res_dict[key][num] = value

    return res_dict


class ComputeInterfaceMoment:
    def __call__(self, data_frame: dict[str, np.ndarray], robo: Robot = None) -> np.ndarray:
        raise NotImplemented

    def output_matrix_shape(self) -> Optional[tuple]:
        return None


class ComputeInterface:

    def __call__(self, data_dict: DataDict, robo: Robot = None):
        raise NotImplemented


class ImfCompute(ComputeInterfaceMoment):

    def __init__(self, projection: ImfProjections) -> None:
        self.projection = projection

    def __call__(self, data_frame: dict[str, np.ndarray], robo: Robot = None) -> np.ndarray:
        imf = calc_IMF(data_frame["M"], data_frame["dq"],
                       data_frame["J_closed"], self.projection)
        return imf


class ManipCompute(ComputeInterfaceMoment):

    def __init__(self, surface: MovmentSurface) -> None:
        self.surface = surface

    def __call__(self, data_frame: dict[str, np.ndarray], robo: Robot = None) -> np.ndarray:
        if self.surface == MovmentSurface.XZ:
            target_J = data_frame["J_closed"]
            target_J = convert_full_J_to_planar_xz(target_J)
            target_J = target_J[:2, :2]
        else:
            raise NotImplemented
        manip_space = calc_manipulability(target_J)
        return manip_space


class NeutralPoseMass(ComputeInterface):

    def __init__(self) -> None:
        pass

    def __call__(self, data_dict: DataDict, robo: Robot = None):
        return calculate_mass(robo)


class ForceEllProjections(ComputeInterface):

    def __init__(self) -> None:
        pass

    def __call__(self, data_dict: DataDict, robo: Robot = None):
        ell_params = calc_force_ell_projection_along_trj(
            data_dict["J_closed"], data_dict["traj_6d"])
        return ell_params


class TranslationErrorMSE(ComputeInterface):

    def __init__(self) -> None:
        pass

    def __call__(self, data_dict: DataDict, robo: Robot = None):

        errors = np.sum(
            norm(data_dict["traj_6d"][:,:3] - data_dict["traj_6d_ee"][:,:3], axis=1))
        mse = np.mean(errors)
        return mse


def moment_criteria_calc(calculate_desription: dict[str,
                                                    ComputeInterfaceMoment],
                         data_dict: DataDict,
                         robo: Robot = None) -> DataDict:
    res_dict = DataDict()
    for key, criteria in calculate_desription.items():
        shape = criteria.output_matrix_shape()
        if shape:
            res_dict[key] = np.zeros((data_dict.get_data_len(), *shape),
                                     dtype=np.float32)
        else:
            frame_data = data_dict.get_frame(0)
            zero_step = criteria(frame_data)
            res_dict[key] = np.zeros((data_dict.get_data_len(), *zero_step.shape),
                                     dtype=np.float32)
            # Need implement alocate from zero step data size
            # raise NotImplemented
    for index in range(data_dict.get_data_len()):
        data_frame = data_dict.get_frame(index)
        for key, criteria in calculate_desription.items():
            res_dict[key][index] = criteria(data_frame, robo)
    return res_dict


def along_criteria_calc(calculate_desription: dict[str, ComputeInterface],
                        data_dict: DataDict,
                        robo: Robot = None) -> dict:
    res_dict = {}
    for index in range(data_dict.get_data_len()):
 
        for key, criteria in calculate_desription.items():
            res_dict[key] = criteria(data_dict, robo)
    return res_dict


