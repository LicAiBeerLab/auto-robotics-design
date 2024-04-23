import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

from auto_robot_design.pinokla.closed_loop_jacobian import (
    constraint_jacobian_active_to_passive,
    inverseConstraintKinematicsSpeed,
    jacobian_constraint,
)


class TorqueComputedControl:
    def __init__(
        self, robot, Kp: np.ndarray, Kd: np.ndarray, use_dJdt_term: bool = False
    ):
        self.robot = robot
        self.use_dJdt = use_dJdt_term

        nmot = len(robot.actuation_model.idqmot)
        assert Kp.shape == (nmot, nmot)
        assert Kd.shape == (nmot, nmot)

        self.Kp = Kp
        self.Kd = Kd

        self.ids_mot = robot.actuation_model.idqmot
        self.ids_vmot = robot.actuation_model.idvmot

        self.ids_free = self.robot.actuation_model.idqfree
        self.ids_vfree = self.robot.actuation_model.idvfree

        self.tauq = np.zeros(robot.model.nv)

        self.prev_Jmot = np.zeros((len(self.ids_vfree), len(self.ids_vmot)))
        self.prev_Jfree = np.zeros((len(self.ids_vfree), len(self.ids_vfree)))

    def compute(
        self,
        q: np.ndarray,
        vq: np.ndarray,
        q_a_ref: np.ndarray,
        vq_a_ref: np.ndarray,
        ddq_a_ref: np.ndarray,
    ):

        if self.use_dJdt:
            Jmot, Jfree = jacobian_constraint(
                self.robot.model,
                self.robot.data,
                self.robot.constraint_models,
                self.robot.constraint_data,
                self.robot.actuation_model,
                q,
            )
            epsilon = 1e-6
            Jmot_ = np.zeros((self.robot.nq, Jmot.shape[0], Jmot.shape[1]))
            Jfree_ = np.zeros((self.robot.nq, Jfree.shape[0], Jfree.shape[1]))
            for i in range(len(self.robot.nq)):
                q_ = q.copy()
                q_[i] += epsilon
                Jmot_i, Jfree_i = jacobian_constraint(
                    self.robot.model,
                    self.robot.data,
                    self.robot.constraint_models,
                    self.robot.constraint_data,
                    self.robot.actuation_model,
                    q_,
                )
                Jmot_[i, :, :] = Jmot_i
                Jfree_[i, :, :] = Jfree_i
            d_Jmot = np.dot(Jmot_, vq)
            d_Jfree = np.dot(Jfree_, vq)
            # d_Jmot = ((Jmot - self.prev_Jmot) / DT).round(6)
            # d_Jfree = ((Jfree - self.prev_Jfree) / DT).round(6)
            a_d = -np.linalg.pinv(Jfree) @ (
                d_Jmot @ vq[self.ids_vmot] + d_Jfree @ vq[self.ids_vfree]
            )
        else:
            a_d = np.zeros(len(self.ids_vfree))

        q_a = q[self.ids_mot]
        vq_a = vq[self.ids_vmot]

        M = pin.crba(self.robot.model, self.robot.data, q)
        g = pin.computeGeneralizedGravity(self.robot.model, self.robot.data, q)
        C = pin.computeCoriolisMatrix(self.robot.model, self.robot.data, q, vq)

        Jda, E_tau = constraint_jacobian_active_to_passive(
            self.robot.model,
            self.robot.data,
            self.robot.constraint_models,
            self.robot.constraint_data,
            self.robot.actuation_model,
            q,
        )

        Ma = Jda.T @ E_tau.T @ M @ E_tau @ Jda
        ga = Jda.T @ E_tau.T @ g
        Ca = Jda.T @ E_tau.T @ C @ E_tau @ Jda

        tau_a = (
            Ma @ (ddq_a_ref + self.Kp @ (q_a_ref - q_a) + self.Kd @ (vq_a_ref - vq_a))
            + Ca @ vq_a
            + ga
            + Jda.T
            @ E_tau.T
            @ M
            @ E_tau
            @ np.concatenate((np.zeros(len(self.ids_vmot)), a_d))
        )

        self.tauq[self.ids_mot] = tau_a

        return self.tauq


class OperationSpacePDControl:
    def __init__(
        self, robot, Kp: np.ndarray, Kd: np.ndarray, id_frame_end_effector: int
    ):
        self.robot = robot
        self.id_frame = id_frame_end_effector

        assert Kp.shape == (6, 6)
        assert Kd.shape == (6, 6)

        self.Kp = Kp
        self.Kd = Kd

        self.ids_mot = robot.actuation_model.idqmot
        self.ids_vmot = robot.actuation_model.idvmot

        self.ids_free = self.robot.actuation_model.idqfree
        self.ids_vfree = self.robot.actuation_model.idvfree

        self.tauq = np.zeros(robot.model.nv)

    def compute(self, q: np.ndarray, vq, x_ref: np.ndarray, dx_ref: np.ndarray):

        vq_cstr, J_closed = inverseConstraintKinematicsSpeed(
            self.robot.model,
            self.robot.data,
            self.robot.constraint_models,
            self.robot.constraint_data,
            self.robot.actuation_model,
            q,
            self.id_frame,
            self.robot.data.oMf[self.id_frame].action @ dx_ref,
        )
        x_body_curr = np.concatenate(
            (
                self.robot.data.oMf[self.id_frame].translation,
                R.from_matrix(self.robot.data.oMf[self.id_frame].rotation).as_rotvec(),
            )
        )

        g = pin.computeGeneralizedGravity(self.robot.model, self.robot.data, q)

        vq_a = vq[self.ids_vmot]
        # vq_a_ref = vq_cstr[self.ids_vmot]
        Jda, E_tau = constraint_jacobian_active_to_passive(
            self.robot.model,
            self.robot.data,
            self.robot.constraint_models,
            self.robot.constraint_data,
            self.robot.actuation_model,
            q,
        )
        tau_a = (
            J_closed.T
            @ (self.Kp @ (x_ref - x_body_curr) + self.Kd @ (dx_ref - J_closed @ vq_a))
            + Jda.T @ E_tau.T @ g
        )

        self.tauq[self.ids_mot] = tau_a

        return self.tauq