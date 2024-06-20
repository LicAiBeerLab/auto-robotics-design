from logging import warning
import os
import numpy as np
import pinocchio as pin

import multiprocess
from pymoo.core.problem import StarmapParallelization, ElementwiseProblem

from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from scipy.optimize import shgo

import meshcat
from pinocchio.visualize import MeshcatVisualizer

from auto_robot_design.control.model_based import OperationSpacePDControl
from auto_robot_design.control.trajectory_planning import trajectory_planning
from auto_robot_design.pinokla.closed_loop_kinematics import (
    closedLoopInverseKinematicsProximal,
    closedLoopProximalMount,
)
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz


class SimNControlData:

    def __init__(
        self, sim_time, time_step, q, vq, acc, tau, pos_ee_frame, power, coeffs
    ) -> None:
        self.sim_time: float = sim_time
        self.time_step: float = time_step
        self.time_arr: np.ndarray = np.arange(0, sim_time + time_step, time_step)
        self.q: np.ndarray = q
        self.vq: np.ndarray = vq
        self.acc: np.ndarray = acc
        self.tau: np.ndarray = tau
        self.pos_ee_frame: np.ndarray = pos_ee_frame
        self.power: np.ndarray = power
        self.coeffs: tuple[np.ndarray, np.ndarray] = coeffs

        self.id_act_q: list[int] = []
        self.id_act_vq: list[int] = []

        self.desired_traj: np.ndarray = None
        self.desired_d_traj: np.ndarray = None


# class ControlOptProblem(ElementwiseProblem):
#     def __init__ (self, simulation, robot, *args, **kwargs):
#         self.robot = robot
#         self.simulation = simulation
#         super().__init__(n_var=4, n_obj=1, n_constr=0, elementwise_evaluation=True, *args, **kwargs)

#     def _evaluate(self, x, out, *args, **kwargs):
#         old_Kp = self.simulation.Kp
#         old_Kd = self.simulation.Kd

#         self.simulation.Kp = np.zeros((6,6))
#         self.simulation.Kd = np.zeros((6,6))

#         self.simulation.Kp[0,0] = x[0]
#         self.simulation.Kp[2,2] = x[1]
#         self.simulation.Kd[0,0] = x[2]
#         self.simulation.Kd[2,2] = x[3]

#         __, __, __, tau_act, pos_ee_frame, __ = self.simulation.simulate(self.robot,False)
#         __, des_traj_6d, __ = self.simulation.prepare_trajectory(self.robot)


#         pos_error = np.sum(np.linalg.norm(pos_ee_frame - des_traj_6d[:,:3], axis=1)**2)

#         norm_tau = np.sum(np.linalg.norm(tau_act, axis=1)**2)/6e4

#         self.Kp = old_Kp
#         self.Kd = old_Kd

#         rew = pos_error + norm_tau
#         if rew > 1e8:
#             rew = 1e8

#         out["F"] = rew


class ControlOptProblem:
    def __init__(self, simulation, robot, *args, **kwargs):
        self.robot = robot
        self.simulation = simulation

    def evaluate(self, x):
        old_Kp = self.simulation.Kp
        old_Kd = self.simulation.Kd

        self.simulation.Kp = np.zeros((6, 6))
        self.simulation.Kd = np.zeros((6, 6))

        self.simulation.Kp[0, 0] = x[0]
        self.simulation.Kp[2, 2] = x[1]
        self.simulation.Kd[0, 0] = x[2]
        self.simulation.Kd[2, 2] = x[3]

        simout: SimNControlData = self.simulation.simulate(self.robot, False)

        pos_error = np.sum(
            np.linalg.norm(simout.pos_ee_frame - simout.desired_traj[:, :3], axis=1)
            ** 2
        )

        norm_tau = (
            np.sum(np.linalg.norm(simout.tau[simout.id_act_vq], axis=1) ** 2) / 6e4
        )

        self.Kp = old_Kp
        self.Kd = old_Kd

        rew = pos_error + norm_tau
        if rew > 1e8:
            rew = 1e8

        return rew


class TrajectoryMovements:
    def __init__(self, trajectory, final_time, time_step, name_ee_frame) -> None:
        """
        Initialization of the class for modeling the mechanism by the trajectory of end effector movement.
        For trajectory movement class use PD control in operational space with coefficients Kp = (7000, 4200) and Kd = (74, 90).

        Method `simulate` simulates the movement of the mechanism along the trajectory and returns the
        position, velocity, acceleration in configuration space,
        torque, position of the end effector frame and power.

        For tune the control coefficients use `optimize_control` method. `optimize_control` method uses the scipy.optimize.shgo
        method for optimization of the control coefficients. The optimization function is the sum of the squared error of the position of the end effector frame and the sum of the squared torque.

        Args:
            trajectory (numpy.ndarray): The desired trajectory via points in x-z plane.
            final_time (numpy.ndarray): The time of the final point in trajectory.
            time_step (float): The time step for simulation.
            name_ee_frame (str): The name of the end-effector frame.

        """

        Kp = np.zeros((6, 6))
        Kp[0, 0] = 7000
        Kp[2, 2] = 4200

        Kd = np.zeros((6, 6))
        Kd[0, 0] = 74
        Kd[2, 2] = 90

        self.Kp = Kp
        self.Kd = Kd

        self.name_ee_frame = name_ee_frame
        self.traj = trajectory
        self.time = final_time
        self.time_step = time_step

        self.num_sim_steps = int(self.time / self.time_step)

    def setup_dynamic(self):
        """Initializes the dynamics calculator, also set time_step."""
        accuracy = 1e-8
        mu_sim = 1e-8
        max_it = 10000
        self.dynamic_settings = pin.ProximalSettings(accuracy, mu_sim, max_it)

    def prepare_trajectory(self, robot):
        """
        Prepare the trajectory for simulation.

        Args:
            robot: The robot object.

        Returns:
            time_arr: Array of time values.
            des_traj_6d: Desired 6D trajectory.
            des_d_traj_6d: Desired 6D trajectory derivative.
        """

        x_max = self.traj[-1, 0] - self.traj[0, 0]

        time_arr = np.linspace(0, self.time, self.num_sim_steps)
        t_s = self.time / 2

        ax_1 = 4 * x_max / self.time**2

        # ax_2 = 2 * self.traj[-1, 0] / (self.time **2/2 - self.time)

        des_dtraj_2d = np.zeros((self.num_sim_steps, 2))

        for idx, t in enumerate(time_arr):
            if t <= t_s:
                des_dtraj_2d[idx, 0] = ax_1 * t
            else:
                des_dtraj_2d[idx, 0] = -ax_1 * t + ax_1 * self.time

        des_traj_by_t = np.zeros((self.num_sim_steps, 2))

        des_traj_by_t[time_arr <= t_s, 0] = 0.5 * ax_1 * time_arr[time_arr <= t_s] ** 2
        des_traj_by_t[time_arr > t_s, 0] = (
            -ax_1 * time_arr[time_arr > t_s] ** 2 / 2
            + ax_1 * self.time * time_arr[time_arr > t_s]
            + x_max
            - ax_1 * self.time**2 / 2
        )

        des_traj_by_t[:, 0] = des_traj_by_t[:, 0] - x_max / 2

        if self.traj[:, 0].max() - self.traj[:, 0].min() < 1e-3:
            des_traj_by_t[:, 1] = np.linspace(
                self.traj[0, 1], self.traj[-1, 1], self.num_sim_steps
            )
        else:
            cs_z_by_x = np.polyfit(self.traj[:, 0], self.traj[:, 1], 3)
            des_traj_by_t[:, 1] = np.polyval(cs_z_by_x, des_traj_by_t[:, 0])

        des_traj_6d = convert_x_y_to_6d_traj_xz(
            des_traj_by_t[:, 0], des_traj_by_t[:, 1]
        )

        d_y =  np.diff(des_traj_by_t[:, 1]) / self.time_step
        d_y = np.hstack([d_y, 0])
        des_dtraj_2d[:,1] = d_y
        
        des_d_traj_6d = convert_x_y_to_6d_traj_xz(
            des_dtraj_2d[:, 0], des_dtraj_2d[:, 1]
        )

        # des_trajectories[:, 0] = np.linspace(
        #     self.traj[0, 0], self.traj[-1, 0], self.num_sim_steps
        # )
        # if self.traj[:, 0].max() - self.traj[:, 0].min() < 1e-3:
        #     des_trajectories[:, 1] = np.linspace(
        #         self.traj[0, 1], self.traj[-1, 1], self.num_sim_steps
        #     )
        # else:
        #     cs_z_by_x = np.polyfit(self.traj[:, 0], self.traj[:, 1], 3)
        #     des_trajectories[:, 1] = np.polyval(cs_z_by_x, des_trajectories[:, 0])

        # des_traj_6d = convert_x_y_to_6d_traj_xz(
        #     des_trajectories[:, 0], des_trajectories[:, 1]
        # )

        # t_s = self.time / 2
        # Vs = (
        #     np.array(
        #         [
        #             des_trajectories[:, 0][-1] - des_trajectories[:, 0][0],
        #             (des_trajectories[:, 1].max() - des_trajectories[:, 1].min()) * 2,
        #         ]
        #     )
        #     / t_s
        # )

        # des_dtraj_2d = np.zeros((self.num_sim_steps, 2))

        # for idx, t in enumerate(time_arr):
        #     if t <= t_s:
        #         des_dtraj_2d[idx, :] = Vs * t
        #     else:
        #         des_dtraj_2d[idx, :] = -Vs * (t - t_s) + Vs * t_s

        # des_d_traj_6d = convert_x_y_to_6d_traj_xz(
        #     des_dtraj_2d[:, 0], des_dtraj_2d[:, 1]
        # )

        # des_d_traj_6d = np.diff(des_traj_6d, axis=0) / self.time_step
        # des_d_traj_6d = np.vstack((des_d_traj_6d, des_d_traj_6d[-1]))

        # q = np.zeros(robot.model.nq)
        # Trajectory by points in joint space

        # traj_points = convert_x_y_to_6d_traj_xz(self.traj[:,0], self.traj[:,1])
        # q_des_points = np.zeros((len(traj_points), robot.model.nq))

        # frame_id = robot.model.getFrameId(self.name_ee_frame)
        # for num, i_pos in enumerate(traj_points):
        #     q, min_feas, is_reach = closedLoopInverseKinematicsProximal(
        #         robot.model,
        #         robot.data,
        #         robot.constraint_models,
        #         robot.constraint_data,
        #         i_pos,
        #         frame_id,
        #         onlytranslation=True,
        #         q_start=q,
        #     )
        #     if not is_reach:
        #         q = closedLoopProximalMount(
        #             robot.model, robot.data, robot.constraint_models, robot.constraint_data, q
        #         )
        #     q_des_points[num] = q.copy()

        # q = q_des_points[0]

        # __, q_des_traj, dq_des_traj, ddq_des_traj = trajectory_planning(
        # q_des_points.T, 0, 0, 0, self.times[-1], self.time_step, False
        # )

        # self.des_trajectories = {
        #     "time": time_arr,
        #     # "q_ref": q_des_traj,
        #     # "dq_ref": dq_des_traj,
        #     # "ddq_ref": ddq_des_traj,
        #     "traj_6d_ref": des_traj_6d,
        #     "d_traj_6d_ref": des_d_traj_6d,
        # }
        return time_arr, des_traj_6d, des_d_traj_6d

    def simulate(self, robot, is_vis=False, control_obj_value=None, path_to_save_frame=None) -> SimNControlData:
        """
        Simulates the trajectory movements of a robot.

        Args:
            robot (RobotModel): The robot model.
            is_vis (bool, optional): Whether to visualize the simulation. Defaults to False.
            control_obj_value(float, optional): The value of the control objective function. Defaults to None. If None, the function check optimized result and compare with the current result.

        Returns:
            tuple: A tuple containing the following simulation data:
                - q (numpy.ndarray): The joint positions at each simulation step.
                - vq (numpy.ndarray): The joint velocities at each simulation step.
                - acc (numpy.ndarray): The joint accelerations at each simulation step.
                - tau_act (numpy.ndarray): The actuator torques at each simulation step.
                - pos_ee_frame (numpy.ndarray): The end-effector frame positions at each simulation step.
                - power (numpy.ndarray): The mechanical power actuators exerted at each simulation step.
        """

        self.setup_dynamic()
        frame_id = robot.model.getFrameId(self.name_ee_frame)
        q = np.zeros(robot.model.nq)

        __, des_traj_6d, des_d_traj_6d = self.prepare_trajectory(robot)
        q, __, is_reach = closedLoopInverseKinematicsProximal(
            robot.model,
            robot.data,
            robot.constraint_models,
            robot.constraint_data,
            des_traj_6d[0],
            frame_id,
            onlytranslation=True,
            q_start=q,
        )
        if not is_reach:
            q = closedLoopProximalMount(
                robot.model,
                robot.data,
                robot.constraint_models,
                robot.constraint_data,
                q,
            )

        control = OperationSpacePDControl(robot, self.Kp, self.Kd, frame_id)

        pin.initConstraintDynamics(robot.model, robot.data, robot.constraint_models)

        vq = np.zeros(robot.model.nv)
        tau_q = np.zeros(robot.model.nv)

        q_act = np.zeros((self.num_sim_steps, robot.model.nq))
        vq_act = np.zeros((self.num_sim_steps, robot.model.nv))
        acc_act = np.zeros((self.num_sim_steps, robot.model.nv))
        tau_act = np.zeros((self.num_sim_steps, 2))

        power = np.zeros((self.num_sim_steps, len(robot.actuation_model.idvmot)))

        pos_ee_frame = np.zeros((self.num_sim_steps, 3))

        if is_vis:
            viz = MeshcatVisualizer(robot.model, robot.visual_model, robot.visual_model)
            # viz.viewer = meshcat.Visualizer().open()
            viz.initViewer(open=True)
            # viz.initViewer()
            viz.viewer["/Background"].set_property("visible", False)
            viz.viewer["/Grid"].set_property("visible", False)
            viz.viewer["/Axes"].set_property("visible", False)
            viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", [0,0,0.5])
            viz.clean()
            viz.loadViewerModel()
            viz.display(q)

        idx_frave_image_save = np.linspace(0, self.num_sim_steps, 10, dtype=int)
        frame_image = []
        for i in range(self.num_sim_steps):
            a = pin.constraintDynamics(
                robot.model,
                robot.data,
                q,
                vq,
                tau_q,
                robot.constraint_models,
                robot.constraint_data,
                self.dynamic_settings,
            )

            vq += a * self.time_step

            q = pin.integrate(robot.model, q, vq * self.time_step)
            try:
                tau_q = control.compute(q, vq, des_traj_6d[i], des_d_traj_6d[i])
            except np.linalg.LinAlgError:
                return q_act, vq_act, acc_act, tau_act, pos_ee_frame, power
            # First coordinate is root_joint
            tau_a = tau_q[control.ids_vmot]

            vq_a = vq[control.ids_vmot]
            q_act[i] = q
            vq_act[i] = vq
            acc_act[i] = a
            tau_act[i] = tau_a
            pos_ee_frame[i] = robot.data.oMf[frame_id].translation
            power[i] = tau_a * vq_a

            if is_vis:
                viz.display(q)
                if i in idx_frave_image_save:
                    frame_image.append(viz.viewer["/Cameras/default/rotated/"].get_image())
                    
        if is_vis:
            # viz.viewer.close()
            viz.viewer.delete()
        simout = SimNControlData(
            self.time,
            self.time_step,
            q_act,
            vq_act,
            acc_act,
            tau_act,
            pos_ee_frame,
            power,
            (self.Kp, self.Kd),
        )
        simout.id_act_q = control.ids_mot
        simout.id_act_vq = control.ids_vmot
        simout.desired_traj = des_traj_6d
        simout.desired_d_traj = des_d_traj_6d

        if control_obj_value is not None:
            pos_error = np.sum(
                np.linalg.norm(simout.pos_ee_frame - simout.desired_traj[:, :3], axis=1)
                ** 2
            )
            norm_tau = (
                np.sum(np.linalg.norm(simout.tau[simout.id_act_vq], axis=1) ** 2) / 1e5
            )
            rew = pos_error + norm_tau

            if not np.isclose(rew, control_obj_value, atol=1e-3):
                warning(
                    f"Control objective value is not equal to the current simulation value. {rew:.3f} != {control_obj_value:.3f}"
                )

        if path_to_save_frame:
            for f_id, img in enumerate(frame_image):
                img.save(os.path.join(path_to_save_frame, f"{f_id}.png"))
        return simout

    def optimize_control(self, robot):
        """
        Optimize the control coefficients for the robot. The optimization function is the sum of the squared error of the position of the end effector frame and the sum of the squared torque.
        The `scipy.optimize.shgo` method is used for optimization.

        Args:
            robot: The robot object.

        Returns:
            Kp: The optimized proportional gain matrix.
            Kd: The optimized derivative gain matrix.
        """

        bounds = [[0, 5e3] for __ in range(2)]
        bounds = np.vstack((bounds, [[0, 1e3] for __ in range(2)]))

        ## N_PROCESS = 8
        ## pool = multiprocess.Pool(N_PROCESS)
        ## runner = StarmapParallelization(pool.starmap)

        ## problem = ControlOptProblem(self, robot, xl=bounds[:,0], xu=bounds[:,1])#, elementwise_runner=runner)
        ## algorithm = PSO(25, max_velocity_rate=500)

        problem = ControlOptProblem(self, robot)
        results = shgo(
            problem.evaluate,
            bounds,
            n=10,
            options={"disp": True, "f_min": 1, "f_tol": 0.5},
        )

        ## results = minimize(problem, algorithm, termination=("n_gen",5), seed=1, verbose=True)

        Kp = np.zeros((6, 6))
        Kd = np.zeros((6, 6))

        # Kp[0, 0] = np.random.uniform(300, 1000)
        # Kp[2, 2] = np.random.uniform(300, 1000)
        # Kd[0, 0] = np.random.uniform(10, 200)
        # Kd[2, 2] = np.random.uniform(10,200)

        Kp[0, 0] = results.x[0]
        Kp[2, 2] = results.x[1]
        Kd[0, 0] = results.x[2]
        Kd[2, 2] = results.x[3]

        ## Kp[0,0] = results.X[0]
        ## Kp[2,2] = results.X[1]
        ## Kd[0,0] = results.X[2]
        ## Kd[2,2] = results.X[3]
        return Kp, Kd, results.fun  # np.random.randint(1, 100) #


if __name__ == "__main__":
    from auto_robot_design.generator.restricted_generator.two_link_generator import (
        TwoLinkGenerator,
    )
    from auto_robot_design.description.builder import (
        ParametrizedBuilder,
        URDFLinkCreator,
        jps_graph2pinocchio_robot,
    )

    gen = TwoLinkGenerator()
    builder = ParametrizedBuilder(URDFLinkCreator)
    graphs_and_cons = gen.get_standard_set()
    np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

    graph_jp, constrain = graphs_and_cons[2]

    robo, __ = jps_graph2pinocchio_robot(graph_jp, builder)

    name_ee = "EE"

    x_point = np.array([-0.5, 0, 0.25]) * 0.5
    y_point = np.array([-0.4, -0.1, -0.4]) * 0.5
    y_point = y_point - 0.7

    trajectory = np.array([x_point, y_point]).T

    test = TrajectoryMovements(trajectory, 1, 0.01, name_ee)

    # test.prepare_trajectory(robo)
    # Kp, Kd = test.optimize_control(robo)

    # test.Kp = Kp
    # test.Kd = Kd

    test.simulate(robo, True)
