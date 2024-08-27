import pinocchio as pin
import numpy as np
from auto_robot_design.motion_planning.ik_calculator import open_loop_ik, closed_loop_ik_pseudo_inverse

from functools import partial

IK_METHODS = {"Open_Loop":open_loop_ik,
              "Closed_Loop_PI":closed_loop_ik_pseudo_inverse}

class TrajectoryIKManager():
    def __init__(self) -> None:
        self.model = None
        self.constraint_model = None
        self.solver = None
        self.default_name = "Closed_Loop_PI"
        self.frame_name = "EE"

    def register_model(self, model, constraint_models):
        self.model = model
        self.constraint_models = constraint_models
    
    def set_solver(self, name, **params):
        try:
            self.solver = partial(IK_METHODS[name], **params)
        except KeyError:
            print(f'Cannot set solver - wrong name: {name}. Solver set to default value: {self.default_name}')
            self.solver = partial(IK_METHODS[self.default_name], {})
        except TypeError:
            print(f"Cannot set solver - wrong parameters for solver: {name}. Solver set to default value: {self.default_name}")
            self.solver = partial(IK_METHODS[self.default_name], {})

    def follow_trajectory(self, trajectory, q_start=None):
        if self.solver:
            ik_solver = self.solver
        else:
            raise Exception("set a solver before an attempt to follow a trajectory")
        frame_id = self.model.getFrameId(self.frame_name)
        # create a copy of a registered model
        model = pin.Model(self.model)
        data = model.createData()
        
        if q_start:
            q = q_start
        else:
            q = pin.neutral(self.model)
        # 3D coordinates of the following frame, TODO: consider a way to specify what kind of positioning we need
        poses = np.zeros((len(trajectory), 3))
        # reach mask
        reach_array = np.zeros(len(trajectory))
        # calculated positions in configuration space
        q_array = np.zeros((len(trajectory), len(q)))
        # final error for each point
        constraint_errors = np.zeros((len(trajectory), 1))
        for idx, point in enumerate(trajectory):
            q, min_feas, is_reach = ik_solver(
                model,
                self.constraint_models,
                point,
                frame_id,
                q_start=q,
            )
            if not is_reach:
                break

            pin.framesForwardKinematics(model, data, q)
            poses[idx] = data.oMf[frame_id].translation
            q_array[idx] = q
            constraint_errors[idx] = min_feas
            reach_array[idx] = is_reach

        return poses, q_array, constraint_errors, reach_array

