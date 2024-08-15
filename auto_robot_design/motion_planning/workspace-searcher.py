import time
from collections import UserDict
from enum import IntFlag, auto
from typing import NamedTuple, Optional

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

from collections import deque
from numpy.linalg import norm

from auto_robot_design.pinokla.closed_loop_jacobian import (
    jacobian_constraint,
    constraint_jacobian_active_to_passive,
)
from auto_robot_design.pinokla.closed_loop_kinematics import (
    ForwardK,
    closedLoopProximalMount,
    closed_loop_ik_grad,
    closed_loop_ik_pseudo_inverse,
)
from auto_robot_design.pinokla.criterion_math import (
    calc_manipulability,
    ImfProjections,
    calc_actuated_mass,
    calc_effective_inertia,
    calc_force_ell_projection_along_trj,
    calc_IMF,
    calculate_mass,
    convert_full_J_to_planar_xz,
)
from auto_robot_design.pinokla.loader_tools import Robot

from auto_robot_design.pinokla.calc_criterion import (
    folow_traj_by_proximal_inv_k,
    closed_loop_pseudo_inverse_follow,
)


class BreadthFirstSearchPlanner:

    class Node:
        def __init__(self, pos, cost, q_arr, parent_index, parent, is_reach):
            self.pos = pos
            self.cost = cost
            self.q_arr = q_arr
            self.parent_index = parent_index
            self.parent = parent
            self.is_reach = is_reach

        def __str__(self):
            return (
                str(self.pos)
                + ", "
                + str(self.q_arr)
                + ", "
                + str(self.cost)
                + ", "
                + str(self.parent_index)
            )

    def __init__(
        self,
        robot: Robot,
        bounds: np.ndarray,
        grid_resolution: float,
        singularity_threshold: float,
    ) -> None:

        self.robot = robot
        self.resolution = grid_resolution
        self.threshold = singularity_threshold

        num_indexes = np.max(bounds[:3], 1) - np.min(bounds[:3], 1) / self.resolution

        self.bounds = np.zeros_like(bounds)
        for id, idx_value in enumerate(num_indexes):
            residue_div = idx_value % 1
            if residue_div != 0.0:
                self.bounds[id, 0] = bounds[id, 0] - residue_div / 2
                self.bounds[id, 1] = bounds[id, 1] + residue_div / 2

                num_indexes[id] = np.ceil(num_indexes[id])
        self.num_indexes = np.asarray(num_indexes, dtype=int)

        self.motion = self.get_motion_model()

    def find_workspace(self, start_pos, viz=None):

        start_n = self.create_node(start_pos, -1, None)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_n)] = start_n

        while len(open_set) != 0:

            current = open_set.pop(list(open_set.keys())[0])

            c_id = self.calc_grid_index(current)
            
            closed_set[c_id] = current
            
            if current.is_reach:
                plt.plot(current.pos[0],current.pos[1], "xc")
                viz.display(current.q_arr)
                # time.sleep(0.5)
            else:
                plt.plot(current.pos[0],current.pos[1], "xr")
                viz.display(current.q_arr)
                # time.sleep(2)
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                            lambda event:
                                            [exit(0) if event.key == 'escape'
                                            else None])
            if len(closed_set.keys()) % 1 == 0:
                plt.pause(0.001)

            for i, moving in enumerate(self.motion):
                new_pos = current.pos + moving[:-1] * self.resolution

                if np.all(self.bounds[:, 0] <= new_pos) and np.all(new_pos <= self.bounds[:, 1]):
                    node = self.create_node(new_pos, c_id, None, current.q_arr)
                    n_id = self.calc_grid_index(node)
                else:
                    continue
                
                if (n_id not in closed_set) and (n_id not in open_set):
                    node.parent = current
                    open_set[n_id] = node

    def create_node(self, pos, parent_index, parent, prev_q = None):
        robot = self.robot

        robot_ms = robot.motion_space
        poses_6d, q_fixed, constraint_errors, reach_array = (
            closed_loop_pseudo_inverse_follow(
                robot.model,
                robot.data,
                robot.constraint_models,
                robot.constraint_data,
                robot.ee_name,
                [robot_ms.get_6d_point(pos)],
                q_start=prev_q
            )
        )

        dq_dqmot, __ = constraint_jacobian_active_to_passive(
            robot.model,
            robot.data,
            robot.constraint_models,
            robot.constraint_data,
            robot.actuation_model,
            q_fixed,
        )
        ee_id = robot.model.getFrameId(robot.ee_name)
        Jfclosed = (
            pin.computeFrameJacobian(
                robot.model, robot.data, q_fixed[0], ee_id, pin.LOCAL_WORLD_ALIGNED
            )
            @ dq_dqmot
        )

        __, S, __ = np.linalg.svd(Jfclosed)

        dext_index = np.abs(S).max() / np.abs(S).min()

        real_pos = robot_ms.rewind_6d_point(poses_6d[0])

        node = self.Node(pos, 1 / dext_index, q_fixed[0], parent_index, parent, reach_array[0])

        return node

    def calc_grid_index(self, node):
        idx = self.calc_index(node.pos)

        grid_index = 0
        for k, id in enumerate(idx):
            grid_index += id * np.prod(self.num_indexes[k:])

        return grid_index

    def calc_grid_position(self, indexes):

        pos_6d = indexes * self.resolution + self.bounds[:, 0]

        return pos_6d

    def calc_index(self, pos):
        return (np.round((pos - self.bounds[:, 0]) / self.resolution)).astype(int)

    def verify_node(self, node):
        pos = node.pos
        if np.any(pos < self.bounds[:, 0]) or np.any(pos > self.bounds[:, 1]):
            return False
        return True

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [
            [1, -1, np.sqrt(2)],
            [1, 0, 1],
            [1, 1, np.sqrt(2)],
            [-1, 1, np.sqrt(2)],
            [-1, 0, 1],
            [-1, -1, np.sqrt(2)],
            [0, -1, 1],
            [0, 1, 1],
        ]

        return motion


if __name__== "__main__":
    from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
    from auto_robot_design.description.builder import (
        ParametrizedBuilder,
        URDFLinkCreator,
        jps_graph2pinocchio_robot
        )
    import meshcat
    from pinocchio.visualize import MeshcatVisualizer
    from auto_robot_design.pinokla.closed_loop_kinematics import (
        closedLoopInverseKinematicsProximal,
        openLoopInverseKinematicsProximal,
        closedLoopProximalMount,
    )

    builder = ParametrizedBuilder(URDFLinkCreator)
    
    gm = get_preset_by_index_with_bounds(0)
    x_centre = gm.generate_central_from_mutation_range()
    graph_jp = gm.get_graph(x_centre)
    

    robo, __ = jps_graph2pinocchio_robot(graph_jp, builder=builder)
    
    start_pos = np.array([-0.12, -0.33])
    pos_6d = np.zeros(6)
    pos_6d[[0,2]] = start_pos
    
    id_ee = robo.model.getFrameId(robo.ee_name)
    
    poses_6d, q_fixed, constraint_errors, reach_array = closed_loop_pseudo_inverse_follow(
        robo.model,
        robo.data,
        robo.constraint_models,
        robo.constraint_data,
        robo.ee_name,
        [pos_6d],
    )
    
    q = q_fixed[0]
    is_reach = reach_array[0]
    pin.forwardKinematics(robo.model, robo.data, q)
    ballID = "world/ball" + "_start"
    material = meshcat.geometry.MeshPhongMaterial()
    if not is_reach:
        q = closedLoopProximalMount(
            robo.model, robo.data, robo.constraint_models, robo.constraint_data, q
        )
        material.color = int(0xFF0000)
    else:
        material.color = int(0x00FF00)
    

    viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
    viz.viewer = meshcat.Visualizer().open()
    viz.clean()
    viz.loadViewerModel()
    
    material.opacity = 1
    viz.viewer[ballID].set_object(meshcat.geometry.Sphere(0.01), material)
    T = np.r_[np.c_[np.eye(3), poses_6d[0][:3]], np.array([[0, 0, 0, 1]])]
    viz.viewer[ballID].set_transform(T)
    pin.framesForwardKinematics(robo.model, robo.data, q)
    viz.display(q)

    bounds = np.array([[-0.09/2, 0.09/2], [-0.24/2, 0.24/2]])
    bounds[0,: ] += -0.12
    bounds[1,: ] += -0.33
    
    
    ws_bfs = BreadthFirstSearchPlanner(robo, bounds, np.array([0.001, 0.001]), 0.5)
    ws_bfs.find_workspace(start_pos, viz)