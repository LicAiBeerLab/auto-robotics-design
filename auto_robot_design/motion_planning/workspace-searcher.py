from itertools import product
from collections import deque
from math import isclose
from tabnanny import check

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

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
from auto_robot_design.pinokla.default_traj import add_auxilary_points_to_trajectory
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

        num_indexes = (np.max(bounds, 1) - np.min(bounds, 1)) / self.resolution
        self.num_indexes = np.zeros_like(num_indexes)
        self.bounds = np.zeros_like(bounds)
        for id, idx_value in enumerate(num_indexes):
            residue_div = np.round(idx_value % 1, 6)
            
            check_bound_size = np.isclose(residue_div, 0.0)
            check_min_bound = np.isclose(bounds[id,0] % self.resolution[id], 0)
            check_max_bound = np.isclose(bounds[id,1] % self.resolution[id], 0)
            if check_bound_size and check_min_bound and check_max_bound:
                self.bounds[id,:] = bounds[id,:]
                self.num_indexes[id] = num_indexes[id]
            else:
                self.bounds[id,1] = bounds[id,1] + bounds[id,1] % self.resolution[id]
                self.bounds[id,0] = bounds[id,0] - bounds[id,0] % self.resolution[id]
                self.num_indexes[id] = np.ceil((self.bounds[id,1]-self.bounds[id,0]) / self.resolution[id])

        self.num_indexes = np.asarray(self.num_indexes, dtype=int)

        self.motion = self.get_motion_model()

    def find_workspace(self, start_pos, prev_q, viz=None):

        start_n = self.create_node(start_pos, -1, None, prev_q)

        open_set, closed_set = dict(), dict()
        queue = deque()
        open_set[self.calc_grid_index(start_n)] = start_n

        queue.append(self.calc_grid_index(start_n))
        while len(queue) != 0:
            
            c_id = queue.popleft()
            current = open_set.pop(c_id)
            
            closed_set[c_id] = current
            
            viz.display(current.q_arr)
            boxID = "world/box" + "_ws_" + str(c_id)
            # material = meshcat.geometry.MeshPhongMaterial()
            # material.opacity = 0.5
            # if current.is_reach:
            #     plt.plot(current.pos[0],current.pos[1], "xc")
            #     # time.sleep(0.5)
            #     material.color = int(0x00FF00)
            # else:
            #     material.color = int(0xFF0000)
            #     plt.plot(current.pos[0],current.pos[1], "xr")
            # pos_3d = np.array([current.pos[0], 0, current.pos[1]])
            # size_box = np.array([self.resolution[0], 0.001, self.resolution[1]])
            # viz.viewer[boxID].set_object(meshcat.geometry.Box(size_box), material)
            # T = np.r_[np.c_[np.eye(3), pos_3d], np.array([[0, 0, 0, 1]])]
            # viz.viewer[boxID].set_transform(T)
                # time.sleep(2)
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                            lambda event:
                                            [exit(0) if event.key == 'escape'
                                            else None])
            if len(closed_set.keys()) % 1 == 0:
                plt.pause(0.001)

            all_direct_reach = []
            close_rachable_node = []
            for i, moving in enumerate(self.motion):
                new_pos = current.pos + moving[:-1] * self.resolution

                if np.all(self.bounds[:, 0] <= new_pos) and np.all(new_pos <= self.bounds[:, 1]):
                    node = self.create_node(new_pos, c_id, None, current.q_arr)
                    n_id = self.calc_grid_index(node)
                    all_direct_reach.append(node.is_reach)
                else:
                    continue
                if n_id in closed_set and closed_set[n_id].is_reach:
                    close_rachable_node.append(closed_set[n_id])
                
                line_id = "world/line" + "_from_" + str(c_id) + "_to_" + str(n_id)
                verteces = np.zeros((2,3))
                verteces[0,:] = self.robot.motion_space.get_6d_point(current.pos)[:3]
                verteces[1,:] = self.robot.motion_space.get_6d_point(node.pos)[:3]
            
                material = meshcat.geometry.LineBasicMaterial()
                material.opacity = 1
                material.linewidth = 50
                if bool(node.is_reach):
                    material.color = 0x66FFFF
                else:
                    material.color = 0x990099
                pts_meshcat = meshcat.geometry.PointsGeometry(verteces.astype(np.float32).T)
                viz.viewer[line_id].set_object(meshcat.geometry.Line(pts_meshcat, material))
                
                if (n_id not in closed_set) and (n_id not in open_set):
                    node.parent = current
                    open_set[n_id] = node
                    # if bool(current.is_reach):
                    queue.append(n_id)
                    
            
            if not bool(current.is_reach) and len(close_rachable_node) > 0:
                q_new_start = close_rachable_node[-1].q_arr
                fix_current = self.create_node(current.pos, current.parent_index, current.parent,
                                               q_new_start)
                
                if fix_current.is_reach:
                    closed_set[c_id] = fix_current
                    current = fix_current
                    all_direct_reach = []
                    for i, moving in enumerate(self.motion):
                        new_pos = current.pos + moving[:-1] * self.resolution

                        if np.all(self.bounds[:, 0] <= new_pos) and np.all(new_pos <= self.bounds[:, 1]):
                            node = self.create_node(new_pos, c_id, None, current.q_arr)
                            n_id = self.calc_grid_index(node)
                            all_direct_reach.append(node.is_reach)
                        else:
                            continue
                        line_id = "world/line" + "_from_" + str(c_id) + "_to_" + str(n_id)
                        verteces = np.zeros((2,3))
                        verteces[0,:] = self.robot.motion_space.get_6d_point(current.pos)[:3]
                        verteces[1,:] = self.robot.motion_space.get_6d_point(node.pos)[:3]
                    
                        material = meshcat.geometry.LineBasicMaterial()
                        material.opacity = 1
                        material.linewidth = 50
                        if bool(node.is_reach):
                            material.color = 0x66FFFF
                        else:
                            material.color = 0x990099
                        pts_meshcat = meshcat.geometry.PointsGeometry(verteces.astype(np.float32).T)
                        viz.viewer[line_id].set_object(meshcat.geometry.Line(pts_meshcat, material))
                        
                        if n_id not in open_set:
                            node.parent = current
                            open_set[n_id] = node
                            # queue.append(n_id)

            
            material = meshcat.geometry.MeshPhongMaterial()
            material.opacity = 0.1
            if not bool(current.is_reach):
                # viz.viewer[boxID].delete()
                material.color = int(0xFF0000)
                plt.plot(current.pos[0],current.pos[1], "xr")
            elif np.all(all_direct_reach):
                plt.plot(current.pos[0],current.pos[1], "xc")
                material.color = int(0x00FF00)
            else:
                material.color = 0xFFFF33
                plt.plot(current.pos[0],current.pos[1], "xy")
                
            pos_3d = np.array([current.pos[0], 0, current.pos[1]])
            size_box = np.array([self.resolution[0], 0.001, self.resolution[1]])
            viz.viewer[boxID].set_object(meshcat.geometry.Box(size_box), material)
            T = np.r_[np.c_[np.eye(3), pos_3d], np.array([[0, 0, 0, 1]])]
            viz.viewer[boxID].set_transform(T)
                    
                    # cls_boxID = "world/box" + "_ws_" + str(n_id)
                    # viz.viewer[cls_boxID].delete()
                    # prev_node = closed_set[n_id]
                    # pos_3d = np.array([prev_node.pos[0], 0, prev_node.pos[1]])
                    # size_box = np.array([self.resolution[0], 0.001, self.resolution[1]])
                    # viz.viewer[cls_boxID].set_object(meshcat.geometry.Box(size_box), material)
                    # T = np.r_[np.c_[np.eye(3), pos_3d], np.array([[0, 0, 0, 1]])]
                    # viz.viewer[cls_boxID].set_transform(T)
                    # plt.plot(prev_node.pos[0],prev_node.pos[1], "xy")
                    
                

        return closed_set

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
            q_fixed[0],
        )
        ee_id = robot.model.getFrameId(robot.ee_name)
        pin.framesForwardKinematics(robot.model, robot.data, q_fixed[0])
        Jfclosed = (
            pin.computeFrameJacobian(
                robot.model, robot.data, q_fixed[0], ee_id, pin.LOCAL_WORLD_ALIGNED
            )
            @ dq_dqmot
        )

        __, S, __ = np.linalg.svd(Jfclosed[self.robot.motion_space.indexes,:],hermitian=True)

        # if np.isclose(np.abs(S).min(), 0):
        #     dext_index = 100
        # else:
        dext_index = np.abs(S).max() / np.abs(S).min()

        real_pos = robot_ms.rewind_6d_point(poses_6d[0])

        node = self.Node(pos, 1 / dext_index, q_fixed[0], parent_index, parent, reach_array[0])

        return node

    def calc_grid_index(self, node):
        idx = self.calc_index(node.pos)

        grid_index = 0
        for k, id in enumerate(idx):
            grid_index += id * np.prod(self.num_indexes[:k])

        return grid_index

    def calc_grid_position(self, indexes):

        pos_6d = indexes * self.resolution + self.bounds[:, 0]

        return pos_6d

    def calc_index(self, pos):
        return np.round((pos - self.bounds[:, 0]) / self.resolution).astype(int)

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
            [0, 1, 1],
            [-1, 1, np.sqrt(2)],
            [-1, 0, 1],
            [-1, -1, np.sqrt(2)],
            [0, -1, 1],
        ]

        return motion


if __name__== "__main__":
    from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
    from auto_robot_design.description.builder import (
        ParametrizedBuilder,
        URDFLinkCreator,
        URDFLinkCreater3DConstraints,
        jps_graph2pinocchio_robot,
        jps_graph2pinocchio_robot_3d_constraints
        )
    import meshcat
    from pinocchio.visualize import MeshcatVisualizer
    from auto_robot_design.pinokla.closed_loop_kinematics import (
        closedLoopInverseKinematicsProximal,
        openLoopInverseKinematicsProximal,
        closedLoopProximalMount,
    )

    builder = ParametrizedBuilder(URDFLinkCreater3DConstraints)
    
    gm = get_preset_by_index_with_bounds(8)
    x_centre = gm.generate_central_from_mutation_range()
    graph_jp = gm.get_graph(x_centre)
    

    robo, __ = jps_graph2pinocchio_robot_3d_constraints(graph_jp, builder=builder)
    
    center_bound = np.array([0, -0.25])
    size_box_bound = np.array([0.05, 0.05])
    
    start_pos = center_bound
    pos_6d = np.zeros(6)
    pos_6d[[0,2]] = start_pos
    
    id_ee = robo.model.getFrameId(robo.ee_name)
    
    pin.framesForwardKinematics(robo.model, robo.data, np.zeros(robo.model.nq))
    
    init_pos = robo.data.oMf[id_ee].translation[[0,2]]
    traj_init_to_center = add_auxilary_points_to_trajectory(([start_pos[0]],[start_pos[1]]), init_pos)
    
    point_6d = robo.motion_space.get_6d_traj(np.array(traj_init_to_center).T)
    poses_6d, q_fixed, constraint_errors, reach_array = closed_loop_pseudo_inverse_follow(
        robo.model,
        robo.data,
        robo.constraint_models,
        robo.constraint_data,
        robo.ee_name,
        point_6d,
    )
    
    q = q_fixed[-1]
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
    viz.viewer[ballID].set_object(meshcat.geometry.Sphere(0.001), material)
    T = np.r_[np.c_[np.eye(3), pos_6d[:3]], np.array([[0, 0, 0, 1]])]
    viz.viewer[ballID].set_transform(T)
    pin.framesForwardKinematics(robo.model, robo.data, q)
    viz.display(q)

    bounds = np.array([[-size_box_bound[0]/2-.001, size_box_bound[0]/2], [-size_box_bound[1]/2, size_box_bound[1]/2]])
    bounds[0,: ] += center_bound[0]
    bounds[1,: ] += center_bound[1]
    
    bound_pos = product(bounds[0,:], bounds[1,:])
    
    for k, pos in enumerate(bound_pos):
        ballID = "world/ball" + "_bound_" + str(k)
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = int(0x0000FF)
        material.opacity = 1
        pos_3d = np.array([pos[0], 0, pos[1]])
        viz.viewer[ballID].set_object(meshcat.geometry.Sphere(0.003), material)
        T = np.r_[np.c_[np.eye(3), pos_3d], np.array([[0, 0, 0, 1]])]
        viz.viewer[ballID].set_transform(T)

    pin.framesForwardKinematics(robo.model, robo.data, q)
    viz.display(q)
        
    
    ws_bfs = BreadthFirstSearchPlanner(robo, bounds, np.array([0.002, 0.002]), 0.5)
    ws_bfs.vis = viz
    viewed_nodes = ws_bfs.find_workspace(start_pos, q, viz)
    
    dext_index = [1/n.cost for n in viewed_nodes.values()]
    
    print(np.max(dext_index), np.min(dext_index))