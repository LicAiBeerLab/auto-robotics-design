import time
from collections import UserDict
from enum import IntFlag, auto
from typing import NamedTuple, Optional

import numpy as np
import pinocchio as pin
from numpy.linalg import norm

from auto_robot_design.pinokla.closed_loop_jacobian import (
    closedLoopInverseKinematicsProximal, dq_dqmot,
    inverseConstraintKinematicsSpeed)
from auto_robot_design.pinokla.closed_loop_kinematics import (
    ForwardK, closedLoopProximalMount, closed_loop_ik_grad, closed_loop_ik_pseudo_inverse)
from auto_robot_design.pinokla.criterion_math import (calc_manipulability,
                                                      ImfProjections, calc_actuated_mass, calc_effective_inertia,
                                                      calc_force_ell_projection_along_trj, calc_IMF, calculate_mass,
                                                      convert_full_J_to_planar_xz)
from auto_robot_design.pinokla.loader_tools import Robot

from auto_robot_design.pinokla.calc_criterion import (
    folow_traj_by_proximal_inv_k, closed_loop_pseudo_inverse_follow)

class BreadthFirstSearchPlanner:


    class Node:
        def __init__(self, pos, cost, q_arr, criteria, parent_index, parent):
            self.pos = pos
            self.cost = cost
            self.q_arr = q_arr
            self.criteria = criteria
            self.parent_index = parent_index
            self.parent = parent
            
        
        def __str__(self):
            return str(self.pos) + ", " + str(self.q_arr) + ", " + str(
                self.cost) + ", " + str(self.parent_index)
            
    def __init__(self, robot, bounds, grid_resolution, singularity_threshold) -> None:
        
        self.robot = robot
        self.resolution = grid_resolution
        self.threshold = singularity_threshold
        
        num_indexes = np.max(bounds[:3], 1) - np.min(bounds[:3], 1) / self.resolution

        self.bounds = np.zeros_like(bounds)
        for id, idx_value in enumerate(num_indexes):
            residue_div = idx_value % 1
            if residue_div != 0.0:
                self.bounds[id,0] = bounds[id,0] - residue_div/2
                self.bounds[id,1] = bounds[id,1] + residue_div/2
                
                num_indexes[id] = np.ceil(num_indexes[id])
        self.num_indexes = np.asarray(num_indexes, dtype=int)
        
    
    def find_workspace(self, start_pos):
        
        poses, q_fixed, constraint_errors,reach_array = closed_loop_pseudo_inverse_follow(
            fixed_robot.model, fixed_robot.data, fixed_robot.constraint_models,
            fixed_robot.constraint_data, ee_frame_name, traj_6d, viz)
        start_n = self.Node(self.calc_index(start_pos), 0.0, )
    
    
    def create_node():
        pass
    
    def calc_grid_position(self, indexes):
        
        pos_6d = indexes * self.resolution + self.bounds[:,0]
        
        return pos_6d
    
    def calc_index(self, pos):
        return np.round((pos - self.bounds[:,0]) / self.resolution)
    
    def verify_node(self, node):
        pos = node.pos
        if np.any(pos < self.bounds[:,0]) or np.any(pos > self.bounds[:,1]):
            return False
        return True
    
    
    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, np.sqrt(2)],
                  [-1, 1, np.sqrt(2)],
                  [1, -1, np.sqrt(2)],
                  [1, 1, np.sqrt(2)]]

        return motion
