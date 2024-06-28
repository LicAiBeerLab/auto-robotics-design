from copy import deepcopy
import time
import numpy as np
import matplotlib.pyplot as plt
import hppfcl
import meshcat
from auto_robot_design.description.builder import MIT_CHEETAH_PARAMS_DICT, jps_graph2pinocchio_robot
from auto_robot_design.optimization.problems import CalculateMultiCriteriaProblem
from auto_robot_design.pinokla.calc_criterion import MovmentSurface, folow_traj_by_proximal_inv_k
from auto_robot_design.pinokla.default_traj import get_workspace_trajectory
from apps.widjetdemo import traj_graph_setup
from apps.widjetdemo import create_reward_manager
from auto_robot_design.pinokla.loader_tools import Robot, make_Robot_copy
from auto_robot_design.user_interface.check_in_ellips import Ellipse, check_points_in_ellips
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import meshcat.geometry as mg
from scipy.spatial.transform import Rotation as R

class RobotVisualizer():
    def __init__(self, robot: Robot, motion_surface=MovmentSurface.XZ) -> None:
        self.robot = robot  # Robot(*make_Robot_copy(robot))

    def play_animation(self, q_vec: np.ndarray):
        self.viz = MeshcatVisualizer(self.robot.model,
                                     self.robot.visual_model,
                                     self.robot.visual_model)
        self.viz.viewer = meshcat.Visualizer()
        self.viz.loadViewerModel()
        self.viz.viewer.open()
        self.viz.displayVisuals(True)

        for q_i in q_vec:
            time.sleep(0.03)
            self.viz.display(q_i)

    
    def add_ellips_to_viz(self, pos_xyz: np.ndarray, angle, r1, r2):
        ell_clear = hppfcl.Ellipsoid(r1, 0.001, r2)
        pos = pin.SE3.Identity()
        pos.translation = pos_xyz
        pos.rotation = R.from_euler('y', angle).as_matrix()
        ell_obj = pin.GeometryObject(
            "ell", 0, pos, ell_clear)
        ell_obj.meshColor[3] = 0.3
        ell_obj.meshColor[0] = 0.1
        ell_obj.meshColor[0] = 0.1
        self.robot.visual_model.addGeometryObject(ell_obj)
