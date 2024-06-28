from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import hppfcl
import meshcat
from auto_robot_design.description.builder import MIT_CHEETAH_PARAMS_DICT, jps_graph2pinocchio_robot
from auto_robot_design.optimization.problems import CalculateMultiCriteriaProblem
from auto_robot_design.pinokla.calc_criterion import folow_traj_by_proximal_inv_k
from auto_robot_design.pinokla.default_traj import get_workspace_trajectory
from  apps.widjetdemo import traj_graph_setup  
from apps.widjetdemo import create_reward_manager
from auto_robot_design.pinokla.loader_tools import Robot, make_Robot_copy
from auto_robot_design.user_interface.check_in_ellips import Ellipse, check_points_in_ellips
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import meshcat.geometry as mg
import time


model = pin.Model()
collision_model = pin.GeometryModel()

ell_clear = hppfcl.Ellipsoid(1, 0.5, 0.5)
octree_object = pin.GeometryObject("octree", 0, pin.SE3.Identity(), ell_clear)
octree_object.meshColor[3] = 0.3
octree_object.meshColor[0] = 0.1
octree_object.meshColor[0] = 0.1
collision_model.addGeometryObject(octree_object)

viz = MeshcatVisualizer(model, collision_model, collision_model)
viz.viewer = meshcat.Visualizer().open()
viz.loadViewerModel()
viz.display()
time.sleep(1)
print("FFFF")
