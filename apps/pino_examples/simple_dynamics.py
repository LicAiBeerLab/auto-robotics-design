from auto_robot_design.pinokla.robot_utils import freezeJointsWithoutVis
from auto_robot_design.pinokla.loader_tools import Robot, build_model_with_extensions
from auto_robot_design.pinokla.criterion_agregator import ComputeConfg, calc_criterion_on_workspace_simple_input, save_criterion_traj
from auto_robot_design.pinokla.closed_loop_jacobian import dq_dqmot, inverseConstraintKinematicsSpeed
from auto_robot_design.pinokla.calc_criterion import calc_IMF, calc_foot_inertia, calc_manipulability, convert_full_J_to_planar_xz
from auto_robot_design.pino_adapter import pino_adapter
from auto_robot_design.description.utils import draw_joint_frames, draw_joint_point, draw_link_frames
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph
from auto_robot_design.description.builder import Builder, URDFLinkCreator, add_branch, jps_graph2urdf, jps_graph2urdf_parametrized
import networkx as nx
import pinocchio as pin
import numpy as np
import meshcat
from pinocchio.visualize import MeshcatVisualizer
from numpy.linalg import norm

import os
import sys

from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, "../utils")
sys.path.append(mymodule_dir)


# Load robot
gen = TwoLinkGenerator()
builder = Builder(URDFLinkCreator)
graphs_and_cons = gen.get_standard_set()
np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

graph_jp, constrain = graphs_and_cons[0]

robot_urdf, ative_joints, constraints = jps_graph2urdf_parametrized(graph_jp)
robo = build_model_with_extensions(robot_urdf, ative_joints, constraints)

# Visualizer

viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
viz.viewer = meshcat.Visualizer().open()
viz.clean()
viz.loadViewerModel()


# Montable configuration
q = closedLoopProximalMount(
    robo.model, robo.data, robo.constraint_models, robo.constraint_data)
viz.display(q)

# free fall dynamics
pin.initConstraintDynamics(robo.model, robo.data, robo.constraint_models)
DT = 1e-4
N_it = 10000
tauq = np.zeros(robo.model.nv)
id_mt1 = robo.actuation_model.idMotJoints[0]
id_mt2 = robo.actuation_model.idqmot[1]
tauq[id_mt1] = 1
tauq[id_mt2] = 1
vq = np.zeros(robo.model.nv)

accuracy = 1e-8
mu_sim = 1e-8
max_it = 100
dyn_set = pin.ProximalSettings(accuracy, mu_sim, max_it)


for i in range(N_it):
    a = pin.constraintDynamics(robo.model, robo.data, q, vq,
                               tauq, robo.constraint_models, robo.constraint_data, dyn_set)
    vq += a*DT
    q = pin.integrate(robo.model, q, vq * DT)
    viz.display(q)

# Check constraint

err = np.sum([norm(pin.log(cd.c1Mc2).np[:cm.size()])
             for (cd, cm) in zip(robo.constraint_data, robo.constraint_models)])
print(err)
