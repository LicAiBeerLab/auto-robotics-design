from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount

from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
from auto_robot_design.pinokla.calc_criterion import search_workspace
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull

from auto_robot_design.description.builder import Builder, DetalizedURDFCreaterFixedEE, jps_graph2urdf_parametrized
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator


gen = TwoLinkGenerator()
builder = Builder(DetalizedURDFCreaterFixedEE)
graphs_and_cons = gen.get_standard_set()
np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

graph_jp, constrain = graphs_and_cons[0]
robot_urdf, ative_joints, constraints = jps_graph2urdf_parametrized(graph_jp)


robo = build_model_with_extensions(robot_urdf,
                                   joint_description=ative_joints,
                                   loop_description=constraints,
                                   fixed=True)


robo_free = build_model_with_extensions(
    robot_urdf,
    joint_description=ative_joints,
    loop_description=constraints,
    fixed=False
)

q0 = closedLoopProximalMount(
    robo.model,
    robo.data,
    robo.constraint_models,
    robo.constraint_data,
    max_it=100,
)


# viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
# viz.viewer = meshcat.Visualizer().open()
# viz.clean()
# viz.loadViewerModel()
# # viz.display(q0)


EFFECTOR_NAME = "EE"
BASE_FRAME = "G"

q_space_mot_1 = np.linspace(-np.pi, np.pi, 10)
q_space_mot_2 = np.linspace(-np.pi, np.pi, 10)
q_mot_double_space = list(product(q_space_mot_1, q_space_mot_2))

workspace_xyz, available_q = search_workspace(robo.model, robo.data, EFFECTOR_NAME, BASE_FRAME, np.array(
    q_mot_double_space), robo.actuation_model, robo.constraint_models)


print("Coverage q " + str(len(available_q)/(len(q_mot_double_space))))


plt.figure()
plt.scatter(workspace_xyz[:, 0],  workspace_xyz[:, 2])
plt.title("WorkScape")
plt.xlabel("X")
plt.ylabel("Ya popravil")


plt.show()
