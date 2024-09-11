from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount

from auto_robot_design.pinokla.calc_criterion import search_workspace
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator


gen = TwoLinkGenerator()
builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE)
graphs_and_cons = gen.get_standard_set()
np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

graph_jp, __ = graphs_and_cons[0]
robo, robo_free = jps_graph2pinocchio_robot(graph_jp, builder)

q0 = closedLoopProximalMount(
    robo.model,
    robo.data,
    robo.constraint_models,
    robo.constraint_data,
    max_it=100,
)


# viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
# viz.viewer = meshcat.Visualizer().open()
# viz.viewer["/Background"].set_property("visible", False)
# viz.viewer["/Grid"].set_property("visible", False)
# viz.viewer["/Axes"].set_property("visible", False)
# viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", [0,0,0.5])
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
