import meshcat
from pinocchio.visualize import MeshcatVisualizer


def create_meshcat_vizualizer(robot):
    viz = MeshcatVisualizer(robot.model, robot.visual_model, robot.visual_model)
    viz.viewer = meshcat.Visualizer().open()
    viz.viewer["/Background"].set_property("visible", False)
    viz.viewer["/Grid"].set_property("visible", False)
    viz.viewer["/Axes"].set_property("visible", False)
    viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", [0,0,0.5])
    viz.clean()
    viz.loadViewerModel()