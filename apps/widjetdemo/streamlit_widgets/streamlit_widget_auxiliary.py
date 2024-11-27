import streamlit as st

import meshcat
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
from auto_robot_design.description.mesh_builder.mesh_builder import (
    MeshBuilder,
    jps_graph2pinocchio_meshes_robot,
)
import dill

@st.cache_resource
def get_visualizer(_visualization_builder):
    builder = _visualization_builder
    gm = st.session_state.gm
    values = gm.generate_central_from_mutation_range()
    graph = gm.get_graph(values)
    fixed_robot, free_robot = jps_graph2pinocchio_meshes_robot(graph, builder)
    # create a pinocchio visualizer object with current value of a robot
    visualizer = MeshcatVisualizer(
        fixed_robot.model, fixed_robot.visual_model, fixed_robot.visual_model
    )
    # create and setup a meshcat visualizer
    visualizer.viewer = meshcat.Visualizer()
    # visualizer.viewer["/Background"].set_property("visible", False)
    visualizer.viewer["/Grid"].set_property("visible", False)
    visualizer.viewer["/Axes"].set_property("visible", False)
    visualizer.viewer["/Cameras/default/rotated/<object>"].set_property(
        "position", [0, 0.0, 1]
    )
    # load a model to the visualizer and set it into the neutral position
    visualizer.clean()
    visualizer.loadViewerModel()
    visualizer.display(pin.neutral(fixed_robot.model))

    return visualizer


def send_graph_to_visualizer(graph, visualization_builder):
    fixed_robot, _ = jps_graph2pinocchio_meshes_robot(
        graph, visualization_builder)
    visualizer = get_visualizer(visualization_builder)
    visualizer.model = fixed_robot.model
    visualizer.collision = fixed_robot.visual_model
    visualizer.visual_model = fixed_robot.visual_model
    visualizer.rebuildData()
    visualizer.clean()
    visualizer.loadViewerModel()
    visualizer.display(pin.neutral(fixed_robot.model))

