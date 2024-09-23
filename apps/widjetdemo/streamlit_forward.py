import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from forward_init import add_trajectory_to_vis, build_constant_objects
from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
from auto_robot_design.description.utils import draw_joint_point
import pinocchio as pin
import meshcat
from auto_robot_design.description.mesh_builder.mesh_builder import (
    MeshBuilder,
    jps_graph2pinocchio_meshes_robot,
)
from pinocchio.visualize import MeshcatVisualizer

# build and cache constant objects 
graph_managers, optimization_builder, visualization_builder, crag, workspace_trajectory, ground_symmetric_step1, ground_symmetric_step2, ground_symmetric_step3, central_vertical, left_vertical, right_vertical = build_constant_objects()

# create gm variable that will be used to store the current graph manager and set it to be update for a session
if 'gm' not in st.session_state:
    st.session_state.gm = get_preset_by_index_with_bounds(-1)

@st.cache_resource
def create_visualizer():
    global visualization_builder
    builder = visualization_builder
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
    visualizer.viewer["/Background"].set_property("visible", False)
    visualizer.viewer["/Grid"].set_property("visible", False)
    visualizer.viewer["/Axes"].set_property("visible", False)
    visualizer.viewer["/Cameras/default/rotated/<object>"].set_property(
        "position", [0, 0.0, 0.8]
    )
    # load a model to the visualizer and set it into the neutral position
    visualizer.clean()
    visualizer.loadViewerModel()
    visualizer.display(pin.neutral(fixed_robot.model))

    return visualizer

visualizer = create_visualizer()


def topology_choice():
    global visualizer
    st.session_state.gm = graph_managers[st.session_state.topology_choice]
    values = st.session_state.gm.generate_central_from_mutation_range()
    graph = st.session_state.gm.get_graph(values)
    draw_joint_point(graph)
    st.pyplot(plt.gcf())
    components.iframe(visualizer.viewer.url())

st.radio(label="Select topology:", options=graph_managers.keys(), index=None, on_change=topology_choice, key='topology_choice')



def confirm_topology():
    gm = st.session_state.gm
    
    print(gm.mutation_ranges)

st.button(label='Confirm topology', key='confirm_topology',on_click=confirm_topology)