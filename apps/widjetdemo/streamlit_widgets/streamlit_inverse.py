import streamlit as st
from copy import deepcopy
import numpy as np
from auto_robot_design.utils.configs import get_mesh_builder, get_standard_builder, get_standard_rewards
from streamlit_widget_auxiliary import get_visualizer, send_graph_to_visualizer
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

@st.cache_resource
def get_items():
    gms = {f"Topology_{i}": get_preset_by_index_with_bounds(i) for i in range(9)}
    return gms, get_mesh_builder(jupyter=False), get_standard_builder(), get_standard_rewards()

graph_managers, visualization_builder, standard_builder, standard_rewards = get_items()


def confirm_topology():
    """Confirm the selected topology and move to the next stage."""
    st.session_state.stage = "ranges_choice"
    # create a deep copy of the graph manager for further updates
    st.session_state.gm_clone = deepcopy(st.session_state.gm)


def topology_choice():
    """Update the graph manager based on the selected topology."""
    st.session_state.gm = graph_managers[st.session_state.topology_choice]

if not hasattr(st.session_state, "stage"):
    st.session_state.stage = 'class_choice'
    st.session_state.gm = get_preset_by_index_with_bounds(-1)

def type_choice(type):
    if type == 'free':
        st.session_state.type = 'free'
    elif type == 'suspension':
        st.session_state.type = 'suspension'
    elif type == 'manipulator':
        st.session_state.type = 'manipulator'
    st.session_state.stage = 'topology_choice'

if st.session_state.stage == 'class_choice':
    col_1, col_2, col_3 = st.columns(3, gap="medium")
    with col_1:
        st.button(label='свободный выбор', key='free',on_click=type_choice, args=['free'])
        st.image('.\\apps\\rogue.jpg')
    with col_2:
        st.button(label='подвеска', key='suspension',on_click=type_choice, args=['suspension'])
        st.image('.\\apps\\wizard.jpg')
    with col_3:
        st.button(label='манипулятор', key='manipulator',on_click=type_choice, args=['manipulator'])
        st.image('.\\apps\\warrior.jpg')


if st.session_state.stage == "topology_choice":
    with st.sidebar:
        st.radio(label="Select topology:", options=graph_managers.keys(),
                 index=None, key='topology_choice', on_change=topology_choice)
        st.button(label='Confirm topology', key='confirm_topology',
                  on_click=confirm_topology)

    gm = st.session_state.gm
    values = gm.generate_central_from_mutation_range()
    graph = st.session_state.gm.get_graph(values)
    send_graph_to_visualizer(graph, visualization_builder)
    col_1, col_2 = st.columns(2, gap="medium")
    with col_1:
        st.header("Graph representation")
        draw_joint_point(graph)
        plt.gcf().set_size_inches(4, 4)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.header("Robot visualization")
        components.iframe(get_visualizer(visualization_builder).viewer.url(), width=400,
                          height=400, scrolling=True)