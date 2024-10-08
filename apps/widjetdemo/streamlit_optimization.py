from copy import deepcopy
import time
import streamlit as st
import numpy as np
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from forward_init import add_trajectory_to_vis, build_constant_objects, set_criteria_and_crag
from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
from auto_robot_design.description.utils import draw_joint_point
import pinocchio as pin
import meshcat
from auto_robot_design.description.mesh_builder.mesh_builder import (
    MeshBuilder,
    jps_graph2pinocchio_meshes_robot,
)
from auto_robot_design.motion_planning.trajectory_ik_manager import TrajectoryIKManager
from auto_robot_design.pinokla.default_traj import (
    add_auxilary_points_to_trajectory,
    convert_x_y_to_6d_traj_xz,
    get_vertical_trajectory,
    create_simple_step_trajectory,
    get_workspace_trajectory,
)
from pinocchio.visualize import MeshcatVisualizer
from auto_robot_design.motion_planning.bfs_ws import Workspace, BreadthFirstSearchPlanner
from auto_robot_design.description.builder import (
    ParametrizedBuilder,
    URDFLinkCreater3DConstraints,
    jps_graph2pinocchio_robot_3d_constraints,
)
from auto_robot_design.optimization.rewards.reward_base import PositioningConstrain, PositioningErrorCalculator, RewardManager
from streamlit_widget_auxiliary import get_visualizer, send_graph_to_visualizer, run_simulation

full_crag, rewards, motor = set_criteria_and_crag()
# build and cache constant objects
graph_managers, optimization_builder, visualization_builder, crag, workspace_trajectory, ground_symmetric_step1, ground_symmetric_step2, ground_symmetric_step3, central_vertical, left_vertical, right_vertical = build_constant_objects()
trajectories = {"ground_symmetric_step1": ground_symmetric_step1,
                "ground_symmetric_step2": ground_symmetric_step2,
                "ground_symmetric_step3": ground_symmetric_step3,
                "central_vertical": central_vertical,
                "left_vertical": left_vertical,
                "right_vertical": right_vertical}


st.title("mechanical linkage mechanism optimization")

if 'gm' not in st.session_state:
    st.session_state.gm = get_preset_by_index_with_bounds(-1)
    st.session_state.reward_manager = RewardManager(crag=full_crag)
    st.session_state.trajectory_idx = 0
    st.session_state.topology = None
    st.session_state.ranges = False
    st.session_state.trajectory_groups = []
    error_calculator = PositioningErrorCalculator(error_key='error', jacobian_key="Manip_Jacobian")
    st.session_state.soft_constraint = PositioningConstrain(error_calculator=error_calculator, points = [])


def confirm_topology():
    st.session_state.topology = graph_managers[st.session_state.topology_choice]


col_1, col_2 = st.columns(2, gap="medium")
# the radio button and confirm button are only visible until the topology is selected
if st.session_state.topology is None:
    with st.sidebar:
        st.radio(label="Select topology:", options=graph_managers.keys(),
                 index=None, key='topology_choice')
        st.button(label='Confirm topology', key='confirm_topology',
                  on_click=confirm_topology)

    if st.session_state.topology_choice:
        st.session_state.gm = graph_managers[st.session_state.topology_choice]

    gm = st.session_state.gm
    values = gm.generate_central_from_mutation_range()
    graph = st.session_state.gm.get_graph(values)
    send_graph_to_visualizer(graph, visualization_builder)
    with col_1:
        st.header("Graph representation")
        draw_joint_point(graph)
        plt.gcf().set_size_inches(4, 4)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.header("Robot visualization")
        components.iframe(get_visualizer(visualization_builder).viewer.url(), width=400,
                          height=400, scrolling=True)


def confirm_ranges():
    st.session_state.topology = False
    st.session_state.ranges = True


# second stage
if st.session_state.topology and not st.session_state.ranges:
    # form for optimization ranges
    gm = st.session_state.gm
    mut_ranges = gm.mutation_ranges
    current_values = []
    gm_clone = deepcopy(gm.mutation_ranges)
    with st.sidebar:
        for key, value in mut_ranges.items():
            current_on = st.toggle(f"Activate feature {key}", value=True)
            if current_on:
                current_value = st.slider(
                    label=key, min_value=value[0], max_value=value[1], value=(value[0], value[1]))
                current_values.append(current_value)
            else:
                current_value = st.number_input("Insert a value")
                current_values.append(current_value)

        for idx, value in enumerate(current_values):
            if isinstance(value, tuple):
                gm_clone[list(gm.mutation_ranges.keys())[idx]] = value
            else:
                gm_clone[list(gm.mutation_ranges.keys())[idx]] = (value, value)
        st.button(label="Confirm optimization ranges",
                  key='ranges_confirm', on_click=confirm_ranges)
    # here should be some kind of visualization for ranges
    st.write(gm_clone)


def add_trajectory(trajectory, idx):
    st.session_state.reward_manager.add_trajectory(trajectory, idx)
    st.session_state.trajectory_groups.append([idx])
    st.session_state.soft_constraint.add_points_set(trajectory) 
    st.session_state.trajectory_idx += 1

    # st.session_state.reward_manager.

# def create_new_group(trajectory, idx):
#     st.session_state.reward_manager.add_trajectory(trajectory, idx)
#     st.session_state.trajectory_groups.append([idx])
#     st.session_state.trajectory_idx += 1


def add_to_group(trajectory, idx):
    st.session_state.reward_manager.add_trajectory(trajectory, idx)
    st.session_state.trajectory_groups[-1].append(idx)
    st.session_state.soft_constraint.add_points_set(trajectory) 
    st.session_state.trajectory_idx += 1


def start_optimization(rewards_tf):
    print(st.session_state.trajectory_groups)
    for trj_list_idx, trajectory_list in enumerate(st.session_state.trajectory_groups):
        for trj in trajectory_list:
            for reward_idx, reward in enumerate(rewards_tf[trj_list_idx]):
                if reward:
                    st.session_state.reward_manager.add_reward(
                        rewards[reward_idx], trj, 1)
        st.session_state.reward_manager.add_trajectory_aggregator(
            trajectory_list, 'mean')
    
    run_simulation(st.session_state.gm, optimization_builder, st.session_state.soft_constraint, motor)


    # when ranges are set we start to choose the reward+trajectory
    # each trajectory should be added to the manager
if st.session_state.ranges:
    graph = st.session_state.gm.graph
    trajectory = None
    with st.sidebar:
        trajectory_type = st.radio(label='Select trajectory type', options=[
            "vertical", "step"], index=1, key="trajectory_type")
        if trajectory_type == "vertical":
            height = st.slider(
                label="height", min_value=0.02, max_value=0.3, value=0.1)
            x = st.slider(label="x", min_value=-0.3,
                          max_value=0.3, value=0.0)
            z = st.slider(label="z", min_value=-0.4,
                          max_value=-0.2, value=-0.3)
            trajectory = convert_x_y_to_6d_traj_xz(
                *add_auxilary_points_to_trajectory(get_vertical_trajectory(z, height, x, 100)))

        if trajectory_type == "step":
            start_x = st.slider(
                label="start_x", min_value=-0.3, max_value=0.3, value=-0.14)
            start_z = st.slider(
                label="start_z", min_value=-0.4, max_value=-0.2, value=-0.34)
            height = st.slider(
                label="height", min_value=0.02, max_value=0.3, value=0.1)
            width = st.slider(label="width", min_value=0.1,
                              max_value=0.6, value=0.2)
            trajectory = convert_x_y_to_6d_traj_xz(
                *add_auxilary_points_to_trajectory(
                    create_simple_step_trajectory(
                        starting_point=[start_x, start_z],
                        step_height=height,
                        step_width=width,
                        n_points=100,
                    )
                )
            )
        st.button(label="Add trajectory", key="add_trajectory", args=(
            trajectory, st.session_state.trajectory_idx), on_click=add_trajectory)
        # st.button(label="Create new group",key="create_new_group", args=[trajectory,st.session_state.trajectory_idx],on_click=create_new_group)
        if st.session_state.trajectory_idx:
            st.button(label="Add to group", key="add_to_group", args=[
                trajectory, st.session_state.trajectory_idx], on_click=add_to_group)
        # for each reward trajectories should be assigned

    with col_1:
        st.header("Graph representation")
        draw_joint_point(graph)
        plt.gcf().set_size_inches(4, 4)
        plt.plot(trajectory[:, 0], trajectory[:, 2])
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.header("Robot visualization")
        add_trajectory_to_vis(get_visualizer(
            visualization_builder), trajectory)
        components.iframe(get_visualizer(visualization_builder).viewer.url(), width=400,
                          height=400, scrolling=True)

    if st.session_state.trajectory_idx:
        cols = st.columns(len(st.session_state.trajectory_groups))
    trajectories = [[0]*len(rewards)]*len(st.session_state.trajectory_groups)
    for i in range(len(st.session_state.trajectory_groups)):
        with cols[i]:
            reward_idxs = [0]*len(rewards)
            for reward_idx, reward in enumerate(rewards):
                current_checkbox = st.checkbox(
                    label=reward.reward_name, value=False, key=reward.reward_name+str(i))
                reward_idxs[reward_idx] = current_checkbox
            trajectories[i] = reward_idxs
    if st.session_state.trajectory_idx:
        st.button(label="Start optimization",
                  key="start_optimization", on_click=start_optimization, args=[trajectories])
