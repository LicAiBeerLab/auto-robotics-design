import time
from copy import deepcopy
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
from streamlit_widget_auxiliary import get_visualizer, send_graph_to_visualizer

full_crag, rewards, motor = set_criteria_and_crag()
# build and cache constant objects
graph_managers, optimization_builder, visualization_builder, crag, reward_dict = build_constant_objects()

st.title("Оценка рычажных механизмов")
# create gm variable that will be used to store the current graph manager and set it to be update for a session
if 'gm' not in st.session_state:
    st.session_state.gm = get_preset_by_index_with_bounds(-1)
    # the session variable for chosen topology, it gets a value after topology confirmation button is clicked
    st.session_state.stage = 'topology_choice'
    # st.session_state.trajectory_choice = False
    st.session_state.run_simulation_flag = False


def confirm_topology():
    st.session_state.stage = 'joint_point_choice'
    st.session_state.gm = deepcopy(graph_managers[st.session_state.topology_choice])


# the radio button and confirm button are only visible until the topology is selected
if st.session_state.stage == 'topology_choice':
    with st.sidebar:
        st.radio(label="Выбор топологии рычажного механизма:", options=graph_managers.keys(), index=0, key='topology_choice')
        st.button(label='Подтвердить выбор топологии', key='confirm_topology',
                  on_click=confirm_topology)

    if st.session_state.topology_choice:
        st.session_state.gm = graph_managers[st.session_state.topology_choice]

    gm = st.session_state.gm
    values = gm.generate_central_from_mutation_range()
    graph = st.session_state.gm.get_graph(values)
    send_graph_to_visualizer(graph, visualization_builder)
    # send_graph_to_visualizer(graph, optimization_builder)
    col_1, col_2 = st.columns(2, gap="medium")
    with col_1:
        st.header("Граф выбранной топологии")
        draw_joint_point(graph, labels=2)
        plt.gcf().set_size_inches(4, 4)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.header("Визуализация")
        components.iframe(get_visualizer(visualization_builder).viewer.url(), width=400,
                          height=400, scrolling=True)


def evaluate_construction():
    """Calculate the workspace of the robot and display it"""
    st.session_state.stage = 'workspace_visualization'
    gm = st.session_state.gm
    graph = gm.graph
    robo, __ = jps_graph2pinocchio_robot_3d_constraints(
        graph, builder=optimization_builder)
    size_box_bound = np.array([0.5, 0.42])
    center_bound = np.array([0, -0.21])
    bounds = np.array(
        [
            [-size_box_bound[0] / 2 - 0.001, size_box_bound[0] / 2],
            [-size_box_bound[1] / 2, size_box_bound[1] / 2],
        ]
    )
    bounds[0, :] += center_bound[0]
    bounds[1, :] += center_bound[1]
    start_pos = np.array([0, -0.4])
    q = np.zeros(robo.model.nq)
    workspace_obj = Workspace(robo, bounds, np.array([0.01, 0.01]))
    ws_bfs = BreadthFirstSearchPlanner(workspace_obj, 0)
    workspace = ws_bfs.find_workspace(start_pos, q)
    points = []
    point = workspace.bounds[:, 0]
    k, m = 0, 0
    while point[1] <= workspace.bounds[1, 1]:

        while point[0] <= workspace.bounds[0, 1]:
            points.append(point)
            m += 1
            point = workspace.bounds[:, 0] + np.array(
                workspace.resolution) * np.array([m, k])
        k += 1
        m = 0
        point = workspace.bounds[:, 0] + np.array(
            workspace.resolution) * np.array([m, k])

    points = np.array(points)
    st.session_state.workspace = workspace
    st.session_state.points = points


def to_trajectory_choice():
    st.session_state.stage = 'trajectory_choice'


def run_simulation():
    st.session_state.run_simulation_flag = True


def calculate_and_display_rewards(trajectory, reward_mask):
    gm = st.session_state.gm
    fixed_robot, free_robot = jps_graph2pinocchio_robot_3d_constraints(
        gm.graph, optimization_builder)
    point_criteria_vector, trajectory_criteria, res_dict_fixed = full_crag.get_criteria_data(
        fixed_robot, free_robot, trajectory, viz=None)
    for i, reward in enumerate(rewards):
        if reward_mask[i]:
            try:
                st.text_area(label=reward.reward_name, value=str(reward.calculate(
                    point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator=motor)[0]))
            except ValueError:
                st.text_area(
                    label="", value="The trajectory is not feasible, please choose a trajectory within the workspace")
                break

# choose the mechanism for optimization
if st.session_state.stage == 'joint_point_choice':
    st.text('Установите необходимые положения для координат центров сичленений')
    gm = st.session_state.gm
    gm.set_mutation_ranges()
    mut_ranges = gm.mutation_ranges
    current_values = []
    # sliders for joint position choice
    with st.sidebar:
        st.button(label='Вернуться к выбору топологии', key='return_to_topology_choice',
                on_click=lambda: st.session_state.__setitem__('stage', 'topology_choice'))
    with st.sidebar.form("jp_coordinates"):
        st.header('Выбор положений сочленений')
        for key, value in mut_ranges.items():
            slider = st.slider(
                label=key, min_value=value[0], max_value=value[1], value=(value[1]+value[0])/2)
            current_values.append(slider)
        st.form_submit_button('Подтвердить выбор механизма')
    with st.sidebar:
        st.button(label="Рассчитать рабочее пространство",
                  on_click=evaluate_construction, key="get_workspace")

    graph = gm.get_graph(current_values)
    send_graph_to_visualizer(graph, visualization_builder)
    col_1, col_2 = st.columns(2, gap="medium")
    with col_1:
        st.header("Графовое представление механизма")
        draw_joint_point(graph, labels=1)
        plt.gcf().set_size_inches(4, 4)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.header("Робот")
        components.iframe(get_visualizer(visualization_builder).viewer.url(), width=400,
                          height=400, scrolling=True)


if st.session_state.stage == 'workspace_visualization':
    gm = st.session_state.gm
    graph = gm.graph
    points = st.session_state.points
    workspace = st.session_state.workspace
    x = points[:, 0]
    y = points[:, 1]
    values = workspace.reachabilty_mask.T.flatten()
    x_0 = x[values == 0]
    y_0 = y[values == 0]
    x_1 = x[values == 1]
    y_1 = y[values == 1]
    # # Plot the points
    plt.plot(x_0, y_0, "xr")
    plt.plot(x_1, y_1, "xb")

    # trajectory setting script
    trajectory = None
    with st.sidebar:
        st.button(label="Return to joint point choice",key="return_to_joint_point_choice",on_click=lambda: st.session_state.__setitem__('stage', 'joint_point_choice'))
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
        st.button(label="run simulation", key="run_simulation",
                  on_click=run_simulation)
        with st.form(key="rewards"):
            st.header("Rewards")
            reward_mask = []
            for reward in rewards:
                reward_mask.append(st.checkbox(
                    label=reward.reward_name, value=False))
            cr = st.form_submit_button("Calculate rewards")
        

    col_1, col_2 = st.columns(2, gap="medium")
    with col_1:
        st.header("Graph representation")
        draw_joint_point(graph, draw_labels=False)
        plt.gcf().set_size_inches(4, 4)
        plt.plot(trajectory[:, 0], trajectory[:, 2], 'yo', markersize=5)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.header("Robot visualization")
        add_trajectory_to_vis(get_visualizer(
            visualization_builder), trajectory)
        components.iframe(get_visualizer(visualization_builder).viewer.url(), width=400,
                          height=400, scrolling=True)

    if st.session_state.run_simulation_flag or cr:
        calculate_and_display_rewards(trajectory, reward_mask)

    if st.session_state.run_simulation_flag:
        ik_manager = TrajectoryIKManager()
        # fixed_robot, free_robot = jps_graph2pinocchio_robot(gm.graph, builder)
        fixed_robot, _ = jps_graph2pinocchio_meshes_robot(
            graph, visualization_builder)
        ik_manager.register_model(
            fixed_robot.model, fixed_robot.constraint_models, fixed_robot.visual_model
        )
        ik_manager.set_solver("Closed_Loop_PI")
        # with st.status("simulation..."):
        _ = ik_manager.follow_trajectory(
            trajectory, viz=get_visualizer(visualization_builder)
        )
        time.sleep(1)
        get_visualizer(visualization_builder).display(
            pin.neutral(fixed_robot.model))
        st.session_state.run_simulation_flag = False

    # trajectory_choice = st.sidebar.radio(
    #     label='Select trajectory for criteria evaluation', options=trajectories.keys(), key="trajectory_choice")
