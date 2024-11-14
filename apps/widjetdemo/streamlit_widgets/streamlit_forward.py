import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import streamlit as st
import streamlit.components.v1 as components
from forward_init import add_trajectory_to_vis, build_constant_objects, get_russian_reward_description
from streamlit_widget_auxiliary import get_visualizer, send_graph_to_visualizer

from auto_robot_design.description.builder import jps_graph2pinocchio_robot_3d_constraints
from auto_robot_design.description.mesh_builder.mesh_builder import (
    jps_graph2pinocchio_meshes_robot)
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.generator.topologies.bounds_preset import \
    get_preset_by_index_with_bounds
from auto_robot_design.motion_planning.bfs_ws import (
    BreadthFirstSearchPlanner, Workspace)
from auto_robot_design.motion_planning.trajectory_ik_manager import \
    TrajectoryIKManager
from auto_robot_design.pinokla.default_traj import (
    add_auxilary_points_to_trajectory, convert_x_y_to_6d_traj_xz,
    create_simple_step_trajectory, get_vertical_trajectory)

# build and cache constant objects
graph_managers, optimization_builder, visualization_builder, crag, reward_dict = build_constant_objects()
reward_description = get_russian_reward_description()
st.title("Оценка рычажных механизмов")
# create gm variable that will be used to store the current graph manager and set it to be update for a session
if 'gm' not in st.session_state:
    st.session_state.gm = get_preset_by_index_with_bounds(-1)
    # the session variable for chosen topology, it gets a value after topology confirmation button is clicked
    st.session_state.stage = 'topology_choice'
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
    col_1, col_2 = st.columns(2, gap="medium")
    with col_1:
        st.header("Граф выбранной топологии")
        draw_joint_point(graph, labels=2, draw_lines=True)
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
    graph = gm.graph
    labels = {n:i for i,n in enumerate(graph.nodes())}
    with st.sidebar.form("jp_coordinates"):
        st.header('Выбор положений сочленений')
        for key, value in mut_ranges.items():
            slider = st.slider(
                label=str(labels[key[0]])+'_'+key[1], min_value=value[0], max_value=value[1], value=(value[1]+value[0])/2)
            current_values.append(slider)
        st.form_submit_button('Внести изменения')
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

def to_trajectory_choice():
    st.session_state.stage = 'trajectory_choice'


def run_simulation():
    st.session_state.run_simulation_flag = True

def calculate_and_display_rewards(trajectory, reward_mask):
    gm = st.session_state.gm
    fixed_robot, free_robot = jps_graph2pinocchio_robot_3d_constraints(
        gm.graph, optimization_builder)
    point_criteria_vector, trajectory_criteria, res_dict_fixed = crag.get_criteria_data(
        fixed_robot, free_robot, trajectory, viz=None)
    some_text = """ Критерии представлены в виде поточечных значений вдоль траектории. """
    st.text(some_text)
    for i, reward in enumerate(reward_dict.items()):
        if reward_mask[i]:
            try:
                calculate_result = reward[1].calculate(
                    point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator=optimization_builder.actuator['default'])
                # st.text(reward_description[reward[0]][0]+":\n   " )
                reward_vector = np.array(calculate_result[1])
                plt.gcf().set_figheight(2.5)
                plt.gcf().set_figwidth(2.5)
                plt.plot(reward_vector)
                plt.xticks(fontsize=4)
                plt.yticks(fontsize=4)
                plt.xlabel('шаг траектории', fontsize=6)
                plt.ylabel('значение критерия на шаге', fontsize=6)
                plt.title(reward_description[reward[0]][0], fontsize=8)
                plt.legend([f'Итоговое значение критерия: {calculate_result[0]:.2f}'], fontsize=4)

                st.pyplot(plt.gcf(), clear_figure=True, use_container_width=False)
            except ValueError:
                st.text_area(
                    label="", value="Траектория содержит точки за пределами рабочего пространства. Для рассчёта критериев укажите траекторию внутри рабочей области.")
                break

if st.session_state.stage == 'workspace_visualization':
    st.text("Жёлтая область - рабочее пространство механизма\nКрасные область - недостижимые точки\nВсе критерии рассчитываются вдоль траектории и для успешного рассчёта необходимо,\nчтобы траектория лежала внутри рабочей области.")
    st.text("Выберите траекторию для оценки критериев:")
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
    plt.plot(x_0, y_0, "sr", markersize=3)
    plt.plot(x_1, y_1, "sy", markersize=3)

    # trajectory setting script
    trajectory = None
    with st.sidebar:
        st.button(label="Вернуться к выбору механизма",key="return_to_joint_point_choice",on_click=lambda: st.session_state.__setitem__('stage', 'joint_point_choice'))
        trajectory_type = st.radio(label='Выберите тип траектории', options=[
            "vertical", "step"], index=None, key="trajectory_type")
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
        if trajectory_type is not None:
            st.button(label="Симуляция движения по траектории", key="run_simulation",
                    on_click=run_simulation)
            with st.form(key="rewards"):
                st.header("Критерии")
                reward_mask = []
                for key, reward in reward_dict.items():
                    reward_mask.append(st.checkbox(
                        label=reward_description[key][0], value=False,help=reward_description[key][1]))
                cr = st.form_submit_button("Рассчитать значения выбранных критериев")
        

    col_1, col_2 = st.columns(2, gap="medium")
    with col_1:
        st.header("Графовое представление механизма")
        draw_joint_point(graph, labels=2, draw_legend=False)
        plt.gcf().set_size_inches(6, 6)
        if trajectory_type is not None:
            plt.plot(trajectory[50:, 0], trajectory[50:, 2], 'green', markersize=2)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.header("Робот")
        if trajectory_type is not None: add_trajectory_to_vis(get_visualizer(visualization_builder), trajectory[50:])
        components.iframe(get_visualizer(visualization_builder).viewer.url(), width=400,
                          height=400, scrolling=True)

    if trajectory_type is not None:
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
        _ = ik_manager.follow_trajectory(
            trajectory, viz=get_visualizer(visualization_builder)
        )
        time.sleep(1)
        get_visualizer(visualization_builder).display(
            pin.neutral(fixed_robot.model))
        st.session_state.run_simulation_flag = False
