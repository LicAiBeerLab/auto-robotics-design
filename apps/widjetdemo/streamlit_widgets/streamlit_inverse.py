import streamlit as st
import os
from copy import deepcopy
import numpy as np
from auto_robot_design.utils.configs import get_mesh_builder, get_standard_builder, get_standard_rewards
from streamlit_widget_auxiliary import get_visualizer, send_graph_to_visualizer
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from pathlib import Path
from auto_robot_design.user_interface.check_in_ellips import Ellipse, check_points_in_ellips
from auto_robot_design.motion_planning.bfs_ws import Workspace, BreadthFirstSearchPlanner
from matplotlib.patches import Circle
from auto_robot_design.motion_planning.dataset_generator import Dataset
from auto_robot_design.motion_planning.many_dataset_api import ManyDatasetAPI 
from auto_robot_design.generator.topologies.graph_manager_2l import plot_2d_bounds, MutationType
# preparations
@st.cache_resource
def get_items():
    gms = {f"Топология_{i}": get_preset_by_index_with_bounds(i) for i in range(9) if i not in [0,1,2,3,4,6,7]}
    return gms, get_mesh_builder(jupyter=False), get_standard_builder(), get_standard_rewards()

dataset_paths = ["./top_5", "./top_8"]
graph_managers, visualization_builder, standard_builder, standard_rewards = get_items()

st.title("Генерация механизмов по заданной рабочей области")
# starting stage
if not hasattr(st.session_state, "stage"):
    st.session_state.stage = 'class_choice'
    st.session_state.gm = get_preset_by_index_with_bounds(-1)

def type_choice(t):
    if t == 'free':
        st.session_state.type = 'free'
    elif t == 'suspension':
        st.session_state.type = 'suspension'
    elif t == 'manipulator':
        st.session_state.type = 'manipulator'
    st.session_state.stage = 'topology_choice'

# chose the class of optimization
if st.session_state.stage == 'class_choice':
    col_1, col_2, col_3 = st.columns(3, gap="medium")
    with col_1:
        st.button(label='свободный выбор', key='free',on_click=type_choice, args=['free'])
        st.image('./apps/rogue.jpg')
    with col_2:
        st.button(label='подвеска', key='suspension',on_click=type_choice, args=['suspension'])
        st.image('./apps/wizard.jpg')
    with col_3:
        st.button(label='манипулятор', key='manipulator',on_click=type_choice, args=['manipulator'])
        st.image('./apps/warrior.jpg')

def confirm_topology(topology_list, topology_mask):
    """Confirm the selected topology and move to the next stage."""
    if len(topology_list) == 1:
        st.session_state.stage = 'jp_ranges'
        st.session_state.gm = topology_list[0][1]
        st.session_state.gm_clone = deepcopy(st.session_state.gm)
        st.session_state.datasets = [x for  x in dataset_paths if topology_mask[i] is True]
    else:
        st.session_state.gm_clone = deepcopy(st.session_state.gm)
        st.session_state.stage = "ellipsoid"
        st.session_state.datasets = [x for  x in dataset_paths if topology_mask[i] is True]

    # create a deep copy of the graph manager for further updates
    st.session_state.topology_list = topology_list
    st.session_state.topology_mask = topology_mask


# def topology_choice():
#     """Update the graph manager based on the selected topology."""
#     st.session_state.gm = graph_managers[st.session_state.topology_choice]

if st.session_state.stage == "topology_choice":
    with st.sidebar:
        # st.radio(label="Select topology:", options=graph_managers.keys(),
        #          index=None, key='topology_choice', on_change=topology_choice)
        st.header("Выбор топологии")
        st.write("При выборе только одной топологии доступна опция выбора границ для параметров генерации")
        topology_mask = []
        for i, gm in enumerate(graph_managers.items()):
            topology_mask.append(st.checkbox(label=gm[0], value=True))
        chosen_topology_list=[x for i, x in enumerate(graph_managers.items()) if topology_mask[i] is True]

        if sum(topology_mask)>0:
            st.button(label='Подтвердить выбор', key='confirm_topology',
                    on_click=confirm_topology, args=[chosen_topology_list, topology_mask])

    plt.figure(figsize=(5, 5))
    st.header("Выбранные топологии")
    for i, gm_t in enumerate(chosen_topology_list):
        gm = gm_t[1]
        plt.subplot((sum(topology_mask)//3)+1,3,i+1)
        gm.get_graph(gm.generate_central_from_mutation_range())
        draw_joint_point(gm.graph, labels=0)
        plt.title(gm_t[0])
    # plt.gcf().set_size_inches(15, 15)
    st.pyplot(plt.gcf(), clear_figure=True, use_container_width=False)
    # if sum(topology_mask)==1:
    #     gm = st.session_state.gm
    #     values = gm.generate_central_from_mutation_range()
    #     graph = st.session_state.gm.get_graph(values)
    #     send_graph_to_visualizer(graph, visualization_builder)
    #     col_1, col_2 = st.columns(2, gap="medium")
    #     with col_1:
    #         st.header("Graph representation")
    #         draw_joint_point(graph)
    #         plt.gcf().set_size_inches(4, 4)
    #         st.pyplot(plt.gcf(), clear_figure=True)
    #     with col_2:
    #         st.header("Robot visualization")
    #         components.iframe(get_visualizer(visualization_builder).viewer.url(), width=400,
    #                         height=400, scrolling=True)
def confirm_ranges():
    """Confirm the selected ranges and move to the next stage."""
    st.session_state.stage = "ellipsoid"
    gm_clone = st.session_state.gm_clone
    for key, value in gm_clone.generator_dict.items():
        for i, values in enumerate(value.mutation_range):
            if values is None:
                continue
            if values[0] == values[1]:
                current_fp = gm.generator_dict[key].freeze_pos
                current_fp[i] = values[0]
                gm_clone.freeze_joint(key, current_fp)

    for key, value in gm.generator_dict.items():
        print(gm.generator_dict[key].freeze_pos)
    gm_clone.set_mutation_ranges()
    # print(gm.mutation_ranges)


def return_to_topology():
    """Return to the topology choice stage."""
    st.session_state.stage = "topology_choice"

if st.session_state.stage == 'jp_ranges':
    initial_generator_info = st.session_state.gm.generator_dict
    gm = st.session_state.gm_clone
    generator_info = gm.generator_dict
    with st.sidebar:
        # return button
        st.button(label="Return to topology choice",
                  key="return_to_topology", on_click=return_to_topology)
        for jp, gen_info in generator_info.items():
            for i, mut_range in enumerate(gen_info.mutation_range):
                # i is from 0 to 2, 0 is x, 1 is y, 2 is z. None value means that the joint is fixed alone an axis
                if mut_range is None:
                    continue
                # create a toggle for each moveable axis. If toggle is off the coordinate is fixed to the value and should be freezed
                if i == 0:
                    name = f"{jp.name}_x"
                    current_on = st.toggle(
                        f"Activate feature "+name, value=True)
                elif i == 1:
                    name = f"{jp.name}_y"
                    current_on = st.toggle(
                        f"Activate feature "+name, value=True)
                else:
                    name = f"{jp.name}_z"
                    current_on = st.toggle(
                        f"Activate feature "+name, value=True)
                # initial values constrain slider range. The same jp can be used in both dicts because of the hash function type, joint copies have the same hash
                init_values = initial_generator_info[jp].mutation_range[i]
                if current_on:
                    gen_info.mutation_range[i] = st.slider(
                        label=name, min_value=init_values[0], max_value=init_values[1], value=(init_values[0], init_values[1]))
                else:
                    current_value = st.number_input(label="Insert a value", value=(
                        init_values[0] + init_values[1])/2, key=name)
                    # at further stage the same values will be used to signal for joint freezing
                    gen_info.mutation_range[i] = (
                        current_value, current_value)

        st.button(label="Confirm optimization ranges",
                  key='ranges_confirm', on_click=confirm_ranges)
    # here should be some kind of visualization for ranges
    gm.set_mutation_ranges()
    center = gm.generate_central_from_mutation_range()
    graph = gm.get_graph(center)
    draw_joint_point(graph)
    # here gm is a clone
    plot_2d_bounds(gm)
    st.pyplot(plt.gcf(), clear_figure=True)
    # this way we set ranges after each step, but without freezing joints
    st.write(gm.mutation_ranges)

def reward_choice():
    st.session_state.stage = 'rewards'

if st.session_state.stage == "ellipsoid":
    st.header("Выбор рабочего пространства")
    with st.sidebar:
        with st.form(key = 'ellipse'):
            x = st.slider(label="х координата центра", min_value=-0.3,
                            max_value=0.3, value=0.0)
            y = st.slider(label="y координата центра", min_value=-0.4,
                            max_value=-0.2, value=-0.3)
            x_rad = st.slider(label="х радиус", min_value=0.02,
                            max_value=0.3, value=0.1)
            y_rad = st.slider(label="y радиус", min_value=0.02,
                            max_value=0.3, value=0.1)
            angle = st.slider(label='наклон', min_value = 0, max_value=180, value = 0)
            st.form_submit_button(label="Задать рабочее пространство")
        st.button(label = "Перейти к целевой функции", key='rewards', on_click=reward_choice)
    st.session_state.ellipsoid_params = [x, y, x_rad, y_rad, angle]
    ellipse = Ellipse(np.array([x, y]), np.deg2rad(angle), np.array([x_rad, y_rad]))
    point_ellipse = ellipse.get_points()
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
    workspace_obj = Workspace(None, bounds, np.array([0.01, 0.01]))
    points = workspace_obj.points
    mask = check_points_in_ellips(points, ellipse, 0.02)
    rev_mask = np.array(1 - mask, dtype="bool")
    plt.figure(figsize=(10, 10))
    plt.plot(point_ellipse[0, :], point_ellipse[1, :], "g", linewidth=1)
    plt.scatter(points[rev_mask, :][:, 0], points[rev_mask, :][:, 1],s=2)
    plt.scatter(points[mask, :][:, 0], points[mask, :][:, 1],s=2)
    graph = st.session_state.gm.get_graph(st.session_state.gm.generate_central_from_mutation_range())
    draw_joint_point(graph, labels=0)
    plt.gcf().set_size_inches(4, 4)
    st.pyplot(plt.gcf(), clear_figure=True)

def generate():
    st.session_state.stage = 'generate'

if st.session_state.stage == 'rewards':
    x, y, x_rad, y_rad, angle = st.session_state.ellipsoid_params
    ellipse = Ellipse(np.array([x, y]), np.deg2rad(angle), np.array([x_rad, y_rad]))
    point_ellipse = ellipse.get_points()
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
    workspace_obj = Workspace(None, bounds, np.array([0.01, 0.01]))
    points = workspace_obj.points
    mask = check_points_in_ellips(points, ellipse, 0.02)
    rev_mask = np.array(1 - mask, dtype="bool")
    plt.figure(figsize=(10, 10))
    plt.plot(point_ellipse[0, :], point_ellipse[1, :], "g", linewidth=1)
    plt.scatter(points[rev_mask, :][:, 0], points[rev_mask, :][:, 1],s=2)
    plt.scatter(points[mask, :][:, 0], points[mask, :][:, 1],s=2)
    with st.sidebar:
        st.header('Выбор точки вычисления')
        x_p = st.slider(label="х координата центра", min_value=-0.25,
                        max_value=0.25, value=0.0)
        y_p = st.slider(label="y координата центра", min_value=-0.42,
                        max_value=0., value=-0.21)
        if st.session_state.type == 'free':
            rewards = standard_rewards.values()
            chosen_reward = st.radio(label='Выбор целевой функции', options=rewards, index=0, format_func=lambda x: x.reward_name)
            st.session_state.chosen_reward = chosen_reward
        if st.session_state.type == 'suspension':
            rewards = list(standard_rewards.values())[:3]
            chosen_reward = st.radio(label='Выбор целевой функции', options=rewards, index=0, format_func=lambda x: x.reward_name)
            st.session_state.chosen_reward = chosen_reward
        
        st.button(label='Сгенерировать механизмы', key='generate',on_click=generate)
        
    st.session_state.point = [x_p, y_p]
    Drawing_colored_circle = Circle((x_p, y_p),radius=0.02, color='r')
    plt.gca().add_artist( Drawing_colored_circle)
    plt.gcf().set_size_inches(4, 4)
    plt.gca().axes.set_aspect( 1 )
    st.pyplot(plt.gcf(), clear_figure=True)

from auto_robot_design.pinokla.default_traj import add_auxilary_points_to_trajectory
from auto_robot_design.motion_planning.dataset_generator import (
    Dataset,
    set_up_reward_manager,
)
if st.session_state.stage == 'generate':
    dataset_api = ManyDatasetAPI(st.session_state.datasets)
    # dataset = Dataset("./top_5")
    x, y, x_rad, y_rad, angle = st.session_state.ellipsoid_params
    ellipse = Ellipse(np.array([x, y]), np.deg2rad(angle), np.array([x_rad, y_rad]))
    index_list=dataset_api.get_indexes_cover_ellipse(ellipse)
    print(len(index_list))
    des_point = np.array(st.session_state.point)
    traj = np.array(
        add_auxilary_points_to_trajectory(([des_point[0]], [des_point[1]]))
    ).T
    test_ws = dataset_api.datasets[0].get_workspace_by_indexes([0])[0]
    traj_6d = test_ws.robot.motion_space.get_6d_traj(traj)
    reward_manager = set_up_reward_manager(traj_6d, st.session_state.chosen_reward)
    sorted_indexes = dataset_api.sorted_indexes_by_reward(
        index_list, 9, reward_manager
    )
    print(sorted_indexes)
    graphs = []
    for topology_idx, index, value in sorted_indexes:
        gm = dataset_api.datasets[topology_idx].graph_manager
        x = dataset_api.datasets[topology_idx].get_design_parameters_by_indexes([index])
        graph = gm.get_graph(x[0])
        graphs.append(graph)
    print(len(graphs), len(sorted_indexes))
    for i, graph in enumerate(graphs):
        plt.subplot(3,6,i+1)
        draw_joint_point(graph, labels=0)
    plt.gcf().set_size_inches(10, 10)
    st.pyplot(plt.gcf(), clear_figure=True)
    # print(index_list)
    # x = dataset.get_design_parameters_by_indexes([index_list[-1]])
    # print(x)
