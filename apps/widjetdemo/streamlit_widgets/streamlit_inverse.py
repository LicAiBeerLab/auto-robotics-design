import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import streamlit as st
import streamlit.components.v1 as components
from forward_init import (
    add_trajectory_to_vis,
    build_constant_objects,
    get_russian_reward_description,
)
from matplotlib.patches import Circle
from streamlit_widget_auxiliary import get_visualizer, send_graph_to_visualizer

from auto_robot_design.description.builder import (
    jps_graph2pinocchio_robot_3d_constraints,
)
from auto_robot_design.description.mesh_builder.mesh_builder import (
    jps_graph2pinocchio_meshes_robot,
)
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.generator.topologies.bounds_preset import (
    get_preset_by_index_with_bounds,
)
from auto_robot_design.generator.topologies.graph_manager_2l import (
    plot_2d_bounds,
    plot_one_jp_bounds,
)
from auto_robot_design.motion_planning.bfs_ws import Workspace
from auto_robot_design.motion_planning.dataset_generator import set_up_reward_manager
from auto_robot_design.motion_planning.many_dataset_api import ManyDatasetAPI
from auto_robot_design.motion_planning.trajectory_ik_manager import TrajectoryIKManager
from auto_robot_design.pinokla.default_traj import (
    add_auxilary_points_to_trajectory,
    convert_x_y_to_6d_traj_xz,
)
from auto_robot_design.user_interface.check_in_ellips import (
    Ellipse,
    SnakePathFinder,
    check_points_in_ellips,
)

# constant objects
(
    graph_managers,
    optimization_builder,
    manipulation_builder,
    suspension_builder,
    crag,
    reward_dict,
) = build_constant_objects()
reward_description = get_russian_reward_description()
general_reward_keys = [
    "actuated_inertia_matrix",
    "z_imf",
    "manipulability",
    "min_manipulability",
    "min_force",
    "trajectory_zrr",
    "dexterity",
    "min_acceleration",
    "mean_heavy_lifting",
    "min_heavy_lifting",
]
suspension_reward_keys = [
    "z_imf",
    "trajectory_zrr",
    "min_acceleration",
    "mean_heavy_lifting",
    "min_heavy_lifting",
]
manipulator_reward_keys = [
    "manipulability",
    "min_manipulability",
    "min_force",
    "dexterity",
    "min_acceleration",
]
# dataset_paths = ["./top_0", "./top_1","./top_2", "./top_3","top_4","./top_5","./top_6", "./top_7", "./top_8"]
dataset_paths = [
    "/run/media/yefim-work/Samsung_data1/top_0",
    "/run/media/yefim-work/Samsung_data1/top_1",
    "/run/media/yefim-work/Samsung_data1/top_2",
    "/run/media/yefim-work/Samsung_data1/top_3",
    "/run/media/yefim-work/Samsung_data1/top_4",
    "/run/media/yefim-work/Samsung_data1/top_5",
    "/run/media/yefim-work/Samsung_data1/top_6",
    "/run/media/yefim-work/Samsung_data1/top_7",
    "/run/media/yefim-work/Samsung_data1/top_8",
]

st.title("Генерация механизмов по заданной рабочей области")
# starting stage
if not hasattr(st.session_state, "stage"):
    st.session_state.stage = "class_choice"
    st.session_state.gm = get_preset_by_index_with_bounds(-1)
    st.session_state.run_simulation_flag = False


def type_choice(t):
    if t == "free":
        st.session_state.type = "free"
        st.session_state.visualization_builder = optimization_builder
        st.session_state.reward_keys = general_reward_keys
    elif t == "suspension":
        st.session_state.type = "suspension"
        st.session_state.visualization_builder = suspension_builder
        st.session_state.reward_keys = suspension_reward_keys
    elif t == "manipulator":
        st.session_state.type = "manipulator"
        st.session_state.visualization_builder = manipulation_builder
        st.session_state.reward_keys = manipulator_reward_keys
    st.session_state.stage = "topology_choice"


# chose the class of optimization
if st.session_state.stage == "class_choice":
    some_text = """В данном сценарии происходит генерация механизмов по заданной рабочей области.
Предлагается выбрать один из трёх типов механизмов: замкнутая кинематическая структура, 
подвеска колёсного робота, робот манипулятор.
Для каждого типа предлагается свой набор критериев, используемых при генерации
механизма и модель визуализации"""
    st.text(some_text)
    col_1, col_2, col_3 = st.columns(3, gap="medium")
    with col_1:
        st.button(
            label="замкнутая кинематическая структура",
            key="free",
            on_click=type_choice,
            args=["free"],
        )
        st.image("./apps/rogue.jpg")
    with col_2:
        st.button(
            label="подвеска",
            key="suspension",
            on_click=type_choice,
            args=["suspension"],
        )
        st.image("./apps/wizard.jpg")
    with col_3:
        st.button(
            label="манипулятор",
            key="manipulator",
            on_click=type_choice,
            args=["manipulator"],
        )
        st.image("./apps/warrior.jpg")


def confirm_topology(topology_list, topology_mask):
    """Confirm the selected topology and move to the next stage."""
    # if only one topology is chosen, there is an option to choose the optimization ranges
    if len(topology_list) == 1:
        st.session_state.stage = "jp_ranges"
        st.session_state.gm = topology_list[0][1]
        st.session_state.gm_clone = deepcopy(st.session_state.gm)
        st.session_state.current_generator_dict = deepcopy(
            st.session_state.gm.generator_dict
        )
        # st.session_state.gm_clone = deepcopy(st.session_state.gm)
        st.session_state.datasets = [
            x for x in dataset_paths if topology_mask[i] is True
        ]
    else:
        st.session_state.gm_clone = deepcopy(st.session_state.gm)
        st.session_state.stage = "ellipsoid"
        st.session_state.datasets = [
            x for i, x in enumerate(dataset_paths) if topology_mask[i] is True
        ]
    # create a deep copy of the graph manager for further updates
    st.session_state.topology_list = topology_list
    st.session_state.topology_mask = topology_mask


if st.session_state.stage == "topology_choice":
    some_text = """Предлагается выбор из девяти топологических структур механизмов.
В процессе генерации будут учитываться только выбранные топологические структуры.
Для визуализации выбора предлагаются примеры механизмов каждой структуры."""
    st.text(some_text)
    with st.sidebar:
        st.header("Выбор структуры")
        st.write(
            "При выборе только одной структуры доступна опция выбора границ для параметров генерации"
        )
        topology_mask = []
        for i, gm in enumerate(graph_managers.items()):
            topology_mask.append(st.checkbox(label=gm[0], value=True))
        chosen_topology_list = [
            x for i, x in enumerate(graph_managers.items()) if topology_mask[i] is True
        ]

        if sum(topology_mask) > 0:
            st.button(
                label="Подтвердить выбор",
                key="confirm_topology",
                on_click=confirm_topology,
                args=[chosen_topology_list, topology_mask],
            )

    plt.figure(figsize=(10, 10))
    for i in range(9):
        if i < len(chosen_topology_list):
            gm = chosen_topology_list[i][1]
            plt.subplot(3, 3, i + 1)
            gm.get_graph(gm.generate_central_from_mutation_range())
            draw_joint_point(gm.graph, labels=2, draw_legend=False)
            plt.title(chosen_topology_list[i][0])
        else:
            plt.subplot(3, 3, i + 1)
            plt.axis("off")

    st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)


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

    gm_clone.set_mutation_ranges()


def return_to_topology():
    """Return to the topology choice stage."""
    st.session_state.stage = "topology_choice"


def joint_choice():
    st.session_state.current_generator_dict = deepcopy(
        st.session_state.gm_clone.generator_dict
    )


# second stage
if st.session_state.stage == "jp_ranges":
    axis = ["x", "y", "z"]
    # form for optimization ranges. All changes affects the gm_clone and it should be used for optimization
    # initial nodes
    initial_generator_info = st.session_state.gm.generator_dict
    initial_mutation_ranges = st.session_state.gm.mutation_ranges
    gm = st.session_state.gm_clone
    generator_info = gm.generator_dict
    graph = gm.graph
    labels = {n: i for i, n in enumerate(graph.nodes())}
    with st.sidebar:
        # return button
        st.button(
            label="Назад к выбору топологии",
            key="return_to_topology",
            on_click=return_to_topology,
        )

        # set of joints that have mutation range in initial generator and get current jp and its index on the graph picture

        mutable_jps = [key[0] for key in initial_mutation_ranges.keys()]
        options = [(jp, idx) for jp, idx in labels.items() if jp in mutable_jps]
        current_jp = st.radio(
            label="Выбор сочленения для установки границ",
            options=options,
            index=0,
            format_func=lambda x: x[1],
            key="joint_choice",
            on_change=joint_choice,
        )
        # we can get current jp generator info in the cloned gm which contains all the changes
        current_generator_info = generator_info[current_jp[0]]
        for i, mut_range in enumerate(current_generator_info.mutation_range):
            if mut_range is None:
                continue
            # we can get mutation range from previous activation of the corresponding radio button
            left_value, right_value = st.session_state.current_generator_dict[
                current_jp[0]
            ].mutation_range[i]
            name = f"{labels[current_jp[0]]}_{axis[i]}"
            toggle_value = not left_value == right_value
            current_on = st.toggle(f"Отключить оптимизацию " + name, value=toggle_value)
            init_values = initial_generator_info[current_jp[0]].mutation_range[i]
            if current_on:
                mut_range = st.slider(
                    label=name,
                    min_value=init_values[0],
                    max_value=init_values[1],
                    value=(left_value, right_value),
                )
                generator_info[current_jp[0]].mutation_range[i] = mut_range
            else:
                current_value = st.number_input(
                    label="Insert a value",
                    value=(left_value + right_value) / 2,
                    key=name,
                    min_value=init_values[0],
                    max_value=init_values[1],
                )
                # if current_value < init_values[0]:
                #     current_value = init_values[0]
                # if current_value > init_values[1]:
                #     current_value = init_values[1]
                mut_range = (current_value, current_value)
                generator_info[current_jp[0]].mutation_range[i] = mut_range

        st.button(
            label="подтвердить диапазоны оптимизации",
            key="ranges_confirm",
            on_click=confirm_ranges,
        )
    # here should be some kind of visualization for ranges
    gm.set_mutation_ranges()
    plot_one_jp_bounds(gm, current_jp[0].name)
    center = gm.generate_central_from_mutation_range()
    graph = gm.get_graph(center)
    # here I can insert the visualization for jp bounds

    draw_joint_point(graph, labels=1, draw_legend=False, draw_lines=True)
    # here gm is a clone

    # plot_2d_bounds(gm)
    st.pyplot(plt.gcf(), clear_figure=True)
    # this way we set ranges after each step, but without freezing joints
    some_text = """Диапазоны оптимизации определяют границы пространства поиска механизмов в процессе 
оптимизации. x - горизонтальные координаты, z - вертикальные координаты.
Отключенные координаты не будут участвовать в оптимизации и будут иметь постоянные 
значения во всех механизмах."""
    st.text(some_text)
    # st.text("x - горизонтальные координаты, z - вертикальные координаты")


def reward_choice():
    st.session_state.stage = "rewards"


if st.session_state.stage == "ellipsoid":
    some_text = """Задайте необходимую рабочую область для генерации механизмов.
Рабочее пространство всех сгенерированных решений будет включать заданную область.
Область задаётся в виде эллипса, определяемого своим центром, радиусами и углом."""
    st.text(some_text)
    with st.sidebar:
        st.header("Выбор рабочего пространства")
        with st.form(key="ellipse"):
            x = st.slider(
                label="х координата центра", min_value=-0.3, max_value=0.3, value=0.0
            )
            y = st.slider(
                label="y координата центра", min_value=-0.4, max_value=-0.2, value=-0.33
            )
            x_rad = st.slider(
                label="х радиус", min_value=0.02, max_value=0.3, value=0.06
            )
            y_rad = st.slider(
                label="y радиус", min_value=0.02, max_value=0.3, value=0.05
            )
            angle = st.slider(label="наклон", min_value=0, max_value=180, value=0)
            st.form_submit_button(label="Задать рабочее пространство")
        st.button(
            label="Перейти к целевой функции", key="rewards", on_click=reward_choice
        )
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
    plt.scatter(points[rev_mask, :][:, 0], points[rev_mask, :][:, 1], s=2)
    plt.scatter(points[mask, :][:, 0], points[mask, :][:, 1], s=2)
    graph = st.session_state.gm.get_graph(
        st.session_state.gm.generate_central_from_mutation_range()
    )
    draw_joint_point(graph, labels=2, draw_legend=False)
    plt.gcf().set_size_inches(4, 4)
    st.pyplot(plt.gcf(), clear_figure=True)


def generate():
    st.session_state.stage = "generate"


if st.session_state.stage == "rewards":
    some_text = """Укажите критерий оценки для обтбора лучших механизмов.
Необходимо задать точку рассчёта критерия в рабочей области механизма.
Используйте боковую панель для установки точки расчёта."""
    st.text(some_text)
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
    st.session_state.ws = workspace_obj
    points = workspace_obj.points
    mask = check_points_in_ellips(points, ellipse, 0.02)
    rev_mask = np.array(1 - mask, dtype="bool")
    plt.figure(figsize=(10, 10))
    plt.plot(point_ellipse[0, :], point_ellipse[1, :], "g", linewidth=1)
    plt.scatter(points[rev_mask, :][:, 0], points[rev_mask, :][:, 1], s=2)
    plt.scatter(points[mask, :][:, 0], points[mask, :][:, 1], s=2)
    with st.sidebar:
        st.header("Выбор точки вычисления")
        x_p = st.slider(
            label="х координата", min_value=-0.25, max_value=0.25, value=0.0
        )
        y_p = st.slider(
            label="y координата", min_value=-0.42, max_value=0.0, value=-0.3
        )
        if st.session_state.type == "free":
            rewards = list(reward_dict.items())
            chosen_reward_idx = st.radio(
                label="Выбор целевой функции",
                options=range(len(rewards)),
                index=0,
                format_func=lambda x: reward_description[rewards[x][0]][0],
            )
            st.session_state.chosen_reward = rewards[chosen_reward_idx][1]
        if st.session_state.type == 'suspension':
            rewards = list(reward_dict.items())
            chosen_reward_idx = st.radio(label='Выбор целевой функции', options=range(len(rewards)), index=0, format_func=lambda x: reward_description[rewards[x][0]][0])
            st.session_state.chosen_reward = rewards[chosen_reward_idx][1]
        if st.session_state.type == "manipulator":
            rewards = list(reward_dict.items())
            chosen_reward_idx = st.radio(label='Выбор целевой функции', options=range(len(rewards)), index=0, format_func=lambda x: reward_description[rewards[x][0]][0])
            st.session_state.chosen_reward = rewards[chosen_reward_idx][1]
        st.button(label="Сгенерировать механизмы", key="generate", on_click=generate)
    st.session_state.point = [x_p, y_p]
    Drawing_colored_circle = Circle((x_p, y_p), radius=0.01, color="r")
    plt.gca().add_artist(Drawing_colored_circle)
    plt.gcf().set_size_inches(4, 4)
    plt.gca().axes.set_aspect(1)
    st.pyplot(plt.gcf(), clear_figure=True)


def show_results():
    st.session_state.stage = "results"


def reset():
    delattr(st.session_state, "stage")


if st.session_state.stage == "generate":
    empt = st.empty()
    with empt:
        st.image(str(Path("./apps/widjetdemo/loading.gif").absolute()))
    dataset_api = ManyDatasetAPI(st.session_state.datasets)

    x, y, x_rad, y_rad, angle = st.session_state.ellipsoid_params
    ellipse = Ellipse(np.array([x, y]), np.deg2rad(angle), np.array([x_rad, y_rad]))
    index_list = dataset_api.get_indexes_cover_ellipse(ellipse)
    print(len(index_list))
    des_point = np.array(st.session_state.point)
    traj = np.array(
        add_auxilary_points_to_trajectory(([des_point[0]], [des_point[1]]))
    ).T
    dataset = dataset_api.datasets[0]
    graph = dataset.graph_manager.get_graph(
        dataset.graph_manager.generate_random_from_mutation_range()
    )
    robot, __ = jps_graph2pinocchio_robot_3d_constraints(graph, dataset.builder)
    traj_6d = robot.motion_space.get_6d_traj(traj)
    reward_manager = set_up_reward_manager(traj_6d, st.session_state.chosen_reward)
    sorted_indexes = dataset_api.sorted_indexes_by_reward(
        index_list, 10, reward_manager
    )
    if len(sorted_indexes) == 0:
        st.markdown(
            """Для заданного рабочего пространства и топологий не удалось найти решений, рекомендуется изменить требуемую рабочую область и/или топологии"""
        )
        st.button(label="Перезапуск сценария", on_click=reset)
    else:
        n = min(len(sorted_indexes), 10)
        graphs = []
        for topology_idx, index, value in sorted_indexes[:n]:
            gm = dataset_api.datasets[topology_idx].graph_manager
            x = dataset_api.datasets[topology_idx].get_design_parameters_by_indexes(
                [index]
            )
            graph = gm.get_graph(x[0])
            graphs.append(deepcopy(graph))
        st.session_state.graphs = graphs
        with empt:
            st.button(
                label="Результаты генерации", key="show_results", on_click=show_results
            )


def run_simulation(**kwargs):
    st.session_state.run_simulation_flag = True


if st.session_state.stage == "results":
    vis_builder = st.session_state.visualization_builder
    idx = st.select_slider(
        label="Лучшие по заданному критерию механизмы:",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        value=1,
        help="двигайте ползунок чтобы выбрать один из 10 лучших дизайнов",
    )
    graph = st.session_state.graphs[idx - 1]
    send_graph_to_visualizer(graph, vis_builder)
    col_1, col_2 = st.columns(2, gap="medium")
    x, y, x_rad, y_rad, angle = st.session_state.ellipsoid_params
    ellipse = Ellipse(np.array([x, y]), np.deg2rad(angle), np.array([x_rad, y_rad]))
    points_on_ellps = ellipse.get_points(0.1).T
    ws = st.session_state.ws
    reach_ws_points = ws.points
    mask_ws_n_ellps = check_points_in_ellips(reach_ws_points, ellipse, 0.1)
    # plt.plot(points_on_ellps[:,0], points_on_ellps[:,1], "r", linewidth=3)
    # plt.scatter(pts[:,0],pts[:,1])
    snake_finder = SnakePathFinder(
        points_on_ellps[0], ellipse, coef_reg=np.prod(ws.resolution)
    )  # max_len_btw_pts= np.linalg.norm(dataset.workspace.resolution),
    traj = snake_finder.create_snake_traj(reach_ws_points[mask_ws_n_ellps, :])

    final_trajectory = convert_x_y_to_6d_traj_xz(
        *add_auxilary_points_to_trajectory((traj[:, 0], traj[:, 1]))
    )

    with col_1:
        st.header("Графовое представление")
        draw_joint_point(graph, labels=2, draw_legend=False)
        rev_mask = np.array(1 - mask_ws_n_ellps, dtype="bool")
        plt.plot(points_on_ellps[:, 0], points_on_ellps[:, 1], "g")
        plt.scatter(
            reach_ws_points[rev_mask, :][:, 0], reach_ws_points[rev_mask, :][:, 1], s=2
        )
        plt.scatter(
            reach_ws_points[mask_ws_n_ellps, :][:, 0],
            reach_ws_points[mask_ws_n_ellps, :][:, 1],
        )
        plt.plot(traj[:, 0], traj[:, 1], "r")
        plt.gcf().set_size_inches(4, 4)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.header("Робот")
        add_trajectory_to_vis(get_visualizer(vis_builder), final_trajectory)
        components.iframe(
            get_visualizer(vis_builder).viewer.url(),
            width=400,
            height=400,
            scrolling=True,
        )
    st.button(
        label="Визуализация движения", key="run_simulation", on_click=run_simulation
    )
    if st.session_state.type == "free":
        if st.session_state.run_simulation_flag:
            ik_manager = TrajectoryIKManager()
            # fixed_robot, free_robot = jps_graph2pinocchio_robot(gm.graph, builder)
            fixed_robot, _ = jps_graph2pinocchio_robot_3d_constraints(
                graph, vis_builder
            )
            ik_manager.register_model(
                fixed_robot.model,
                fixed_robot.constraint_models,
                fixed_robot.visual_model,
            )
            ik_manager.set_solver("Closed_Loop_PI")
            # with st.status("simulation..."):
            _ = ik_manager.follow_trajectory(
                final_trajectory, viz=get_visualizer(vis_builder)
            )
            time.sleep(1)
            get_visualizer(vis_builder).display(pin.neutral(fixed_robot.model))
            st.session_state.run_simulation_flag = False
    else:
        if st.session_state.run_simulation_flag:
            ik_manager = TrajectoryIKManager()
            # fixed_robot, free_robot = jps_graph2pinocchio_robot(gm.graph, builder)
            fixed_robot, _ = jps_graph2pinocchio_meshes_robot(graph, vis_builder)
            ik_manager.register_model(
                fixed_robot.model,
                fixed_robot.constraint_models,
                fixed_robot.visual_model,
            )
            ik_manager.set_solver("Closed_Loop_PI")
            # with st.status("simulation..."):
            _ = ik_manager.follow_trajectory(
                final_trajectory, viz=get_visualizer(vis_builder)
            )
            time.sleep(1)
            get_visualizer(vis_builder).display(pin.neutral(fixed_robot.model))
            st.session_state.run_simulation_flag = False
