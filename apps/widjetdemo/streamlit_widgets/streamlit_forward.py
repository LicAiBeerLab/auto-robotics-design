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

full_crag, rewards, motor = set_criteria_and_crag()
# build and cache constant objects
graph_managers, optimization_builder, visualization_builder, crag, workspace_trajectory, ground_symmetric_step1, ground_symmetric_step2, ground_symmetric_step3, central_vertical, left_vertical, right_vertical = build_constant_objects()
trajectories = {"ground_symmetric_step1": ground_symmetric_step1,
                "ground_symmetric_step2": ground_symmetric_step2,
                "ground_symmetric_step3": ground_symmetric_step3,
                "central_vertical": central_vertical,
                "left_vertical": left_vertical,
                "right_vertical": right_vertical}
# set widget title
st.title("robotic leg evaluation")
# create gm variable that will be used to store the current graph manager and set it to be update for a session
if 'gm' not in st.session_state:
    st.session_state.gm = get_preset_by_index_with_bounds(-1)
    # the session variable for chosen topology, it gets a value after topology confirmation button is clicked
    st.session_state.topology = None
    st.session_state.workspace = None
    st.session_state.trajectory_choice = False


# the function for getting visualizer. At first call it creates a visualizer object and caches it, at any further call it returns the cached object
@st.cache_resource
def get_visualizer():
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
    # visualizer.viewer["/Background"].set_property("visible", False)
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


def send_graph_to_visualizer(graph):
    fixed_robot, _ = jps_graph2pinocchio_meshes_robot(
        graph, visualization_builder)
    visualizer = get_visualizer()
    visualizer.model = fixed_robot.model
    visualizer.collision = fixed_robot.visual_model
    visualizer.visual_model = fixed_robot.visual_model
    visualizer.rebuildData()
    visualizer.clean()
    visualizer.loadViewerModel()
    visualizer.display(pin.neutral(fixed_robot.model))


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
    send_graph_to_visualizer(graph)
    with col_1:
        st.header("Graph representation")
        draw_joint_point(graph)
        plt.gcf().set_size_inches(4, 4)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.header("Robot visualization")
        components.iframe(get_visualizer().viewer.url(), width=400,
                          height=400, scrolling=True)


def evaluate_construction():
    st.session_state.workspace = None
    gm = st.session_state.gm
    graph = gm.graph
    robo, __ = jps_graph2pinocchio_robot_3d_constraints(
        graph, builder=optimization_builder)
    size_box_bound = np.array([0.5, 0.5])
    center_bound = np.array([0, -0.4])
    bounds = np.array(
        [
            [-size_box_bound[0] / 2 - 0.001, size_box_bound[0] / 2],
            [-size_box_bound[1] / 2, size_box_bound[1] / 2],
        ]
    )
    bounds[0, :] += center_bound[0]
    bounds[1, :] += center_bound[1]
    start_pos = center_bound
    q = np.zeros(robo.model.nq)
    workspace = Workspace(robo, bounds, np.array([0.01, 0.01]))
    ws_bfs = BreadthFirstSearchPlanner(workspace, 0)
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
    x = points[:, 0]
    y = points[:, 1]
    values = workspace.reachabilty_mask.T.flatten()
    x_0 = x[values == 0]
    y_0 = y[values == 0]
    x_1 = x[values == 1]
    y_1 = y[values == 1]
    # # Plot the points
    # plt.plot(x_0, y_0, "xr")
    # plt.plot(x_1, y_1, "xb")
    plt.scatter(x_0, y_0, color='blue')
    plt.scatter(x_1, y_1, color='red')
    #plt.plot(points[:, 0], points[:, 1], "xy")
    st.session_state.workspace = workspace


def to_trajectory_choice():
    st.session_state.trajectory_choice = True


def run_simulation(**kwargs):
    trajectory = kwargs["trajectory"]
    gm = st.session_state.gm
    ik_manager = TrajectoryIKManager()
    # fixed_robot, free_robot = jps_graph2pinocchio_robot(gm.graph, builder)
    fixed_robot, _ = jps_graph2pinocchio_meshes_robot(
        gm.graph, visualization_builder)
    ik_manager.register_model(
        fixed_robot.model, fixed_robot.constraint_models, fixed_robot.visual_model
    )
    ik_manager.set_solver("Closed_Loop_PI")
    _ = ik_manager.follow_trajectory(
        trajectory, viz=get_visualizer()
    )
    time.sleep(1)
    get_visualizer().display(pin.neutral(fixed_robot.model))
    # st.session_state.trajectory_choice = False
    # st.session_state.workspace = None
    # st.session_state.topology = None


def calculate_and_display_rewards(trajectory, reward_mask):
    gm = st.session_state.gm
    fixed_robot, free_robot = jps_graph2pinocchio_robot_3d_constraints(
        gm.graph, optimization_builder)
    point_criteria_vector, trajectory_criteria, res_dict_fixed = full_crag.get_criteria_data(
        fixed_robot, free_robot, trajectory, viz=None)
    for i, reward in enumerate(rewards):
        if reward_mask[i]:
            st.text_area(label=reward.reward_name, value=str(reward.calculate(
                point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator=motor)[0]))


if st.session_state.topology:
    if not st.session_state.trajectory_choice:
        gm = st.session_state.gm
        mut_ranges = gm.mutation_ranges
        gm = st.session_state.gm
        mut_ranges = gm.mutation_ranges
        current_values = []
        with st.sidebar.form("jp_coordinates"):
            for key, value in mut_ranges.items():
                slider = st.slider(
                    label=key, min_value=value[0], max_value=value[1], value=(value[1]+value[0])/2)
                current_values.append(slider)
            st.form_submit_button('Set joint points')
        with st.sidebar:
            st.button(label="Get workspace",
                      on_click=evaluate_construction, key="get_workspace")
        if st.session_state.workspace:
            with st.sidebar:
                st.button(label="to trajectory choice",
                          key="to trajectory choice", on_click=to_trajectory_choice)

        graph = gm.get_graph(current_values)
        send_graph_to_visualizer(graph)
        with col_1:
            st.header("Graph representation")
            draw_joint_point(graph)
            plt.gcf().set_size_inches(4, 4)
            st.pyplot(plt.gcf(), clear_figure=True)
        with col_2:
            st.header("Robot visualization")
            components.iframe(get_visualizer().viewer.url(), width=400,
                              height=400, scrolling=True)
    else:
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
            st.button(label="run simulation", key="run_simulation",
                      on_click=run_simulation, kwargs={'trajectory': trajectory})
            with st.form(key="rewards"):
                st.header("Rewards")
                reward_mask = []
                for reward in rewards:
                    reward_mask.append(st.checkbox(
                        label=reward.reward_name, value=False))
                st.form_submit_button("Submit")
        
        graph = st.session_state.gm.graph
        with col_1:
            st.header("Graph representation")
            draw_joint_point(graph)
            plt.gcf().set_size_inches(4, 4)
            plt.plot(trajectory[:, 0], trajectory[:, 2])
            st.pyplot(plt.gcf(), clear_figure=True)
        with col_2:
            st.header("Robot visualization")
            add_trajectory_to_vis(get_visualizer(), trajectory)
            components.iframe(get_visualizer().viewer.url(), width=400,
                              height=400, scrolling=True)
        calculate_and_display_rewards(trajectory, reward_mask)
    # trajectory_choice = st.sidebar.radio(
    #     label='Select trajectory for criteria evaluation', options=trajectories.keys(), key="trajectory_choice")
