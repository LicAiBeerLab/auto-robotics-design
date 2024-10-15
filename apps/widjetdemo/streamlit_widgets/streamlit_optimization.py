from pymoo.decomposition.asf import ASF
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.saver import load_checkpoint
from auto_robot_design.optimization.problems import MultiCriteriaProblem, SingleCriterionProblem
from copy import deepcopy
import time
import os
import streamlit as st
import numpy as np
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from forward_init import add_trajectory_to_vis, build_constant_objects, set_criteria_and_crag, OptimizationData
from auto_robot_design.generator.topologies.bounds_preset import get_preset_by_index_with_bounds
from auto_robot_design.description.utils import draw_joint_point
import pinocchio as pin
import meshcat
from auto_robot_design.description.mesh_builder.mesh_builder import (
    MeshBuilder,
    jps_graph2pinocchio_meshes_robot,
)
from auto_robot_design.generator.topologies.graph_manager_2l import GraphManager2L, plot_2d_bounds, MutationType
import dill
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
    ParametrizedBuilder, URDFLinkCreater3DConstraints,
    jps_graph2pinocchio_robot_3d_constraints,
)
from auto_robot_design.optimization.rewards.reward_base import PositioningConstrain, PositioningErrorCalculator, RewardManager
from streamlit_widget_auxiliary import get_visualizer, send_graph_to_visualizer

full_crag, rewards, motor = set_criteria_and_crag()
# build and cache constant objects
graph_managers, optimization_builder, visualization_builder, crag, workspace_trajectory, ground_symmetric_step1, ground_symmetric_step2, ground_symmetric_step3, central_vertical, left_vertical, right_vertical = build_constant_objects()
trajectories = {"ground_symmetric_step1": ground_symmetric_step1,
                "ground_symmetric_step2": ground_symmetric_step2,
                "ground_symmetric_step3": ground_symmetric_step3,
                "central_vertical": central_vertical,
                "left_vertical": left_vertical,
                "right_vertical": right_vertical}


st.title("Mechanical linkage mechanism optimization")

# gm is the first value that gets set. List of all values that should be update for each session
if 'gm' not in st.session_state:
    st.session_state.gm = get_preset_by_index_with_bounds(-1)
    st.session_state.gm_clone = None
    st.session_state.reward_manager = RewardManager(crag=full_crag)
    st.session_state.trajectory_idx = 0
    st.session_state.stage = "topology_choice"
    st.session_state.trajectory_groups = []
    st.session_state.trajectory_buffer = {}
    error_calculator = PositioningErrorCalculator(
        error_key='error', jacobian_key="Manip_Jacobian")
    st.session_state.soft_constraint = PositioningConstrain(
        error_calculator=error_calculator, points=[])
    st.session_state.opt_rewards_dict = {}

# top columns that usually show the chosen topology in both graph and mesh forms
col_1, col_2 = st.columns(2, gap="medium")


def confirm_topology():
    """Confirm the selected topology and move to the next stage."""
    st.session_state.stage = "ranges_choice"
    # create a deep copy of the graph manager for further updates
    st.session_state.gm_clone = deepcopy(st.session_state.gm)


def topology_choice():
    """Update the graph manager based on the selected topology."""
    st.session_state.gm = graph_managers[st.session_state.topology_choice]


# the radio button and confirm button are only visible until the topology is selected
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
    """Confirm the selected ranges and move to the next stage."""
    st.session_state.stage = "trajectory_choice"
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
    print(gm.mutation_ranges)


def return_to_topology():
    """Return to the topology choice stage."""
    st.session_state.stage = "topology_choice"


# second stage
if st.session_state.stage == "ranges_choice":
    # form for optimization ranges. All changes affects the gm_clone and it should be used for optimization
    # initial nodes
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
    graph = st.session_state.gm.graph
    draw_joint_point(graph)
    # here gm is a clone
    plot_2d_bounds(gm)
    st.pyplot(plt.gcf(), clear_figure=True)
    # this way we set ranges after each step, but without freezing joints
    gm.set_mutation_ranges()
    st.write(gm.mutation_ranges)


def add_trajectory(trajectory, idx):
    """Create a new trajectory group with a single trajectory."""
    # trajectory buffer is necessary to store all trajectories until the confirmation and adding to reward manager
    st.session_state.trajectory_buffer[idx] = trajectory
    st.session_state.trajectory_groups.append([idx])
    st.session_state.trajectory_idx += 1


def remove_trajectory_group():
    """Remove the last added trajectory group."""
    # we only allow to remove the last added group and that should be enough
    for idx in st.session_state.trajectory_groups[-1]:
        del st.session_state.trajectory_buffer[idx]
    st.session_state.trajectory_groups.pop()


def add_to_group(trajectory, idx):
    """Add a trajectory to the last added group."""
    st.session_state.trajectory_buffer[idx] = trajectory
    st.session_state.trajectory_groups[-1].append(idx)
    st.session_state.trajectory_idx += 1


def start_optimization(rewards_tf):
    """Start the optimization process."""
    # print(st.session_state.trajectory_groups)
    st.session_state.stage = "optimization"
    # rewards_tf = trajectories
    # add all trajectories to the reward manager and soft constraint
    for idx_trj, trj in st.session_state.trajectory_buffer.items():
        st.session_state.reward_manager.add_trajectory(trj, idx_trj)
        st.session_state.soft_constraint.add_points_set(trj)
    # add all rewards to the reward manager according to trajectory groups
    for trj_list_idx, trajectory_list in enumerate(st.session_state.trajectory_groups):
        for trj in trajectory_list:
            for r_idx, r in enumerate(rewards_tf[trj_list_idx]):
                if r:
                    st.session_state.reward_manager.add_reward(
                        rewards[r_idx], trj, 1)
        # we only allow mean aggregation for now
        st.session_state.reward_manager.add_trajectory_aggregator(
            trajectory_list, 'mean')
    # add all necessary objects to a buffer folder for the optimization script
    graph_manager = deepcopy(st.session_state.gm_clone)
    reward_manager = deepcopy(st.session_state.reward_manager)
    sf = deepcopy(st.session_state.soft_constraint)
    builder = deepcopy(optimization_builder)
    data = OptimizationData(graph_manager, builder,
                            full_crag, reward_manager, sf, motor)
    with open("./results/buffer/data.pkl", "wb+") as f:
        dill.dump(data, f)


def return_to_ranges(reset=False):
    """Return to the ranges choice stage."""
    st.session_state.stage = "ranges_choice"
    if reset:
        st.session_state.trajectory_groups = []
        st.session_state.trajectory_buffer = {}
        st.session_state.trajectory_idx = 0
        st.session_state.reward_manager = RewardManager(crag=full_crag)


    # when ranges are set we start to choose the reward+trajectory
    # each trajectory should be added to the manager
if st.session_state.stage == "trajectory_choice":
    # graph is only for visualization so it still gm
    graph = st.session_state.gm.graph
    trajectory = None
    with st.sidebar:
        st.button(label="Return to ranges choice",
                  key="return_to_ranges", on_click=return_to_ranges)
        st.button(label='return to range choice and reset',
                  key='return_to_ranges_reset', on_click=return_to_ranges, args=[True])
        # currently only choice between predefined parametrized trajectories
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
                              max_value=0.6, value=0.28)
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
        # no more than 2 groups for now
        if len(st.session_state.trajectory_groups) < 2:
            st.button(label="Add trajectory", key="add_trajectory", args=(
                trajectory, st.session_state.trajectory_idx), on_click=add_trajectory)
        # if there is at leas one group we can add to group or remove group
        if st.session_state.trajectory_groups:
            st.button(label="Add to group", key="add_to_group", args=[
                trajectory, st.session_state.trajectory_idx], on_click=add_to_group)
            st.button(label="Remove group", key="remove_group",
                      on_click=remove_trajectory_group)
        # for each reward trajectories should be assigned
    # top visualization of current trajectory
    st.write("Current trajectory visualization:")
    col_1, col_2 = st.columns(2, gap="medium")
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

    trajectories = [[0]*len(rewards)]*len(st.session_state.trajectory_groups)
    if st.session_state.trajectory_groups:
        st.write("Select rewards for each trajectory group:")
    rewards_counter = []
    for i, t_g in enumerate(st.session_state.trajectory_groups):
        st.write(f"Group {i} trajectories and rewards:")
        cols = st.columns(2)
        with cols[0]:
            st.header("Graph and trajectories:")
            draw_joint_point(graph)
            for idx in st.session_state.trajectory_groups[i]:
                current_trajectory = st.session_state.trajectory_buffer[idx]
                plt.plot(current_trajectory[:, 0], current_trajectory[:, 2])
            st.pyplot(plt.gcf(), clear_figure=True)
        with cols[1]:
            st.header("Rewards:")
            reward_idxs = [0]*len(rewards)
            for reward_idx, reward in enumerate(rewards):
                current_checkbox = st.checkbox(
                    label=reward.reward_name, value=False, key=reward.reward_name+str(i))
                reward_idxs[reward_idx] = current_checkbox
            trajectories[i] = reward_idxs
        rewards_counter.append(sum(reward_idxs))
    # we only allow to start optimization if there is at least one group and all groups have at least one reward
    if st.session_state.trajectory_groups and all([r > 0 for r in rewards_counter]):
        st.button(label="Start optimization",
                  key="start_optimization", on_click=start_optimization, args=[trajectories])


def show_results():
    st.session_state.stage = "results"


if st.session_state.stage == "optimization":
    graph = st.session_state.gm.graph
    with col_1:
        st.header("Graph representation")
        draw_joint_point(graph)
        plt.gcf().set_size_inches(4, 4)
        st.pyplot(plt.gcf(), clear_figure=True)
    with col_2:
        st.header("Robot visualization")
        components.iframe(get_visualizer(visualization_builder).viewer.url(), width=400,
                          height=400, scrolling=True)
    # this is a long process that will block the streamlit app
    with st.status("Optimization..."):
        os.system("python apps/widjetdemo/streamlit_widgets/run.py")
    # the button should appear after the optimization is done
    st.button(label="Show results", key="show_results", on_click=show_results)


def run_simulation(**kwargs):
    trajectory = kwargs["trajectory"]
    graph = kwargs['graph']
    ik_manager = TrajectoryIKManager()
    # fixed_robot, free_robot = jps_graph2pinocchio_robot(gm.graph, builder)
    fixed_robot, _ = jps_graph2pinocchio_meshes_robot(
        graph, visualization_builder)
    ik_manager.register_model(
        fixed_robot.model, fixed_robot.constraint_models, fixed_robot.visual_model
    )
    ik_manager.set_solver("Closed_Loop_PI")
    with st.status("simulation..."):
        _ = ik_manager.follow_trajectory(
            trajectory, viz=get_visualizer(visualization_builder)
        )
    time.sleep(1)
    get_visualizer(visualization_builder).display(
        pin.neutral(fixed_robot.model))


def calculate_and_display_rewards(graph, trajectory, reward_mask):
    fixed_robot, free_robot = jps_graph2pinocchio_robot_3d_constraints(
        graph, optimization_builder)
    point_criteria_vector, trajectory_criteria, res_dict_fixed = full_crag.get_criteria_data(
        fixed_robot, free_robot, trajectory, viz=None)
    st.session_state.opt_rewards_dict = {}
    for i, reward in enumerate(rewards):
        if reward_mask[i]:
            st.session_state.opt_rewards_dict[reward.reward_name] = str(reward.calculate(
                point_criteria_vector, trajectory_criteria, res_dict_fixed, Actuator=motor)[0])


if st.session_state.stage == "results":
    n_obj = st.session_state.reward_manager.close_trajectories()
    selected_directory = "./results/optimization_widget/current_results"

    if n_obj == 1:
        problem = SingleCriterionProblem.load(selected_directory)
        checkpoint = load_checkpoint(selected_directory)
        optimizer = PymooOptimizer(problem, checkpoint)
        optimizer.load_history(selected_directory)
        res = optimizer.run()
        ten_best = np.argsort(np.array(optimizer.history["F"]).flatten())[:11]
        print(ten_best)
        
        idx = st.select_slider(label="best results",options=[1,2,3,4,5,6,7,8,9,10], value=1, help='10 best mechanisms with 1 being the best')
        best_id = ten_best[idx]
        best_x = optimizer.history["X"][best_id]
        print(best_x)
        graph = problem.graph_manager.get_graph(best_x)
        send_graph_to_visualizer(graph, visualization_builder)
        with st.sidebar:
            trajectories = problem.rewards_and_trajectories.trajectories
            trj_idx = st.radio(label="Select trajectory", options=trajectories.keys(
            ), index=0, key='opt_trajectory_choice')
            trajectory = trajectories[trj_idx]

            st.button(label='run simulation', key='run_simulation', on_click=run_simulation, kwargs={
                      "graph": graph, "trajectory": trajectory})
            st.header("Rewards:")
            reward_idxs = [0]*len(rewards)
            for reward_idx, reward in enumerate(rewards):
                current_checkbox = st.checkbox(
                    label=reward.reward_name, value=False, key=reward.reward_name+str(reward_idx), help=reward.__doc__)
                reward_idxs[reward_idx] = current_checkbox
        
        with col_1:
            st.header("Graph representation")
            draw_joint_point(graph)
            plt.plot(trajectory[:, 0], trajectory[:, 2])
            plt.gcf().set_size_inches(4, 4)
            st.pyplot(plt.gcf(), clear_figure=True)
        with col_2:
            st.header("Robot visualization")
            add_trajectory_to_vis(get_visualizer(
                visualization_builder), trajectory)
            components.iframe(get_visualizer(visualization_builder).viewer.url(), width=400,
                              height=400, scrolling=True)

        with st.sidebar:
            st.button(label="calculate rewards", key="calculate_rewards",
                      on_click=calculate_and_display_rewards, args=[graph, trajectory, reward_idxs])

        for key, value in st.session_state.opt_rewards_dict.items():
            st.text(f"{key}: {value}")

    if n_obj == 2:
        problem = MultiCriteriaProblem.load(selected_directory)
        checkpoint = load_checkpoint(selected_directory)
        optimizer = PymooOptimizer(problem, checkpoint)
        optimizer.load_history(selected_directory)
        res = optimizer.run()
        F = res.F
        approx_ideal = F.min(axis=0)
        approx_nadir = F.max(axis=0)
        nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
        st.header('Choose the solution weights:')
        w1 = st.slider(label="Select weight", min_value=0.0,
                       max_value=1.0, value=0.5)
        weights = np.array([w1, 1-w1])

        decomp = ASF()
        b = decomp.do(nF, 1/weights).argmin()
        best_x = res.X[b]
        graph = problem.graph_manager.get_graph(best_x)
        with st.sidebar:
            trajectories = st.session_state.reward_manager.trajectories
            trj_idx = st.radio(label="Select trajectory", options=trajectories.keys(
            ), index=0, key='opt_trajectory_choice')
            trajectory = trajectories[trj_idx]

            st.button(label='run simulation', key='run_simulation', on_click=run_simulation, kwargs={
                      "graph": graph, "trajectory": trajectory})
            st.header("Rewards:")
            reward_idxs = [0]*len(rewards)
            for reward_idx, reward in enumerate(rewards):
                current_checkbox = st.checkbox(
                    label=reward.reward_name, value=False, key=reward.reward_name+str(reward_idx), help=reward.__doc__)
                reward_idxs[reward_idx] = current_checkbox
        send_graph_to_visualizer(graph, visualization_builder)
        with col_1:
            st.header("Graph representation")
            draw_joint_point(graph)
            plt.plot(trajectory[:, 0], trajectory[:, 2])
            plt.gcf().set_size_inches(4, 4)
            st.pyplot(plt.gcf(), clear_figure=True)
        with col_2:
            st.header("Robot visualization")
            add_trajectory_to_vis(get_visualizer(
                visualization_builder), trajectory)
            components.iframe(get_visualizer(visualization_builder).viewer.url(), width=400,
                              height=400, scrolling=True)
        # draw_joint_point(graph)
        # st.pyplot(plt.gcf(), clear_figure=True)
        with st.sidebar:
            st.button(label="calculate rewards", key="calculate_rewards",
                      on_click=calculate_and_display_rewards, args=[graph, trajectory, reward_idxs])

        plt.figure(figsize=(7, 5))
        plt.scatter(F[:, 0], F[:, 1], s=30,
                    facecolors='none', edgecolors='blue')
        plt.scatter(approx_ideal[0], approx_ideal[1], facecolors='none',
                    edgecolors='red', marker="*", s=100, label="Ideal Point (Approx)")
        plt.scatter(approx_nadir[0], approx_nadir[1], facecolors='none',
                    edgecolors='black', marker="p", s=100, label="Nadir Point (Approx)")
        plt.scatter(F[b, 0], F[b, 1], marker="x", color="red", s=200)
        plt.title("Objective Space")
        plt.legend()

        # st.pyplot(plt.gcf(), clear_figure=True)
        for key, value in st.session_state.opt_rewards_dict.items():
            st.text(f"{key}: {value}")

    # trajectory = problem.rewards_and_trajectories.trajectories[0]
    # builder = problem.builder
    # crag = problem.rewards_and_trajectories.crag
    # fixed_robot, free_robot = jps_graph2pinocchio_robot_3d_constraints(graph, builder=builder)
    # constrain_error, results = problem.soft_constrain.calculate_constrain_error(
    #     crag, fixed_robot, free_robot)
    # st.warning(constrain_error)
    # plt.plot(np.sum(np.abs(np.diff(results[0][2]['q'], axis=0)),axis=1))
    # st.pyplot(plt.gcf(), clear_figure=True)
