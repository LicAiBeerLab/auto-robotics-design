import streamlit as st
import multiprocessing
import meshcat
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
from auto_robot_design.description.mesh_builder.mesh_builder import (
    MeshBuilder,
    jps_graph2pinocchio_meshes_robot,
)
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.decomposition.asf import ASF
from auto_robot_design.optimization.problems import MultiCriteriaProblem
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.saver import ProblemSaver

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
        "position", [0, 0.0, 0.8]
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

def run_simulation(graph_manager, builder, soft_constraint, actuator):
    N_PROCESS = 16
    pool = multiprocessing.Pool(N_PROCESS)
    runner = StarmapParallelization(pool.starmap)
    population_size = 128
    n_generations = 3
    reward_manager = st.session_state.reward_manager

    # create the problem for the current optimization
    problem = MultiCriteriaProblem(graph_manager, builder, reward_manager,
                                soft_constraint, elementwise_runner=runner, Actuator=actuator)

    saver = ProblemSaver(problem, f"optimization_widget\\res_", True)
    saver.save_nonmutable()
    algorithm = AGEMOEA2(pop_size=population_size, save_history=True)
    optimizer = PymooOptimizer(problem, algorithm, saver)

    res = optimizer.run(
        True, **{
            "seed": 2,
            "termination": ("n_gen", n_generations),
            "verbose": True
        })