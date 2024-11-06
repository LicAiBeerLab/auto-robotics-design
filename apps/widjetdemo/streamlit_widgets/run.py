import multiprocessing
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.age2 import AGEMOEA2
from auto_robot_design.optimization.problems import MultiCriteriaProblem, SingleCriterionProblem
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.saver import ProblemSaver
from pymoo.algorithms.soo.nonconvex.pso import PSO
import dill
if __name__ == "__main__":
    with open("./results/buffer/data.pkl", "rb") as f:
        data = dill.load(f)
    N_PROCESS = 10
    pool = multiprocessing.Pool(N_PROCESS)
    runner = StarmapParallelization(pool.starmap)
    population_size = 128
    n_generations = 2
    graph_manager = data.graph_manager
    builder = data.optimization_builder
    reward_manager = data.reward_manager
    soft_constraint = data.soft_constraint
    actuator = data.actuator
    num_objs = reward_manager.close_trajectories()
    print(num_objs)
    if num_objs>1:
    # create the problem for the current optimization
        problem = MultiCriteriaProblem(graph_manager, builder, reward_manager,
                                    soft_constraint, elementwise_runner=runner, Actuator=actuator)


        algorithm = AGEMOEA2(pop_size=population_size, save_history=True)
    else:
        problem = SingleCriterionProblem(graph_manager, builder, reward_manager,
                                    soft_constraint, elementwise_runner=runner, Actuator=actuator)
        algorithm = PSO(pop_size=population_size, save_history=True)
    saver = ProblemSaver(problem, f"optimization_widget\\current_results", False)
    saver.save_nonmutable()
    optimizer = PymooOptimizer(problem, algorithm, saver)

    res = optimizer.run(
        True, **{
            "seed": 2,
            "termination": ("n_gen", n_generations),
            "verbose": True
        })
