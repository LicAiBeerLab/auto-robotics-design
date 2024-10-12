import multiprocessing
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.decomposition.asf import ASF
from auto_robot_design.optimization.problems import MultiCriteriaProblem
from auto_robot_design.optimization.optimizer import PymooOptimizer
from auto_robot_design.optimization.saver import ProblemSaver
import dill
if __name__ == "__main__":
    with open("./results/buffer/data.pkl", "rb") as f:
        data = dill.load(f)
    N_PROCESS = 16
    pool = multiprocessing.Pool(N_PROCESS)
    runner = StarmapParallelization(pool.starmap)
    population_size = 128
    n_generations = 10
    graph_manager = data.graph_manager
    builder = data.optimization_builder
    reward_manager = data.reward_manager
    soft_constraint = data.soft_constraint
    actuator = data.actuator
    # create the problem for the current optimization
    problem = MultiCriteriaProblem(graph_manager, builder, reward_manager,
                                soft_constraint, elementwise_runner=runner, Actuator=actuator)

    saver = ProblemSaver(problem, f"optimization_widget\\current_results", False)
    saver.save_nonmutable()
    algorithm = AGEMOEA2(pop_size=population_size, save_history=True)
    optimizer = PymooOptimizer(problem, algorithm, saver)

    res = optimizer.run(
        True, **{
            "seed": 2,
            "termination": ("n_gen", n_generations),
            "verbose": True
        })
    print('done')