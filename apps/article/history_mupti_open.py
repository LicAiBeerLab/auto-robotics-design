import matplotlib.pyplot as plt


from auto_robot_design.optimization.analyze import get_optimizer_and_problem, get_pareto_sample_linspace, get_pareto_sample_histogram

optimizer, problem, res = get_optimizer_and_problem(
    r"results/topology_0_2024-05-29_18-48-58")
sample_x, sample_F = get_pareto_sample_linspace(res, 5)
sample_x2, sample_F2 = get_pareto_sample_histogram(res, 5)



plt.figure()
plt.scatter(sample_F[:, 0], sample_F[:, 1])
plt.title("from res1")


plt.figure()
plt.scatter(res.F[:, 0], res.F[:, 1])
plt.title("all")

plt.figure()
plt.scatter(sample_F2[:, 0], sample_F2[:, 1])
plt.title("from res2")
plt.show()
