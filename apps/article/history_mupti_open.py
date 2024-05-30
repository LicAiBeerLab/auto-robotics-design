from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from auto_robot_design.optimization.analyze import get_optimizer_and_problem, get_pareto_sample_linspace, get_pareto_sample_histogram

PATH_CS = "results\\multi_opti_preset2\\topology_8_2024-05-30_10-40-12"
optimizer, problem, res = get_optimizer_and_problem(
    PATH_CS)
 


save_p = Path(str(PATH_CS) + "/" + "plots")
save_p.mkdir(parents=True, exist_ok=True)

history_mean = np.array(optimizer.history["Mean"])

plt.figure()

 
plt.title("Mean generation reward")
plt.xlabel("Generation")
plt.ylabel("Reword")
plt.plot(np.array(history_mean)[:,0])
plt.plot(np.array(history_mean)[:,1])
plt.legend(["ACC capability", "HeavyLifting"])
save_current1 = save_p / "Mean_Generation_reward.svg"
save_current2 = save_p / "Mean_Generation_reward.png"
plt.savefig(save_current1)
plt.savefig(save_current2)
plt.show()

pass
