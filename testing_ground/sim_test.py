import numpy as np
from auto_robot_design.generator.restricted_generator.two_link_generator import (
TwoLinkGenerator,
)
from auto_robot_design.description.builder import (
ParametrizedBuilder,
URDFLinkCreator,
jps_graph2pinocchio_robot,
)
import auto_robot_design.simulation.evaluation as eval
from auto_robot_design.simulation.trajectory_movments import TrajectoryMovements   

gen = TwoLinkGenerator()
builder = ParametrizedBuilder(URDFLinkCreator)
graphs_and_cons = gen.get_standard_set()
np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

graph_jp, constrain = graphs_and_cons[2]

robo, __ = jps_graph2pinocchio_robot(graph_jp, builder)

name_ee = "EE"

x_point = np.array([-0.5, 0, 0.25]) * 0.5
y_point = np.array([-0.4, -0.1, -0.4]) * 0.5
y_point = y_point - 0.7

trajectory = np.array([x_point, y_point]).T

test = TrajectoryMovements(trajectory, 1, 0.001, name_ee)

time_arr, des_traj_6d, __ = test.prepare_trajectory(robo)
# test.prepare_trajectory(robo)
# Kp, Kd = test.optimize_control(robo)

# test.Kp = Kp
# test.Kd = Kd

q, vq, acc, tau, pos_ee, power = test.simulate(robo, True)


print(f"PQ: {eval.power_quality(time_arr, power, True)}")
print(f"Error {eval.movments_in_xz_plane(time_arr, pos_ee, des_traj_6d[:,:3], True)}")
print(f"Max Torque {eval.torque_evaluation(time_arr, tau, True)}")