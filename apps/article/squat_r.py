from auto_robot_design.optimization.analyze import get_optimizer_and_problem
from auto_robot_design.pinokla.analyze_squat_history import load_run_hop_sims
from auto_robot_design.pinokla.squat import SquatHopParameters

optimizer, problem, res = get_optimizer_and_problem("results\\new_op\\topology_8_2024-07-11_17-59-08")
sqh_p = SquatHopParameters(hop_flight_hight=0.10,
                            squatting_up_hight=0.0,
                            squatting_down_hight=-0.04,
                            total_time=0.2)
load_run_hop_sims("results\\new_op\\topology_8_2024-07-11_17-59-08", sqh_p, 10, True)
print("FFF")