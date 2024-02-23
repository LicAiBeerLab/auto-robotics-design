from auto_robot_design.pino_adapter.generate_graph import generate_strange_graph
 
from auto_robot_design.pinokla.criterion_agregator import calc_criterion_on_workspace_simple_input, save_criterion_traj
 

for i in range(10):
    BASE_FRAME = "G"
    EFFECTOR_NAME = "EE"
    LIN_NUM = 100
    try:
        robot_urdf, joint_mot_description, loop_description = generate_strange_graph()
    except:
        continue
    
    #robo_dict, res_dict = calc_criterion_on_workspace_simple_input(robot_urdf, joint_mot_description, loop_description, BASE_FRAME, EFFECTOR_NAME, LIN_NUM)
    #available_q = res_dict["available_q"]
    #coverage = len(available_q)/(LIN_NUM*LIN_NUM)
 
    save_criterion_traj(robot_urdf, "./mehs", loop_description, joint_mot_description, dict())
print("Finish")