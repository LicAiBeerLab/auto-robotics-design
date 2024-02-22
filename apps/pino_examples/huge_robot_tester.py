from joblib import Parallel, delayed, cpu_count
from auto_robot_design.pinokla.criterion_agregator import calc_criterion_on_workspace, calc_criterion_on_workspace_simple_input, save_criterion_traj
from efim_zoo import efim_robot_str7, efim_robot7_loop_description, efim_robot7_joint_description, \
    efim_robot_str6, efim_robot6_loop_description, efim_robot6_joint_description, efim_robot_str5,  efim_robot5_loop_description, efim_robot5_joint_description
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
import time
import os
if __name__ == '__main__':
    urdf_list = [efim_robot_str5, efim_robot_str6, efim_robot_str7]
    loop_des_list = [efim_robot5_loop_description,
                     efim_robot6_loop_description, efim_robot7_loop_description]
    motor_des_list = [efim_robot5_joint_description,
                      efim_robot6_joint_description, efim_robot7_joint_description]

    urdf_list *= 1
    loop_des_list *= 1
    motor_des_list *= 1
    for ur, mt, lp in zip(urdf_list, motor_des_list, loop_des_list):
        build_model_with_extensions(ur, mt, lp)

    models = list([build_model_with_extensions(ur, mt, lp)
                   for ur, mt, lp in zip(urdf_list, motor_des_list, loop_des_list)])
    free_models = list([build_model_with_extensions(ur, mt, lp, False)
                        for ur, mt, lp in zip(urdf_list, motor_des_list, loop_des_list)])

    EFFECTOR_NAME = "EE"
    BASE_FRAME = "G"
    NUM_LINSPACE = 100
    DIR_NAME = "./benis"

    effector_name_list = [EFFECTOR_NAME for i in range(len(models))]
    base_name_list = [BASE_FRAME for i in range(len(models))]
    linspace_num_list = [NUM_LINSPACE for i in range(len(models))]
    dir_names_list = [DIR_NAME for i in range(len(models))]

    packed_args = list(zip(urdf_list, motor_des_list, loop_des_list,
                       base_name_list, effector_name_list, linspace_num_list))

    parallel_results = []
    cpus = 4
    parallel_results = Parallel(cpus, backend="multiprocessing", verbose=100, timeout=60*5)(
        delayed(calc_criterion_on_workspace_simple_input)(*i) for i in packed_args)
    dd = {}
    dd.get
    for robo_dict, res_dict in parallel_results:
        if robo_dict.get("urdf") is None:
            continue
        save_criterion_traj(
            robo_dict["urdf"], DIR_NAME, robo_dict["loop_des"], robo_dict["joint_des"], res_dict)
