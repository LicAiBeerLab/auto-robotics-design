from pathlib import Path
from joblib import Parallel, delayed, cpu_count
from auto_robot_design.pinokla.criterion_agregator import calc_criterion_on_workspace_simple_input, save_criterion_traj, load_criterion_traj
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
import time
import os


def chunk_list(lst, chunk_size):
    """Yield successive chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def test_n_mechs_from_dir(list_parhs_mechs: list[str], dir_for_save: str):
    urdf_list = []
    loop_des_list = []
    motor_des_list = []
    for i_name in list_parhs_mechs:
        res = load_criterion_traj(Path(i_name))
        urdf_list.append(str(res["urdf"]))
        loop_des_list.append(res["loop_description"].item())
        motor_des_list.append(res["mot_description"].item())
        pass

 
    for ur, mt, lp in zip(urdf_list, motor_des_list, loop_des_list):
        build_model_with_extensions(ur, mt, lp)

    models = list([
        build_model_with_extensions(ur, mt, lp)
        for ur, mt, lp in zip(urdf_list, motor_des_list, loop_des_list)
    ])
    free_models = list([
        build_model_with_extensions(ur, mt, lp, False)
        for ur, mt, lp in zip(urdf_list, motor_des_list, loop_des_list)
    ])

    EFFECTOR_NAME = "EE"
    BASE_FRAME = "G"
    NUM_LINSPACE = 100
    DIR_NAME = dir_for_save

    effector_name_list = [EFFECTOR_NAME for i in range(len(models))]
    base_name_list = [BASE_FRAME for i in range(len(models))]
    linspace_num_list = [NUM_LINSPACE for i in range(len(models))]
    dir_names_list = [DIR_NAME for i in range(len(models))]

    packed_args = list(
        zip(urdf_list, motor_des_list, loop_des_list, base_name_list,
            effector_name_list, linspace_num_list))

    parallel_results = []
    cpus = cpu_count(only_physical_cores=True)
    parallel_results = Parallel(
        cpus, backend="multiprocessing", verbose=100, timeout=60 * 1000)(
            delayed(calc_criterion_on_workspace_simple_input)(*i)
            for i in packed_args)

    for robo_dict, res_dict in parallel_results:
        if robo_dict.get("urdf") is None:
            continue
        save_criterion_traj(robo_dict["urdf"], DIR_NAME, robo_dict["loop_des"],
                            robo_dict["joint_des"], res_dict)
    pass


if __name__ == '__main__':
    start_time = time.time()
    DIR_NAME_FOR_TEST = "./mehs"
    DIR_NAME_FOR_SAVE = "result"
    file_list = os.listdir(DIR_NAME_FOR_TEST)
    new_list = [
        Path(DIR_NAME_FOR_TEST + "/" + str(item)) for item in file_list
    ]
    paths_chunks = chunk_list(new_list, 20)
    for num, chunk_i in enumerate(paths_chunks):
        test_n_mechs_from_dir(chunk_i, DIR_NAME_FOR_SAVE)
        time_spent = time.time() - start_time
        print(f"Chunk:{num} is tested. Time spent {time_spent}")
    pass
