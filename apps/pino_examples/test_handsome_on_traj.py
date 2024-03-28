from auto_robot_design.pinokla.calc_criterion import ForceEllProjections, ImfCompute, ManipCompute, MovmentSurface, NeutralPoseMass, TranslationErrorMSE
from auto_robot_design.pinokla.criterion_agregator import load_criterion_traj
from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator
from auto_robot_design.pinokla.criterion_math import ImfProjections
from auto_robot_design.pinokla.default_traj import convert_x_y_to_6d_traj_xz, get_simple_spline
import os
import os
from pathlib import Path

DIR_NAME_FOR_LOAD = "generated_1_select"
file_list = os.listdir(DIR_NAME_FOR_LOAD)

new_list = [Path(DIR_NAME_FOR_LOAD + "/" + str(item)) for item in file_list]

dict_along_criteria = {
    "MASS": NeutralPoseMass(),
    "POS_ERR": TranslationErrorMSE(),
    "ELL_PRJ": ForceEllProjections()
}
dict_moment_criteria = {
    "IMF": ImfCompute(ImfProjections.Z),
    "MANIP": ManipCompute(MovmentSurface.XZ)
}

x_traj, y_traj = get_simple_spline()
traj_6d = convert_x_y_to_6d_traj_xz(x_traj, y_traj)

crag = CriteriaAggregator(dict_moment_criteria, dict_along_criteria, traj_6d)

for num, path_i in enumerate(new_list):
    res_i = load_criterion_traj(path_i)
    urdf_i = str(res_i["urdf"])
    joint_description_i = res_i["mot_description"].item()
    loop_description_i = res_i["loop_description"].item()
    moment_critria_trj, along_critria_trj, res_dict_fixed = crag.get_criteria_data(
        urdf_i, joint_description_i, loop_description_i)
