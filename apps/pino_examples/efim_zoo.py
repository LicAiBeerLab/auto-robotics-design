from copy import deepcopy
from dataclasses import dataclass
import time
import odio_urdf
from auto_robot_design.pinokla.loader_tools import completeRobotLoader
from pinocchio.visualize import MeshcatVisualizer, GepettoVisualizer
import meshcat
import os
import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import random
import os
import sys
import xml

f = open("robots\generated_zoo\\robot_sliva.urdf", "r")
efim_robot_str1 = str(f.read())
f.close()

efim_robot1_loop_description = {"closed_loop": [['J2b1_L4Pseudo', 'J2b1_L5Pseudo'], ['J2b2_L7Pseudo', 'J2b2_EEPseudo']], "type": ["6d", "6d"]}
efim_robot1_joint_description = {
    "name_mot": ['Gabs', 'J1m'],
    "joint_name": [],
    "joint_type": [],
}




f = open("robots\generated_zoo\\robot_rainbow.urdf", "r")
efim_robot_str2 = str(f.read())
f.close()

efim_robot2_loop_description = {"closed_loop": [['J2b1_L5Pseudo', 'J2b1_L4Pseudo'], ['J2b2_L6Pseudo', 'J2b2_EEPseudo']], "type": ["6d", "6d"]}
efim_robot2_joint_description = {
    "name_mot": ['Gabs', 'J1m'],
    "joint_name": [],
    "joint_type": [],
}



f = open("robots\generated_zoo\\robot_longer.urdf", "r")
efim_robot_str3 = str(f.read())
f.close()

efim_robot3_loop_description = {"closed_loop": [['J3_0_L8Pseudo', 'J3_0_L7Pseudo'], ['J5_1_0_L6Pseudo', 'J5_1_0_L5Pseudo']], "type": ["6d", "6d"]}
efim_robot3_joint_description = {
    "name_mot": ['Ground_main', 'J0_0'],
    "joint_name": [],
    "joint_type": [],
}

  
f = open("robots\generated_zoo\\handsome_EE_fix.urdf", "r")
efim_robot_str6 = str(f.read())
f.close()


efim_robot6_joint_description =  {
    "name_mot": ['G0main', 'J0_0'],
    "joint_name": [],
    "joint_type": [],
}
efim_robot6_loop_description = {
    "closed_loop": [['J1m_L4Pseudo', 'J1m_L5Pseudo']],
    "type": ["6d"]
}

f = open("robots\generated_zoo\\urod.urdf", "r")
efim_robot_str7 = str(f.read())
f.close()


efim_robot7_joint_description =  {
    "name_mot": ['G0main', 'J1_0'],
    "joint_name": [],
    "joint_type": [],
}
efim_robot7_loop_description = {
    "closed_loop": [['J2_0_L6Pseudo', 'J2_0_L5Pseudo']],
    "type": ["6d"]
}
f = open("robots\generated_zoo\\urod2.urdf", "r")
efim_robot_str8 = str(f.read())
f.close()


efim_robot8_joint_description =  {
    "name_mot": ['G0main', 'J4_1_0'],
    "joint_name": [],
    "joint_type": [],
}
efim_robot8_loop_description = {
    "closed_loop": [['J0_0_L3Pseudo', 'J0_0_L4Pseudo'], ['J5_1_0_L6Pseudo', 'J5_1_0_L7Pseudo']],
    "type": ["6d", "6d"]
}