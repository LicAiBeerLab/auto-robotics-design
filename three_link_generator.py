from two_link_generator import set_circle_points
import numpy as np
from typing import Tuple, List
import networkx as nx

from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch
from auto_robot_design.description.utils import draw_joint_point
import itertools


class ThreeLinkGenerator():
    def __init__(self) -> None:
        pass
    def build_standard_threelink(self, middle_length=0.4, middle_pos=0.5, nominal_length = 1, right_shift = 0.2):
        ground_joint = JointPoint(
            r=np.zeros(3),
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=True,
            name="Main_ground"
        )
        #coordinate_list = [ground_joint.r]
        constrain_dict = {ground_joint.name:{'optim':False, 'x_range':{-0.2, 0.2}, 'z_range':{-0.2, 0.2}}}
        branch = [ground_joint]
        top_joint_pos = np.array([-right_shift, 0, -(middle_pos-middle_length*0.5)])
        bot_joint_pos = np.array([-right_shift, 0, -(middle_pos+middle_length*0.5)])
        top_joint = JointPoint(r=top_joint_pos, w=np.array([0, 1, 0]),name="Main_top")
        branch.append(top_joint)
        constrain_dict = {top_joint.name:{'optim':False, 'x_range':{-0.2, 0.2}, 'z_range':{-0.2, 0.2}}}
        bot_joint = JointPoint(r=bot_joint_pos, w=np.array([0, 1, 0]),name="Main_bot")
        branch.append(bot_joint)
        constrain_dict = {bot_joint.name:{'optim':False, 'x_range':{-0.2, 0.2}, 'z_range':{-0.2, 0.2}}}
        ee = JointPoint(
            r=np.array([0, 0,-nominal_length]),
            w=np.array([0, 1, 0]),
            attach_endeffector=True,
            name="Main_ee"
        )
        constrain_dict = {ee.name:{'optim':False, 'x_range':{-0.2, 0.2}, 'z_range':{-0.2, 0.2}}}
        branch.append(ee)
        graph = nx.Graph()
        add_branch(graph, branch)

        return graph, branch, constrain_dict
    
    def add_tl_branch(self, graph, main_branch, constrain_dict, inner:bool = True, shift = 0.25, variant = 0, branch_idx=0):
        if inner:
            ground_connection = np.array([-shift,0,0])
        else:
            ground_connection = np.array([-shift,0,0])
        
        link_connection_points = [(main_branch[i-1].r + main_branch[i].r)/2 for i in range(1, len(main_branch))]
        link_connection_points = [ground_connection]+link_connection_points
        pairs = list(itertools.combinations(list(range(len(link_connection_points))),2))
        pairs.pop(0)
        branch = []
        pair = pairs[variant]
        if pair[0] == 0:
            branch_top_joint = JointPoint(r=ground_connection,
                w=np.array([0, 1, 0]),
                attach_ground=True,
                active=True,
                name=f"branch_{branch_idx}_ground")
            constrain_dict = {branch_top_joint.name:{'optim':True, 'x_range':{-0.2, 0.2}}}
            branch.append(branch_top_joint)
        else:
            branch_top_joint = JointPoint(r=link_connection_points[pair[0]],
                w=np.array([0, 1, 0]),
                active=True,
                name=f"branch_{branch_idx}_top")
            branch.append([main_branch[pair[0]-1], main_branch[pair[0]]])
            constrain_dict = {branch_top_joint.name:{'optim':True, 'x_range':{-0.2, 0.2}, 'z_range':{-0.2, 0.2}}}
            branch.append(branch_top_joint)
        
        branch_bot_joint = JointPoint(r=link_connection_points[pair[1]],w=np.array([0, 1, 0]),name=f"branch_{branch_idx}_bot")
        constrain_dict = {branch_bot_joint.name:{'optim':True, 'x_range':{-0.2, 0.2}, 'z_range':{-0.2, 0.2}}}
        if inner:
            knee_point = (branch_top_joint.r + branch_bot_joint.r)/2 + np.array([-shift,0,0])
        else:
            knee_point = (branch_top_joint.r + branch_bot_joint.r)/2 + np.array([shift,0,0])
        branch_knee_joint = JointPoint(r=knee_point, w=np.array([0, 1, 0]),name=f"branch_{branch_idx}_knee")
        constrain_dict = {branch_bot_joint.name:{'optim':True, 'x_range':{-0.4, 0.4}, 'z_range':{-0.4, 0.4}}}
        branch.append(branch_knee_joint)
        branch.append(branch_bot_joint)
        branch.append([main_branch[pair[1]-1],main_branch[pair[1]]])
        add_branch(graph, branch)

        
        return graph, main_branch, constrain_dict, pair[0], pair[1]
    
    def add_fl_branch_type1(self, graph, main_branch, constrain_dict, inner:bool = True, shift = 0.5, ground:bool = True, variant = 0, branch_idx=0):
        if inner:
            ground_connection = np.array([-shift,0,0])
        else:
            ground_connection = np.array([-shift,0,0])
        
        link_connection_points = [(main_branch[i-1].r + main_branch[i].r)/2 for i in range(1, len(main_branch))]
        link_connection_points = [ground_connection]+link_connection_points
        triplets = list(itertools.combinations(list(range(len(link_connection_points))),3))
        branch = []
        if inner:
            triplet_top = JointPoint(r=np.array([-shift, 0, main_branch[1].r[2]]),w=np.array([0, 1, 0]),name=f"branch_{branch_idx}_triplet_top")
            constrain_dict = {triplet_top.name:{'optim':True, 'x_range':{-0.3, 0.3}, 'z_range':{-0.3, 0.3}}}
            triplet_mid = JointPoint(r=np.array([-shift, 0, (main_branch[1].r[2]+main_branch[2].r[2])/2]),w=np.array([0, 1, 0]),name=f"branch_{branch_idx}_triplet_mid")
            constrain_dict = {triplet_mid.name:{'optim':True, 'x_range':{-0.3, 0.3}, 'z_range':{-0.3, 0.3}}}
            triplet_bot = JointPoint(r=np.array([-shift, 0, main_branch[2].r[2]]),w=np.array([0, 1, 0]),name=f"branch_{branch_idx}_triplet_bot")
            constrain_dict = {triplet_bot.name:{'optim':True, 'x_range':{-0.3, 0.3}, 'z_range':{-0.3, 0.3}}}
        else:
            triplet_top = JointPoint(r=np.array([shift, 0, main_branch[1].r[2]]),w=np.array([0, 1, 0]),name=f"branch_{branch_idx}_triplet_top")
            constrain_dict = {triplet_top.name:{'optim':True, 'x_range':{-0.3, 0.3}, 'z_range':{-0.3, 0.3}}}
            triplet_mid = JointPoint(r=np.array([shift, 0, (main_branch[1].r[2]+main_branch[2].r[2])/2]),w=np.array([0, 1, 0]),name=f"branch_{branch_idx}_triplet_mid")
            constrain_dict = {triplet_mid.name:{'optim':True, 'x_range':{-0.3, 0.3}, 'z_range':{-0.3, 0.3}}}
            triplet_bot = JointPoint(r=np.array([shift, 0, main_branch[2].r[2]]),w=np.array([0, 1, 0]),name=f"branch_{branch_idx}_triplet_bot")
            constrain_dict = {triplet_bot.name:{'optim':True, 'x_range':{-0.3, 0.3}, 'z_range':{-0.3, 0.3}}}
        
        triplet =  triplets[variant]
        if triplet[0] == 0:
            branch_top_joint = JointPoint(r=ground_connection,
                w=np.array([0, 1, 0]),
                attach_ground=True,
                active=True,
                name=f"branch_{branch_idx}_ground")
            constrain_dict = {branch_top_joint.name:{'optim':True, 'x_range':{-0.2, 0.2}}}
            branch.append(branch_top_joint)
        else:
            branch_top_joint = JointPoint(r=link_connection_points[triplet[0]],
                w=np.array([0, 1, 0]),
                active=True,
                name=f"branch_{branch_idx}_top")
            branch.append([main_branch[triplet[0]-1], main_branch[triplet[0]]])
            constrain_dict = {branch_top_joint.name:{'optim':True, 'x_range':{-0.2, 0.2}, 'z_range':{-0.2, 0.2}}}
            branch.append(branch_top_joint)
        
        branch.append(triplet_top)
        branch.append(triplet_bot)
        branch_bot_joint = JointPoint(r=link_connection_points[triplet[2]],w=np.array([0, 1, 0]),name=f"branch_{branch_idx}_bot")
        constrain_dict = {branch_bot_joint.name:{'optim':True, 'x_range':{-0.2, 0.2}, 'z_range':{-0.2, 0.2}}}
        branch.append(branch_bot_joint)
        branch.append([main_branch[triplet[2]-1],main_branch[triplet[2]]])
        add_branch(graph, branch)

        branch_mid_joint = JointPoint(r=link_connection_points[triplet[1]],w=np.array([0, 1, 0]),name=f"branch_{branch_idx}_mid")
        constrain_dict = {branch_bot_joint.name:{'optim':True, 'x_range':{-0.2, 0.2}, 'z_range':{-0.2, 0.2}}}
        secondary_branch = [[triplet_top, triplet_bot], triplet_mid, branch_mid_joint,[main_branch[triplet[1]-1],main_branch[triplet[1]]]]
        add_branch(graph, secondary_branch)

        return graph, main_branch, constrain_dict, triplet[0], triplet[2]