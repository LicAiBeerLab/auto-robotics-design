import numpy as np
from typing import Tuple, List
import networkx as nx
from copy import deepcopy
import networkx

from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch
from auto_robot_design.description.utils import draw_joint_point
import itertools


def set_circle_points(pos_1, pos_2, add_pos, n):
    center = (pos_1+pos_2)/2
    vec = pos_1-center
    if np.linalg.norm(add_pos-center) > np.linalg.norm(pos_1-center):
        pos_turn = center + np.array([vec[0]*np.cos(np.pi/n)-vec[2]*np.sin(
            np.pi/n), 0, vec[2]*np.cos(np.pi/n)+vec[0]*np.sin(np.pi/n)])
        neg_turn = center + np.array([vec[0]*np.cos(-np.pi/n)-vec[2]*np.sin(-np.pi/n),
                                     0, vec[2]*np.cos(-np.pi/n)+vec[0]*np.sin(-np.pi/n)])
        new_pos_list = []
        crit = int((-0.5+int(np.linalg.norm(pos_turn-add_pos)
                   < np.linalg.norm(neg_turn-add_pos)))*2)
        for i in range(crit*1, crit * n, crit):
            angle = i*np.pi/n
            new_pos_list.append(center + np.array([vec[0]*np.cos(angle)-vec[2]*np.sin(
                angle), 0, vec[2]*np.cos(angle)+vec[0]*np.sin(angle)]))
    else:
        new_pos_list = []
        for i in range(1, n):
            new_pos_list.append(pos_1 + (pos_2-pos_1)/n*i)
    return new_pos_list


class TwoLinkGenerator():
    def __init__(self) -> None:
        self.variants = list(range(7))

    def build_standard_two_linker(self, knee_pos: float = -0.5, nominal_length=1, right_shift=np.tan(np.pi/6)/2):
        ground_joint = JointPoint(
            r=np.zeros(3),
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=True,
            name="TL_ground"
        )
        graph_dict = {"TL_ground": ground_joint}
        constrain_dict = {ground_joint.name: {'optim': False,
                                              'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}}
        branch = [ground_joint]
        # if variant not in (1,2,3): raise Exception("wrong variant!")
        # knee_joint_pos = np.array([right_shift, 0, -variant * 0.25])
        # knee_joint_pos = np.array([right_shift, 0, -np.random.uniform(0.2,0.8)])
        knee_joint_pos = np.array([right_shift, 0, knee_pos])
        knee_joint = JointPoint(
            r=knee_joint_pos, w=np.array([0, 1, 0]), name=f"TL_knee")
        constrain_dict[knee_joint.name] = {
            'optim': False, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        branch.append(knee_joint)
        graph_dict["TL_knee"] = knee_joint
        ee = JointPoint(
            r=np.array([0, 0, -nominal_length]),
            w=np.array([0, 1, 0]),
            attach_endeffector=True,
            name="TL_ee"
        )
        graph_dict["TL_ee"] = ee
        branch.append(ee)
        constrain_dict[ee.name] = {
            'optim': False, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        graph = nx.Graph()
        add_branch(graph, branch)

        return graph, graph_dict, constrain_dict

    def add_tl_branch(self, graph, graph_dict, constrain_dict, inner: bool = True, shift=0.25, ground: bool = True):
        knee_joint: JointPoint = graph_dict["TL_knee"]
        knee_pos = knee_joint.r
        ee: JointPoint = graph_dict["TL_ee"]
        ee_pos = ee.r
        connection_point = (ee_pos + knee_pos)/2
        if inner:
            branch_knee_pos = knee_pos + np.array([-knee_pos[0]-shift, 0, 0])
        else:
            branch_knee_pos = knee_pos + np.array([shift, 0, 0])

        branch_knee_joint = JointPoint(r=branch_knee_pos,
                                       w=np.array([0, 1, 0]),
                                       name="TL_branch_knee")
        constrain_dict[branch_knee_joint.name] = {
            'optim': True, 'x_range': (-0.5, 0.5), 'z_range': (-0.5, 0.5)}
        branch_connection = JointPoint(r=connection_point,
                                       w=np.array([0, 1, 0]),
                                       name="TL_branch_connection")
        constrain_dict[branch_connection.name] = {
            'optim': True, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}

        if ground:
            ground_connection = np.array([branch_knee_pos[0], 0, 0])
            branch_ground_joint = JointPoint(r=ground_connection,
                                             w=np.array([0, 1, 0]),
                                             attach_ground=True,
                                             active=True,
                                             name="TL_branch_ground")
            constrain_dict[branch_ground_joint.name] = {
                'optim': True, 'x_range': (-0.4, 0.4)}
            add_branch(graph, [
                       branch_ground_joint, branch_knee_joint, branch_connection, [knee_joint, ee]])
            return graph, constrain_dict
        else:
            ground_joint: JointPoint = graph_dict["TL_ground"]
            ground_connection = (ground_joint.r+knee_pos)/2
            branch_ground_joint = JointPoint(r=ground_connection,
                                             w=np.array([0, 1, 0]),
                                             active=True,
                                             name="TL_branch_ground")
            constrain_dict[branch_ground_joint.name] = {
                'optim': True, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
            add_branch(graph, [[ground_joint, knee_joint], branch_ground_joint,
                       branch_knee_joint, branch_connection, [knee_joint, ee]])
            return graph, constrain_dict

    def add_fl_branch(self, graph, graph_dict, constrain_dict, inner: bool = True, shift=0.5, ground: bool = True, variant=0):
        knee_joint: JointPoint = graph_dict["TL_knee"]
        knee_pos = knee_joint.r
        ee: JointPoint = graph_dict["TL_ee"]
        ee_pos = ee.r
        ground_joint: JointPoint = graph_dict["TL_ground"]
        ground_pos = ground_joint.r

        if inner:
            branch_ground_pos = np.array([-shift, 0, 0])
        else:
            branch_ground_pos = np.array([shift, 0, 0])

        connection_top = (ground_pos + knee_pos)/2
        connection_bot = (ee_pos + knee_pos)/2

        branch_ground_joint = JointPoint(r=branch_ground_pos,
                                         w=np.array([0, 1, 0]),
                                         active=True,
                                         attach_ground=True, name="FL_branch_ground")
        constrain_dict[branch_ground_joint.name] = {
            'optim': True, 'x_range': (-0.4, 0.4)}
        top_joint = JointPoint(r=connection_top,
                               w=np.array([0, 1, 0]),
                               name="FL_branch_top")
        constrain_dict[top_joint.name] = {
            'optim': True, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        bot_joint = JointPoint(r=connection_bot,
                               w=np.array([0, 1, 0]),
                               name="FL_branch_bot")
        constrain_dict[bot_joint.name] = {
            'optim': True, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
        new_joint_dict = {branch_ground_joint: [], top_joint: [
            [ground_joint, knee_joint]], bot_joint: [[ee, knee_joint]]}
        # triangle with 3 connections
        if variant == 0:
            pos_1 = np.array([branch_ground_joint.r[0], 0, bot_joint.r[2]])
            pos_2 = np.array([branch_ground_joint.r[0], 0, bot_joint.r[2]/2])
            j1 = JointPoint(r=pos_1,
                            w=np.array([0, 1, 0]),
                            name="FL_branch_knee")
            constrain_dict[j1.name] = {
                'optim': True, 'x_range': (-0.4, 0.4), 'z_range': (-0.4, 0.4)}
            j2 = JointPoint(r=pos_2,
                            w=np.array([0, 1, 0]),
                            name="FL_branch_hip")
            constrain_dict[j2.name] = {
                'optim': True, 'x_range': (-0.4, 0.4), 'z_range': (-0.4, 0.4)}
            add_branch(graph, [[ee, knee_joint], bot_joint,
                       j1, j2, branch_ground_joint])
            j3 = JointPoint(r=(pos_1+pos_2)/2,
                            w=np.array([0, 1, 0]),
                            name="FL_branch_additional")
            constrain_dict[j3.name] = {
                'optim': True, 'x_range': (-0.4, 0.4), 'z_range': (-0.4, 0.4)}
            add_branch(graph, [[ground_joint, knee_joint],
                       top_joint, j3, [j1, j2]])
            return graph, constrain_dict

        else:
            new_joints = [branch_ground_joint, top_joint, bot_joint]
            permutation = list(itertools.permutations(new_joints))[variant-1]

            new_joint_pos = set_circle_points(
                permutation[0].r, permutation[2].r, permutation[1].r, 4)
            branch = new_joint_dict[permutation[0]]
            branch.append(permutation[0])
            triangle_joints = []
            for i, pos in enumerate(new_joint_pos):
                joint = JointPoint(r=pos, w=np.array(
                    [0, 1, 0]), name=f"FL_branch_j{i}")
                constrain_dict[joint.name] = {
                    'optim': True, 'x_range': (-0.4, 0.4), 'z_range': (-0.4, 0.4)}
                branch.append(joint)
                if i<2:
                    triangle_joints.append(joint)

            branch.append(permutation[2])
            branch += new_joint_dict[permutation[2]]
            add_branch(graph, branch)

            branch_2 = [triangle_joints, permutation[1]] + \
                    new_joint_dict[permutation[1]]

            add_branch(graph, branch_2)
            return graph, constrain_dict

    def get_standard_set(self, knee_pos=-0.5, shift=0.5):
        answer_list = []
        for inner in [True, False]:
            for ground in [True, False]:
                graph, graph_dict, constrain_dict = self.build_standard_two_linker(
                    knee_pos=knee_pos)
                graph, constrain_dict = self.add_tl_branch(
                    graph, graph_dict, constrain_dict, inner=inner, ground=ground, shift=shift)
                answer_list.append((graph, constrain_dict))
            for i in self.variants:
                graph, graph_dict, constrain_dict = self.build_standard_two_linker(
                    knee_pos=knee_pos)
                graph, constrain_dict = self.add_fl_branch(
                    graph, graph_dict, constrain_dict, inner=inner, variant=i, shift=shift)
                answer_list.append((graph, constrain_dict))
        return answer_list


def get_constrain_space(constrain_dict: dict):
    space = []
    for key in constrain_dict:
        item = constrain_dict[key]
        if item['optim']:
            space.append(item.get('x_range'))
            space.append(item.get('z_range'))
    space = [x for x in space if x is not None]
    space = np.array(space)
    return space


def get_changed_graph(graph, constrain_dict, change_vector):
    new_graph: networkx.Graph = deepcopy(graph)
    vector_dict = {}
    i = 0
    for key in constrain_dict:
        if constrain_dict[key]['optim']:
            vector = np.zeros(3)
            if constrain_dict[key].get('x_range'):
                vector[0] = change_vector[i]
                i += 1
            if constrain_dict[key].get('z_range'):
                vector[2] = change_vector[i]
                i += 1
            vector_dict[key] = vector

    for node in new_graph.nodes:
        if node.name in vector_dict:
            node.r = node.r + vector_dict[node.name]

    return new_graph


if __name__ == '__main__':
    gen = TwoLinkGenerator()
    graph, constrain_dict = gen.get_standard_set()[0]
    space = get_constrain_space(constrain_dict)
    random_vector = np.zeros(len(space))
    for i, r in enumerate(space):
        random_vector[i] = np.random.uniform(low=r[0], high=r[1])

    get_changed_graph(graph, constrain_dict, random_vector)
    draw_joint_point(graph)

