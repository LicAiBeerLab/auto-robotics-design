from collections import deque

import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import modern_robotics as mr

import networkx as nx

from description.kinematics import JointPoint

graph = nx.Graph()
# https://cad.onshape.com/documents/52eb11422c701d811548a6f5/w/655758bb668dff773a0e7c1a/e/77ff7f84e82d8fb31fe9c30b
# abs_ground = np.array([0.065, 0, -0.015])
abs_ground = np.array([0.065, 0, -0.047])
pos_toeA_joint = np.array([0.065, 0, -0.047]) - abs_ground
pos_toeA_tarus_joint = np.array([-0.273, 0, -0.350]) - abs_ground
pos_shin_joint = np.array([0.021, 0, -0.159]) - abs_ground
pos_knee_spring = np.array([0.011, 0, -0.219]) - abs_ground
pos_tarus_joint = np.array([-0.237, 0, -0.464]) - abs_ground
pos_foot_joint = np.array([-0.080, 0, -0.753]) - abs_ground
pos_molet_joint = np.array([-0.207, 0, -0.552]) - abs_ground
pos_toeB_joint = np.array([-0.257, 0, -0.579]) - abs_ground
pos_toeB_foot_joint = np.array([-0.118, 0, -0.776]) - abs_ground

ground_joint = JointPoint(
    r=np.zeros(3), w=np.array([0, 1, 0]), attach_ground=True, active=True
)
shin_joint = JointPoint(r=pos_shin_joint, w=np.array([0, 1, 0]), active=True)
knee_spring = JointPoint(pos_knee_spring, w=np.array([0, 1, 0]), weld=True)
tarus_joint = JointPoint(r=pos_tarus_joint, w=np.array([0, 1, 0]))
foot_joint = JointPoint(
    r=pos_foot_joint, w=np.array([0, 1, 0]), attach_endeffector=True
)

toeA_joint = JointPoint(r=pos_toeA_joint, w=np.array([0, 1, 0]))
connect_toeA_tarus_joint = JointPoint(
    r=pos_toeA_tarus_joint, w=np.array([0, 1, 0]), weld=True
)

molet_joint = JointPoint(r=pos_molet_joint, w=np.array([0, 1, 0]), active=True)
toeB_joint = JointPoint(r=pos_toeB_joint, w=np.array([0, 1, 0]))
toeB_foot_joint = JointPoint(
    r=pos_toeB_foot_joint, w=np.array([0, 1, 0]), attach_endeffector=True
)

joints = [
    ground_joint,
    shin_joint,
    knee_spring,
    tarus_joint,
    foot_joint,
    toeA_joint,
    connect_toeA_tarus_joint,
    molet_joint,
    toeB_joint,
    toeB_foot_joint,
]

for j in joints:
    graph.add_node(j)
main_branch = [ground_joint, shin_joint, knee_spring, tarus_joint, foot_joint]
add_branch_1 = [
    [ground_joint, shin_joint],
    toeA_joint,
    connect_toeA_tarus_joint,
    [tarus_joint, foot_joint],
]
add_branch_2 = [
    [tarus_joint, foot_joint],
    molet_joint,
    toeB_joint,
    toeB_foot_joint,
    foot_joint,
]


def calc_weight(n0, n1):
    norm = la.norm(n0.r - n1.r)
    norm = norm if not np.isclose(norm, 0) else 1e5
    return 1 / (norm)


for id in range(len(main_branch) - 1):
    graph.add_edge(
        main_branch[id],
        main_branch[id + 1],
        variable=False,
        active=False,
        weight=calc_weight(main_branch[id], main_branch[id + 1]),
    )
for id in range(len(add_branch_1) - 1):
    if isinstance(add_branch_1[id], list):
        for j in add_branch_1[id]:
            graph.add_edge(
                j,
                add_branch_1[id + 1],
                variable=False,
                active=False,
                weight=calc_weight(j, add_branch_1[id + 1]),
            )
    elif isinstance(add_branch_1[id + 1], list):
        for j in add_branch_1[id + 1]:
            graph.add_edge(
                j,
                add_branch_1[id],
                variable=False,
                active=False,
                weight=calc_weight(j, add_branch_1[id]),
            )
    else:
        graph.add_edge(
            add_branch_1[id],
            add_branch_1[id + 1],
            variable=False,
            active=False,
            weight=calc_weight(add_branch_1[id], add_branch_1[id + 1]),
        )
for id in range(len(add_branch_2) - 1):
    if isinstance(add_branch_2[id], list):
        for j in add_branch_2[id]:
            graph.add_edge(
                j,
                add_branch_2[id + 1],
                variable=False,
                active=False,
                weight=calc_weight(j, main_branch[id + 1]),
            )
    elif isinstance(add_branch_2[id + 1], list):
        for j in add_branch_2[id + 1]:
            graph.add_edge(
                j,
                add_branch_2[id],
                variable=False,
                active=False,
                weight=calc_weight(j, add_branch_2[id]),
            )
    else:
        graph.add_edge(
            add_branch_2[id],
            add_branch_2[id + 1],
            variable=False,
            active=False,
            weight=calc_weight(add_branch_2[id], add_branch_2[id + 1]),
        )

R = 0.751618254168963
joint_nodes = list(graph.nodes())
ground_joints = set(
    sorted(filter(lambda n: n.attach_ground, joint_nodes), key=lambda x: la.norm(x.r))
)
ee_joints = set(
    sorted(
        filter(lambda n: n.attach_endeffector, joint_nodes),
        key=lambda x: np.abs(la.norm(x.r) - R),
    )
)

from dataclasses import dataclass, field


@dataclass
class Link:
    name: str = ""
    joints: set = field(default_factory=set)
    instance_counter: int = 0

    def __post_init__(self):
        Link.instance_counter += 1
        self.instance_counter = Link.instance_counter
        if self.name == "":
            self.name = "L" + str(self.instance_counter)

    def __hash__(self) -> int:
        return hash((self.name, *self.joints))
    
    def __eq__(self, __value: object) -> bool:
        self.joints == __value.joints

stack_joints = deque(maxlen=len(graph.nodes()))

ground_link = Link("ground", ground_joints)
ee_link = Link("EndEffector", ee_joints)

j_has_link = {j:0 for j in graph.nodes()}
j2link = {j:[] for j in graph.nodes()}
for j in ground_joints:
    j2link[j].append(ground_link)
    j_has_link[j] += 1
for ee_j in ee_joints:
    j2link[ee_j].append(ee_link)
    j_has_link[ee_j] += 1

stack_joints += list(ground_link.joints)
explored_j = set()
links = [ee_link, ground_link]
while stack_joints:
    curr_j = stack_joints.pop()
    if len(j2link.get(curr_j, [])) == 2:
        continue
    elif len(j2link.get(curr_j, [])) == 1:
        old_link = j2link[j][0]
    else:
        old_link = None
    
    explored_j.add(curr_j)
    neighbor = set(graph.neighbors(curr_j))

    for n in neighbor:
        if len(j2link[n]) == 2:
            nneighbors = set(graph.neighbors(n))
            j_links = nneighbors & (neighbor | set([curr_j]))
            main_link = sorted(j2link[n], lambda x: len(x.joints & j_links), reverse=True)[0]
            main_link.joints.add(curr_j)
            j2link[curr_j].append(main_link)



    if len(neighbor & explored_j) > 1:
        more1neighbor_onelink = filter(lambda link: len(neighbor.intersection(link.joints)) > 1, links)
        for link in more1neighbor_onelink:
            link.joints.add(curr_j)
        j_has_link[curr_j] +=1
    close_j = set(dict(filter(lambda x: x[1] > 1, j_has_link.items())).keys())
    hanging_j = set(dict(filter(lambda x: x[1] == 1, j_has_link.items())).keys())

    new_neighbor = neighbor -  close_j - links[-1].joints
    view_joints = new_neighbor | set([curr_j])
    if len(view_joints) > 2:
        view_joints = set([curr_j])
        neighbor
        for n in neighbor:
            view_joints = view_joints.union(neighbor.intersection(set(graph.neighbors(n))))
    links.append(Link(joints=view_joints))
    stack_joints += list(neighbor - explored_j)
    for j in view_joints:
        j2link[j].append(links[-1])
        j_has_link[j] += 1
    close_j = set(dict(filter(lambda x: x[1] > 1, j_has_link.items())).keys())
    for j in close_j:
        if j in stack_joints:
            stack_joints.remove(j)