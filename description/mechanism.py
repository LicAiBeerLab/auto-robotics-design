from collections import deque
from itertools import combinations

from networkx import neighbors
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt
from pyparsing import List, Tuple
from scipy.spatial.transform import Rotation as R

import modern_robotics as mr

import networkx as nx

from description.kinematics import JointPoint, Link
from description.kinematics import get_rot_matrix_by_vec
import numpy as np

def get_rot_matrix_by_vec(v_l, w):
    ez = np.array([0,0,1])
    ex = np.array([1,0,0])
    angle = np.arccos(np.inner(ez, v_l) / la.norm(ez) / la.norm(v_l))
    out = lambda q: R.from_rotvec(w * (angle + np.sign(np.inner(ex, v_l)) * q), )
    return out, angle

def build_homogeneous_transformation(v_l, w, v_trans):
    Rj, q_init = get_rot_matrix_by_vec(v_l,w)
    Hj = lambda q: np.r_[np.c_[Rj(q), np.array([0,0,la.norm(v_trans)])], np.array([[0,0,0,1]])]
    return Hj, q_init

def calculate_weight(n0: JointPoint, n1: JointPoint):
    length = la.norm(n0.r, n1.r)
    if n1.active:
        pass
    if n1.active:
        pass


def JointPoint2KinematicGraph(jp_graph: nx.Graph):
    JPs = list(jp_graph.nodes())
    
    ground_joints = set(
    sorted(filter(lambda n: n.attach_ground, JPs), key=lambda x: la.norm(x.r))
    )
    ee_joints = set(
    sorted(
        filter(lambda n: n.attach_endeffector, JPs),
        key=lambda x: np.abs(la.norm(x.r) - R0),
        )
    )
    ground_link = Link("G", ground_joints)
    ee_link = Link("EE", ee_joints)
    
    stack_joints: deque[JointPoint] = deque(maxlen=len(jp_graph.nodes()))

    stack_joints += list(ground_joints)
    j2link: dict[JointPoint, set[Link]] = {j: set() for j in jp_graph.nodes()}
    for j in ground_joints:
        j2link[j].add(ground_link)
    for ee_j in ee_joints:
        j2link[ee_j].add(ee_link)

    exped_j = set()
    links: list[Link] = [ee_link, ground_link]

    while stack_joints:
        curr_j = stack_joints.pop()
        print(curr_j.name)
        L = next(iter(j2link[curr_j]))
        exped_j.add(curr_j)
        L1 = jp_graph.subgraph(L.joints)
        N = set(jp_graph.neighbors(curr_j)) - L.joints
        nextN = {}
        lenNN = {}
        for n in N:
            nextN[n] = set(jp_graph.neighbors(n))
            lenNN[n] = len(nextN[n] & L.joints)
            j2link[n]
        if len(L.joints) <= 2:
            L2 = Link(joints=(N | set([curr_j])))
            for j in L2.joints:
                j2link[j].add(L2)
        elif len(N) == 1:
            N = N.pop()
            if lenNN[n] == 1:
                L2 = Link(joints=set([N, curr_j]))
                for j in L2.joints:
                    j2link[j].add(L2)
            else:
                L1.joints.add(n)
                j2link[n].add(L1)
                continue
        else:
            more_one_adj_L1 = set(filter(lambda n: lenNN[n] > 1, N))
            for n in more_one_adj_L1:
                L1.joints.add(n)
                j2link[n].add(L1)
            less_one_adj_L1 = N - more_one_adj_L1
            if len(less_one_adj_L1) > 1:
                N = less_one_adj_L1
                L2 = Link(joints= (N | set([curr_j])))
                for j in L2.joints:
                    j2link[j].add(L2)
            else:
                N = less_one_adj_L1.pop()
                L2 = Link(joints=set([N, curr_j]))
                j2link[N].add(L2)
        links.append(L2)
        if isinstance(N, set):
            intersting_joints = set(filter(lambda n: len(j2link[n]) < 2, N))
            stack_joints += list(intersting_joints)
        else:
            intersting_joints = N if len(j2link[N]) < 2 else set()
            stack_joints.append(N)
        stack_joints = deque(filter(lambda j: len(j2link[j]) < 2, stack_joints))
        
    kin_graph = nx.Graph()
    for l in links:
        kin_graph.add_node(l.name, link=l)
    pairs = combinations(links, 2)
    # list_edges = list(map(lambda x: tuple(x), filter(lambda x: len(x[0].joints & x[1].joints)>0, pairs)))
    list_edges = filter(lambda x: len(x[0].joints & x[1].joints) > 0, pairs)
    list_edges = list(
        map(lambda x: x + tuple([(x[0].joints & x[1].joints).pop()]), list_edges)
    )

class Mechanism:
    def __init__(self, graph: nx.Graph = nx.Graph(), main_branch: List[JointPoint] = []) -> None:
        self.graph_representation = graph
        self.main_branch = main_branch

    def define_generalized_coord(self, branches: List[List[JointPoint]]):
        ez = np.array([0,0,1])
        ex = np.array([1,0,0])
        if not self.main_branch:
            branch_g2ee = list(filter(lambda x: x[0].attach_ground and x[-1].attach_endeffector, branches))
            self.main_branch = branch_g2ee[0] if len(branch_g2ee) > 0 else raise Exception("No main branch found")
        for branch in branches:
            trans = np.zeros(3)
            v_w_prev = np.zeros(3)
            Hprev = np.eye(4)
            out = []
            for m in range(len(branch)-1):
                v_w = branch[m+1].r - branch[m].r
                v_l = Hprev @ np.c_[v_w, 0]
                out.append(build_homogeneous_transformation(v_l[:3], branch[m].omg, trans))
                trans = v_w
                Hprev = mr.TransInv(out[-1][0](out[-1][1])) @ mr.TransInv(Hprev) 

    # def add_branch(self, branch: List[JointPoint] | List[List[JointPoint]]):
    #     is_list  = [isinstance(br, List) for br in branch]
    #     if all(is_list):
    #         for b in branch:
    #             self.add_branch(b)
    #     else:
    #         for i in range(len(branch)-1):
    #             if isinstance(branch[i], List):
    #                 for b in branch[i]:
    #                     self.graph_representation.add_edge(b, branch[i+1])
    #             elif isinstance(branch[i+1], List):
    #                 for b in branch[i+1]:
    #                     self.graph_representation.add_edge(branch[i], b)
    #             else:
    #                 self.graph_representation.add_edge(branch[i], branch[i+1])
        
    # def add_branch_with_attrib(self, branch: List[Tuple(JointPoint, dict)] | List[List[Tuple[JointPoint,dict]]]):
    #     is_list  = [isinstance(br, List) for br in branch]
    #     if all(is_list):
    #         for b in branch:
    #             self.add_branch(b)
    #     else:
    #         for ed in branch:
    #                 self.graph_representation.add_edge(ed[0], ed[1], **ed[2])
        
    # def draw_mechanism(self):
    #     pos = {}
    #     for node in self.graph_representation:
    #         pos[node] = [node.pos[0],node.pos[2]]
    #     nx.draw(self.graph_representation, 
    #             pos, 
    #             node_color="w", 
    #             linewidths=3.5, 
    #             edgecolors="k", 
    #             node_shape="o",
    #             node_size=150,
    #             with_labels=False)
    #     plt.axis("equal")