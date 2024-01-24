from networkx import neighbors
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt
from pyparsing import List, Tuple
from scipy.spatial.transform import Rotation as R

import modern_robotics as mr

import networkx as nx

from description.kinematics import JointPoint
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

def jointpos2kin_tree_repr(graph: nx.Graph):
    kin_tree = nx.DiGraph()

    joint_nodes = graph.nodes(data=True)

    ground_joints = list(sorted(filter(lambda n: n.attach_ground, joint_nodes), key= lambda x: la.norm(x.r)))
    ee_joints = list(filter(lambda n: n.attach_endeffector, joint_nodes))

    ground_link = {"joints":ground_joints, "main":ground_joints[0]}

    ee_link = {"joints":ee_joints, "main":ee_joints[0]}

    neighbor = set(graph.neighbors(ground_link.get("main",ground_joints[0])))
    link = {ground_link.get("main",ground_joints[0])}
    if len(neighbor) > 1:
        for n in neighbor:
            link = link.union(neighbor.intersection(set(graph.neighbors(n))))
    link1 = {"joints":list(link), "main":list(link)[0]}

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

    def add_branch(self, branch: List[JointPoint] | List[List[JointPoint]]):
        is_list  = [isinstance(br, List) for br in branch]
        if all(is_list):
            for b in branch:
                self.add_branch(b)
        else:
            for i in range(len(branch)-1):
                if isinstance(branch[i], List):
                    for b in branch[i]:
                        self.graph_representation.add_edge(b, branch[i+1])
                elif isinstance(branch[i+1], List):
                    for b in branch[i+1]:
                        self.graph_representation.add_edge(branch[i], b)
                else:
                    self.graph_representation.add_edge(branch[i], branch[i+1])
        
    def add_branch_with_attrib(self, branch: List[Tuple(JointPoint, dict)] | List[List[Tuple[JointPoint,dict]]]):
        is_list  = [isinstance(br, List) for br in branch]
        if all(is_list):
            for b in branch:
                self.add_branch(b)
        else:
            for ed in branch:
                    self.graph_representation.add_edge(ed[0], ed[1], **ed[2])
        
    def draw_mechanism(self):
        pos = {}
        for node in self.graph_representation:
            pos[node] = [node.pos[0],node.pos[2]]
        nx.draw(self.graph_representation, 
                pos, 
                node_color="w", 
                linewidths=3.5, 
                edgecolors="k", 
                node_shape="o",
                node_size=150,
                with_labels=False)
        plt.axis("equal")