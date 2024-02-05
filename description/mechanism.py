from collections import deque
from itertools import combinations
from copy import deepcopy

import numpy as np
import numpy.linalg as la

from scipy.spatial.transform import Rotation as R

import modern_robotics as mr

import networkx as nx

from description.kinematics import (
    JointPoint,
    Link,
    get_ground_joints,
    get_endeffector_joints,
)
from description.utils import calc_weight_for_span


def get_rot_matrix_by_vec(v_l, w):
    ez = np.array([0, 0, 1])
    ex = np.array([1, 0, 0])
    angle = np.arccos(np.inner(ez, v_l) / la.norm(ez) / la.norm(v_l))
    out = lambda q: R.from_rotvec(
        w * (angle + np.sign(np.inner(ex, v_l)) * q),
    )
    return out, angle


def build_homogeneous_transformation(v_l, w, v_trans):
    Rj, q_init = get_rot_matrix_by_vec(v_l, w)
    Hj = lambda q: np.r_[
        np.c_[Rj(q), np.array([0, 0, la.norm(v_trans)])], np.array([[0, 0, 0, 1]])
    ]
    return Hj, q_init


def JointPoint2KinematicGraph(jp_graph: nx.Graph):

    ground_joints = set(get_ground_joints(jp_graph))
    ee_joints = set(get_endeffector_joints(jp_graph))
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
                L2 = Link(joints=(N | set([curr_j])))
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

    list_edges = filter(lambda x: len(x[0].joints & x[1].joints) > 0, pairs)
    list_edges = list(
        map(lambda x: x + tuple([(x[0].joints & x[1].joints).pop()]), list_edges)
    )

    for edge in list_edges:
        kin_graph.add_edge(edge[0].name, edge[1].name, joint=edge[-1])
    return kin_graph


def get_span_tree_n_main_branch(graph: nx.Graph, f_weight=calc_weight_for_span):
    weighted_graph = deepcopy(graph)
    for edge in weighted_graph.edges(data=True):
        weighted_graph.add_weighted_edges_from(
            [(edge[0], edge[1], calc_weight_for_span(edge, weighted_graph))]
        )
    span_tree = nx.maximum_spanning_tree(weighted_graph, algorithm="prim")
    main_branch = nx.all_shortest_paths(span_tree, "G", "EE")
    main_branch = sorted([path for path in main_branch], key=lambda x: len(x), reverse=True)[0]
    return span_tree, main_branch


def define_link_frames(
    graph,
    span_tree,
    init_link="G",
    in_joint=None,
    main_branch=[],
    all_joints=set(),
    **kwargs
):
    if init_link == "G" and in_joint is None:
        kwargs = {}
        kwargs["ez"] = np.array([0, 0, 1, 0])
        kwargs["joint2edge"] = {
            data[2]["joint"]: set((data[0], data[1]))
            for data in span_tree.edges(data=True)
        }

        kwargs["get_next_link"] = lambda joint, prev_link: (
            (kwargs["joint2edge"][joint] - set([prev_link])).pop()
        )

        graph.nodes()["EE"]["frame_geom"] = (
            np.array([0, 0, 0]),
            np.array([0, 0, 0, 1]),
        )

        graph.nodes()["G"]["frame"] = (np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
        graph.nodes()["G"]["frame_geom"] = (np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
        graph.nodes()["G"]["H_w_l"] = mr.RpToTrans(np.eye(3), np.zeros(3))
        graph.nodes()["G"]["m_out"] = (
            span_tree[main_branch[0]][main_branch[1]]["joint"],
            main_branch[1],
        )
        graph.nodes()["G"]["out"] = {
            j: kwargs["get_next_link"](j, "G")
            for j in graph.nodes()["G"]["link"].joints
        }
        for j in graph.nodes()["G"]["out"]:
            define_link_frames(
                graph, span_tree, "G", j, main_branch, all_joints, **kwargs
            )
        return graph

    data_prev_link = graph.nodes()[init_link]
    link = kwargs["get_next_link"](in_joint, init_link)

    graph.nodes()[link]["in"] = (in_joint, init_link)
    sorted_out_jj = sorted(
        list(
            graph.nodes()[link]["link"].joints
            & set(kwargs["joint2edge"].keys()) - set([in_joint])
        ),
        key=lambda x: la.norm(x.r - in_joint.r),
        reverse=True,
    )

    H_w_L1 = data_prev_link["H_w_l"]
    if sorted_out_jj:
        if link in main_branch:
            i = np.argwhere(np.array(main_branch) == link).squeeze()
            graph.nodes()[link]["m_out"] = (
                span_tree[main_branch[i]][main_branch[i + 1]]["joint"],
                main_branch[i + 1],
            )
        else:
            graph.nodes()[link]["m_out"] = (
                sorted_out_jj[0],
                kwargs["get_next_link"](sorted_out_jj[0], link),
            )
        graph.nodes()[link]["out"] = {
            j: kwargs["get_next_link"](j, link) for j in sorted_out_jj
        }
        ee_jj = graph.nodes()[link]["m_out"][0].r
        v_w = graph.nodes()[link]["m_out"][0].r - in_joint.r
    else:
        if link == "EE":
            ee_jj = all_joints - set(
                map(lambda x: x[2]["joint"], graph.edges(data=True))
            )
        else:
            ee_jj = (all_joints - set(kwargs["joint2edge"].keys())) & graph.nodes()[
                link
            ]["link"].joints
        if ee_jj:
            # G.nodes()[link]["out"] = {j for j in ee_jj}
            ee_jj = sorted(
                list(ee_jj),
                key=lambda x: la.norm(x.r - in_joint.r),
                reverse=True,
            )
            graph.nodes()[link]["m_out"] = (ee_jj[0],)
            ee_jj = ee_jj[0].r
            v_w = ee_jj - in_joint.r
        else:
            ee_jj = in_joint.r
            v_w = np.array([0, 0, 1])
    ez_l_w = H_w_L1 @ kwargs["ez"]
    angle = np.arccos(np.inner(ez_l_w[:3], v_w) / la.norm(v_w) / la.norm(ez_l_w[:3]))
    axis = mr.VecToso3(ez_l_w[:3]) @ v_w
    axis /= la.norm(axis)

    pos = mr.TransInv(H_w_L1) @ np.array([*in_joint.r.tolist(), 1])
    pos = np.round(pos, 15)
    rot = R.from_rotvec(axis * angle)
    H_w_L2 = H_w_L1 @ mr.RpToTrans(rot.as_matrix(), pos[:3])
    graph.nodes()[link]["H_w_l"] = H_w_L2
    graph.nodes()[link]["frame"] = (pos[:3], rot.as_quat())
    graph.nodes()[link]["frame_geom"] = (
        ((mr.TransInv(H_w_L2) @ np.array([*ee_jj.tolist(), 1])) / 2)[:3],
        np.array([0, 0, 0, 1]),
    )
    if link == "EE":
        return graph
    if graph.nodes()[link].get("out", {}):
        for jj_out in graph.nodes()[link]["out"]:
            if jj_out in kwargs["joint2edge"].keys():
                define_link_frames(
                    graph, span_tree, link, jj_out, main_branch, all_joints, **kwargs
                )
    return graph


# class Mechanism:
#     def __init__(self, graph: nx.Graph = nx.Graph(), main_branch: List[JointPoint] = []) -> None:
#         self.graph_representation = graph
#         self.main_branch = main_branch

#     def define_generalized_coord(self, branches: List[List[JointPoint]]):
#         ez = np.array([0,0,1])
#         ex = np.array([1,0,0])
#         if not self.main_branch:
#             branch_g2ee = list(filter(lambda x: x[0].attach_ground and x[-1].attach_endeffector, branches))
#             self.main_branch = branch_g2ee[0] if len(branch_g2ee) > 0 else raise Exception("No main branch found")
#         for branch in branches:
#             trans = np.zeros(3)
#             v_w_prev = np.zeros(3)
#             Hprev = np.eye(4)
#             out = []
#             for m in range(len(branch)-1):
#                 v_w = branch[m+1].r - branch[m].r
#                 v_l = Hprev @ np.c_[v_w, 0]
#                 out.append(build_homogeneous_transformation(v_l[:3], branch[m].omg, trans))
#                 trans = v_w
#                 Hprev = mr.TransInv(out[-1][0](out[-1][1])) @ mr.TransInv(Hprev)

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
