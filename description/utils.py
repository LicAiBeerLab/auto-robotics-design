import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from description.kinematics import Link

from scipy.spatial.transform import Rotation as R
import modern_robotics as mr

def calc_weight_for_span(edge, graph: nx.Graph):
    # if "EE" in (e.name for e in edge):
    #     length_to_EE = 0
    # else:
    #     length_to_EE = nx.shortest_path_length(graph, source="EE", target=edge)
    length_to_EE = min([nx.shortest_path_length(graph, source="EE", target=e) for e in edge[:2]])
    if edge[-1].active:
        weight = np.round(len(graph.nodes())*10+length_to_EE, 3)
    else:
        weight = np.round(length_to_EE, 3)
    return weight

def calc_weight_for_main_branch(edge, graph: nx.Graph):
    pass

def get_pos(G: nx.Graph):
    pos = {}
    for node in G:
        pos[node] = [node.r[0], node.r[2]]

    return pos

def plot_link(L: Link, graph: nx.Graph, color):
    sub_g_l = graph.subgraph(L.joints)
    pos = get_pos(sub_g_l)
    nx.draw(
        sub_g_l,
        pos,
        node_color= color,
        linewidths=1.5,
        edge_color= color,
        node_shape="o",
        node_size=100,
        width=5,
        with_labels=False,
    )
    
def draw_joint_point(graph: nx.Graph):
    pos = get_pos(graph)
    for node in graph:
        pos[node] = [node.pos[0],node.pos[2]]
    nx.draw(graph, 
            pos, 
            node_color="w", 
            linewidths=3.5, 
            edgecolors="k", 
            node_shape="o",
            node_size=150,
            with_labels=False)
    plt.axis("equal")
    
def draw_kinematic_graph(graph: nx.Graph): 
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["joint"].active]
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["joint"].active]
    pos = nx.spring_layout(graph, seed=7)
    nx.draw_networkx_nodes(graph, pos, node_size=700)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family="sans-serif")

    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    
def draw_link_frames(graph: nx.graph):
    ex = np.array([1, 0, 0, 0])
    ez = np.array([0, 0, 1, 0])
    p = np.array([0, 0, 0, 1])
    H = np.eye(4)
    plt.figure(figsize=(15, 15))
    plt.axis("equal")
    nx.draw(
        graph,
        get_pos(graph),
        node_color="w",
        linewidths=2.5,
        edgecolors="k",
        node_shape="o",
        node_size=150,
        with_labels=False,
    )
    for name in graph.nodes():
        data = graph.nodes(data=True)[name]
        frame = data.get("frame", np.zeros(3))
        geom_l = data.get("frame_geom", np.zeros(3))
        H_w_l = data.get("H_w_l", np.eye(4))
        print(name, frame)
        if frame:
            H = H_w_l 
            ex_l = H @ ex
            ez_l = H @ ez
            p_l = H @ p

            ex_g_l = H @ mr.RpToTrans(R.from_quat(geom_l[1]).as_matrix(), geom_l[0]) @ ex
            ez_g_l = H @ mr.RpToTrans(R.from_quat(geom_l[1]).as_matrix(), geom_l[0]) @ ez
            p_g_l = H @ mr.RpToTrans(R.from_quat(geom_l[1]).as_matrix(), geom_l[0]) @ p

            plt.arrow(p_l[0], p_l[2], ex_l[0] * 0.05, ex_l[2] * 0.05, color="r")
            plt.arrow(p_l[0], p_l[2], ez_l[0] * 0.05, ez_l[2] * 0.05, color="b")
            plt.arrow(p_g_l[0], p_g_l[2], ex_g_l[0] * 0.05, ex_g_l[2] * 0.05, color="g")
            plt.arrow(p_g_l[0], p_g_l[2], ez_g_l[0] * 0.05, ez_g_l[2] * 0.05, color="c")