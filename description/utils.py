from matplotlib import legend, scale
import networkx as nx
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from description.kinematics import Link

from scipy.spatial.transform import Rotation as R
import modern_robotics as mr


def calc_weight_for_span(edge, graph: nx.Graph):
    length_to_EE = [nx.shortest_path_length(graph, source="EE", target=e) for e in edge[:2]]
    edge_min_length = np.argmin(length_to_EE)
    min_length_to_EE = min(length_to_EE)
    next_joints_link = graph.nodes()[edge[edge_min_length]]["link"].joints - set([edge[-1]["joint"]])
    if next_joints_link:
        length_next_j_to_j = max([la.norm(edge[-1]["joint"].r - next_j.r) for next_j in next_joints_link])
    else:
        length_next_j_to_j = 0
    if edge[-1]["joint"].active:
        weight = np.round(len(graph.nodes()) * 100 + min_length_to_EE * 10 + length_next_j_to_j/10, 3)
    else:
        weight = np.round(min_length_to_EE * 10 + length_next_j_to_j/10, 3)
    print(edge[0], edge[1], weight)
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
        node_color=color,
        linewidths=1.5,
        edge_color=color,
        node_shape="o",
        node_size=100,
        width=5,
        with_labels=False,
    )

def draw_links(kinematic_graph: nx.Graph, JP_graph: nx.Graph):
    links = kinematic_graph.nodes()
    colors = range(len(links))
    draw_joint_point(JP_graph) 
    for link, color in zip(links, colors):
        sub_graph_l = JP_graph.subgraph(links[link]["link"].joints)
        name_link = links[link]["link"].name
        options = {
            "node_color": "orange",
            "edge_color": "orange",
            "alpha": color/len(links),
            "width": 5,
            "edge_cmap": plt.cm.Blues,
            "linewidths": 1.5,
            "node_shape": "o",
            "node_size": 100,
            "with_labels": False,
        }
        pos = get_pos(sub_graph_l)
        list_pos = [p for p in pos.values()]
        if len(list_pos) == 1:
            pos_name = np.array(list_pos).squeeze() + np.ones(2) * 0.2
        else:
            pos_name = np.mean([p for p in pos.values()], axis=0)
        nx.draw(sub_graph_l, pos, **options)
        plt.text(pos_name[0],pos_name[1], name_link, fontsize=15)

def draw_joint_point(graph: nx.Graph):
    pos = get_pos(graph)
    G_pos = np.array(
        list(
        map(
            lambda n: [n.r[0], n.r[2]],
            filter(lambda n: n.attach_ground, graph),
        )
        )
    )
    EE_pos = np.array(
        list(
        map(
            lambda n: [n.r[0], n.r[2]],
            filter(lambda n: n.attach_endeffector, graph),
        )
        )
    )
    active_j_pos = np.array(
        list(
        map(
            lambda n: [n.r[0], n.r[2]],
            filter(lambda n: n.active, graph),
        )
        )
    )
    labels = {n:n.name for n in graph.nodes()}
    nx.draw(
        graph,
        pos,
        node_color="w",
        linewidths=3.5,
        edgecolors="k",
        node_shape="o",
        node_size=150,
        with_labels=False,
    )
    pos_labels = {g:np.array(p) + np.array([-0.2, 0.2]) for g, p in pos.items()}
    nx.draw_networkx_labels(
        graph,
        pos_labels,
        labels,
        font_color = "r",
        font_family = "monospace"

    )
    plt.plot(G_pos[:,0], G_pos[:,1], "ok", label="Ground")
    plt.plot(EE_pos[:,0], EE_pos[:,1], "ob", label="EndEffector")
    plt.plot(active_j_pos[:,0], active_j_pos[:,1], "og",
             markersize=20, 
             fillstyle="none", label="Active")
    plt.legend()
    plt.axis("equal")


def draw_kinematic_graph(graph: nx.Graph):
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["joint"].active]
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if not d["joint"].active]
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


def draw_link_frames(kinematic_graph: nx.Graph):
    ex = np.array([1, 0, 0, 0])
    ez = np.array([0, 0, 1, 0])
    p = np.array([0, 0, 0, 1])
    H = np.eye(4)
    max_length = np.max([la.norm(kinematic_graph.nodes()[n]["frame_geom"][0]) for n in  kinematic_graph.nodes()])
    scale = max_length/4
    plt.figure(figsize=(15,15))
    for name in kinematic_graph.nodes():
        data = kinematic_graph.nodes(data=True)[name]
        frame = data.get("frame", np.zeros(3))
        geom_l = data.get("frame_geom", np.zeros(3))
        H_w_l = data.get("H_w_l", np.eye(4))
        if frame:
            H = H_w_l
            ex_l = H @ ex
            ez_l = H @ ez
            p_l = H @ p

            ex_g_l = (
                H @ mr.RpToTrans(R.from_quat(geom_l[1]).as_matrix(), geom_l[0]) @ ex
            )
            ez_g_l = (
                H @ mr.RpToTrans(R.from_quat(geom_l[1]).as_matrix(), geom_l[0]) @ ez
            )
            p_g_l = H @ mr.RpToTrans(R.from_quat(geom_l[1]).as_matrix(), geom_l[0]) @ p

            plt.arrow(p_l[0], p_l[2], ex_l[0] * scale, ex_l[2] * scale, color="r")
            plt.arrow(p_l[0], p_l[2], ez_l[0] * scale, ez_l[2] * scale, color="b")
            plt.arrow(p_g_l[0], p_g_l[2], ex_g_l[0] * scale, ex_g_l[2] * scale, color="g")
            plt.arrow(p_g_l[0], p_g_l[2], ez_g_l[0] * scale, ez_g_l[2] * scale, color="c")


def calculate_inertia(length):
    Ixx = 1 / 12 * 1 * (0.001**2 * length**2)
    Iyy = 1 / 12 * 1 * (0.001**2 * length**2)
    Izz = 1 / 12 * 1 * (0.001**2 * 0.001**2)
    return {"ixx": Ixx, "ixy": 0, "ixz": 0, "iyy": Iyy, "iyz": 0, "izz": Izz}
