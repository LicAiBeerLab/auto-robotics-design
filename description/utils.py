import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from description.kinematics import Link

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
    
def draw_mechanism(graph: nx.Graph):
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