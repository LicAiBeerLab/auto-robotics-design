"""This module contains the class for the five bar mechanism topology manager."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union
import networkx as nx
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch


class MutationType(Enum):
    """Enumerate for mutation types."""
    #UNMOVABLE = 0  # Unmovable joint that is not used for optimization
    ABSOLUTE = 1  # The movement of the joint are in the absolute coordinate system and are relative to the initial position
    RELATIVE = 2  # The movement of the joint are relative to some other joint or joints and doesn't have an initial position
    # The movement of the joint are relative to some other joint or joints and doesn't have an initial position. The movement is in percentage of the distance between the joints.
    RELATIVE_PERCENTAGE = 3


@dataclass
class GeneratorInfo:
    """Information for node of a generator."""
    mutation_type: int = MutationType.ABSOLUTE
    initial_coordinate: np.ndarray = np.zeros(3)
    mutation_range: List[Optional[Tuple]] = field(
        default_factory=lambda: [None, None, None])
    relative_to: Optional[Union[JointPoint, List[JointPoint]]] = None
    freeze_pos: List[Optional[Tuple]] = field(
        default_factory=lambda: [None, 0, None])


@dataclass
class ConnectionInfo:
    """Description of a point for branch connection."""
    connection_jp: JointPoint
    jp_connection_to_main: List[JointPoint]
    relative_mutation_range: List[Optional[Tuple]] # this parameter is used to set the mutation range of the branch joints


class TopologyManager2D():
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.mutation_ranges = {}
        self.branch_ends = []
        self.edges = []
        self.generator_dict = {}


    def add_absolute_node(self, jp:JointPoint, initial_coordinates, mutation_range, freeze_pos=[None,0,None],parent_branch_idx=None):
        if jp.name in self.mutation_ranges:
            raise Exception(f"Joint point {jp.name} already exists in the graph.")

        if parent_branch_idx == None:
            self.branch_ends.append(jp)
        else:
            self.graph.add_edge(self.branch_ends[parent_branch_idx], jp)
            self.branch_ends[parent_branch_idx] = jp
            self.edges.append((self.branch_ends[parent_branch_idx], jp))
        self.generator_dict[jp.name] = GeneratorInfo(mutation_type=MutationType.ABSOLUTE, initial_coordinates=initial_coordinates, mutation_range=mutation_range,freeze_pos=freeze_pos)
    
    def add_relative_node(self, jp:JointPoint, mutation_range, relative_to, parent_branch_idx=None, freeze_pos=[None,0,None]):
        if jp.name in self.mutation_ranges:
            raise Exception(f"Joint point {jp.name} already exists in the graph.")

        if parent_branch_idx == None:
            raise Exception("Relative node must have a parent branch")
        else:
            self.graph.add_edge(self.branch_ends[parent_branch_idx], jp)
            self.branch_ends[parent_branch_idx] = jp

        self.generator_dict[jp.name] = GeneratorInfo(mutation_type=MutationType.RELATIVE, relative_to=relative_to, mutation_range=mutation_range,freeze_pos=freeze_pos)

    def add_connection_node(self,jp,mutation_range, parent_pair, freeze_pos=[None,0,None]):
        self.graph.add_edge(parent_pair[0], jp)
        self.graph.add_edge(parent_pair[1], jp)
        self.generator_dict[jp.name] = GeneratorInfo(mutation_type=MutationType.RELATIVE, relative_to=parent_pair, mutation_range=mutation_range,freeze_pos=freeze_pos)
    

    def get_pos(self):
        """Return the dictionary of type {label: [x_coordinate, z_coordinate]} for the JP graph

        Args:
            G (nx.Graph): a graph with JP nodes

        Returns:
            dict: dictionary of type {node: [x_coordinate, z_coordinate]}
        """
        pos = {}
        for node in self.graph:
            pos[node] = [node.r[0], node.r[2]]

        return pos

    def visualize(self, legend=True,draw_labels=True):
        'Visualize the current graph'
        pos = self.get_pos()
        # get positions of all ground nodes
        G_pos = np.array(
            list(
            map(
                lambda n: [n.r[0], n.r[2]],
                filter(lambda n: n.attach_ground, self.graph),
            )
            )
        )
        # get positions of all end effector nodes
        EE_pos = np.array(
            list(
            map(
                lambda n: [n.r[0], n.r[2]],
                filter(lambda n: n.attach_endeffector, self.graph),
            )
            )
        )
        # get positions of all active joints
        active_j_pos = np.array(
            list(
            map(
                lambda n: [n.r[0], n.r[2]],
                filter(lambda n: n.active, self.graph),
            )
            )
        )
        if draw_labels: 
            labels = {n:n.name for n in self.graph.nodes()}
        else:
            labels = {n:str() for n in self.graph.nodes()}
        nx.draw(
            self.graph,
            pos,
            node_color="w",
            linewidths=3.5,
            edgecolors="k",
            node_shape="o",
            node_size=150,
            with_labels=False,
        )
        #pos_labels = {g:np.array(p) + np.array([-0.2, 0.2])*la.norm(EE_pos)/5 for g, p in pos.items()}
        pos_labels = {}
        pos_additions = [np.array([0.2, 0.2])*la.norm(EE_pos)/5, np.array([0.2, -0.2])*la.norm(EE_pos)/5, 
                        np.array([0.2,-0.2])*la.norm(EE_pos)/5, np.array([-0.2, -0.2])*la.norm(EE_pos)/5]
        for g,p in pos.items():
            pos_flag = False
            for pos_addition in pos_additions:
                new_pos = np.array(p) + pos_addition
                if all([la.norm(new_pos-op)>la.norm(EE_pos)/5 for op in pos_labels.values()]):
                    pos_labels[g] = new_pos
                    pos_flag = True
                    break
            if not pos_flag:
                pos_labels[g] = np.array(p)

        nx.draw_networkx_labels(
            graph,
            pos_labels,
            labels,
            font_color = "r",
            font_family = "monospace"

        )
        if nx.is_weighted(graph):
            edge_labels = nx.get_edge_attributes(graph, "weight")
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels,
                font_color = "c",
                font_family = "monospace"

            )
        plt.plot(G_pos[:,0], G_pos[:,1], "ok", label="Ground")
        plt.plot(EE_pos[:,0], EE_pos[:,1], "ob", label="EndEffector")
        plt.plot(active_j_pos[:,0], active_j_pos[:,1], "og",
                markersize=20, 
                fillstyle="none", label="Active")
        if draw_labels: plt.legend()
        plt.axis("equal")