"""This module contains the class for the five bar mechanism topology manager."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from auto_robot_design.description.kinematics import JointPoint
from auto_robot_design.description.builder import add_branch


class MutationType(Enum):
    """Enumerate for mutation types."""
    UNMOVABLE = 0  # Unmovable joint that is not used for optimization
    ABSOLUTE = 1  # The movement of the joint are in the absolute coordinate system and are relative to the initial position
    RELATIVE = 2  # The movement of the joint are relative to some other joint or joints and doesn't have an initial position


@dataclass
class GeneratorInfo:
    """Information for node of a generator."""
    mutation_type: int = MutationType.UNMOVABLE
    initial_coordinate: np.ndarray = np.zeros(3)
    mutation_range: List[Optional[Tuple]] = field(
        default_factory=lambda: [None, None, None])
    relative_to: Optional[Union[JointPoint, List[JointPoint]]] = None


class GraphManager():
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.generator_dict = {}
        self.current_main_branch = []
        self.main_connections = []
        self.mutation_ranges = {}

    def reset(self):
        """Reset the graph builder."""
        self.generator_dict = {}
        self.current_main_branch = []
        self.graph = nx.Graph()
        self.mutation_ranges = {}

    def build_main(self, length: float):
        """Builds the main branch and create nodes for the connections.

        Args:
            length (float): length of the main branch that we use as a reference for all sizes.
        """
        ground_joint = JointPoint(
            r=np.zeros(3),
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=True,
            name="Main_ground"
        )
        self.current_main_branch.append(ground_joint)
        self.generator_dict[ground_joint] = GeneratorInfo()

        ground_connection = JointPoint(
            r=np.array([0, 0, 0.001]),
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=False,  # initial value is false, it should be changed in branch attachment process
            name="Ground_connection"
        )

        self.generator_dict[ground_connection] = GeneratorInfo(MutationType.ABSOLUTE, np.array(
            [0, 0, 0.001]), mutation_range=[(-0.15, 0.15), None, (-0.15, 0.15)])
        self.main_connections.append((ground_connection, []))

        knee_joint_pos = np.array([0, 0, -length/2])
        knee_joint = JointPoint(
            r=knee_joint_pos, w=np.array([0, 1, 0]), name="Main_knee")
        self.current_main_branch.append(knee_joint)
        self.generator_dict[knee_joint] = GeneratorInfo(
            MutationType.ABSOLUTE, knee_joint_pos, mutation_range=[None, None, (-0.1, 0.1)])

        first_connection = JointPoint(r=None, w=np.array([
                                      0, 1, 0]), name="Main_connection_1")
        self.generator_dict[first_connection] = GeneratorInfo(MutationType.RELATIVE, None, mutation_range=[
                                                              (-0.05, 0.05), None, (-0.15, 0.15)], relative_to=[ground_joint, knee_joint])
        self.main_connections.append(
            (first_connection, [ground_joint, knee_joint]))

        ee = JointPoint(
            r=np.array([0, 0, -length]),
            w=np.array([0, 1, 0]),
            attach_endeffector=True,
            name="Main_ee"
        )
        self.current_main_branch.append(ee)
        self.generator_dict[ee] = GeneratorInfo(
            initial_coordinate=np.array([0, 0, -length]))

        second_connection = JointPoint(r=None, w=np.array([
                                       0, 1, 0]), name="Main_connection_2")
        self.generator_dict[second_connection] = GeneratorInfo(MutationType.RELATIVE, None, mutation_range=[
                                                               (-0.05, 0.05), None, (-0.15, 0.15)], relative_to=[knee_joint, ee])
        self.main_connections.append((second_connection, [knee_joint, ee]))

        add_branch(self.graph, self.current_main_branch)

    def build_branch(self, connection_list: List[int]):
        """Generate a trivial branch that only have one node.

        Args:
            connection_list (List[int]): List of connection point indexes that we want to connect the branch to.
        """
        first_joint = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            name="branch_1"
        )
        self.generator_dict[first_joint] = GeneratorInfo(MutationType.RELATIVE, None,
            mutation_range=[(-0.1, 0.1), None, (0, -0.15)],
            relative_to=self.main_connections[connection_list[0]][0])

        for connection in connection_list:
            jp, connect_description = self.main_connections[connection]
            if len(connect_description) == 0:
                # if the connection_description is empty, it means that the connection is directly to the ground
                jp.active = True
                self.graph.add_edge(jp, first_joint)
            else:
                self.graph.add_edge(jp, first_joint)
                for cd in connect_description:
                    self.graph.add_edge(cd, jp)

    def set_topology(self, branch_code=0, connection_list=[0, 2]):
        self.reset()
        self.build_main(0.3)
        self.build_branch(connection_list)

    def set_mutation_ranges(self):
        """Traverse the generator_dict to get all mutable parameters and their ranges.
        """
        keys =list(self.generator_dict)
        for key in keys:
            if key not in self.graph.nodes:
                del self.generator_dict[key]

        for key, value in self.generator_dict.items():
            if value.mutation_type == MutationType.RELATIVE:
                for i, r in enumerate(value.mutation_range):
                    if r is not None:
                        self.mutation_ranges[key.name+'_'+str(i)] = r
            elif value.mutation_type == MutationType.ABSOLUTE:
                for i, r in enumerate(value.mutation_range):
                    if r is not None:
                        self.mutation_ranges[key.name+'_'+str(i)] = (r[0]+value.initial_coordinate[i], r[1]+value.initial_coordinate[i])

    def generate_random_from_mutation_range(self):
        """Sample random values from the mutation ranges.

        Returns:
            List[float]: a vector of parameters that are sampled from the mutation ranges.
        """
        result = []
        for _, value in self.mutation_ranges.items():
            result.append(np.random.uniform(value[0], value[1]))
        return result

    def get_graph(self, parameters:List[float]):
        """Produce a graph of the set topology from the given parameters.

        Args:
            parameters List[float]: list of mutations.

        Raises:
            Exception: raise an exception if the number of parameters is not equal to the number of mutation ranges.

        Returns:
            nx.Graph: the graph of a mechanism with the given parameters.
        """
        if len(parameters) != len(list(self.mutation_ranges.keys())):
            raise ValueError(
                'Wrong number of parameters for graph specification!')

        parameter_counter = 0
        for jp, gi in self.generator_dict.items():
            if jp.r is None:
                jp.r = np.zeros(3)
            if gi.mutation_type == MutationType.ABSOLUTE:
                for i, r in enumerate(gi.mutation_range):
                    if r is not None:
                        jp.r[i] = parameters[parameter_counter]
                        parameter_counter += 1

            elif gi.mutation_type == MutationType.RELATIVE:
                for i, r in enumerate(gi.mutation_range):
                    if r is not None:
                        if isinstance(gi.relative_to, JointPoint):
                            jp.r[i] = gi.relative_to.r[i] + \
                                parameters[parameter_counter]
                            parameter_counter += 1
                        else:
                            if len(gi.relative_to)==2:
                                link_direction = gi.relative_to[0].r - gi.relative_to[1].r
                                if i ==0:
                                    jp.r[i] = np.mean([jp.r[i] for jp in gi.relative_to]) + parameters[parameter_counter]*np.linalg.norm(np.array([-link_direction[2],link_direction[1],link_direction[0]]))
                                elif i == 2:
                                    jp.r[i] = np.mean([jp.r[i] for jp in gi.relative_to]) + parameters[parameter_counter]*np.linalg.norm(link_direction)
                            parameter_counter += 1

        return self.graph
