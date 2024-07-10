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
    # The movement of the joint are relative to some other joint or joints and doesn't have an initial position. The movement is in percentage of the distance between the joints.
    RELATIVE_PERCENTAGE = 3


@dataclass
class GeneratorInfo:
    """Information for node of a generator."""
    mutation_type: int = MutationType.UNMOVABLE
    initial_coordinate: np.ndarray = np.zeros(3)
    mutation_range: List[Optional[Tuple]] = field(
        default_factory=lambda: [None, None, None])
    relative_to: Optional[Union[JointPoint, List[JointPoint]]] = None


@dataclass
class ConnectionInfo:
    """Description of a point for branch connection."""
    connection_jp: JointPoint
    jp_connection_to_main: List[JointPoint]
    relative_mutation_range: List[Optional[Tuple]]


class GraphManager2L():
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.generator_dict = {}
        self.current_main_branch = []
        self.main_connections: List[ConnectionInfo] = []
        self.mutation_ranges = {}

    def reset(self):
        """Reset the graph builder."""
        self.generator_dict = {}
        self.current_main_branch = []
        self.graph = nx.Graph()
        self.mutation_ranges = {}

    def build_main(self, length: float, fully_actuated: bool = False):
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

        ground_connection_jp = JointPoint(
            r=np.array([0, 0, 0.001]),
            w=np.array([0, 1, 0]),
            attach_ground=True,
            active=False,  # initial value is false, it should be changed in branch attachment process
            name="Ground_connection"
        )

        self.generator_dict[ground_connection_jp] = GeneratorInfo(MutationType.ABSOLUTE, np.array(
            [0, 0, 0.001]), mutation_range=[(-0.15, 0.), None, (-0.1, 0.1)])
        ground_connection_description = ConnectionInfo(
            ground_connection_jp, [], [(-0.1, 0.1), None, (-0.15, 0)])
        self.main_connections.append(ground_connection_description)

        knee_joint_pos = np.array([0, 0, -length/2])
        knee_joint = JointPoint(
            r=knee_joint_pos, w=np.array([0, 1, 0]), active=fully_actuated, name="Main_knee")
        self.current_main_branch.append(knee_joint)
        self.generator_dict[knee_joint] = GeneratorInfo(
            MutationType.ABSOLUTE, knee_joint_pos.copy(), mutation_range=[None, None, (-0.1, 0.1)])

        first_connection = JointPoint(r=None, w=np.array([
                                      0, 1, 0]), name="Main_connection_1")
        # self.generator_dict[first_connection] = GeneratorInfo(MutationType.RELATIVE, None, mutation_range=[
        #                                                       (-0.05, 0.05), None, (-0.15, 0.15)], relative_to=[ground_joint, knee_joint])
        self.generator_dict[first_connection] = GeneratorInfo(MutationType.RELATIVE_PERCENTAGE, None, mutation_range=[
                                                              (-0.2, 0.2), None, (-0.45, 0.45)], relative_to=[ground_joint, knee_joint])
        first_connection_description = ConnectionInfo(
            first_connection, [ground_joint, knee_joint], [(-0.1, 0.0), None, (-0.1, 0.1)])
        self.main_connections.append(first_connection_description)

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
        # self.generator_dict[second_connection] = GeneratorInfo(MutationType.RELATIVE, None, mutation_range=[
        #                                                        (-0.05, 0.05), None, (-0.15, 0.15)], relative_to=[knee_joint, ee])
        self.generator_dict[second_connection] = GeneratorInfo(MutationType.RELATIVE_PERCENTAGE, None, mutation_range=[
                                                               (-0.2, 0.2), None, (-0.45, 0.45)], relative_to=[knee_joint, ee])
        second_connection_description = ConnectionInfo(
            second_connection, [knee_joint, ee], [(-0.1, 0), None, (0, 0.1)])
        self.main_connections.append(second_connection_description)

        add_branch(self.graph, self.current_main_branch)

    def build_3n2p_branch(self, connection_list: List[int]):
        """Generate a trivial branch that only have one node.

        Args:
            connection_list (List[int]): List of connection point indexes that we want to connect the branch to.
        """
        branch_joint = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            name="branch_1"
        )
        self.generator_dict[branch_joint] = GeneratorInfo(MutationType.RELATIVE, None,
                                                          mutation_range=self.main_connections[
                                                              connection_list[0]].relative_mutation_range,
                                                          relative_to=self.main_connections[connection_list[0]].connection_jp)

        for connection in connection_list:
            connection_description = self.main_connections[connection]
            jp = connection_description.connection_jp
            jp_connection_to_main = connection_description.jp_connection_to_main
            if len(jp_connection_to_main) == 0:
                # if the connection_description is empty, it means that the connection is directly to the ground
                self.graph.add_edge(jp, branch_joint)

            else:
                self.graph.add_edge(jp, branch_joint)
                for cd in jp_connection_to_main:
                    self.graph.add_edge(cd, jp)

            if connection == min(connection_list):
                jp.active = True

    def build_6n4p_symmetric(self, connection_list: List[int]):
        branch_jp_counter = 0
        branch_joints = []
        for connection in connection_list:
            connection_description = self.main_connections[connection]
            jp = connection_description.connection_jp
            jp_connection_to_main = connection_description.jp_connection_to_main
            branch_jp = JointPoint(
                r=None,
                w=np.array([0, 1, 0]),
                name=f"branch_{branch_jp_counter}"
            )
            branch_joints.append(branch_jp)
            branch_jp_counter += 1
            self.generator_dict[branch_jp] = GeneratorInfo(MutationType.RELATIVE, None,
                                                           mutation_range=self.main_connections[
                                                               connection].relative_mutation_range,
                                                           relative_to=self.main_connections[connection].connection_jp)
            if len(jp_connection_to_main) == 0:
                self.graph.add_edge(jp, branch_jp)
                jp.active = True

            elif len(jp_connection_to_main) == 2:
                self.graph.add_edge(jp, branch_jp)
                for cd in jp_connection_to_main:
                    self.graph.add_edge(cd, jp)

        self.graph.add_edge(branch_joints[0], branch_joints[1])
        self.graph.add_edge(branch_joints[1], branch_joints[2])
        self.graph.add_edge(branch_joints[2], branch_joints[0])

    def build_6n4p_asymmetric(self, connection_list: float):
        """Connects the 4l asymmetric branch to the main branch

        Args:
            connection_list (float): list of connecting points indexes for branch connection to main. Linkage chain of the largest length between the first and the third indices
        """
        if connection_list[0]+connection_list[1] == 1:
            branch_1_active = True
        else:
            branch_1_active = False

        connection_description = self.main_connections[connection_list[0]]
        jp = connection_description.connection_jp
        jp_connection_to_main = connection_description.jp_connection_to_main
        branch_jp_0 = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            name="branch_0"
        )
        self.generator_dict[branch_jp_0] = GeneratorInfo(MutationType.RELATIVE, None,
                                                         mutation_range=connection_description.relative_mutation_range,
                                                         relative_to=jp)
        if len(jp_connection_to_main) == 0:
            self.graph.add_edge(jp, branch_jp_0)
            jp.active = not branch_1_active
        else:
            self.graph.add_edge(jp, branch_jp_0)
            for cd in jp_connection_to_main:
                self.graph.add_edge(cd, jp)

        connection_description = self.main_connections[connection_list[1]]
        jp = connection_description.connection_jp
        jp_connection_to_main = connection_description.jp_connection_to_main

        branch_jp_1 = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            name="branch_1"
        )
        branch_jp_1.active = branch_1_active
        self.generator_dict[branch_jp_1] = GeneratorInfo(MutationType.RELATIVE, None,
                                                         mutation_range=connection_description.relative_mutation_range,
                                                         relative_to=jp)
        if len(jp_connection_to_main) == 0:
            self.graph.add_edge(jp, branch_jp_1)
            jp.active = not branch_1_active  
        else:
            self.graph.add_edge(jp, branch_jp_1)
            for cd in jp_connection_to_main:
                self.graph.add_edge(cd, jp)
        self.graph.add_edge(branch_jp_0, branch_jp_1)
        self.graph.add_edge(branch_jp_0, jp)
        connection_description = self.main_connections[connection_list[2]]
        jp = connection_description.connection_jp
        jp_connection_to_main = connection_description.jp_connection_to_main
        branch_jp_2 = JointPoint(
            r=None,
            w=np.array([0, 1, 0]),
            name="branch_2"
        )
        self.generator_dict[branch_jp_2] = GeneratorInfo(MutationType.RELATIVE, None,
                                                         mutation_range=connection_description.relative_mutation_range,
                                                         relative_to=jp)
        if len(jp_connection_to_main) == 0:
            self.graph.add_edge(jp, branch_jp_2)
            if not branch_1_active:
                jp.active = True
        else:
            self.graph.add_edge(jp, branch_jp_2)
            for cd in jp_connection_to_main:
                self.graph.add_edge(cd, jp)
        self.graph.add_edge(branch_jp_2, branch_jp_1)

    def set_mutation_ranges(self):
        """Traverse the generator_dict to get all mutable parameters and their ranges.
        """
        keys = list(self.generator_dict)
        for key in keys:
            if key not in self.graph.nodes:
                del self.generator_dict[key]

        for key, value in self.generator_dict.items():
            if value.mutation_type == MutationType.RELATIVE or value.mutation_type == MutationType.RELATIVE_PERCENTAGE:
                for i, r in enumerate(value.mutation_range):
                    if r is not None:
                        self.mutation_ranges[key.name+'_'+str(i)] = r
            elif value.mutation_type == MutationType.ABSOLUTE:
                for i, r in enumerate(value.mutation_range):
                    if r is not None:
                        self.mutation_ranges[key.name+'_'+str(i)] = (
                            r[0]+value.initial_coordinate[i], r[1]+value.initial_coordinate[i])

    def generate_random_from_mutation_range(self):
        """Sample random values from the mutation ranges.

        Returns:
            List[float]: a vector of parameters that are sampled from the mutation ranges.
        """
        result = []
        for _, value in self.mutation_ranges.items():
            result.append(np.random.uniform(value[0], value[1]))
        return result

    def generate_central_from_mutation_range(self):
        """Return values from center of the mutation ranges.

        Returns:
            List[float]: a vector of parameters that are centered on the mutation ranges.
        """
        result = []
        for _, value in self.mutation_ranges.items():
            result.append((value[0]+value[1])/2)
        return result

    def get_graph(self, parameters: List[float]):
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
                if isinstance(gi.relative_to, list) and len(gi.relative_to) == 2:
                    jp.r = (gi.relative_to[0].r + gi.relative_to[1].r)/2

                for i, r in enumerate(gi.mutation_range):
                    if r is not None:
                        if isinstance(gi.relative_to, JointPoint):
                            jp.r[i] = gi.relative_to.r[i] + \
                                parameters[parameter_counter]
                            parameter_counter += 1
                        else:
                            if len(gi.relative_to) == 2:
                                link_direction = gi.relative_to[0].r - \
                                    gi.relative_to[1].r
                                link_ortogonal = np.array(
                                    [-link_direction[2], link_direction[1], link_direction[0]])
                                link_length = np.linalg.norm(link_direction)
                                if i == 0:
                                    jp.r += parameters[parameter_counter] * \
                                        link_ortogonal/link_length
                                if i == 2:
                                    jp.r += parameters[parameter_counter] * \
                                        link_direction/link_length
                            #     jp.r += parameters[parameter_counter]*link_direction/link_length
                            #     jp.r += parameters[parameter_counter]*np.array([-link_direction[2],link_direction[1],link_direction[0]])/link_length
                            parameter_counter += 1

            elif gi.mutation_type == MutationType.RELATIVE_PERCENTAGE:
                if isinstance(gi.relative_to, list) and len(gi.relative_to) == 2:
                    jp.r = (gi.relative_to[0].r + gi.relative_to[1].r)/2
                for i, r in enumerate(gi.mutation_range):
                    if r is not None:
                        if isinstance(gi.relative_to, JointPoint):
                            raise ValueError(
                                'Relative percentage mutation type should have a list of joints as relative_to')
                        else:
                            if len(gi.relative_to) == 2:
                                link_direction = gi.relative_to[0].r - \
                                    gi.relative_to[1].r
                                link_ortogonal = np.array(
                                    [-link_direction[2], link_direction[1], link_direction[0]])
                                link_length = np.linalg.norm(link_direction)
                                if i == 0:
                                    jp.r += parameters[parameter_counter] * \
                                        link_ortogonal
                                if i == 2:
                                    jp.r += parameters[parameter_counter] * \
                                        link_direction
                            parameter_counter += 1
        return self.graph


def get_preset_by_index(idx: int):
    if idx == -1:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4, fully_actuated=True)
        gm.set_mutation_ranges()
        return gm

    if idx == 0:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_3n2p_branch([0, 2])
        gm.set_mutation_ranges()
        return gm

    if idx == 1:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_3n2p_branch([1, 2])
        gm.set_mutation_ranges()
        return gm

    if idx == 2:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_symmetric([0, 1, 2])
        gm.set_mutation_ranges()
        return gm

    if idx == 3:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([0, 1, 2])
        gm.set_mutation_ranges()
        return gm

    if idx == 4:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([0, 2, 1])
        gm.set_mutation_ranges()
        return gm

    if idx == 5:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([1, 0, 2])
        gm.set_mutation_ranges()
        return gm

    if idx == 6:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([1, 2, 0])
        gm.set_mutation_ranges()
        return gm

    if idx == 7:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([2, 0, 1])
        gm.set_mutation_ranges()
        return gm

    if idx == 8:
        gm = GraphManager2L()
        gm.reset()
        gm.build_main(0.4)
        gm.build_6n4p_asymmetric([2, 1, 0])
        gm.set_mutation_ranges()
        return gm
