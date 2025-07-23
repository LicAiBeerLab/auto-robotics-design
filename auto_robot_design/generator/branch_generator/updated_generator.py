from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum
import itertools
from copy import deepcopy
from auto_robot_design.generator.branch_generator.graph_scheme import MutationType, MutationCoordinate, SchemeEE, SchemeJoint, SchemeConnectionJoint
import matplotlib.pyplot as plt
from auto_robot_design.generator.branch_generator.graph_manager import MutableGraphManager
from auto_robot_design.description.utils import draw_joint_point
# We need a dataclass that incorporates all the information about the joints and can be used to build a graph

@dataclass 
class GeneratorConnection:
    scheme_connection: SchemeConnectionJoint = SchemeConnectionJoint()

    start_open:Tuple[int] = ()
    end_open: Tuple[int] = ()
    connected: Optional[Tuple[int, int]] = None  # branch idx and first joint index
    


class Generator2DRotational():
    def __init__(self):
        self.scheme = {0: []}
        # self.branch_connection_dict = {}
        self.connections = {0: []}
        self.topologies = []
        self.branch_idx = 0

    def create_sub_branch(self):
        self.branch_idx += 1
        self.scheme[self.branch_idx] = ["Connection", "Connection"]
        self.connections[self.branch_idx] = []

    def add_joint(self, sp, branch_idx: int = 0):
        if branch_idx not in self.scheme:
            raise ValueError(f"Branch index {branch_idx} does not exist.")
        if branch_idx == 0:
            self.scheme[branch_idx].append(sp)
        else:
            self.scheme[branch_idx].insert(-1, sp)

    def add_connection(self, connection: GeneratorConnection):
        if connection.scheme_connection.connected_to[0] not in self.connections:
            raise ValueError(f"Branch index {connection.scheme_connection.connected_to[0] } does not exist.")
        self.connections[connection.scheme_connection.connected_to[0]].append(connection)
        
    def build_all_topologies(self):
        if self.branch_idx == 0:
            self.topologies.append(self.scheme)
            return self.topologies

        def connect_branch(branch_idx, current_connections=self.connections):
            start_points = []
            for b_idx in range(branch_idx):
                for connection_idx, generator_connection in enumerate(current_connections[b_idx]):
                    if branch_idx in generator_connection.start_open and generator_connection.connected is None:
                        start_points.append((b_idx, connection_idx))

            if not start_points:
                raise ValueError(f"No start connection found for branch {branch_idx}.")

            end_points = []
            for b_idx in range(branch_idx):
                for connection_idx, generator_connection in enumerate(current_connections[b_idx]):
                    if branch_idx in generator_connection.end_open and generator_connection.connected is None:
                        end_points.append((b_idx, connection_idx))

            if not end_points:
                raise ValueError(f"No end connection found for branch {branch_idx}.")

            product = itertools.product(start_points, end_points)
            for pair in product:
                if pair[0][0]== pair[1][0] and pair[0][1] == pair[1][1]:
                    continue
                new_connections = deepcopy(current_connections)
                new_connections[pair[0][0]][pair[0][1]].connected = (0, branch_idx)
                new_connections[pair[1][0]][pair[1][1]].connected = (1, branch_idx)
                if branch_idx+1 in self.scheme:
                    connect_branch(branch_idx + 1, new_connections)
                else:
                    scheme = deepcopy(self.scheme)
                    for b_idx in range(branch_idx + 1):
                        for connection in new_connections[b_idx]:
                            if connection.connected:
                                scheme_entry = connection.scheme_connection
                                if connection.connected[0] == 0:
                                    scheme[connection.connected[1]][0] = scheme_entry
                                    scheme_entry.name = f"b_{connection.connected[1]}_j_0"
                                elif connection.connected[0] == 1:
                                    scheme[connection.connected[1]][-1] = scheme_entry
                                    scheme_entry.name = f"b_{connection.connected[1]}_j_{len(scheme[connection.connected[1]]) - 1}"

                    for key in scheme:
                        if isinstance(scheme[key][-1], str) or isinstance(scheme[key][0], str):
                            raise ValueError(f"Branch {key} not connected properly.")
                    
                    self.topologies.append(scheme)


        connect_branch(1)
        return self.topologies

    
    def vis_branches(self):
        pass

    def filter_cycles(self):
        pass


if __name__ == "__main__":
    generator = Generator2DRotational()
    joint = SchemeJoint(name="b_0_j_0", mutation_type=MutationType.ABSOLUTE, mutation_x=MutationCoordinate(
        freeze=0.0), mutation_y=MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(freeze=0.0), active=True, attach_ground=True)
    generator.add_joint(joint)
    joint = SchemeJoint(name="b_0_j_1", mutation_type=MutationType.ABSOLUTE,
                        mutation_x=MutationCoordinate(mutation_origin=0.05, lower_bound=-0.1, upper_bound=0.1), mutation_y = MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(mutation_origin=-0.2, lower_bound=-0.1, upper_bound=0.1))
    generator.add_joint(joint)
    ee = SchemeEE(name="b_0_j_2", mutation_x=MutationCoordinate(mutation_origin=0.0, lower_bound=-0.1, upper_bound=0.1), mutation_y=MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(mutation_origin=-0.4, lower_bound=-0.1, upper_bound=0.1))
    generator.add_joint(ee)

    scheme_connection = SchemeConnectionJoint(name="G",
                                                mutation_type=MutationType.ABSOLUTE,
                                                mutation_x = MutationCoordinate(mutation_origin=-0.2, upper_bound=0.1, lower_bound=-0.1), 
                                                mutation_y = MutationCoordinate(freeze=0.0),
                                                mutation_z = MutationCoordinate(mutation_origin=0.0, lower_bound=0.0, upper_bound=0.2),
                                            active=True, attach_ground=True, connected_to=(0,-1))
    connection = GeneratorConnection(scheme_connection, start_open=tuple([1]), end_open=(), connected=None)
    generator.add_connection(connection)
    scheme_connection = SchemeConnectionJoint(name="CJ_0",
                                                mutation_type=MutationType.RELATIVE_PERCENTAGE,
                                                mutation_x = MutationCoordinate(mutation_origin=0, lower_bound=-0.2, upper_bound=0.2),
                                                mutation_y = MutationCoordinate(freeze=0.0),
                                                mutation_z = MutationCoordinate(mutation_origin=0.0, lower_bound=-0.4, upper_bound=0.4),
                                            active=True, attach_ground=False, connected_to=(0, 0))
    connection = GeneratorConnection(scheme_connection, start_open=tuple([1]), end_open=(), connected=None)
    generator.add_connection(connection)
    scheme_connection = SchemeConnectionJoint(name="CJ_1",
                                                mutation_type=MutationType.RELATIVE_PERCENTAGE, 
                                                mutation_x = MutationCoordinate(mutation_origin=0, lower_bound=-0.2, upper_bound=0.2),
                                                mutation_y = MutationCoordinate(freeze=0.0),
                                                mutation_z = MutationCoordinate(mutation_origin=0.0, lower_bound=-0.4, upper_bound=0.4),
                                            active=False, attach_ground=False, connected_to=(0, 1))
    connection = GeneratorConnection(scheme_connection, start_open=(), end_open=tuple([1]), connected=None)
    generator.add_connection(connection)

    generator.create_sub_branch()
    joint = SchemeJoint(name="b_1_j_0", mutation_type=MutationType.RELATIVE, mutation_x=MutationCoordinate(mutation_origin=0.0, lower_bound=-0.1, upper_bound=0.1), mutation_y=MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(mutation_origin=-0.2, lower_bound=-0.1, upper_bound=0.1))
    generator.add_joint(joint, 1)


    generator.build_all_topologies()
    print(len(generator.topologies))

    for topology in generator.topologies:
        manager = MutableGraphManager(topology)
        manager.build_graph()
        manager.get_mutation_ranges()
        graph = manager.get_central_graph()
        draw_joint_point(graph)
        plt.show()


    
