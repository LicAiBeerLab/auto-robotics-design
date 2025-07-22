from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum
import itertools
from copy import deepcopy
from auto_robot_design.generator.user_generator.graph_scheme import MutationType, MutationCoordinate, SchemePoint, SchemeEE, SchemeJoint, SchemeConnectionJoint

class PointType(Enum):
    """Enumerate for point types."""
    JOINT = 0  # A joint that can be moved
    CONNECTION = 1  # A fixed point that cannot be moved
    END_EFFECTOR = 2  # The end effector of the robot, which is the last joint in the chain

class MutationType(Enum):
    """Enumerate for mutation types."""
    # UNMOVABLE = 0  # Unmovable joint that is not used for optimization
    ABSOLUTE = 1  # The movement of the joint are in the absolute coordinate system and are relative to the initial position
    RELATIVE = 2  # The movement of the joint are relative to some other joint or joints and doesn't have an initial position
    # The movement of the joint are relative to some other joint or joints and doesn't have an initial position. The movement is in percentage of the distance between the joints.
    RELATIVE_PERCENTAGE = 3

# We need a dataclass that incorporates all the information about the joints and can be used to build a graph 



@dataclass 
class GeneratorConnection:
    scheme_connection: SchemeConnectionJoint = SchemeConnectionJoint()
    dependent_open: bool = False
    independent_open: bool = False
    connected: Optional[Tuple[int, int]] = None  # branch idx and first joint index
    


class Generator2DRotational():
    def __init__(self):
        self.scheme = {0: []}
        self.branch_connection_dict = {}
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
            dependent_points = []
            for branch in self.branch_connection_dict[branch_idx][0]:
                for connection_idx, generator_connection in enumerate(current_connections[branch]):
                    if generator_connection.dependent_open and generator_connection.connected is None:
                        dependent_points.append((branch, connection_idx))
            if not dependent_points:
                raise ValueError(f"No dependent connection found for branch {branch_idx}.")
            
            independent_points = []
            for branch in self.branch_connection_dict[branch_idx][1]:
                for connection_idx, generator_connection in enumerate(current_connections[branch]):
                    if generator_connection.independent_open and generator_connection.connected is None:
                        independent_points.append((branch, connection_idx))
            if not independent_points:
                raise ValueError(f"No independent connection found for branch {branch_idx}.")

            product = itertools.product(dependent_points, independent_points)
            for pair in product:
                if pair[0][0]== pair[1][0] and pair[0][1] == pair[1][1]:
                    continue
                new_connections = deepcopy(current_connections)
                new_connections[pair[0][0]][pair[0][1]].connected = (0, branch_idx)
                new_connections[pair[1][0]][pair[1][1]].connected = (1, branch_idx)
                if branch_idx+1 in self.branch_connection_dict:
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
    connection = GeneratorConnection(scheme_connection, dependent_open=True, independent_open=False, connected=None)
    generator.add_connection(connection)
    scheme_connection = SchemeConnectionJoint(name="CJ_0",
                                              mutation_type=MutationType.RELATIVE_PERCENTAGE, 
                                              mutation_x = MutationCoordinate(mutation_origin=0, lower_bound=-0.2, upper_bound=0.2), 
                                              mutation_y = MutationCoordinate(freeze=0.0), 
                                              mutation_z = MutationCoordinate(mutation_origin=0.0, lower_bound=-0.4, upper_bound=0.4),
                                            active=True, attach_ground=False, connected_to=(0, 0))
    connection = GeneratorConnection(scheme_connection, dependent_open=True, independent_open=False, connected=None)
    generator.add_connection(connection)
    scheme_connection = SchemeConnectionJoint(name="CJ_1",
                                              mutation_type=MutationType.RELATIVE_PERCENTAGE, 
                                              mutation_x = MutationCoordinate(mutation_origin=0, lower_bound=-0.2, upper_bound=0.2), 
                                              mutation_y = MutationCoordinate(freeze=0.0), 
                                              mutation_z = MutationCoordinate(mutation_origin=0.0, lower_bound=-0.4, upper_bound=0.4),
                                            active=True, attach_ground=False, connected_to=(0, 1))
    connection = GeneratorConnection(scheme_connection, dependent_open=False, independent_open=True, connected=None)
    generator.add_connection(connection)

    generator.create_sub_branch()
    joint = SchemeJoint(name="b_1_j_0", mutation_type=MutationType.RELATIVE, mutation_x=MutationCoordinate(mutation_origin=0.0, lower_bound=-0.1, upper_bound=0.1), mutation_y=MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(mutation_origin=-0.2, lower_bound=-0.1, upper_bound=0.1))
    generator.add_joint(joint, 1)

    generator.branch_connection_dict[1] = ([0], [0])

    generator.build_all_topologies()
    print(len(generator.topologies))




