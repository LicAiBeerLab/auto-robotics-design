from dataclasses import dataclass
from typing import Tuple
from enum import Enum
import itertools

class PointType(Enum):
    """Enumerate for point types."""
    JOINT = 0  # A joint that can be moved
    FIXED = 1  # A fixed point that cannot be moved
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
class GeneratorPoint:
    name:str = "J"
    point_type: PointType = PointType.JOINT

    mutation_type: int = MutationType.ABSOLUTE
    coordinates: Tuple[float, float, float]= (0.0, 0.0, 0.0)
    mutation_x:Tuple[float, float]= (0.0, 0.0)
    mutation_y: Tuple[float, float]= (0.0, 0.0)
    mutation_z: Tuple[float, float]= (0.0, 0.0)


@dataclass 
class GeneratorConnection:
    name: str = "C"
    branch_idx: int = 0 
    start_joint: int = -1
    dependent_open: bool = False
    independent_open: bool = False





class Generator2DRotational():
    def __init__(self):
        self.branch_point_dict = {0: []}
        self.branch_connection_dict = {}
        self.connections = []
        self.topologies = []


    def create_sub_branch(self, idx):
        if idx not in self.branch_dict:
            self.branch_dict[idx] = []
        else:
            raise ValueError(f"Branch {idx} already exists.")

    def add_joint(self, joint: GeneratorPoint, idx: int = 0):
        self.branch_dict[idx].append(joint)


    def add_connection(self, connection: GeneratorConnection):
        if connection.branch_idx not in self.branch_connection_dict:
            self.branch_connection_dict[connection.branch_idx] = [(connection.start_joint, connection.dependent_open, connection.independent_open)]
        else:
            self.branch_connection_dict[connection.branch_idx].append((connection.start_joint, connection.dependent_open, connection.independent_open)) 
    
    def build_all_topologies(self):
        def connect_branch(branch_idx, current_connections=self.branch_connection_dict):
            dependent_points = []
            dependent_points+=[c for c in current_connections if c.branch_idx in self.branch_connection_dict[branch_idx][0] and c.dependent_open]
            independent_points = []
            independent_points+=[c for c in current_connections if c.branch_idx in self.branch_connection_dict[branch_idx][1] and c.independent_open]
            
            product = itertools.product(dependent_points, independent_points)
            for pair in product:
                new_connections = current_connections.deepcopy()
                new_connections.remove(pair[0])
                new_connections.remove(pair[1])
                if branch_idx+1 in self.branch_connection_dict:
                    connect_branch(branch_idx + 1, new_connections)
                


            if branch_idx not in self.topologies:
                self.topologies.append((branch_idx, start_joint, dependent_open, independent_open))
            else:
                raise ValueError(f"Topology for branch {branch_idx} already exists.")
    
    def vis_branches(self):
        pass

    def filter_cycles(self):
        pass


if __name__ == "__main__":
    generator = Generator2DRotational()
    generator.open_main_branch()
    joint = GeneratorPoint(name = "G0",mutation_type=1, coordinates=(0.0, 0.0, 0.0), mutation_x=(0.0, 0.0), mutation_y=(0.0, 0.0), mutation_z=(0.0, 0.0))
    generator.add_joint(joint)
    joint = GeneratorPoint(name = "J1", mutation_type=1, coordinates=(0.05, 0.0, -0.2), mutation_x=(-0.1, 0.1), mutation_y=(0.0, 0.0), mutation_z=(-0.1, 0.1))
    generator.add_joint(joint)
    ee = GeneratorPoint(name="EE", point_type=PointType.END_EFFECTOR, mutation_type=1, coordinates=(0., 0.0, -0.4), mutation_x=(0.0, 0.0), mutation_y=(0.0, 0.0), mutation_z=(0.0, 0.0))
    generator.add_joint(ee)

    connection = GeneratorConnection(name="G1", branch_idx=0, start_joint=-1, dependent_open=True, independent_open=False)
    generator.add_connection(connection)
    connection = GeneratorConnection(name="C1", branch_idx=0, start_joint=1, dependent_open=False, independent_open=True)
    generator.add_connection(connection)

    generator.create_sub_branch(1)
    joint = GeneratorPoint(name="J2", mutation_type=2, coordinates=(0.0, 0.0, -0.2), mutation_x=(-0.1, 0.1), mutation_y=(0.0, 0.0), mutation_z=(-0.1, 0.1))
    generator.branch_connection_dict[1] = ([0], [0])

    generator.build_all_topologies()




