# from updated_generator import GeneratorPoint
from dataclasses import dataclass
from typing import Tuple
from graph_scheme import SchemePoint, SchemeEE, SchemeJoint, SchemeConnectionJoint, MutationType
from auto_robot_design.description.kinematics import JointPoint
import networkx as nx

@dataclass
class MutationEntry:
    mutation_type: int = MutationType.ABSOLUTE
    freeze_x = None
    freeze_y = 0
    freeze_z = None
    starting_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mutation_x: Tuple[float, float] = (0.0, 0.0)
    mutation_y: Tuple[float, float] = (0.0, 0.0)
    mutation_z: Tuple[float, float] = (0.0, 0.0)



class MutableGraphManager:
    def __init__(self, graph_scheme=None):
        self.graph_scheme = graph_scheme
        self.mutations = None
        self.joint_points = {}
        self.graph = nx.Graph()

    def get_mutation_ranges(self):
        pass

    def create_jp_from_sp(self, scheme_point) -> Tuple(JointPoint, Tuple(float)):
        if isinstance(scheme_point, SchemeEE):
            jp = JointPoint(r=None, attach_endeffector=True,name=scheme_point.name)
        else:
            jp = JointPoint(r=None, name=scheme_point.name, attach_ground=scheme_point.attach_ground, active=scheme_point.active)

        mutation_entry = MutationEntry(mutation_type=scheme_point.mutation_type)
        if scheme_point.freeze_coordinates is not None:
            if scheme_point.freeze_coordinates[0] is not None:
                mutation_entry.freeze_x = scheme_point.freeze_coordinates[0]
            if scheme_point.freeze_coordinates[1] is not None:
                mutation_entry.freeze_y = scheme_point.freeze_coordinates[1]
            if scheme_point.freeze_coordinates[2] is not None:
                mutation_entry.freeze_z = scheme_point.freeze_coordinates[2]

        return jp, mutation_entry

    def build_graph(self):
        branch_idx = 0
             
        while branch_idx in self.graph_scheme:
            branch = self.graph_scheme[branch_idx]
            self.joint_points[branch_idx] = []

            for idx, point in enumerate(branch):
                jp, mutation_entry = self.create_jp_from_sp(point)
                






                        
                        

    def get_graph(self, params):
        pass
    def get_central_graph(self):
        pass
    def get_random_graph(self):
        pass
    def set_mutation_range(self, point_idx, mutation_range):
        pass
    def freeze_point(self, point_idx, freeze_coordinates):
        pass


if __name__ == "__main__":
    # example with manually built scheme
    graph_scheme = {0: [GeneratorPoint(name="G0", mutation_type=1, coordinates=(0.0, 0.0, 0.0), mutation_x=(0.0, 0.0), mutation_y=(0.0, 0.0), mutation_z=(0.0, 0.0)),
                    GeneratorPoint(name="J1", mutation_type=1, coordinates=(0.05, 0.0, -0.2), mutation_x=(-0.1, 0.1), mutation_y=(0.0, 0.0), mutation_z=(-0.1, 0.1)),
                    GeneratorPoint(name="EE", mutation_type=1, coordinates=(0.0, 0.0, -0.4), mutation_x=(-0.1, 0.1), mutation_y=(0.0, 0.0), mutation_z=(-0.1, 0.1))],