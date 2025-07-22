from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class MutationType(Enum):
    """Enumerate for mutation types."""
    # UNMOVABLE = 0  # Unmovable joint that is not used for optimization
    ABSOLUTE = 1  # The movement of the joint are in the absolute coordinate system and are relative to the initial position
    RELATIVE = 2  # The movement of the joint are relative to some other joint or joints and doesn't have an initial position
    # The movement of the joint are relative to some other joint or joints and doesn't have an initial position. The movement is in percentage of the distance between the joints.
    RELATIVE_PERCENTAGE = 3

@dataclass
class MutationCoordinate:
    freeze: Optional[float] = None
    mutation_origin: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    shift: Optional[float] = None


@dataclass
class SchemePoint:
    name: str = "P"
    mutation_type: int = MutationType.ABSOLUTE
    mutation_x: MutationCoordinate = MutationCoordinate()
    mutation_y: MutationCoordinate = MutationCoordinate()
    mutation_z: MutationCoordinate = MutationCoordinate()


@dataclass
class SchemeEE(SchemePoint):
    name: str = "EE"


@dataclass
class SchemeJoint(SchemePoint):
    name: str = "J"
    active: bool = False
    attach_ground: bool = False


@dataclass
class SchemeConnectionJoint(SchemePoint):
    name: str = "CJ"
    attach_ground: bool = False
    active: bool = False
    dependent_shift: Tuple[float, float, float] = (0, 0, 0)
    connected_to: Optional[Tuple[int, int]] = None #branch idx and first joint index


class ManualSchemeBuilder:
    def __init__(self):
        self.graph_scheme = {0: []}
        self.counter = 0
        self.branch_idx = 0

    def add_point(self, point: SchemePoint, branch_idx: int = 0):
        if branch_idx not in self.graph_scheme:
            raise ValueError(f"Branch {branch_idx} does not exist.")

        branch = self.graph_scheme[branch_idx]
        if len(branch) > 0:
            last_point = branch[-1]
            if isinstance(last_point, SchemeEE):
                raise ValueError("Cannot add a point after a EE.")

        if point.name != "EE":
            point.name += f"_{self.counter}"
            self.counter += 1

        self.graph_scheme[branch_idx].append(point)

    def add_branch(self):
        self.branch_idx += 1
        self.graph_scheme[self.branch_idx] = []


if __name__ == "__main__":
    # example with manually built scheme
    builder = ManualSchemeBuilder()
    builder.add_point(SchemeJoint(name="G", mutation_type=MutationType.ABSOLUTE, mutation_x=MutationCoordinate(
        freeze=0.0), mutation_y=MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(freeze=0.0), active=True, attach_ground=True))
    builder.add_point(SchemeJoint(name="J", mutation_type=MutationType.ABSOLUTE,
                      mutation_x=MutationCoordinate(mutation_origin=0.05, lower_bound=-0.1, upper_bound=0.1), mutation_y = MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(mutation_origin=-0.2, lower_bound=-0.1, upper_bound=0.1)))
    builder.add_point(SchemeEE(name="EE", mutation_type=MutationType.ABSOLUTE, mutation_x=MutationCoordinate(
        freeze=0.0), mutation_y=MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(freeze=-0.4)))
    builder.add_branch()
    builder.add_point(SchemeConnectionJoint(name="G", mutation_type=MutationType.ABSOLUTE, mutation_x = MutationCoordinate(mutation_origin=-0.2, upper_bound=0.1, lower_bound=-0.1), mutation_y = MutationCoordinate(freeze=0.0), mutation_z = MutationCoordinate(mutation_origin=0.0, lower_bound=0.0, upper_bound=0.2),
                      active=True, attach_ground=True), branch_idx=1)
    builder.add_point(SchemeJoint(name="J", mutation_type=MutationType.RELATIVE, mutation_x=MutationCoordinate(mutation_origin=0.0, lower_bound=-0.1, upper_bound=0.1),mutation_y=MutationCoordinate(0.0), mutation_z=MutationCoordinate(None, -0.2, -0.1, 0.1)), branch_idx=1)
    builder.add_point(SchemeConnectionJoint(name="CJ", mutation_type=MutationType.RELATIVE_PERCENTAGE, connected_to=(0, 1), mutation_x=MutationCoordinate(None, -0.05,-0.1, 0.1), mutation_y=MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(None, 0.0, -0.4, 0.4)), branch_idx=1)
    print(builder.graph_scheme)
