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


class TopologyGenerator():
    def __init__(self) -> None:
        self.graph = nx.Graph()

    def add_node(label:str='JP',mutation_type=MutationType.ABSOLUTE,active=False, ground=False):
        pass