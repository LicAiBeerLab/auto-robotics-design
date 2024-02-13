from abc import abstractmethod
import array
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.core.multiarray import zeros as zeros
import numpy.linalg as la

import networkx as nx
import modern_robotics as mr
from trimesh import Trimesh
from trimesh.convex import convex_hull


@dataclass
class JointPoint:
    """Describe a point in global frame where a joint is attached"""

    r: np.ndarray = field(default_factory=np.zeros(3))
    w: np.ndarray = field(default_factory=np.zeros(3))
    active: bool = False
    attach_ground: bool = False
    attach_endeffector: bool = False
    name: str = ""  # noqa: F811

    instance_counter: int = 0

    def __post_init__(self):
        JointPoint.instance_counter += 1
        self.__instance_counter = JointPoint.instance_counter
        if self.name == "":
            self.name = "J" + str(self.__instance_counter)

    def reser_id_counter(self):
        JointPoint.instance_counter = 0

    def __hash__(self) -> int:
        return hash(
            (
                self.r[0],
                self.r[1],
                self.r[2],
                self.w[0],
                self.w[1],
                self.w[2],
                self.attach_ground,
                self.attach_endeffector,
                self.active,
                self.__instance_counter,
            )
        )

    def __eq__(self, __value: object) -> bool:
        return hash(self) == hash(__value)


def create_mesh_from_joints(points: list[np.ndarray], thickness) -> Trimesh:
    vertices = []
    for p in points:
        v1 = deepcopy(p)
        v2 = deepcopy(p)
        v1[1] = thickness / 2
        v2[1] = -thickness / 2
        vertices += [v1, v2]

    return convex_hull(vertices)


class Geometry:

    def __init__(
        self,
        density: float = 0,
        size: list[float] | Trimesh = [],
        mass: float = 0,
        inertia: np.ndarray = np.zeros((3, 3)),
    ) -> None:
        self.shape: str = ""
        self._size: list[float] | Trimesh = size
        self._density: float = density
        if mass == 0 and np.sum(inertia) == 0:
            self.calculate_inertia()
        else:
            self.mass: float = mass
            self.inertia = inertia

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, values: list[float] | Trimesh):  # noqa: F811
        self._size = values
        self.calculate_inertia()

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, values: float):  # noqa: F811
        self._density = values
        self.calculate_inertia()

    @abstractmethod
    def calculate_inertia(self) -> tuple[float, np.ndarray]:
        return self.mass, self.inertia


class Box(Geometry):
    def __init__(
        self,
        density: float = 0,
        size: list[float] | Trimesh = [],
        mass: float = 0,
        inertia: np.ndarray = np.zeros((3, 3)),
    ) -> None:
        super().__init__(density, size, mass, inertia)
        self.shape = "box"

    def calculate_inertia(self):
        self.mass = np.prod(self.size) * self.density
        inertia = lambda a1, a2: 1 / 12 * self.mass * (a1**2 + a2**2)

        inertia_xx = inertia(*self.size[1:])
        inertia_yy = inertia(self.size[1], self.size[2])
        inertia_zz = inertia(*self.size[:-1])

        self.inertia = np.diag([inertia_xx, inertia_yy, inertia_zz])

        return self.mass, self.inertia


class Sphere(Geometry):
    def __init__(
        self,
        density: float = 0,
        size: list[float] = [],
        mass: float = 0,
        inertia: np.ndarray = np.zeros((3, 3)),
    ) -> None:
        super().__init__(density, size, mass, inertia)
        self.shape = "sphere"

    def calculate_inertia(self):
        self.mass = 4 / 3 * np.pi * self.size[0] ** 3 * self.density
        central_inertia = 2 / 5 * self.mass * self.size[0] ** 2

        self.inertia = np.diag([central_inertia for __ in range(3)])

        return self.mass, self.inertia


class Mesh(Geometry):
    def __init__(
        self,
        density: float = 0.0,
        size: Trimesh = Trimesh(),
        mass: float = 0.0,
        inertia: np.ndarray = np.zeros((3, 3)),
    ) -> None:
        super().__init__(density, size, mass, inertia)
        self._size.density = density
        self.shape = "mesh"

    def calculate_inertia(self):
        self.mass = self._size.mass

        self.inertia = self._size.moment_inertia

        return self.mass, self.inertia

class Link:
    instance_counter: int = 0

    def __init__(
        self,
        joints: set[JointPoint],
        name: str = "",
        geometry: Optional[Geometry] = None,
        frame: np.ndarray = np.eye(4),
        inertial_frame: np.ndarray = np.eye(4),
        density: float = 0,
        thickness: float = 0,
    ) -> None:
        self.joints: set[JointPoint] = joints
        self.name: str = name
        self.geometry = geometry

        self._frame: np.ndarray = deepcopy(frame)
        self.inertial_frame: np.ndarray = deepcopy(inertial_frame)

        self._density: float = density
        self._thickness: float = thickness
        self.define_geometry()

        Link.instance_counter += 1
        self.instance_counter = Link.instance_counter
        if self.name == "":
            self.name = "L" + str(self.instance_counter)
            
    @property
    def density(self):
        return self._density
    
    @density.setter
    def density(self, value: float):
        self._density = value
        self.define_geometry()

    @property
    def thickness(self):
        return self._thickness
    
    @thickness.setter
    def thickness(self, value: float):
        self._thickness = value
        self.define_geometry()

    @property
    def frame(self):
        return self._frame
    
    @frame.setter
    def frame(self, value: np.ndarray):
        self._frame = value
        self.define_geometry()

    def define_geometry(self):
        num_joint = len(self.joints)
        if num_joint == 1 or self.name == "G":
            self.geometry = Sphere(self._density, [self._thickness])
        elif num_joint == 2:
            joint_list = list(self.joints)
            vector = joint_list[1].jp.r - joint_list[0].jp.r
            size =  [self._thickness, self._thickness, la.norm(vector)]
            self.geometry = Box(self._density, size)
        elif num_joint > 2:
            points = []
            for j in self.joints:
                points.append((mr.TransInv(self.frame) @ np.r_[j.jp.r, 1])[:3])
            mesh = create_mesh_from_joints(points, self._thickness)
            self.geometry = Mesh(self._density, mesh)
        else:
            raise Exception("Zero joints")

    def str(self):
        return {self.name: tuple(j.jp.name for j in self.joints)}
        
    def __hash__(self) -> int:
        return hash((self.name, *self.joints))

    def __eq__(self, __value: object) -> bool:
        return self.joints == __value.joints


class Joint:
    def __init__(self, joint_point: JointPoint,
                is_constraint: bool = False,
                links: set[Link] = set(),
                frame: np.ndarray = np.eye(4) ) -> None:
        self.jp = joint_point
        self.is_constraint = is_constraint
        self.links = deepcopy(links)
        self._link_in = None
        self._link_out = None
        self.frame = deepcopy(frame)
        self.limits = deepcopy({})
        self.damphing_friction = deepcopy((0,0))
    
    @property
    def link_in(self):
        return self._link_in
    
    @link_in.setter
    def link_in(self, value: Link):  # noqa: F811
        self._link_in = value
        self.links = self.links | set([value])
        
    @property
    def link_out(self):
        return self._link_out
    
    @link_out.setter
    def link_out(self, value: Link):  # noqa: F811
        self._link_out = value
        self.links = self.links | set([value])

    def str(self):
        str_repr = {self.jp.name: tuple(l.name for l in self.links)}
        if self.link_in:
            str_repr["in"] =  self.link_in.name
        if self.link_out:
            str_repr["out"] =  self.link_out.name
        return str_repr
    
    def __hash__(self) -> int:
        return hash((
            self.jp,
            # self.is_constraint,
        ))
    
    def __eq__(self, __value: object) -> bool:
        return hash(self) == hash(__value)


def get_ground_joints(graph: nx.Graph):
    if isinstance(list(graph.nodes())[0], JointPoint):
        joint_nodes = graph.nodes()
        return filter(lambda n: n.attach_ground, joint_nodes)
    else:
        joint_nodes = graph.nodes()
        return filter(lambda n: n.jp.attach_ground, joint_nodes)


def get_endeffector_joints(graph: nx.Graph):
    if isinstance(list(graph.nodes())[0], JointPoint):
        joint_nodes = graph.nodes()
        return filter(lambda n: n.attach_endeffector, joint_nodes)
    else:
        joint_nodes = graph.nodes()
        return filter(lambda n: n.jp.attach_endeffector, joint_nodes)

if __name__ == "__main__":
    print("Kinematic description of the mechanism")
    # Define the joint points
    joint_points = [
        JointPoint(r=np.array([0, 0, 0]), attach_ground=True),
        JointPoint(r=np.array([1, 0, 0])),
        JointPoint(r=np.array([0, 1, 0])),
        JointPoint(r=np.array([0, 0, 1])),
        JointPoint(r=np.array([1, 1, 0])),
        JointPoint(r=np.array([1, 0, 1])),
        JointPoint(r=np.array([0, 1, 1])),
        JointPoint(r=np.array([1, 1, 1]), attach_endeffector=True),
    ]
    print(joint_points[0] == joint_points[1])
    print(joint_points[0] == joint_points[0])
