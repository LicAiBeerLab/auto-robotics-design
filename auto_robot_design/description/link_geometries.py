from abc import ABC, abstractmethod
import array
from copy import deepcopy
from dataclasses import dataclass, field
from collections import namedtuple
from itertools import combinations, product
from typing import Optional


from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy.core.multiarray import zeros as zeros
import numpy.linalg as la

import networkx as nx
import modern_robotics as mr
from trimesh import Trimesh
from trimesh.convex import convex_hull
from trimesh.boolean import union
from auto_robot_design.description.actuators import RevoluteUnit
from typing import Union

Box = namedtuple("Box", ["a", "b", "length"])
Cylinder = namedtuple("Cylinder", ["radius", "length"])
Sphere = namedtuple("Sphere", ["radius"])


def create_mesh_from_joints(joints, thickness, frame=np.eye(4)) -> Trimesh:
    points = {}
    for j in joints:
        points[j] = ((mr.TransInv(frame) @ np.r_[j.jp.r, 1])[:3], j.jp.w)
    pairs_p = combinations(points.keys(), 2)
    mesh = Trimesh()
    for p1, p2 in pairs_p:
        link_points = []
        vector = points[p2][0] - points[p1][0]
        vector = vector / la.norm(vector) if la.norm(vector) != 0 else vector
        ort_vector = np.cross(points[p1][1], vector)
        ort_vector = (
            ort_vector / la.norm(ort_vector) if la.norm(ort_vector) != 0 else ort_vector
        )
        variants = product((ort_vector, -ort_vector), (1, -1))
        for v in variants:
            link_points.append(points[p1][0] + (v[0] - v[1] * p1.jp.w) * thickness / 2)
            link_points.append(points[p2][0] + (v[0] - v[1] * p2.jp.w) * thickness / 2)
        mesh = mesh.union(
            [convex_hull(link_points), convex_hull(link_points)], check_volume=False
        )

    return mesh


def calculate_transform_with_2points(
    p1: np.ndarray, p2: np.ndarray, vec: np.ndarray = np.array([0, 0, 1])
):
    """Calculate transformation from `vec` to vector build with points `p1` and `p2`

    Args:
        p1 (np.ndarray): point of vector's start
        p2 (np.ndarray): point of vector's end
        vec (np.ndarray, optional): Vector tansform from. Defaults to np.array([0, 0, 1]).

    Returns:
        tuple: position: np.ndarray, rotation: scipy.spatial.rotation, length: float
    """
    v_l = p2 - p1
    angle = np.arccos(np.inner(vec, v_l) / la.norm(v_l) / la.norm(vec))
    axis = mr.VecToso3(vec[:3]) @ v_l[:3]
    if not np.isclose(np.sum(axis), 0):
        axis /= la.norm(axis)

    rot = R.from_rotvec(axis * angle)
    pos = (p2 + p1) / 2
    length = la.norm(v_l)

    return pos, rot, length


@dataclass
class LinkGeometryCreator(ABC):
    def __init__(self) -> None:
        pass


class LinkGeometry(ABC):
    # ==== TODO: REFACTORING =================
    # Remove inertia and mass from constructor
    # =======================================
    def __init__(
        self,
        params: dict,
    ) -> None:
        self.params: dict = params
        self.color: list[float] = params.get("color", [0, 0, 0, 0])
        self.principal_axis: R = R.identity()
        self.name: str = "undefined"

    @abstractmethod
    def create_geometry(self, *args, **kwargs) -> list:
        return []


class Truss(LinkGeometry):
    def __init__(
        self,
        params: dict,
    ) -> None:
        super().__init__(params)
        self.name = "truss"

    def create_geometry(self, *args, **kwargs):
        """Create truss elements between given points
        Args:
            points (list[np.ndarray]): list of points
        Kwargs:
            shape (namedtuple, optional): shape of truss element. If shape is not set, it use shape from params.
            one_point_size (float): Size along length of link which contain one point.
            one_point_rotation (scipy.spatial.transform.Rotation): Orientation of shape which contain one point.
            other shape parameters (float, optional): other shape parameters.
        Returns:
            list: list of tuples (position, rotation matrix, shape)
        """
        points = args[0]

        if len(points) == 1:
            pos = points[0]
            rot = kwargs.get("one_point_rotation", self.params["one_point_rotation"])
            length = kwargs.get("one_point_size", self.params["one_point_size"])
            t_frame_length = ((pos, rot, length),)
        else:
            points_pairs = combinations(points, 2)
            t_frame_length = []
            for p1, p2 in points_pairs:
                pos, rot, length = calculate_transform_with_2points(p1, p2)
                t_frame_length.append((pos, rot, length))

        l_truss_elements = []
        for pos, rot, length in t_frame_length:
            if "shape" in kwargs:
                shape_init = kwargs["shape"]
            else:
                try:
                    shape_init = self.params["shape"]
                except KeyError:
                    raise Exception("Shape is not defined")

            shape_params = {"length": length}
            for name_field in shape_init._fields:
                if name_field in kwargs:
                    shape_params[name_field] = kwargs[name_field]
                elif name_field in self.params:
                    shape_params[name_field] = self.params[name_field]
                else:
                    raise Exception(f"Parameter {name_field} is not defined")

            l_truss_elements.append(
                (
                    pos,
                    rot.as_matrix(),
                    shape_init(
                        **shape_params,
                    ),
                )
            )
        return l_truss_elements
