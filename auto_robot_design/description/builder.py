from copy import deepcopy
from itertools import combinations
from typing import Union
from pyparsing import List, Tuple

import odio_urdf as urdf
import networkx as nx
import numpy.linalg as la
import numpy as np
from scipy.spatial.transform import Rotation as R
import modern_robotics as mr

from auto_robot_design.description.actuators import Actuator, RevoluteUnit, TMotor_AK80_9
from auto_robot_design.description.kinematics import (
    Joint,
    JointPoint,
    Link,
    Mesh,
    Sphere,
    Box,
)
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph, KinematicGraph
from auto_robot_design.description.utils import tensor_inertia_sphere_by_mass
from auto_robot_design.pino_adapter.pino_adapter import get_pino_description

DEFAULT_DENSITY = 2700 / 2.8
DEFAULT_THICKNESS = 0.04
DEFAULT_JOINT_DAMPING = 0.05
DEFAULT_JOINT_FRICTION = 0
DEFAULT_ACTUATOR = TMotor_AK80_9()

DEFAULT_PARAMS_DICT = {
    "density": DEFAULT_DENSITY,
    "thickness": DEFAULT_THICKNESS,
    "joint_damping": DEFAULT_JOINT_DAMPING,
    "joint_friction": DEFAULT_JOINT_FRICTION,
    "actuator": DEFAULT_ACTUATOR,
}

def add_branch(G: nx.Graph, branch: Union[List[JointPoint], List[List[JointPoint]]]):
    """
    Add a branch to the given graph.

    Parameters:
    - G (nx.Graph): The graph to which the branch will be added.
    - branch (Union[List[JointPoint], List[List[JointPoint]]]): The branch to be added. It can be a list of JointPoints or a list of lists of JointPoints.

    Returns:
    None
    """

    is_list = [isinstance(br, List) for br in branch]
    if all(is_list):
        for b in branch:
            add_branch(G, b)
    else:
        for i in range(len(branch) - 1):
            if isinstance(branch[i], List):
                for b in branch[i]:
                    G.add_edge(b, branch[i + 1])
            elif isinstance(branch[i + 1], List):
                for b in branch[i + 1]:
                    G.add_edge(branch[i], b)
            else:
                G.add_edge(branch[i], branch[i + 1])


def add_branch_with_attrib(
    G: nx.Graph,
    branch: Union[List[Tuple[JointPoint, dict]], List[List[Tuple[JointPoint, dict]]]],
):
    is_list = [isinstance(br, List) for br in branch]
    if all(is_list):
        for b in branch:
            add_branch_with_attrib(G, b)
    else:
        for ed in branch:
            G.add_edge(ed[0], ed[1], **ed[2])


class URDFLinkCreater:
    """
    Class responsible for creating URDF links and joints.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def create_link(cls, link: Link):
        """
        Create a URDF link based on the given Link object.

        Args:
            link (Link): The Link object containing the link information.

        Returns:
            urdf_link: The created URDF link.
        """
        if link.geometry.shape == "mesh":
            pos_joint_in_local = []
            H_l_w = mr.TransInv(link.frame)
            for j in link.joints:
                pos_joint_in_local.append(H_l_w @ np.r_[j.jp.r, 1])

            joint_pos_pairs = combinations(pos_joint_in_local, 2)
            ez = np.array([0, 0, 1, 0])
            body_origins = []
            for j_p in joint_pos_pairs:
                v_l = j_p[1] - j_p[0]
                angle = np.arccos(np.inner(ez, v_l) / la.norm(v_l) / la.norm(ez))
                axis = mr.VecToso3(ez[:3]) @ v_l[:3]
                if not np.isclose(np.sum(axis), 0):
                    axis /= la.norm(axis)

                rot = R.from_rotvec(axis * angle)
                pos = (j_p[1][:3] + j_p[0][:3]) / 2
                if la.norm(v_l) > link.geometry.get_thickness():
                    length = la.norm(v_l) - link.geometry.get_thickness()
                else:
                    length = la.norm(v_l)
                body_origins.append(
                    (pos.tolist(), rot.as_euler("xyz").tolist(), length)
                )
            inertia = (
                link.inertial_frame,
                link.geometry.size.moment_inertia_frame(link.inertial_frame),
            )
            urdf_link = cls._create_mesh(
                link.geometry, link.name, inertia, body_origins
            )
        elif link.geometry.shape == "box":
            origin = cls.trans_matrix2xyz_rpy(link.inertial_frame)
            urdf_link = cls._create_box(link.geometry, link.name, origin, origin)
        elif link.geometry.shape == "sphere":
            origin = cls.trans_matrix2xyz_rpy(link.inertial_frame)
            urdf_link = cls._create_sphere(link.geometry, link.name, origin, origin)
        else:
            pass
        return urdf_link

    @classmethod
    def create_joint(cls, joint: Joint):
        """
        Create a URDF joint based on the given Joint object.

        Args:
            joint (Joint): The Joint object containing the joint information.

        Returns:
            dict: A dictionary containing the created URDF joint and additional information.
        """
        if joint.link_in is None or joint.link_out is None:
            return {"joint": []}
        origin = cls.trans_matrix2xyz_rpy(joint.frame)
        if joint.is_constraint:
            color1 = joint.link_in.geometry.color
            color1[3] = 0.5
            color2 = joint.link_out.geometry.color
            color2[3] = 0.5

            name_link_in = joint.jp.name + "_" + joint.link_in.name + "Pseudo"
            rad_in = joint.link_in.geometry.get_thickness() / 1.4
            urdf_pseudo_link_in = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(urdf.Sphere(float(rad_in))),
                    urdf.Material(
                        urdf.Color(rgba=color1), name=name_link_in + "_Material"
                    ),
                    # name=name_link_in + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **cls.convert_inertia(
                            tensor_inertia_sphere_by_mass(
                                joint.actuator.mass / 2, rad_in
                            )
                        )
                    ),
                ),
                name=name_link_in,
            )
            urdf_joint_in = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=name_link_in),
                urdf.Origin(
                    xyz=origin[0],
                    rpy=origin[1],
                ),
                urdf.Axis(joint.jp.w.tolist()),
                urdf.Limit(
                    lower=joint.pos_limits[0],
                    upper=joint.pos_limits[1],
                    effort=joint.actuator.get_max_effort(),
                    velocity=joint.actuator.get_max_vel(),
                ),
                urdf.Dynamics(
                    damping=joint.damphing_friction[0],
                    friction=joint.damphing_friction[1],
                ),
                name=joint.jp.name + "_" + joint.link_in.name + "_revolute",
                type="revolute",
            )

            name_link_out = joint.jp.name + "_" + joint.link_out.name + "Pseudo"
            rad_out = joint.link_out.geometry.get_thickness() / 1.4
            urdf_pseudo_link_out = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(urdf.Sphere(float(rad_out))),
                    urdf.Material(
                        urdf.Color(rgba=color2), name=name_link_out + "_Material"
                    ),
                    # name=name_link_out + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **cls.convert_inertia(
                            tensor_inertia_sphere_by_mass(
                                joint.actuator.mass / 2, rad_out
                            )
                        )
                    ),
                ),
                name=name_link_out,
            )

            H_in_j = joint.frame
            H_w_in = joint.link_in.frame

            H_w_out = joint.link_out.frame

            H_out_j = mr.TransInv(H_w_out) @ H_w_in @ H_in_j

            out_origin = cls.trans_matrix2xyz_rpy(H_out_j)

            urdf_joint_out = urdf.Joint(
                urdf.Parent(link=joint.link_out.name),
                urdf.Child(link=name_link_out),
                urdf.Origin(
                    xyz=out_origin[0],
                    rpy=out_origin[1],
                ),
                name=joint.jp.name + "_" + joint.link_in.name + "_Weld",
                type="fixed",
            )

            out = {
                "joint": [
                    urdf_pseudo_link_in,
                    urdf_joint_in,
                    urdf_joint_out,
                    urdf_pseudo_link_out,
                ],
                "constraint": [name_link_in, name_link_out],
            }
        else:
            urdf_joint = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=joint.link_out.name),
                urdf.Origin(
                    xyz=origin[0],
                    rpy=origin[1],
                ),
                urdf.Axis(joint.jp.w.tolist()),
                urdf.Limit(
                    lower=joint.pos_limits[0],
                    upper=joint.pos_limits[1],
                    effort=joint.actuator.get_max_effort(),
                    velocity=joint.actuator.get_max_vel(),
                ),
                urdf.Dynamics(
                    damping=joint.damphing_friction[0],
                    friction=joint.damphing_friction[1],
                ),
                name=joint.jp.name,
                type="revolute",
            )
            out = {"joint": [urdf_joint]}
            if joint.jp.active:
                connected_unit = RevoluteUnit()
                connected_unit.size = [
                    joint.link_in.geometry.get_thickness() / 2,
                    joint.link_in.geometry.get_thickness(),
                ]
            elif not joint.actuator.size:
                unit_size = [
                    joint.link_in.geometry.get_thickness() / 2,
                    joint.link_in.geometry.get_thickness(),
                ]
                joint.actuator.size = unit_size
                connected_unit = joint.actuator
            else:
                connected_unit = joint.actuator

            name_joint_link = joint.jp.name + "_" + joint.link_in.name + "Unit"
            name_joint_weld = joint.jp.name + "_" + joint.link_in.name + "_WeldUnit"
            Rp_j = mr.TransToRp(joint.frame)
            color = joint.link_in.geometry.color
            color[3] = 0.9
            rot_a = R.from_matrix(
                Rp_j[0] @ R.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
            ).as_euler("xyz")
            urdf_joint_weld = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=name_joint_link),
                urdf.Origin(
                    xyz=Rp_j[1].tolist(),
                    rpy=rot_a.tolist(),
                ),
                name=name_joint_weld,
                type="fixed",
            )
            urdf_unit_link = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(
                        urdf.Cylinder(
                            length=connected_unit.size[1], radius=connected_unit.size[0]
                        )
                    ),
                    urdf.Material(
                        urdf.Color(rgba=color), name=name_joint_link + "_Material"
                    ),
                    # name=name_joint_link + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Inertia(
                        **cls.convert_inertia(connected_unit.calculate_inertia())
                    ),
                    urdf.Mass(float(connected_unit.mass)),
                ),
                name=name_joint_link,
            )

            if joint.jp.active:
                out["active"] = joint.jp.name
                name_actuator_link = (
                    joint.jp.name + "_" + joint.link_in.name + "Actuator"
                )
                name_actuator_weld = (
                    joint.jp.name + "_" + joint.link_in.name + "_WeldActuator"
                )
                pos = Rp_j[1] + joint.jp.w * (
                    joint.actuator.size[1] / 2 + connected_unit.size[1] / 2
                )
                urdf_actuator_weld = urdf.Joint(
                    urdf.Parent(link=joint.link_in.name),
                    urdf.Child(link=name_actuator_link),
                    urdf.Origin(
                        xyz=pos.tolist(),
                        rpy=rot_a.tolist(),
                    ),
                    name=name_actuator_weld,
                    type="fixed",
                )
                urdf_actuator_link = urdf.Link(
                    urdf.Visual(
                        urdf.Geometry(
                            urdf.Cylinder(
                                length=joint.actuator.size[1],
                                radius=joint.actuator.size[0],
                            )
                        ),
                        urdf.Material(
                            urdf.Color(rgba=color),
                            name=name_actuator_link + "_Material",
                        ),
                        # name=name_actuator_link + "_Visual",
                    ),
                    urdf.Inertial(
                        urdf.Inertia(
                            **cls.convert_inertia(joint.actuator.calculate_inertia())
                        ),
                        urdf.Mass(float(joint.actuator.mass)),
                    ),
                    name=name_actuator_link,
                )
                out["joint"].append(urdf_actuator_weld)
                out["joint"].append(urdf_actuator_link)

            out["joint"].append(urdf_unit_link)
            out["joint"].append(urdf_joint_weld)
        return out

    @classmethod
    def trans_matrix2xyz_rpy(cls, H):
        """
        Convert a transformation matrix to XYZ and RPY representation.

        Args:
            H: The transformation matrix.

        Returns:
            tuple: A tuple containing the XYZ position and RPY orientation.
        """
        Rp = mr.TransToRp(H)
        rpy = R.from_matrix(Rp[0]).as_euler("xyz").tolist()
        return (Rp[1].tolist(), rpy)

    @classmethod
    def convert_inertia(cls, tensor_inertia):
        """
        Convert the tensor inertia to a dictionary representation.

        Args:
            tensor_inertia: The tensor inertia.

        Returns:
            dict: A dictionary containing the converted inertia values.
        """
        x, y, z = tuple(range(3))
        Ixx = tensor_inertia[x][x]
        Iyy = tensor_inertia[y][y]
        Izz = tensor_inertia[z][z]
        Ixy = tensor_inertia[x][y]
        Ixz = tensor_inertia[x][z]
        Iyz = tensor_inertia[y][z]
        return {"ixx": Ixx, "ixy": Ixy, "ixz": Ixz, "iyy": Iyy, "iyz": Iyz, "izz": Izz}

    @classmethod
    def _create_box(cls, geometry: Box, name, origin, inertia_origin):
        """
        Create a URDF box based on the given Box geometry.

        Args:
            geometry (Box): The Box geometry object.
            name: The name of the box.
            origin: The origin of the box.
            inertia_origin: The origin of the inertia.

        Returns:
            urdf.Link: The created URDF link.
        """
        name_m = name + "_" + "Material"
        urdf_material = urdf.Material(urdf.Color(rgba=geometry.color), name=name_m)
        name_c = name + "_" + "Collision"
        name_v = name + "_" + "Visual"
        urdf_geometry = urdf.Geometry(urdf.Box(geometry.size))
        urdf_inertia_origin = urdf.Origin(
            xyz=inertia_origin[0],
            rpy=inertia_origin[1],
        )
        urdf_origin = urdf.Origin(
            xyz=origin[0],
            rpy=origin[1],
        )

        visual = urdf.Visual(
            urdf_origin,
            urdf_geometry,
            urdf_material,
            # name = name_v
        )
        collision = urdf.Collision(urdf_origin, urdf_geometry, name=name_c)
        inertial = urdf.Inertial(
            urdf_inertia_origin,
            urdf.Mass(float(geometry.mass)),
            urdf.Inertia(**cls.convert_inertia(geometry.inertia)),
        )

        return urdf.Link(visual, collision, inertial, name=name)

    @classmethod
    def _create_sphere(cls, geometry: Sphere, name, origin, inertia_origin):
        """
        Create a URDF sphere based on the given Sphere geometry.

        Args:
            geometry (Sphere): The Sphere geometry object.
            name: The name of the sphere.
            origin: The origin of the sphere.
            inertia_origin: The origin of the inertia.

        Returns:
            urdf.Link: The created URDF link.
        """
        name_m = name + "_" + "Material"
        urdf_material = urdf.Material(urdf.Color(rgba=geometry.color), name=name_m)

        name_c = name + "_" + "Collision"
        name_v = name + "_" + "Visual"
        urdf_geometry = urdf.Geometry(urdf.Sphere(geometry.size[0]))
        urdf_inertia_origin = urdf.Origin(
            xyz=inertia_origin[0],
            rpy=inertia_origin[1],
        )
        urdf_origin = urdf.Origin(
            xyz=origin[0],
            rpy=origin[1],
        )

        visual = urdf.Visual(
            urdf_origin,
            urdf_geometry,
            urdf_material,
            # name = name_v
        )
        collision = urdf.Collision(urdf_origin, urdf_geometry, name=name_c)
        inertial = urdf.Inertial(
            urdf_inertia_origin,
            urdf.Mass(geometry.mass),
            urdf.Inertia(**cls.convert_inertia(geometry.inertia)),
        )

        return urdf.Link(visual, collision, inertial, name=name)

    @classmethod
    def _create_mesh(cls, geometry: Mesh, name, inertia, body_origins):
        """
        Create a URDF mesh based on the given Mesh geometry.

        Args:
            geometry (Mesh): The Mesh geometry object.
            name: The name of the mesh.
            inertia: The inertia of the mesh.
            body_origins: The origins of the mesh bodies.

        Returns:
            urdf.Link: The created URDF link.
        """
        name_m = name + "_" + "Material"
        urdf_material = urdf.Material(urdf.Color(rgba=geometry.color), name=name_m)
        origin_I = cls.trans_matrix2xyz_rpy(inertia[0])
        urdf_inertia_origin = urdf.Origin(xyz=origin_I[0], rpy=origin_I[1])
        visual_n_collision = []
        for id, origin in enumerate(body_origins):
            name_c = name + "_" + str(id) + "_Collision"
            name_v = name + "_" + str(id) + "_Visual"
            thickness = geometry.get_thickness()
            urdf_geometry = urdf.Geometry(urdf.Box([thickness, thickness, origin[2]]))
            urdf_origin = urdf.Origin(
                xyz=origin[0],
                rpy=origin[1],
            )
            visual = urdf.Visual(
                urdf_origin,
                urdf_geometry,
                urdf_material,
                # name = name_v
            )

            collision = urdf.Collision(urdf_origin, urdf_geometry, name=name_c)
            visual_n_collision += [visual, collision]
        inertial = urdf.Inertial(
            urdf_inertia_origin,
            urdf.Mass(float(geometry.size.mass)),
            urdf.Inertia(**cls.convert_inertia(inertia[1])),
        )
        return urdf.Link(*visual_n_collision, inertial, name=name)


class DetalizedURDFCreaterFixedEE(URDFLinkCreater):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def create_joint(cls, joint: Joint):
        if joint.link_in is None or joint.link_out is None:
            return {"joint": []}
        origin = cls.trans_matrix2xyz_rpy(joint.frame)
        if joint.is_constraint:
            color1 = joint.link_in.geometry.color
            color1[3] = 0.5
            color2 = joint.link_out.geometry.color
            color2[3] = 0.5

            name_link_in = joint.jp.name + "_" + joint.link_in.name + "Pseudo"
            rad_in = joint.link_in.geometry.get_thickness() / 1.4
            urdf_pseudo_link_in = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(urdf.Sphere(float(rad_in))),
                    urdf.Material(
                        urdf.Color(rgba=color1), name=name_link_in + "_Material"
                    ),
                    # name=name_link_in + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **cls.convert_inertia(
                            tensor_inertia_sphere_by_mass(
                                joint.actuator.mass / 2, rad_in
                            )
                        )
                    ),
                ),
                name=name_link_in,
            )
            urdf_joint_in = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=name_link_in),
                urdf.Origin(
                    xyz=origin[0],
                    rpy=origin[1],
                ),
                urdf.Axis(joint.jp.w.tolist()),
                urdf.Limit(
                    lower=joint.pos_limits[0],
                    upper=joint.pos_limits[1],
                    effort=joint.actuator.get_max_effort(),
                    velocity=joint.actuator.get_max_vel(),
                ),
                urdf.Dynamics(
                    damping=joint.damphing_friction[0],
                    friction=joint.damphing_friction[1],
                ),
                name=joint.jp.name + "_" + joint.link_in.name + "_revolute",
                type="revolute",
            )

            name_link_out = joint.jp.name + "_" + joint.link_out.name + "Pseudo"
            rad_out = joint.link_out.geometry.get_thickness() / 1.4
            urdf_pseudo_link_out = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(urdf.Sphere(float(rad_out))),
                    urdf.Material(
                        urdf.Color(rgba=color2), name=name_link_out + "_Material"
                    ),
                    # name=name_link_out + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Mass(float(joint.actuator.mass / 2)),
                    urdf.Inertia(
                        **cls.convert_inertia(
                            tensor_inertia_sphere_by_mass(
                                joint.actuator.mass / 2, rad_out
                            )
                        )
                    ),
                ),
                name=name_link_out,
            )

            H_in_j = joint.frame
            H_w_in = joint.link_in.frame

            H_w_out = joint.link_out.frame

            H_out_j = mr.TransInv(H_w_out) @ H_w_in @ H_in_j

            out_origin = cls.trans_matrix2xyz_rpy(H_out_j)

            urdf_joint_out = urdf.Joint(
                urdf.Parent(link=joint.link_out.name),
                urdf.Child(link=name_link_out),
                urdf.Origin(
                    xyz=out_origin[0],
                    rpy=out_origin[1],
                ),
                name=joint.jp.name + "_" + joint.link_in.name + "_Weld",
                type="fixed",
            )

            out = {
                "joint": [
                    urdf_pseudo_link_in,
                    urdf_joint_in,
                    urdf_joint_out,
                    urdf_pseudo_link_out,
                ],
                "constraint": [name_link_in, name_link_out],
            }
        else:
            if "EE" in [l.name for l in joint.links]:
                urdf_joint = urdf.Joint(
                    urdf.Parent(link=joint.link_in.name),
                    urdf.Child(link=joint.link_out.name),
                    urdf.Origin(
                        xyz=origin[0],
                        rpy=origin[1],
                    ),
                    name=joint.jp.name,
                    type="fixed",
                )
            else:
                urdf_joint = urdf.Joint(
                    urdf.Parent(link=joint.link_in.name),
                    urdf.Child(link=joint.link_out.name),
                    urdf.Origin(
                        xyz=origin[0],
                        rpy=origin[1],
                    ),
                    urdf.Axis(joint.jp.w.tolist()),
                    urdf.Limit(
                        lower=joint.pos_limits[0],
                        upper=joint.pos_limits[1],
                        effort=joint.actuator.get_max_effort(),
                        velocity=joint.actuator.get_max_vel(),
                    ),
                    urdf.Dynamics(
                        damping=joint.damphing_friction[0],
                        friction=joint.damphing_friction[1],
                    ),
                    name=joint.jp.name,
                    type="revolute",
                )
            out = {"joint": [urdf_joint]}
            if joint.jp.active:
                connected_unit = RevoluteUnit()
                if joint.link_in.name == "G":
                    connected_unit.size = [
                        joint.link_out.geometry.get_thickness() / 2,
                        joint.link_out.geometry.get_thickness(),
                    ]
                else:
                    connected_unit.size = [
                        joint.link_in.geometry.get_thickness() / 2,
                        joint.link_in.geometry.get_thickness(),
                    ]
            elif not joint.actuator.size:
                if joint.link_in.name == "G":
                    unit_size = [
                        joint.link_out.geometry.get_thickness() / 2,
                        joint.link_out.geometry.get_thickness(),
                    ]
                else:
                    unit_size = [
                        joint.link_in.geometry.get_thickness() / 2,
                        joint.link_in.geometry.get_thickness(),
                    ]
                joint.actuator.size = unit_size
                connected_unit = joint.actuator
            else:
                connected_unit = joint.actuator

            name_joint_link = joint.jp.name + "_" + joint.link_in.name + "Unit"
            name_joint_weld = joint.jp.name + "_" + joint.link_in.name + "_WeldUnit"
            Rp_j = mr.TransToRp(joint.frame)
            color = joint.link_in.geometry.color
            color[3] = 0.9
            rot_a = R.from_matrix(
                Rp_j[0] @ R.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
            ).as_euler("xyz")
            urdf_joint_weld = urdf.Joint(
                urdf.Parent(link=joint.link_in.name),
                urdf.Child(link=name_joint_link),
                urdf.Origin(
                    xyz=Rp_j[1].tolist(),
                    rpy=rot_a.tolist(),
                ),
                name=name_joint_weld,
                type="fixed",
            )
            urdf_unit_link = urdf.Link(
                urdf.Visual(
                    urdf.Geometry(
                        urdf.Cylinder(
                            length=connected_unit.size[1], radius=connected_unit.size[0]
                        )
                    ),
                    urdf.Material(
                        urdf.Color(rgba=color), name=name_joint_link + "_Material"
                    ),
                    # name=name_joint_link + "_Visual",
                ),
                urdf.Inertial(
                    urdf.Inertia(
                        **cls.convert_inertia(connected_unit.calculate_inertia())
                    ),
                    urdf.Mass(float(connected_unit.mass)),
                ),
                name=name_joint_link,
            )

            if joint.jp.active:
                out["active"] = joint.jp.name
                name_actuator_link = (
                    joint.jp.name + "_" + joint.link_in.name + "Actuator"
                )
                name_actuator_weld = (
                    joint.jp.name + "_" + joint.link_in.name + "_WeldActuator"
                )
                pos = Rp_j[1] + joint.jp.w * (
                    joint.actuator.size[1] / 2 + connected_unit.size[1] / 2
                )
                urdf_actuator_weld = urdf.Joint(
                    urdf.Parent(link=joint.link_in.name),
                    urdf.Child(link=name_actuator_link),
                    urdf.Origin(
                        xyz=pos.tolist(),
                        rpy=rot_a.tolist(),
                    ),
                    name=name_actuator_weld,
                    type="fixed",
                )
                urdf_actuator_link = urdf.Link(
                    urdf.Visual(
                        urdf.Geometry(
                            urdf.Cylinder(
                                length=joint.actuator.size[1],
                                radius=joint.actuator.size[0],
                            )
                        ),
                        urdf.Material(
                            urdf.Color(rgba=color),
                            name=name_actuator_link + "_Material",
                        ),
                        # name=name_actuator_link + "_Visual",
                    ),
                    urdf.Inertial(
                        urdf.Inertia(
                            **cls.convert_inertia(joint.actuator.calculate_inertia())
                        ),
                        urdf.Mass(float(joint.actuator.mass)),
                    ),
                    name=name_actuator_link,
                )
                out["joint"].append(urdf_actuator_weld)
                out["joint"].append(urdf_actuator_link)

            out["joint"].append(urdf_unit_link)
            out["joint"].append(urdf_joint_weld)
        return out


class Builder:
    def __init__(self, creater) -> None:
        self.creater = creater

    def create_kinematic_graph(self, kinematic_graph, name="Robot"):

        links = kinematic_graph.nodes()
        joints = dict(
            filter(lambda kv: len(kv[1]) > 0, kinematic_graph.joint2edge.items())
        )

        urdf_links = []
        urdf_joints = []
        for link in links:
            urdf_links.append(self.creater.create_link(link))

        active_joints = []
        constraints = []
        for joint in joints:
            info_joint = self.creater.create_joint(joint)

            urdf_joints += info_joint["joint"]

            if "active" in info_joint.keys():
                active_joints.append(info_joint["active"])

            if "constraint" in info_joint.keys():
                constraints.append(info_joint["constraint"])

        urdf_objects = urdf_links + urdf_joints

        urdf_robot = urdf.Robot(*urdf_objects, name=name)

        return urdf_robot, active_joints, constraints


class ParametrizedBuilder(Builder):
    """
    A builder class that allows for parameterized construction of objects.

    Args:
        creater: The object that creates the instance of the builder.
        density (Union[float, dict]): The density of the object being built. Defaults to 2700 / 2.8.
        thickness (float): The thickness of the object being built. Defaults to 0.04.
        joint_damping (Union[float, dict]): The damping of the joints in the object being built. Defaults to 0.05.
        joint_friction (Union[float, dict]): The friction of the joints in the object being built. Defaults to 0.
        size_ground (np.ndarray): The size of the ground for the object being built. Defaults to np.zeros(3).
        actuator: The actuator used in the object being built. Defaults to TMotor_AK80_9().

    Attributes:
        density (Union[float, dict]): The density of the object being built.
        actuator: The actuator used in the object being built.
        thickness (float): The thickness of the object being built.
        size_ground (np.ndarray): The size of the ground for the object being built.
        joint_damping (Union[float, dict]): The damping of the joints in the object being built.
        joint_friction (Union[float, dict]): The friction of the joints in the object being built.
    """

    def __init__(
        self,
        creater,
        density: Union[float, dict] = 2700 / 2.8,
        thickness: Union[float, dict] = 0.04,
        joint_damping: Union[float, dict] = 0.05,
        joint_friction: Union[float, dict] = 0,
        joint_limits: Union[dict, tuple] = (-np.pi, np.pi),
        size_ground: np.ndarray = np.zeros(3),
        actuator: Union[Actuator, dict]=TMotor_AK80_9(),
    ) -> None:
        super().__init__(creater)
        self.density = density
        self.actuator = actuator
        self.thickness = thickness
        self.size_ground = size_ground
        self.joint_damping = joint_damping
        self.joint_friction = joint_friction
        self.joint_limits = joint_limits
        self.attributes = ["density", "joint_damping", "joint_friction", "joint_limits", "actuator", "thickness"]
        self.joint_attributes = ["joint_damping", "joint_friction", "actuator", "joint_limits"]
        self.link_attributes = ["density", "thickness"]
    
    def create_kinematic_graph(self, kinematic_graph: KinematicGraph, name="Robot"):
        # kinematic_graph = deepcopy(kinematic_graph)
        # kinematic_graph.G = list(filter(lambda n: n.name == "G", kinematic_graph.nodes()))[0]
        # kinematic_graph.EE = list(filter(lambda n: n.name == "EE", kinematic_graph.nodes()))[0]
        for attr in self.attributes:
            self.check_default(getattr(self, attr), attr)
        joints = kinematic_graph.joint_graph.nodes()
        for joint in joints:
            self._set_joint_attributes(joint)
        links = kinematic_graph.nodes()
        for link in links:
            self._set_link_attributes(link)
        
        return super().create_kinematic_graph(kinematic_graph, name)
        
    def _set_joint_attributes(self, joint):
        if joint.jp.active:
            joint.actuator = self.actuator[joint.jp.name] if joint.jp.name in self.actuator else self.actuator["default"]
        damping = self.joint_damping[joint.jp.name] if joint.jp.name in self.joint_damping else self.joint_damping["default"]
        friction = self.joint_friction[joint.jp.name] if joint.jp.name in self.joint_friction else self.joint_friction["default"]
        limits = self.joint_limits[joint.jp.name] if joint.jp.name in self.joint_limits else self.joint_limits["default"]
        joint.damphing_friction = (damping, friction)
        joint.pos_limits = limits
        
    def _set_link_attributes(self, link):
        if link.name == "G" and self.size_ground.any():
            link.geometry.size = list(self.size_ground)
            pos = np.zeros(3)
            pos[1] = self.size_ground[1] / 2
            link.inertial_frame = mr.RpToTrans(np.eye(3), pos)
        else:
            link.thickness = self.thickness[link.name] if link.name in self.thickness else self.thickness["default"]
        link.geometry.density = self.density[link.name] if link.name in self.density else self.density["default"]

    
    def check_default(self, params, name):
        if not isinstance(params, dict):
            setattr(self, name, {"default": params})
        if "default" not in getattr(self, name):
            getattr(self, name)["default"] = DEFAULT_PARAMS_DICT[name]


def jps_graph2urdf(graph: nx.Graph):
    kinematic_graph = JointPoint2KinematicGraph(graph)
    kinematic_graph.define_main_branch()
    kinematic_graph.define_span_tree()
    thickness = 0.04
    density = 2700 / 2.8

    for n in kinematic_graph.nodes():
        n.thickness = thickness
        n.density = density

    for j in kinematic_graph.joint_graph.nodes():
        j.pos_limits = (-np.pi, np.pi)
        if j.jp.active:
            j.actuator = TMotor_AK80_9()
        j.damphing_friction = (0.05, 0)
    kinematic_graph.define_link_frames()
    builder = Builder(URDFLinkCreater)

    robot, ative_joints, constraints = builder.create_kinematic_graph(kinematic_graph)

    act_description, constraints_descriptions = get_pino_description(
        ative_joints, constraints
    )

    return robot.urdf(), act_description, constraints_descriptions


def jps_graph2urdf_parametrized(
    graph: nx.Graph,
    density_EE: float = 2700 / 2.8,
    density_G: float = 2700 / 2.8,
    actuator=TMotor_AK80_9(),
):
    kinematic_graph = JointPoint2KinematicGraph(graph)
    kinematic_graph.define_main_branch()
    kinematic_graph.define_span_tree()
    thickness = 0.04
    density = 2700 / 2.8

    for n in kinematic_graph.nodes():
        n.thickness = thickness
        n.density = density
        if n.name == "EE":
            n.density = density_EE
        if n.name == "G":
            n.density = density_G

    for j in kinematic_graph.joint_graph.nodes():
        j.pos_limits = (-np.pi, np.pi)
        if j.jp.active:
            j.actuator = actuator
        j.damphing_friction = (0.05, 0)
    kinematic_graph.define_link_frames()
    # builder = Builder(DetalizedURDFCreater)
    builder = Builder(DetalizedURDFCreaterFixedEE)

    robot, ative_joints, constraints = builder.create_kinematic_graph(kinematic_graph)

    act_description, constraints_descriptions = get_pino_description(
        ative_joints, constraints
    )

    return robot.urdf(), act_description, constraints_descriptions


def jps_graph2urdf_by_bulder(
    graph: nx.Graph,
    builder: ParametrizedBuilder
):
    """
    Converts a graph representation of a robot's kinematic structure to a URDF file using a builder.

    Args:
        graph (nx.Graph): The graph representation of the robot's kinematic structure.
        builder (ParametrizedBuilder): The builder object used to create the kinematic graph.

    Returns:
        tuple: A tuple containing the URDF representation of the robot, the actuator description, and the constraints descriptions.
    """
    kinematic_graph = JointPoint2KinematicGraph(graph)
    kinematic_graph.define_main_branch()
    kinematic_graph.define_span_tree()

    kinematic_graph.define_link_frames()

    robot, ative_joints, constraints = builder.create_kinematic_graph(kinematic_graph)

    act_description, constraints_descriptions = get_pino_description(
        ative_joints, constraints
    )

    return robot.urdf(), act_description, constraints_descriptions


def create_dict_jp_limit(joints, limit):
    jp2limits = {}
    for jp, lim in zip(joints, limit):
        jp2limits[jp] = lim
    return jp2limits
