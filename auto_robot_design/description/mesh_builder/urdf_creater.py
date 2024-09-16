
from itertools import combinations
import numpy as np
from scipy.spatial.transform import Rotation as R

import manifold3d as m3d
import trimesh
import modern_robotics as mr
import odio_urdf as urdf

from auto_robot_design.description.actuators import RevoluteUnit
from auto_robot_design.description.builder import URDFLinkCreator
from auto_robot_design.description.kinematics import (
    Box,
    Joint,
    Link,
    Mesh,
    Sphere
)
from auto_robot_design.description.utils import tensor_inertia_sphere_by_mass
from auto_robot_design.utils.geom import calculate_rot_vec2_to_vec1, calculate_transform_with_2points

def manifold2trimesh(manifold):
    mesh = manifold.to_mesh()

    if mesh.vert_properties.shape[1] > 3:
        vertices = mesh.vert_properties[:, :3]
        colors = (mesh.vert_properties[:, 3:] * 255).astype(np.uint8)
    else:
        vertices = mesh.vert_properties
        colors = None

    return trimesh.Trimesh(
        vertices=vertices, faces=mesh.tri_verts, vertex_colors=colors
    )

class MeshCreator:
    def __init__(self, predefind_mesh: dict[str, str] = {}):
        self.predefind_mesh = predefind_mesh
    
    def build_link_mesh(self, link: Link):
        if link.name in self.predefind_mesh:
            body_link = trimesh.
        body = self.build_link_m3d(link)
        mesh = manifold2trimesh(body)
        
        return mesh
    
    def build_link_m3d(self, link: Link):
        
        in_joints = [j for j in link.joints if j.link_in == link]
        out_joints = [j for j in link.joints if j.link_out == link]
        joint_bodies = []
        substract_bodies = []
        frame = mr.TransInv(link.frame @ link.inertial_frame)
        for j in in_joints:
            pos = (frame @ np.hstack((j.jp.r, [1])))[:3]
            ort_move = frame[:3,:3] @ j.jp.w
            rot = calculate_rot_vec2_to_vec1(ort_move)
            size = j.actuator.size
            if len(size) == 0:
                size = [0.03, 0.05]
            tranform = mr.RpToTrans(rot.as_matrix(), pos)
            joint_bodies.append(
                m3d.Manifold.
                cylinder(size[1], size[0], circular_segments=32, center=True).
                transform(tranform[:3,:])
            )
            substract_bodies.append(
                m3d.Manifold.
                cylinder(size[1]-0.03, size[0]+0.01, circular_segments=32, center=True).
                transform(tranform[:3,:])
            )
        for j in out_joints:
            pos = (frame @ np.hstack((j.jp.r, [1])))[:3]
            ort_move = frame[:3,:3] @ j.jp.w
            rot = calculate_rot_vec2_to_vec1(ort_move)
            size = j.actuator.size
            if len(size) == 0:
                size = [0.03, 0.02]
            tranform = mr.RpToTrans(rot.as_matrix(), pos)
            joint_bodies.append(
                m3d.Manifold.
                cylinder(size[1], size[0], circular_segments=32, center=True).
                transform(tranform[:3,:])
            )
            
            

        js = in_joints + out_joints
        num_joint = len(in_joints + out_joints)
        
        j_points = []
        for j in js:
            j_points.append(
                (frame @ np.hstack((j.jp.r, [1])))[:3]
            ) 
        
        if num_joint == 2:
            pos, rot, vec_len = calculate_transform_with_2points(j_points[1], j_points[0])
            thickness = link.geometry.get_thickness()
            if vec_len > thickness:
                length = vec_len - thickness
            else:
                length = vec_len
            Hb = mr.RpToTrans(rot.as_matrix(), pos)
            link_part1 = (m3d.Manifold.cylinder(length, thickness, circular_segments=32, center=True).
                        transform(Hb[:3,:]))
            link_part2 = (m3d.Manifold.cylinder(length, thickness, circular_segments=32, center=True).
                        transform(Hb[:3,:]))
            
            body_1 = m3d.Manifold.batch_hull([link_part1, joint_bodies[0]])
            body_2 = m3d.Manifold.batch_hull([link_part2, joint_bodies[1]])
            
            body_link = body_1 + body_2
        elif num_joint == 3:
            joint_pos_pairs = combinations(range(len(j_points)), 2)
            body_link = m3d.Manifold()
            for k, m in joint_pos_pairs:
                oth_point = (set(range(3)) - set([k,m])).pop()
                pos, rot, vec_len = calculate_transform_with_2points(j_points[k], j_points[m])
                thickness = link.geometry.get_thickness()
                if vec_len > thickness:
                    length = vec_len - thickness
                else:
                    length = vec_len
                Hb = mr.RpToTrans(rot.as_matrix(), pos)
                link_part = (m3d.Manifold.cylinder(length, thickness, circular_segments=32, center=True).
                        transform(Hb[:3,:]))
                body_part = m3d.Manifold.batch_hull([link_part, joint_bodies[oth_point]])
                # body_part = link_part
                body_link = body_link + body_part
        for sub_body in substract_bodies:
            body_link = body_link - sub_body
            
        return body_link

class URDFMeshCreator(URDFLinkCreator):
    def __init__(self) -> None:
        super().__init__()
        self.mesh_path = None
        self.prefix_name = None
        
    def set_path_to_mesh(self, path_to_mesh):
        self.mesh_path = path_to_mesh
        
    def set_prefix_name_mesh(self, prefix):
        self.prefix_name = prefix

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
                name=joint.jp.name + "_" + joint.link_in.name + "_Weld",
                type="fixed",
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
                name=joint.jp.name + "_" + joint.link_out.name + "_Weld",
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
            # urdf_joint = urdf.Joint(
            #     urdf.Parent(link=joint.link_in.name),
            #     urdf.Child(link=joint.link_out.name),
            #     urdf.Origin(
            #         xyz=origin[0],
            #         rpy=origin[1],
            #     ),
            #     urdf.Axis(joint.jp.w.tolist()),
            #     urdf.Limit(
            #         lower=joint.pos_limits[0],
            #         upper=joint.pos_limits[1],
            #         effort=joint.actuator.get_max_effort(),
            #         velocity=joint.actuator.get_max_vel(),
            #     ),
            #     urdf.Dynamics(
            #         damping=joint.damphing_friction[0],
            #         friction=joint.damphing_friction[1],
            #     ),
            #     name=joint.jp.name,
            #     type="revolute",
            # )
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
        # urdf_geometry = urdf.Geometry(urdf.Box(geometry.size))
        to_mesh = cls.mesh_path + cls.prefix_name + name + ".obj"
        urdf_geometry = urdf.Geometry(urdf.Mesh(to_mesh, 1))
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
        # urdf_geometry = urdf.Geometry(urdf.Sphere(geometry.size[0]))
        to_mesh = cls.mesh_path + cls.prefix_name + name + ".obj"
        urdf_geometry = urdf.Geometry(urdf.Mesh(to_mesh, 1))
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
    def _create_mesh(cls, geometry: Mesh, name, inertia, body_origins, link_origin=None):
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
        to_mesh = cls.mesh_path + cls.prefix_name + name + ".obj"
        urdf_geometry = urdf.Geometry(urdf.Mesh(to_mesh, 1))
        urdf_origin = urdf.Origin(
            xyz=link_origin[0],
            rpy=link_origin[1],
        )
        visual = urdf.Visual(
            urdf_origin,
            urdf_geometry,
            urdf_material,
            # name = name_v
        )
        for id, origin in enumerate(body_origins):
            name_c = name + "_" + str(id) + "_Collision"
            name_v = name + "_" + str(id) + "_Visual"
            thickness = geometry.get_thickness()
            urdf_geometry = urdf.Geometry(urdf.Box([thickness, thickness, origin[2]]))
            urdf_origin = urdf.Origin(
                xyz=origin[0],
                rpy=origin[1],
            )

            collision = urdf.Collision(urdf_origin, urdf_geometry, name=name_c)
            # visual_n_collision += [visual, collision]
            visual_n_collision += [collision]
        visual_n_collision += [visual]
        inertial = urdf.Inertial(
            urdf_inertia_origin,
            urdf.Mass(float(geometry.size.mass)),
            urdf.Inertia(**cls.convert_inertia(inertia[1])),
        )
        return urdf.Link(*visual_n_collision, inertial, name=name)