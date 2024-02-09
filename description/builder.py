from copy import deepcopy
from itertools import combinations
from math import isclose
from shlex import join
from turtle import st
from pyparsing import List, Tuple

import odio_urdf as urdf
import networkx as nx
import numpy.linalg as la
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import modern_robotics as mr

from description.kinematics import Geometry, Joint, JointPoint, Link, Mesh, Sphere, Box

# from description.utils import calculate_inertia


def add_branch(G: nx.Graph, branch: List[JointPoint] | List[List[JointPoint]]):
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
    branch: List[Tuple[JointPoint, dict]] | List[List[Tuple[JointPoint, dict]]],
):
    is_list = [isinstance(br, List) for br in branch]
    if all(is_list):
        for b in branch:
            add_branch_with_attrib(G, b)
    else:
        for ed in branch:
            G.add_edge(ed[0], ed[1], **ed[2])


class URDFLinkCreater:
    def __init__(self) -> None:
        pass

    @classmethod
    def create_link(cls, link: Link):
        if link.geometry.shape == "mesh":
            pos_joint_in_local = []
            H_l_w = mr.TransInv(link.frame)
            for j in link.joints:
                pos_joint_in_local.append(
                    H_l_w @ np.r_[j.jp.r, 1]
                )
                
            joint_pos_pairs = combinations(pos_joint_in_local, 2)
            ez = np.array([0,0,1,0])
            body_origins = []
            for j_p in joint_pos_pairs:
                v_l = j_p[1] - j_p[0]
                angle = np.arccos(np.inner(ez, v_l) / la.norm(v_l) / la.norm(ez))
                axis = mr.VecToso3(ez[:3]) @ v_l[:3]
                if not np.isclose(np.sum(axis), 0):
                    axis /= la.norm(axis)
                
                rot = R.from_rotvec(axis * angle)
                pos = (j_p[1][:3] + j_p[0][:3])/2
                length = la.norm(v_l)
                body_origins.append((pos.tolist(), rot.as_euler("xyz").tolist(), length))
            inertia = (link.frame,
                    link.geometry.size.moment_inertia_frame(link.frame)
                )
            urdf_link = cls._create_mesh(link.geometry, link.name, inertia, body_origins, link.thickness)
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
        if joint.link_in is None or joint.link_out is None:
            return {"joint": []}
        origin = cls.trans_matrix2xyz_rpy(joint.frame)
        if joint.is_constraint:
            color1 = [0,0,0,0.3]
            color2 = [1,1,1,0.3]
            
            name_link_in = joint.jp.name + "_" + joint.link_in.name + "Pseudo"
            urdf_pseudo_link_in = urdf.Link(
                urdf.Visual(urdf.Geometry(urdf.Sphere(joint.link_in.thickness*1.5)),
                            urdf.Material(urdf.Color(rgba=color1))),
                name = name_link_in,  
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
                lower=joint.limits.get("lower", -2*np.pi),
                upper=joint.limits.get("upper", 2*np.pi),
                effort=joint.limits.get("effort", 1),
                velocity=joint.limits.get("velocity", 10),
            ),
            urdf.Dynamics(damping=joint.damphing_friction[0], friction=joint.damphing_friction[1]),
            name=joint.jp.name + "_" + joint.link_in.name + "_revolute",
            type="revolute")

            name_link_out = joint.jp.name + "_" + joint.link_out.name + "Pseudo"
            urdf_pseudo_link_out= urdf.Link(
                urdf.Visual(urdf.Geometry(urdf.Sphere(joint.link_out.thickness*1.5)),
                            urdf.Material(urdf.Color(rgba=color2))),
                name = name_link_out,
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
            name=joint.jp.name + "_" + joint.link_in.name + "_fix",
            type="fixed")

            out = {"joint":[urdf_pseudo_link_in, urdf_joint_in, urdf_joint_out, urdf_pseudo_link_out],
                    "constraint": [name_link_in, name_link_out]}
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
                lower=joint.limits.get("lower", -2*np.pi),
                upper=joint.limits.get("upper", 2*np.pi),
                effort=joint.limits.get("effort", 1),
                velocity=joint.limits.get("velocity", 10),
            ),
            urdf.Dynamics(damping=joint.damphing_friction[0], friction=joint.damphing_friction[1]),
            name=joint.jp.name,
            type="revolute")
            out = {"joint":[urdf_joint]}
        if joint.jp.active:
            out["active"] = joint.jp.name
        return out

    @classmethod
    def trans_matrix2xyz_rpy(cls, H):
        Rp = mr.TransToRp(H)
        rpy = R.from_matrix(Rp[0]).as_euler("xyz").tolist()
        return (Rp[1].tolist(), rpy)

    
    @classmethod
    def convert_inertia(cls, tensor_inertia):
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
        color = np.r_[np.random.uniform(0, 1, 3), 0.7]
        urdf_material = urdf.Material(urdf.Color(rgba=color.tolist()))
        name_c = name + "_" + "collision"
        name_v = name + "_" + "visul"
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
        collision = urdf.Collision(
            urdf_origin,
            urdf_geometry,
            name = name_c
        )
        inertial = urdf.Inertial(
            urdf_inertia_origin,
            urdf.Mass(float(geometry.mass)),
            urdf.Inertia(**cls.convert_inertia(geometry.inertia))
        )
        
        return urdf.Link(
            visual,
            collision,
            inertial,
            name = name
        )

    @classmethod
    def _create_sphere(cls, geometry: Sphere, name, origin, inertia_origin):
        color = np.r_[np.random.uniform(0, 1, 3), 0.7]
        urdf_material = urdf.Material(urdf.Color(rgba=color.tolist()))
        
        name_c = name + "_" + "collision"
        name_v = name + "_" + "visul"
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
        collision = urdf.Collision(
            urdf_origin,
            urdf_geometry,
            name = name_c
        )
        inertial = urdf.Inertial(
            urdf_inertia_origin,
            urdf.Mass(geometry.mass),
            urdf.Inertia(**cls.convert_inertia(geometry.inertia))
        )
        
        return urdf.Link(
            visual,
            collision,
            inertial,
            name = name
        )

    @classmethod
    def _create_mesh(cls, geometry: Mesh, name, inertia, body_origins, thickness):
        color = np.r_[np.random.uniform(0, 1, 3), 0.7]
        urdf_material = urdf.Material(urdf.Color(rgba=color.tolist()))
        origin_I = cls.trans_matrix2xyz_rpy(inertia[0])
        urdf_inertia_origin = urdf.Origin(
            xyz = origin_I[0],
            rpy = origin_I[1]
        )
        visual_n_collision = []
        for id, origin in enumerate(body_origins):
            name_c = name + "_" + str(id) + "collision"
            name_v = name + "_" + str(id) + "visual"
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
            
            collision = urdf.Collision(
                urdf_origin,
                urdf_geometry,
                name = name_c
            )
            visual_n_collision += [visual, collision]
        inertial = urdf.Inertial(
            urdf_inertia_origin,
            urdf.Mass(float(geometry.mass)),
            urdf.Inertia(**cls.convert_inertia(inertia[1]))
        )
        return urdf.Link(
            *visual_n_collision,
            inertial,
            name = name
        )


class Builder:
    def __init__(
        self,
        creater
    ) -> None:
        self.creater = creater
        
    def create_kinematic_graph(self, kinematic_graph, name = "Robot"):
        
        links = kinematic_graph.nodes()
        joints = dict(filter(lambda kv: len(kv[1]) >0 , kinematic_graph.joint2edge.items()))
        
        
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
        
        urdf_robot = urdf.Robot(
            *urdf_objects,
            name = name
        )
        
        return urdf_robot, active_joints, constraints



def create_urdf(graph: nx.Graph):
    urdf_elements = []
    for link in graph.nodes():
        data_link = graph.nodes()[link]
        geom_frame = data_link["frame_geom"]

        if link in ("EE", "G"):
            length = 0.1
        # elif len(data_link["link"].joints) == 2:
        #     length = la.norm(data_link["m_out"][0].r - data_link["in"][0].r)
        #     geometry = urdf.Geometry(urdf.Box([0.1, 0.1, length])),
        # else:
        #     # H_l_w = mr.TransInv(data_link["H_w_l"])
        #     # vertices = []
        #     # for j in data_link["link"].joints:
        #     #     v_l = H_l_w @ np.c_[j.r, 1]
        #     #     v_l_2 = deepcopy(v_l)
        #     #     v_l[1] = 0.05
        #     #     v_l_2[1] = -0.05
        #     #     vertices.append(v_l.tolist())
        #     #     vertices.append(v_l_2.tolist())
        #     # trimesh.Trimesh()
        inertia = calculate_inertia(length)
        urdf_link = urdf.Link(
            urdf.Inertial(
                urdf.Origin(
                    xyz=geom_frame[0].tolist(),
                    rpy=R.from_quat(geom_frame[1]).as_euler("xyz").tolist(),
                ),
                urdf.Mass(1),
                urdf.Inertia(**inertia),
            ),
            urdf.Visual(
                urdf.Origin(
                    xyz=geom_frame[0].tolist(),
                    rpy=R.from_quat(geom_frame[1]).as_euler("xyz").tolist(),
                ),
                urdf.Geometry(urdf.Box([0.1, 0.1, length])),
                urdf.Material("Grey"),
            ),
            urdf.Collision(
                urdf.Origin(
                    xyz=geom_frame[0].tolist(),
                    rpy=R.from_quat(geom_frame[1]).as_euler("xyz").tolist(),
                ),
                urdf.Geometry(urdf.Box([0.1, 0.1, length])),
            ),
            name=link,
        )
        urdf_elements.append(urdf_link)

        if data_link.get("out", {}):
            for out in data_link["out"].items():
                data_child_link = graph.nodes()[out[1]]
                frame = data_child_link["frame"]
                urdf_elements.append(
                    urdf.Joint(
                        urdf.Parent(link=link),
                        urdf.Child(link=out[1]),
                        urdf.Origin(
                            xyz=frame[0].tolist(),
                            rpy=R.from_quat(frame[1]).as_euler("xyz").tolist(),
                        ),
                        urdf.Axis(out[0].w.tolist()),
                        urdf.Limit(
                            lower=-np.pi,
                            upper=np.pi,
                            effort=self.effort_limit,
                            velocity=10,
                        ),
                        urdf.Dynamics(damping=0.05),
                        name=out[0].name,
                        type="revolute",
                    )
                )
    return urdf.Robot(*urdf_elements)
