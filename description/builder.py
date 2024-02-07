from copy import deepcopy
from turtle import st
from pyparsing import List, Tuple

import odio_urdf as urdf
import networkx as nx
import numpy.linalg as la
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import modern_robotics as mr

from description.kinematics import JointPoint, Link

from description.utils import calculate_inertia
from abc import abstractmethod


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


class Builder:
    def __init__(
        self,
        density=100,
        thickness=0.1,
        joint_limit=(-np.pi, np.pi),
        effort_limit=1,
        velocity_limit=10,
        damphing = 0.05,
        stiffness = 0
    ) -> None:
        self.density = density
        self.thickness = thickness
        self.joint_limit = joint_limit
        self.effort_limit = effort_limit
        self.velocity_limit = velocity_limit
        self.stiffness = stiffness
        self.damphing = damphing

    def create_link(self, link: Link):

        urdf_link = urdf.Link(
            urdf.Inertial(
                urdf.Origin(
                    xyz=link.inertial_frame[:3].tolist(),
                    rpy=R.from_quat(link.inertial_frame[3:]).as_euler("xyz").tolist(),
                ),
                urdf.Mass(1),
                urdf.Inertia(**inertia),
            ),
            urdf.Visual(
                urdf.Origin(
                    xyz=link.inertial_frame[:3].tolist(),
                    rpy=R.from_quat(link.inertial_frame[3:]).as_euler("xyz").tolist(),
                ),
                urdf.Geometry(urdf.Box([self.thickness, self.thickness, length])),
                urdf.Material("Grey"),
            ),
            urdf.Collision(
                urdf.Origin(
                    xyz=link.inertial_frame[:3].tolist(),
                    rpy=R.from_quat(link.inertial_frame[3:]).as_euler("xyz").tolist(),
                ),
                urdf.Geometry(urdf.Box([self.thickness, self.thickness, length])),
                urdf.Material("Grey"),
            ),
            name=link.name,
        )
        return urdf_link

    def create_joint(self, edge):
        urdf_joint = urdf.Joint(
            urdf.Parent(link=edge[0].name),
            urdf.Child(link=edge[1].name),
            urdf.Origin(
                xyz=edge[1].frame[:3].tolist(),
                rpy=R.from_quat(edge[1].frame[3:]).as_euler("xyz").tolist(),
            ),
            urdf.Axis(edge[2].w.tolist()),
            urdf.Limit(
                lower=self.joint_limit[0],
                upper=self.joint_limit[1],
                effort=self.effort_limit,
                velocity=self.velocity_limit,
            ),
            urdf.Dynamics(damping=self.damphing, 
                        stiffness=self.stiffness),
            name=edge[2].name,
            type="revolute",
        )
        return urdf_joint


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
                urdf.Material("Grey"),
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
