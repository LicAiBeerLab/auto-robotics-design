# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import manifold3d as m3d
import trimesh
import numpy as np

from auto_robot_design.generator.topologies.bounds_preset import (
    get_preset_by_index_with_bounds,
)
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph, KinematicGraph
from auto_robot_design.description.builder import (
    ParametrizedBuilder,
    URDFLinkCreater3DConstraints,
    jps_graph2pinocchio_robot_3d_constraints,
    MIT_CHEETAH_PARAMS_DICT
)
import meshcat
from pinocchio.visualize import MeshcatVisualizer
from auto_robot_design.pinokla.closed_loop_kinematics import (
    closedLoopProximalMount,
)
# Helper to convert a Manifold into a Trimesh
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


# Helper to display interactive mesh preview with trimesh
def showMesh(mesh):
  scene = trimesh.Scene()
  scene.add_geometry(mesh)
  # scene.add_geometry(trimesh.creation.axis())
  scene.show()


# %%
thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]
actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
density = MIT_CHEETAH_PARAMS_DICT["density"]
body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]


builder = ParametrizedBuilder(URDFLinkCreater3DConstraints,
                              density={"default": density, "G": body_density},
                              thickness={"default": thickness, "EE": 0.033},
                              actuator={"default": actuator},
                              size_ground=np.array(
                                  MIT_CHEETAH_PARAMS_DICT["size_ground"]),
                              offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"]
                              )
builder = ParametrizedBuilder(URDFLinkCreater3DConstraints)
gm = get_preset_by_index_with_bounds(8)
x_centre = gm.generate_central_from_mutation_range()
graph_jp = gm.get_graph(x_centre)

kinematic_graph = JointPoint2KinematicGraph(graph_jp)
kinematic_graph.define_main_branch()
kinematic_graph.define_span_tree()
kinematic_graph.define_link_frames()

links = kinematic_graph.nodes()


# %%
from auto_robot_design.description.builder import jps_graph2urdf_by_bulder


robo_urdf, joint_description, loop_description = jps_graph2urdf_by_bulder(graph_jp, builder)


with open("parametrized_builder_test.urdf", "w") as f:
    f.write(robo_urdf)


# %%
from auto_robot_design.description.utils import draw_joint_point


draw_joint_point(graph_jp)


# %%
from pathlib import Path
import os
dirpath = Path().parent.absolute()
path_to_mesh = dirpath.joinpath("mesh")
if not path_to_mesh.exists():
    os.mkdir(path_to_mesh)
print(path_to_mesh)


# %%
from itertools import combinations
import numpy.linalg as la
import modern_robotics as mr
from scipy.spatial.transform import Rotation as R

def calculate_rot_vec2_to_vec1(vec1: np.ndarray,
                            vec2: np.ndarray = np.array([0, 0, 1])):
    """Calculate transformation from `vec2` to vector `vec1`

    Args:
        p1 (np.ndarray): point of vector's start
        p2 (np.ndarray): point of vector's end
        vec (np.ndarray, optional): Vector transform from. Defaults to np.array([0, 0, 1]).

    Returns:
        tuple: position: np.ndarray, rotation: scipy.spatial.rotation, length: float
    """
    angle = np.arccos(np.inner(vec2, vec1) / la.norm(vec1) / la.norm(vec2))
    axis = mr.VecToso3(vec2) @ vec1
    if not np.isclose(np.sum(axis), 0):
        axis /= la.norm(axis)

    rot = R.from_rotvec(axis * angle)
    
    return rot

def calculate_transform_with_2points(p1: np.ndarray, 
                                     p2: np.ndarray,
                                     vec: np.ndarray = np.array([0, 0, 1])):
    """Calculate transformation from `vec` to vector build with points `p1` and `p2`

    Args:
        p1 (np.ndarray): point of vector's start
        p2 (np.ndarray): point of vector's end
        vec (np.ndarray, optional): Vector transform from. Defaults to np.array([0, 0, 1]).

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

def create_mesh(in_joints, out_joints, link):
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
            cylinder(size[1]-0.03, size[0]+0.015, circular_segments=32, center=True).
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
            print("")
    for sub_body in substract_bodies:
        body_link = body_link - sub_body
    return body_link

meshes = []
for link in links: 
    in_joints = [j for j in link.joints if j.link_in == link]
    out_joints = [j for j in link.joints if j.link_out == link]
    print(link.name)
    num_joint = len(link.joints)
    print(num_joint)
    if num_joint == 1:
        body = m3d.Manifold.sphere(link.geometry.get_thickness(), circular_segments=32)
        mesh = manifold2trimesh(body)
        mesh.apply_scale(1)
        name = link.name + ".obj"
        mesh.export(path_to_mesh.joinpath(name))
    else:
        body_m3d = create_mesh(in_joints, out_joints, link)
        mesh = manifold2trimesh(body_m3d)
        mesh.apply_scale(1)
        name = link.name + ".obj"
        mesh.export(path_to_mesh.joinpath(name))
    meshes.append(mesh)


# %%
for mesh in meshes:
    showMesh(mesh)


# %%



# %%
# help(m3d)


