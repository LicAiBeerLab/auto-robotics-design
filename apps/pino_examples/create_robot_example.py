from copy import deepcopy
from dataclasses import dataclass
import time
import odio_urdf
from auto_robot_design.pinokla.loader_tools import completeRobotLoader
from pinocchio.visualize import MeshcatVisualizer, GepettoVisualizer
import meshcat
import os
import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import random
import os
import sys
import xml 
 
def create_box_link(
    name: str, mass: float, x_size_length: float, y_size: float, z_size: float
):
    random.seed(sum(map(ord, name)))
    box = odio_urdf.Box(size=[x_size_length, y_size, z_size])
    inertia_dict = get_dict_box_inertia(x_size_length, y_size, z_size, mass)
    link = odio_urdf.Link(
        name,
        odio_urdf.Visual(
            odio_urdf.Origin(xyz=[x_size_length / 2, 0, 0], rpy=[0, 0, 0]),
            odio_urdf.Geometry(box),
            odio_urdf.Material(
                name + "_material",
                odio_urdf.Color(
                    rgba=[random.random(), random.random(), random.random(), 0.2]
                ),
            ),
        ),
        odio_urdf.Collision(
            odio_urdf.Origin(xyz=[x_size_length / 2, 0, 0], rpy=[0, 0, 0]),
            odio_urdf.Geometry(box),
 
        ),
        odio_urdf.Inertial(
            odio_urdf.Origin(xyz=[x_size_length / 2, 0, 0], rpy=[0, 0, 0]),
            odio_urdf.Mass(mass),
            odio_urdf.Inertia(**inertia_dict),
        ),
    )
    return link


def create_psedo_sphere_link(
    name: str, r_size_length: float, mass: float = 1e-9, xyz=[0, 0, 0], rpy=[0, 0, 0]
):
    random.seed(sum(map(ord, name)))
    sphere = odio_urdf.Sphere(radius=r_size_length)
    zero_inertia_dict = get_dict_box_inertia(
        r_size_length, r_size_length, r_size_length, 0
    )
    link = odio_urdf.Link(
        name,
        odio_urdf.Visual(
            odio_urdf.Origin(xyz=xyz, rpy=rpy),
            odio_urdf.Geometry(sphere),
            odio_urdf.Material(
                name + "_material",
                odio_urdf.Color(
                    rgba=[random.random(), random.random(), random.random(), 0.2]
                ),
            ),
        ),
        odio_urdf.Inertial(
            odio_urdf.Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
            odio_urdf.Mass(mass),
            odio_urdf.Inertia(**zero_inertia_dict),
        ),
    )
    return link


def get_dict_box_inertia(x_size, y_size, z_size, mass):
    magic_fun = lambda a, b: (a * 2 + b**2) * mass / 12
    return {
        "ixx": magic_fun(y_size, z_size),
        "iyy": magic_fun(x_size, z_size),
        "izz": magic_fun(x_size, y_size),
        "ixy": 0,
        "iyz": 0,
        "ixz": 0,
    }


base_link = create_box_link("base_link", 5, 0.6, 0.2, 0.2)

link1_auto = create_box_link("link1", 5, 0.5, 0.12, 0.12)
link2_auto = create_box_link("link2", 5, 0.5, 0.11, 0.11)
link3_auto = create_box_link("link3", 5, 0.6, 0.14, 0.12)
link4_auto = create_box_link("link4", 5, 0.6, 0.11, 0.16)

link5_psedo_auto = create_psedo_sphere_link("link5_psedo", 0.1)
link6_psedo_auto = create_psedo_sphere_link("link6_psedo", 0.1)


def create_revolve_mid_joint():
    pass


joint1 = odio_urdf.Joint(
    odio_urdf.Origin(xyz=[0.5, 0, 0], rpy=[0, 0, 0]),
    odio_urdf.Parent("link1"),
    odio_urdf.Child("link2"),
    odio_urdf.Axis(xyz=[0, 0, 1]),
    odio_urdf.Limit(effort=5, velocity=50, lower = -np.pi,  upper =  np.pi),
    type="revolute",
    name="joint1",
)

joint2 = odio_urdf.Joint(
    odio_urdf.Origin(xyz=[0.6, 0, 0], rpy=[0, 0, 0]),
    odio_urdf.Parent("base_link"),
    odio_urdf.Child("link1"),
    odio_urdf.Axis(xyz=[0, 0, 1]),
    odio_urdf.Limit(effort=5, velocity=50, lower = -np.pi,  upper =  np.pi),
    type="revolute",
    name="joint2",
)


joint3 = odio_urdf.Joint(
    odio_urdf.Origin(xyz=[0.0, 0, 0], rpy=[0, 0, 3.14159]),
    odio_urdf.Parent("base_link"),
    odio_urdf.Child("link3"),
    odio_urdf.Axis(xyz=[0, 0, 1]),
    odio_urdf.Limit(effort=5, velocity=50, lower = -np.pi,  upper =  np.pi),
    type="revolute",
    name="joint3",
)

joint4 = odio_urdf.Joint(
    odio_urdf.Origin(xyz=[0.6, 0, 0], rpy=[0, 0, 0]),
    odio_urdf.Parent("link3"),
    odio_urdf.Child("link4"),
    odio_urdf.Axis(xyz=[0, 0, 1]),
    odio_urdf.Limit(effort=5, velocity=50, lower = -np.pi,  upper =  np.pi),
    type="revolute",
    name="joint4",
)

joint5 = odio_urdf.Joint(
    odio_urdf.Origin(xyz=[0.6, 0, 0], rpy=[0, 0, 0]),
    odio_urdf.Parent("link4"),
    odio_urdf.Child("link5_psedo"),
    odio_urdf.Axis(xyz=[0, 0, 0]),
    type="fixed",
    name="joint5",
)

joint6 = odio_urdf.Joint(
    odio_urdf.Origin(xyz=[0.5, 0, 0], rpy=[0, 0, 0]),
    odio_urdf.Parent("link2"),
    odio_urdf.Child("link6_psedo"),
    odio_urdf.Axis(xyz=[0, 0, 1]),
    odio_urdf.Limit(effort=5, velocity=50),
    type="revolute",
    name="joint6",
)

loop_description = {"closed_loop": [["link6_psedo", "link5_psedo"]], "type": ["6d"]}
joint_description = {
    "name_mot": ["joint2", "joint3"],
    "joint_name": [],
    "joint_type": [],
}

