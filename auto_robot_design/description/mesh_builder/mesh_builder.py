
from pathlib import Path
import os

from typing import Union
import networkx as nx
import numpy as np
import modern_robotics as mr
from auto_robot_design.description.actuators import Actuator, TMotor_AK80_9
from auto_robot_design.description.builder import BLUE_COLOR, DEFAULT_PARAMS_DICT, GREEN_COLOR, ParametrizedBuilder
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph, KinematicGraph
from auto_robot_design.description.mesh_builder.urdf_creater import MeshCreator, URDFMeshCreator
from auto_robot_design.pino_adapter.pino_adapter import get_pino_description_3d_constraints
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions


class MeshBuilder(ParametrizedBuilder):
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
        creator: URDFMeshCreator,
        mesh_creator: MeshCreator,
        mesh_path = None,
        density: Union[float, dict] = 2700 / 2.8,
        thickness: Union[float, dict] = 0.01,
        joint_damping: Union[float, dict] = 0.05,
        joint_friction: Union[float, dict] = 0,
        joint_limits: Union[dict, tuple] = (-np.pi, np.pi),
        size_ground: np.ndarray = np.zeros(3),
        offset_ground: np.ndarray = np.zeros(3),
        actuator: Union[Actuator, dict]=TMotor_AK80_9(),
    ) -> None:
        super().__init__(creator, density, thickness, joint_damping, joint_damping, joint_friction, joint_limits, size_ground, offset_ground, actuator)
        self.mesh_creator: MeshCreator = mesh_creator
        self.mesh_path = mesh_path

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

    def create_meshes(self, kinematic_graph, prefix=""):
        if self.mesh_path is None:
            dirpath = Path().parent.absolute()
            path_to_mesh = dirpath.joinpath("mesh")
            if not path_to_mesh.exists():
                os.mkdir(path_to_mesh)
            self.mesh_path = path_to_mesh
        
        self.creater.set_path_to_mesh(self.mesh_path)
        self.creater.set_prefix_name_mesh(prefix)
        
        links = kinematic_graph.nodes()
        for link in links:
            link_mesh = self.mesh_creator.build_link_mesh(link)
            link_mesh.apply_scale(1)
            name = prefix + link.name + ".obj"
            link_mesh.export(Path(self.mesh_path).joinpath(name))


def jps_graph2pinocchio_meshes_robot(
    graph: nx.Graph,
    builder: ParametrizedBuilder
):
    """
    Converts a Joint Point Structure (JPS) graph to a Pinocchio robot model.

    Args:
        graph (nx.Graph): The Joint Point Structure (JPS) graph representing the robot's kinematic structure.
        builder (ParametrizedBuilder): The builder object used to create the kinematic graph.

    Returns:
        tuple: A tuple containing the robot model with fixed base and free base.
    """

    kinematic_graph = JointPoint2KinematicGraph(graph)
    kinematic_graph.define_main_branch()
    kinematic_graph.define_span_tree()
    
    # thickness_aux_branch = 0.025
    i = 1
    k = 1
    name_link_in_aux_branch = []
    for link in kinematic_graph.nodes():
        if link in kinematic_graph.main_branch.nodes():
            # print("yes")
            link.geometry.color = BLUE_COLOR[i,:].tolist()
            i = (i + 1) % 6
        else:
            link.geometry.color = GREEN_COLOR[k,:].tolist()
            name_link_in_aux_branch.append(link.name)
            k = (k + 1) % 5

    # builder.thickness = {link: thickness_aux_branch for link in name_link_in_aux_branch}

    kinematic_graph.define_link_frames()

    robot, ative_joints, constraints = builder.create_kinematic_graph(kinematic_graph)

    act_description, constraints_descriptions = get_pino_description_3d_constraints(
        ative_joints, constraints
    )

    fixed_robot = build_model_with_extensions(robot.urdf(),
                                joint_description=act_description,
                                loop_description=constraints_descriptions,
                                actuator_context=kinematic_graph,
                                fixed=True)

    free_robot = build_model_with_extensions(robot.urdf(),
                                joint_description=act_description,
                                loop_description=constraints_descriptions,
                                actuator_context=kinematic_graph,
                                fixed=False)

    return fixed_robot, free_robot