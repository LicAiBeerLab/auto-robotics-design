
import matplotlib.pyplot as plt


from auto_robot_design.description.builder import Builder, URDFLinkCreator, add_branch, jps_graph2urdf
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph
from auto_robot_design.generator.two_link_generator import TwoLinkGenerator
from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.optimization.optimizer import jps_graph2urdf

gen = TwoLinkGenerator()
graphs_and_cons = gen.get_standard_set()
DIR_NAME = "generated_1"
builder = Builder(URDFLinkCreator)
urdf_motors_cons_list = []
for graph_i, constarin_i in graphs_and_cons:
    robot, ative_joints, constraints = jps_graph2urdf(graph_i)
    urdf_motors_cons_tuple = (robot, ative_joints, constraints)
    draw_joint_point(graph_i)
    plt.show()
    urdf_motors_cons_list.append(urdf_motors_cons_tuple)
