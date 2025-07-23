
from dataclasses import dataclass
from typing import Tuple, Optional
from auto_robot_design.generator.branch_generator.graph_scheme import MutationType, MutationCoordinate, SchemeEE, SchemeJoint, SchemeConnectionJoint
from auto_robot_design.description.kinematics import JointPoint
import networkx as nx
import numpy as np
from auto_robot_design.description.utils import draw_joint_point
@dataclass
class AbsoluteMutation:
    mutation_x: MutationCoordinate = MutationCoordinate()
    mutation_y: MutationCoordinate = MutationCoordinate()
    mutation_z: MutationCoordinate = MutationCoordinate()

@dataclass
class RelativeMutation(AbsoluteMutation):
    relative_to:Optional[JointPoint] = None

@dataclass
class RelativePercentageMutation(AbsoluteMutation):
    relative_to: Optional[Tuple[JointPoint,JointPoint]] = None


class MutableGraphManager:
    def __init__(self, graph_scheme=None):
        self.graph_scheme = graph_scheme
        self.mutations = {}
        self.current_mutation_ranges = {}
        self.joint_points = {}
        self.graph = nx.Graph()

    def create_jp_from_sp(self, scheme_point) -> JointPoint:
        if isinstance(scheme_point, SchemeEE):
            jp = JointPoint(r=None, attach_endeffector=True,name=scheme_point.name)
        else:
            jp = JointPoint(r=None, name=scheme_point.name, attach_ground=scheme_point.attach_ground, active=scheme_point.active)

        return jp

    def build_graph(self):
        """Builds the graph from the graph scheme.
        
        Creates JointPoint objects for each SchemePoint and MutationEntry for mutation ranges.
        """
        branch_idx = 0
        while branch_idx in self.graph_scheme:
            branch = self.graph_scheme[branch_idx]
            self.joint_points[branch_idx] = []
            self.mutations[branch_idx] = []
            for idx, point in enumerate(branch):
                # first step - get the joint point without position
                jp = self.create_jp_from_sp(point)
                jp.name = f"b_{branch_idx}_jp_{idx}"
                # second step - solve the graph relation for the new joint point
                if idx == 0: # first point in branch, it can be a jp for main branch or connection for any other branch
                    if jp.attach_ground:
                        self.joint_points[branch_idx].append(jp)
                        self.graph.add_node(jp) #ground points are just added to the graph
                    else: # the only other possibility is the connection point attached to already existing pair of joints
                        # get the joints to connect
                        branch_to_connect = point.connected_to[0]
                        joints = (self.joint_points[branch_to_connect][point.connected_to[1]], self.joint_points[branch_to_connect][point.connected_to[1]+1])
                        self.graph.add_edge(joints[0], jp)
                        self.graph.add_edge(joints[1], jp)
                        self.joint_points[branch_idx].append(jp)
                else:
                    # if joint the next point is joint or ee it is just added with connection to previous point
                    if isinstance(point, SchemeJoint) or isinstance(point, SchemeEE):
                        self.graph.add_edge(jp, self.joint_points[branch_idx][-1])
                        self.joint_points[branch_idx].append(jp)
                    else:
                        # if it is connection joint, it is added to the graph with connection to previous point and connection to the corresponding branch
                        self.graph.add_edge(jp, self.joint_points[branch_idx][-1])
                        self.joint_points[branch_idx].append(jp)
                        branch_to_connect = point.connected_to[0]
                        joints = (self.joint_points[branch_to_connect][point.connected_to[1]], self.joint_points[branch_to_connect][point.connected_to[1]+1])
                        self.graph.add_edge(joints[0], jp)
                        self.graph.add_edge(joints[1], jp)

                # third step - create mutation entry for the point
                if point.mutation_type == MutationType.ABSOLUTE:
                    mutation = AbsoluteMutation(mutation_x=point.mutation_x, mutation_y=point.mutation_y, mutation_z=point.mutation_z)
                    self.mutations[branch_idx].append(mutation)
                elif point.mutation_type == MutationType.RELATIVE:
                    # get relative to what
                    if isinstance(point, SchemeJoint) or isinstance(point, SchemeEE):
                        relative_to = self.joint_points[branch_idx][-2]
                        mutation = RelativeMutation(mutation_x=point.mutation_x, mutation_y=point.mutation_y, mutation_z=point.mutation_z, relative_to=relative_to)
                        if isinstance(branch[idx-1], SchemeConnectionJoint):
                            shift = branch[idx-1].dependent_shift
                            mutation.mutation_x.shift = shift[0]
                            mutation.mutation_y.shift = shift[1]
                            mutation.mutation_z.shift = shift[2]
                        self.mutations[branch_idx].append(mutation)
                    elif isinstance(point, SchemeConnectionJoint):
                        mutation = RelativeMutation(mutation_x=point.mutation_x, mutation_y=point.mutation_y, mutation_z=point.mutation_z)
                        first_joint, second_joint = self.joint_points[point.connected_to[0]][point.connected_to[1]], self.joint_points[point.connected_to[0]][point.connected_to[1]+1] 
                        mutation.relative_to = (first_joint, second_joint)
                        self.mutations[branch_idx].append(mutation)
                elif point.mutation_type == MutationType.RELATIVE_PERCENTAGE:
                    if isinstance(point, SchemeConnectionJoint):
                        mutation = RelativePercentageMutation(mutation_x=point.mutation_x, mutation_y=point.mutation_y, mutation_z=point.mutation_z)
                        first_joint, second_joint = self.joint_points[point.connected_to[0]][point.connected_to[1]], self.joint_points[point.connected_to[0]][point.connected_to[1]+1]
                        mutation.relative_to = (first_joint, second_joint)
                        self.mutations[branch_idx].append(mutation)

            branch_idx += 1

    def get_mutation_ranges(self):
        """returns the current mutation ranges"""
        if len(self.mutations.keys())==0:
            raise ValueError("No mutations defined. Please build the graph first.")
        
        branch_idx = 0
        while branch_idx in self.mutations:
            for mutation_idx, mutation in enumerate(self.mutations[branch_idx]):
                if mutation.mutation_x.freeze is None:
                    if mutation.mutation_x.lower_bound != None and mutation.mutation_x.upper_bound:
                        jp = self.joint_points[branch_idx][mutation_idx]
                        if isinstance(mutation, AbsoluteMutation):
                            if mutation.mutation_x.shift is not None:
                                self.current_mutation_ranges[(jp, 'x')] = (mutation.mutation_x.mutation_origin + mutation.mutation_x.lower_bound + mutation.mutation_x.shift, mutation.mutation_x.mutation_origin + mutation.mutation_x.upper_bound + mutation.mutation_x.shift)
                            else:
                                self.current_mutation_ranges[(jp, 'x')] = (mutation.mutation_x.mutation_origin + mutation.mutation_x.lower_bound, mutation.mutation_x.mutation_origin + mutation.mutation_x.upper_bound)

                        else:
                            self.current_mutation_ranges[(jp, 'x')] = (mutation.mutation_x.lower_bound, mutation.mutation_x.upper_bound)
                
                if mutation.mutation_y.freeze is None:
                    if mutation.mutation_y.lower_bound != None and mutation.mutation_y.upper_bound:
                        jp = self.joint_points[branch_idx][mutation_idx]
                        if isinstance(mutation, AbsoluteMutation):
                            if mutation.mutation_y.shift is not None:
                                self.current_mutation_ranges[(jp, 'y')] = (mutation.mutation_y.mutation_origin + mutation.mutation_y.lower_bound + mutation.mutation_y.shift, mutation.mutation_y.mutation_origin + mutation.mutation_y.upper_bound + mutation.mutation_y.shift)
                            else:
                                self.current_mutation_ranges[(jp, 'y')] = (mutation.mutation_y.mutation_origin + mutation.mutation_y.lower_bound, mutation.mutation_y.mutation_origin + mutation.mutation_y.upper_bound)
                        else:
                            self.current_mutation_ranges[(jp, 'y')] = (mutation.mutation_y.lower_bound, mutation.mutation_y.upper_bound)
                
                if mutation.mutation_z.freeze is None:
                    if mutation.mutation_z.lower_bound != None and mutation.mutation_z.upper_bound:
                        jp = self.joint_points[branch_idx][mutation_idx]
                        if isinstance(mutation, AbsoluteMutation):
                            if mutation.mutation_z.shift is not None:
                                self.current_mutation_ranges[(jp, 'z')] = (mutation.mutation_z.mutation_origin + mutation.mutation_z.lower_bound + mutation.mutation_z.shift, mutation.mutation_z.mutation_origin + mutation.mutation_z.upper_bound + mutation.mutation_z.shift)
                            else:
                                self.current_mutation_ranges[(jp, 'z')] = (mutation.mutation_z.mutation_origin + mutation.mutation_z.lower_bound, mutation.mutation_z.mutation_origin + mutation.mutation_z.upper_bound)
                        else:
                            self.current_mutation_ranges[(jp, 'z')] = (mutation.mutation_z.lower_bound, mutation.mutation_z.upper_bound)
            branch_idx += 1
        return self.current_mutation_ranges

    def get_graph(self, params):
        """Produce a graph of the set topology from the given parameters.

        Args:
            parameters List[float]: list of mutations.

        Raises:
            Exception: raise an exception if the number of parameters is not equal to the number of mutation ranges.

        Returns:
            nx.Graph: the graph of a mechanism with the given parameters.
        """
        if len(params) != len(list(self.current_mutation_ranges.keys())):
            raise ValueError(
                'Wrong number of parameters for graph specification!')
        
        parameter_counter = 0
        # starting to set coordinates of the joint points one by one
        for branch_idx, jp_branch in self.joint_points.items():
            for jp_idx, jp in enumerate(jp_branch):
                jp.r = np.zeros(3)
                mutation = self.mutations[branch_idx][jp_idx]
                if type(mutation) is AbsoluteMutation: 
                # isinstance(mutation, AbsoluteMutation):
                    if not mutation.mutation_x.freeze is None:
                        jp.r[0] = mutation.mutation_x.freeze
                    elif mutation.mutation_x.lower_bound == mutation.mutation_x.upper_bound:
                        jp.r[0] = mutation.mutation_x.lower_bound
                    else:
                        jp.r[0] = params[parameter_counter] + mutation.mutation_x.mutation_origin
                        parameter_counter += 1
                    if mutation.mutation_x.shift is not None:
                        jp.r[0] += mutation.mutation_x.shift

                    if not mutation.mutation_y.freeze is None:
                        jp.r[1] = mutation.mutation_y.freeze
                    elif mutation.mutation_y.lower_bound == mutation.mutation_y.upper_bound:
                        jp.r[1] = mutation.mutation_y.lower_bound
                    else:
                        jp.r[1] = params[parameter_counter] + mutation.mutation_y.mutation_origin
                        parameter_counter += 1
                    if mutation.mutation_y.shift is not None:
                        jp.r[1] += mutation.mutation_y.shift

                    if not mutation.mutation_z.freeze is None:
                        jp.r[2] = mutation.mutation_z.freeze
                    elif mutation.mutation_z.lower_bound == mutation.mutation_z.upper_bound:
                        jp.r[2] = mutation.mutation_z.lower_bound
                    else:
                        jp.r[2] = params[parameter_counter] + mutation.mutation_z.mutation_origin
                        parameter_counter += 1
                    if mutation.mutation_z.shift is not None:
                        jp.r[2] += mutation.mutation_z.shift
                
                elif type(mutation) is RelativeMutation:
                # elif isinstance(mutation, RelativeMutation):
                    relative_to = mutation.relative_to
                    if isinstance(relative_to, JointPoint):
                        if not mutation.mutation_x.freeze is None:
                            jp.r[0] = relative_to.r[0] + mutation.mutation_x.freeze
                        elif mutation.mutation_x.lower_bound == mutation.mutation_x.upper_bound:
                            jp.r[0] = relative_to.r[0] + mutation.mutation_x.lower_bound
                        else:
                            jp.r[0] = relative_to.r[0] + params[parameter_counter] + mutation.mutation_x.mutation_origin
                            parameter_counter += 1
                        if mutation.mutation_x.shift is not None:
                            jp.r[0] += mutation.mutation_x.shift

                        if not mutation.mutation_y.freeze is None:
                            jp.r[1] = relative_to.r[1] + mutation.mutation_y.freeze
                        elif mutation.mutation_y.lower_bound == mutation.mutation_y.upper_bound:
                            jp.r[1] = relative_to.r[1] + mutation.mutation_y.lower_bound
                        else:
                            jp.r[1] = relative_to.r[1] + params[parameter_counter] + mutation.mutation_y.mutation_origin
                            parameter_counter += 1
                        if mutation.mutation_y.shift is not None:
                            jp.r[1] += mutation.mutation_y.shift

                        if not mutation.mutation_z.freeze is None:
                            jp.r[2] = relative_to.r[2] + mutation.mutation_z.freeze
                        elif mutation.mutation_z.lower_bound == mutation.mutation_z.upper_bound:
                            jp.r[2] = relative_to.r[2] + mutation.mutation_z.lower_bound
                        else:
                            jp.r[2] = relative_to.r[2] + params[parameter_counter] + mutation.mutation_z.mutation_origin
                            parameter_counter += 1
                        if mutation.mutation_z.shift is not None:
                            jp.r[2] += mutation.mutation_z.shift

                    elif isinstance(relative_to, Tuple):
                        first_joint, second_joint = relative_to
                        if not mutation.mutation_x.freeze is None:
                            jp.r[0] = (first_joint.r[0] + second_joint.r[0]) / 2 + mutation.mutation_x.freeze
                        elif mutation.mutation_x.lower_bound == mutation.mutation_x.upper_bound:
                            jp.r[0] = (first_joint.r[0] + second_joint.r[0]) / 2 + mutation.mutation_x.lower_bound
                        else:
                            jp.r[0] = (first_joint.r[0] + second_joint.r[0]) / 2 + params[parameter_counter] + mutation.mutation_x.mutation_origin
                            parameter_counter += 1
                        if mutation.mutation_x.shift is not None:
                            jp.r[0] += mutation.mutation_x.shift

                        if not mutation.mutation_y.freeze is None:
                            jp.r[1] = (first_joint.r[1] + second_joint.r[1]) / 2 + mutation.mutation_y.freeze
                        elif mutation.mutation_y.lower_bound == mutation.mutation_y.upper_bound:
                            jp.r[1] = (first_joint.r[1] + second_joint.r[1]) / 2 + mutation.mutation_y.lower_bound
                        else:
                            jp.r[1] = (first_joint.r[1] + second_joint.r[1]) / 2 + params[parameter_counter] + mutation.mutation_y.mutation_origin
                            parameter_counter += 1
                        if mutation.mutation_y.shift is not None:
                            jp.r[1] += mutation.mutation_y.shift

                        if not mutation.mutation_z.freeze is None:
                            jp.r[2] = (first_joint.r[2] + second_joint.r[2]) / 2 + mutation.mutation_z.freeze
                        elif mutation.mutation_z.lower_bound == mutation.mutation_z.upper_bound:
                            jp.r[2] = (first_joint.r[2] + second_joint.r[2]) / 2 + mutation.mutation_z.lower_bound
                        else:
                            jp.r[2] = (first_joint.r[2] + second_joint.r[2]) / 2 + params[parameter_counter] + mutation.mutation_z.mutation_origin
                            parameter_counter += 1
                        if mutation.mutation_z.shift is not None:
                            jp.r[2] += mutation.mutation_z.shift
                
                elif type(mutation) is RelativePercentageMutation:
                # elif isinstance(mutation, RelativePercentageMutation):
                    first_joint, second_joint =  mutation.relative_to
                    jp.r = (first_joint.r + second_joint.r) / 2
                    link_direction = first_joint.r - second_joint.r
                    link_orthogonal = np.array(
                        [link_direction[2], link_direction[1], -link_direction[0]])
                    if not mutation.mutation_x.freeze is None:
                        jp.r +=  mutation.mutation_x.freeze * link_orthogonal
                    elif mutation.mutation_x.lower_bound == mutation.mutation_x.upper_bound:
                        jp.r +=  mutation.mutation_x.lower_bound * link_orthogonal
                    else:
                        jp.r += params[parameter_counter] * link_orthogonal 
                        parameter_counter += 1
                    if mutation.mutation_x.shift is not None:
                        jp.r[0] += mutation.mutation_x.shift 
                    
                    if mutation.mutation_y.freeze is None:
                        raise ValueError("Relative percentage mutation for y coordinate is not supported.")
                    else:
                        jp.r[1] +=  mutation.mutation_y.freeze
                    if mutation.mutation_y.shift is not None:
                        jp.r[1] += mutation.mutation_y.shift

                    if not mutation.mutation_z.freeze is None:
                        jp.r +=  mutation.mutation_z.freeze * link_direction
                    elif mutation.mutation_z.lower_bound == mutation.mutation_z.upper_bound:
                        jp.r +=  mutation.mutation_z.lower_bound * link_direction
                    else:
                        jp.r += params[parameter_counter] * link_direction 
                        parameter_counter += 1
                    if mutation.mutation_z.shift is not None:
                        jp.r[2] += mutation.mutation_z.shift
                    
        # finally return the graph with all the joint points set
        return self.graph


    def get_central_graph(self):
        params = []
        for key, value in self.current_mutation_ranges.items():
            params.append((value[0] + value[1]) / 2)
        return self.get_graph(params)
    def get_random_graph(self):
        params = []
        for key, value in self.current_mutation_ranges.items():
            params.append(np.random.uniform(value[0], value[1]))
        return self.get_graph(params)
    
    def reset_mutation_range(self, branch_idx, joint_idx, mutation_range):
        self.mutations[branch_idx][joint_idx] = mutation_range
        self.get_mutation_ranges()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # example with manually built scheme
    graph_scheme = {0: [SchemeJoint(name="G0", attach_ground = True,mutation_x=MutationCoordinate(freeze=0.0), mutation_y=MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(freeze=0.0), active=True),
                    SchemeJoint(name="J0",  mutation_x=MutationCoordinate(mutation_origin=0.05, lower_bound=-0.1, upper_bound=0.1), mutation_y=MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(mutation_origin=-0.2, lower_bound=-0.1, upper_bound=0.1)),
                    SchemeEE(name="EE", mutation_x=MutationCoordinate(mutation_origin=0.0, lower_bound=-0.1, upper_bound=0.1), mutation_y=MutationCoordinate(freeze=0.0), mutation_z=MutationCoordinate(mutation_origin=-0.4, lower_bound=-0.1, upper_bound=0.1))]}
    manager = MutableGraphManager(graph_scheme)
    manager.build_graph()
    manager.get_mutation_ranges()
    graph = manager.get_central_graph()
    draw_joint_point(graph)
    plt.show()
