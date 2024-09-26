
import dill
import pathlib
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pinocchio as pin

from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.description.builder import jps_graph2pinocchio_robot_3d_constraints, MIT_CHEETAH_PARAMS_DICT, ParametrizedBuilder, URDFLinkCreater3DConstraints
from auto_robot_design.motion_planning.utils import Workspace
from auto_robot_design.motion_planning.bfs_ws import BreadthFirstSearchPlanner



class DatasetGenerator:
    FIELD_NAMES = ['joint_positions', 'initial_position_ws', 'resolution_ws', 'grid_size', 'COUNTRY']
    WORKSPACE_ARGS_NAMES = ["bounds", "resolution", "dexterous_tolerance"]
    def __init__(self, graph_manager, path, workspace_args):
        
        self.ws_args = workspace_args
        self.graph_manager = graph_manager
        self.path = pathlib.Path(path)
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
        draw_joint_point(self.graph_manager.get_graph(self.graph_manager.generate_central_from_mutation_range()))
        plt.savefig(self.path / "graph.png")
        with open(self.path / "graph.pkl", "wb") as file:
            dill.dump(self.graph_manager, file)
        
        wrt_lines = []
        for nm, vls in zip(self.WORKSPACE_ARGS_NAMES, self.ws_args):
            wrt_lines.append(nm +": " + str(np.round(vls, 3)) + '\n')
        
        with open(self.path / "info.txt", "w") as file:
                file.writelines(wrt_lines)
                

        self.builder = ParametrizedBuilder(URDFLinkCreater3DConstraints)

    def _find_workspace(self, joint_positions: np.ndarray):
        
        graph = self.graph_manager.get_graph(joint_positions)
        robot, __ = jps_graph2pinocchio_robot_3d_constraints(graph, self.builder)
        ws = Workspace(robot, *self.ws_args[:-1])
        ws_search =  BreadthFirstSearchPlanner(ws, 0, self.ws_args[-1])
        
        pin.framesForwardKinematics(robot.model, robot.data, )
        
        workspace = ws_bfs.find_workspace(start_pos, q)
        
        
        
    def _calculate_batch(self):
        pass


#     def 
    

# # List of column names

# # Dictionary that you want to add as a new row
# new_row = {'ID': 6, 'NAME': 'William', 'RANK': 5532, 'ARTICLE': 1, 'COUNTRY': 'UAE'}

# # Open the CSV file in append mode
# with open('your_file.csv', 'a', newline='') as f_object:
#     # Pass the file object and a list of column names to DictWriter()
#     dict_writer_object = csv.DictWriter(f_object, fieldnames=field_names)
#     # If the file is empty or you are adding the first row, write the header
#     if f_object.tell() == 0:
#         dict_writer_object.writeheader()
#     # Pass the dictionary as an argument to the writerow() function
#     dict_writer_object.writerow(new_row)

if __name__=="__main__":
    names = ["raz", "dwa"]
    values = [np.array([0.04, 0.02, 5.46]), np.array([0.04, 0.02, 5.46])]
    lines = []
    for name, val in zip(names, values):
        lines.append(name +": " + str(np.round(val, 3)) + '\n')
    with open( "info.txt", "w") as f:
            f.writelines(lines)