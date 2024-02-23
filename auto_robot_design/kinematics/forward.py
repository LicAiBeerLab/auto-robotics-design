import numpy as np
import numpy.linalg as la


from scipy.spatial.transform import rotation as R

import modern_robotics as mr

import networkx as nx

from auto_robot_design.description.kinematics import JointPoint


def forward_kinematics(G: nx.Graph):
    pass
    
    
if __name__=="__main__":
    cassie_graph = nx.Graph()
    # https://cad.onshape.com/documents/52eb11422c701d811548a6f5/w/655758bb668dff773a0e7c1a/e/77ff7f84e82d8fb31fe9c30b
    # abs_ground = np.array([0.065, 0, -0.015])
    abs_ground = np.array([0.065, 0, -0.047])
    pos_toeA_joint = np.array([0.065, 0, -0.047]) - abs_ground
    pos_toeA_tarus_joint = np.array([-0.273, 0, -0.350]) - abs_ground
    pos_shin_joint = np.array([0.065, 0, -0.047]) - abs_ground
    pos_knee_spring = np.array([0.011, 0, -0.219]) - abs_ground
    pos_tarus_joint = np.array([-0.237, 0, -0.464]) - abs_ground
    pos_foot_joint = np.array([-0.080, 0, -0.753]) - abs_ground
    pos_molet_joint = np.array([-0.207, 0, -0.552]) - abs_ground
    pos_toeB_joint = np.array([-0.257, 0, -0.579]) - abs_ground
    pos_toeB_foot_joint = np.array([-0.118, 0, -0.776]) - abs_ground
    
    ground_joint = JointPoint(attach_ground=True, active=True)
    shin_joint = JointPoint(pos=pos_shin_joint, active=True)
    knee_spring = JointPoint(pos_knee_spring, weld=True)
    tarus_joint = JointPoint(pos=pos_tarus_joint)
    foot_joint = JointPoint(pos=pos_foot_joint)
    
    toeA_joint = JointPoint(pos=pos_toeA_joint)
    connect_toeA_tarus_joint = JointPoint(pos=pos_toeA_tarus_joint, weld=True)
    
    molet_joint = JointPoint(pos=pos_molet_joint, active=True)
    toeB_joint = JointPoint(pos=pos_toeB_joint)
    toeB_foot_joint = JointPoint(pos=pos_toeB_foot_joint)
    
    joints = [ground_joint, shin_joint, knee_spring, tarus_joint, foot_joint, toeA_joint, connect_toeA_tarus_joint,
              molet_joint, toeB_joint, toeB_foot_joint]
    
    for j in joints:
        cassie_graph.add_node(j)
    main_branch = [ground_joint, shin_joint, knee_spring, tarus_joint, foot_joint]
    add_branch_1 = [[ground_joint,shin_joint], toeA_joint, connect_toeA_tarus_joint, [tarus_joint,foot_joint]]
    add_branch_2 = [[tarus_joint,foot_joint], molet_joint, toeB_joint, toeB_foot_joint, foot_joint]
    for j0 in main_branch:
        j1 = next(main_branch)
        # print(j0,j1)
        # cassie_graph.add_edge()