import matplotlib.pyplot as plt
from pyparsing import List, Tuple

import networkx as nx

from description.kinematics import JointPoint
import numpy as np
    
def add_branch(G: nx.Graph, branch: List[JointPoint] | List[List[JointPoint]]):
    is_list  = [isinstance(br, List) for br in branch]
    if all(is_list):
        for b in branch:
            add_branch(G, b)
    else:
        for i in range(len(branch)-1):
            if isinstance(branch[i], List):
                for b in branch[i]:
                    G.add_edge(b, branch[i+1])
            elif isinstance(branch[i+1], List):
                for b in branch[i+1]:
                    G.add_edge(branch[i], b)
            else:
                G.add_edge(branch[i], branch[i+1])

def add_branch_with_attrib(G: nx.Graph, branch: List[Tuple(JointPoint, dict)] | List[List[Tuple[JointPoint,dict]]]):
    is_list  = [isinstance(br, List) for br in branch]
    if all(is_list):
        for b in branch:
            add_branch_with_attrib(G, b)
    else:
        for ed in branch:
                G.add_edge(ed[0], ed[1], **ed[2])
