import networkx as nx
import numpy as np

from auto_robot_design.description.builder import add_branch
from auto_robot_design.description.kinematics import JointPoint


def calc_effective_inertia(M, dqdmot, Jcl):
    Mmot = dqdmot.T @ M @ dqdmot
    Lambda = np.linalg.pinv(Jcl[[0,2],:] @ np.linalg.inv(Mmot) @ Jcl[[0,2],:].T)
    return Lambda

def calc_effective_inertia_simpler(Mmot, Jcl):
    Lambda = np.linalg.pinv(Jcl[[0,2],:] @ np.linalg.inv(Mmot) @ Jcl[[0,2],:].T)
    return Lambda

def calc_zrr(Jcl):
    Jcl_translational = Jcl[:3,:]
    z_axis = np.array([0,0,1])
    return np.linalg.norm( abs(Jcl_translational.T @ z_axis) )

def calc_Mmot_inertia(M, dqdmot):
    Mmot = dqdmot.T @ M @ dqdmot
    return Mmot

def is_ground_id(id):
    return id == 'G' or id == 0

def get_ground_group(group_dict):
    try: 
        gr = group_dict['G']
    except KeyError:
        gr = group_dict[0]
    return gr


class MntHandler:
    def __init__(self, link_mounts):
        G_name = None
        for name,mnt in link_mounts.items():
            if is_ground_id(mnt[2]):
                G_name = mnt[2]
                break
        assert G_name is not None
        self.ground_name = G_name
        self.visited_ids = []
        self.welded_bar_groups = {self.ground_name: self.ground_name}
        self.mounts = link_mounts

        self._convert_mount_coords()

        for mnt_id, mnt in self.mounts.items():
            self.find_welded_recursive(mnt_id, mnt)

    def find_welded_recursive(self, mnt_id, mnt):
        p1 = mnt[0]
        p2 = mnt[1]
        parent_id = mnt[2]

        j_buf = mnt[3]

        if mnt_id in self.visited_ids:
            return
        if is_ground_id(parent_id):
            bar_group = self.ground_name
        else:
            self.find_welded_recursive(parent_id, self.mounts[parent_id])
            bar_group = self.welded_bar_groups[parent_id]

        if j_buf is not None:
            # TODO check entire list
            bar_group = mnt_id  # TODO check how bargroups work with names
        self.welded_bar_groups[mnt_id] = bar_group
        self.visited_ids.append(mnt_id)
        # self.bar_solids[mnt_id] = s

    @staticmethod
    def _is_p_local(p):
        return len(p) > 2

    def loc2glob(self, p, parent_id):
        # TODO add account for roll, mb move conversion after solids
        if is_ground_id(parent_id):
            return p
        p_start = self.mounts[parent_id][0]
        p_end = self.mounts[parent_id][1]
        if self._is_p_local(p_start):
            raise NotImplementedError(f'{parent_id} has local p1, unsupported sequence')
        if self._is_p_local(p_end):
            raise NotImplementedError(f'{parent_id} has local p2, unsupported sequence')
        vec = np.asarray(p_end) - np.asarray(p_start)

        is_body_coords = p[2]

        # local coords, units same as global
        length = np.linalg.norm(vec)
        unit_vec_x = vec / length
        ref_z = np.asarray([0, 0, 1])
        unit_vec_y = -np.cross(unit_vec_x, ref_z)
        unit_vec_y = unit_vec_y / np.linalg.norm(unit_vec_y)

        scale = 1.
        if is_body_coords: # body coords, 1 equals the bar length
            scale = length

        rotated = (np.multiply(p[0], unit_vec_x[:2]*scale) +
                   np.multiply(p[1], unit_vec_y[:2]*scale))
        return np.add(rotated, p_start)  # .tolist()
    
    def _convert_mount_coords(self):
        # convert all local or body coords to global
        #TODO it does not work when parent also has non-global coords, fix that
        for key, m in self.mounts.items():
            for i in range(2):
                if self._is_p_local(m[i]):
                    self.mounts[key][i] = self.loc2glob(m[i], m[2])

def build_jp(link_mounts, equalities, trackpoints):
    # constrain_dict = {}  # should be updated after creating each joint
    current_main_branch = []
    graph = nx.Graph()

    is_open_chain = False if len(equalities) else True

    jaxis = np.array([0, 1, 0])

    mnth = MntHandler(link_mounts)
    weld_groups = mnth.welded_bar_groups

    # print(weld_groups)
    
    seen_groups = {}
    for name, group in weld_groups.items():
        if group not in seen_groups:
            seen_groups[group] = []
        seen_groups[group].append(name)
    list_groups = list(seen_groups.values())

    # print('seen_groups',seen_groups)


    # # remove parenting to welded links
    # for i in range(len(link_mounts)):
    #     for k,v in link_mounts.items():
    #         parent_id = v[2]
            
    #         if not is_ground_id(parent_id):
    #             grandparent_id = link_mounts[parent_id][2]
    #             parent_joint = link_mounts[parent_id][3]
    #             if parent_joint is None: # if parent is welded to grandparent
    #                 link_mounts[k][2] = grandparent_id
    #                 link_mounts[k][0] = link_mounts[parent_id][0]  # move starting point of a bar to 

    jp_dict = {}
    jp_groups = {} # only p1 joints corresponding to its link's group
    jp_groups_extended = {} # jp can be a part of several groups
    for group_name,l_group in seen_groups.items():
        jp_groups_extended[group_name] = []

    # add starting joints from welded link groups and make edges to represent links.
    # Only 1 valid joint (p1) from each unwelded bar is used - to remove unnecessary corners.
    for group_name,l_group in seen_groups.items():
        # print(group_name)
        jp_group = []
        parent_group_names = []
        for link_name in l_group: 
            if is_ground_id(link_name): # ignore ground since it has no unique joints
                continue
            is_jp_active = link_mounts[link_name][3]
            if is_jp_active is None:  # ignore any points from welded bars
                continue
            j_pos = link_mounts[link_name][0]
            assert len(j_pos) == 2
            jp_pos = np.array([j_pos[0], 0, j_pos[1]])
            
            link_parent_name = link_mounts[link_name][2]
            jp_name = str(link_name)#[0] #str(link_name)+'-'+str(link_parent_name)

            # neigh_to_g_group = 
            attach_ground = link_parent_name in get_ground_group(seen_groups)

            current_jp = JointPoint(r=jp_pos, w=jaxis, name=jp_name,
                                    attach_ground=attach_ground, active=is_jp_active)
            
            parent_group_names.append(weld_groups[link_parent_name])
            jp_dict[link_name] = current_jp
            jp_group.append(current_jp)
        graph.add_nodes_from(jp_group) # it is expected to see 1 jp in each jp_group
        jp_groups[group_name] = jp_group
        jp_groups_extended[group_name] = jp_group
        # assert parent_group_names[:-1] == parent_group_names[1:] # check if all jps have the same parent group, which is due
        for parent_group, jp in zip(parent_group_names, jp_group):
            jp_groups_extended[parent_group].append(jp) #loop is here in case of processing welded points in the future

        #TODO move this after filling up jp_groups_extended and adding ee to it
        # # connect all joints in the group (same "link") if it isn't ground
        # if not is_ground_id(group_name):
        #     for i in range(len(jp_group)):
        #         for j in range(i + 1, len(jp_group)):
        #             graph.add_edge(jp_group[i], jp_group[j])

    # print('jp.group')
    # for k,v in jp_groups.items():
    #     print(f'{k}:')
    #     for j in v:
    #         print(f'{j.name} {j.r} a={j.active}, g={j.attach_ground}, ee={j.attach_endeffector}')


    # # from trackpoints (only if not coinciding with another joint)
    # tp = trackpoints[0] # only the first is used
    # jpos = tp[0]
    # link_name = tp[1]
    # assert len(jpos) == 2
    # ee_pos = np.array([jpos[0], 0, jpos[1]])

    def check_overlap_jp(graph, pos3d, atol=1e-5):
        for jp in graph.nodes():
            if np.allclose(jp.r, pos3d, atol=atol):
                return jp
        return None

    # from equalities
    i_eq = 0
    for eq in equalities:
        link1_name = eq[1]
        link2_name = eq[2]
        jpos = eq[0]
        assert len(jpos) == 2
        jp_pos = np.array([jpos[0], 0, jpos[1]])

        overlapper = check_overlap_jp(graph, jp_pos)
        if overlapper is None:
            attach_ground = is_ground_id(link1_name) or is_ground_id(link2_name)
            # attach_endeffector = np.allclose(jp_pos, ee_pos, atol=1e-5)
            attach_endeffector = False
            current_jp = JointPoint(r=jp_pos, w=jaxis, name=f'eq{i_eq}',#str(link1_name)+'-'+str(link2_name),
                                    attach_ground=attach_ground, attach_endeffector=attach_endeffector)
            graph.add_node(current_jp)
            i_eq+=1
        else:
            current_jp = overlapper
        graph.add_edge(current_jp, jp_dict[link1_name])
        graph.add_edge(current_jp, jp_dict[link2_name])
        jp_groups_extended[weld_groups[link1_name]].append(current_jp)
        jp_groups_extended[weld_groups[link2_name]].append(current_jp)


    # from trackpoints (only if not coinciding with another joint)
    tp = trackpoints[0] # only the first is used
    jpos = tp[0]
    link_name = tp[1]
    assert len(jpos) == 2
    ee_pos = np.array([jpos[0], 0, jpos[1]])#-0.01])

    jp_ee = JointPoint(r=ee_pos, w=jaxis, name='jEE', attach_endeffector=True)
    # ee_welded_jps = jp_groups[weld_groups[link_name]] #TODO mb add eejp to this group as well
    ee_welded_jps = jp_groups_extended[weld_groups[link_name]] #TODO mb add eejp to this group as well
    add_branch(graph, [jp_ee, ee_welded_jps])

    # ee_candidate = check_overlap_jp(graph, jp_pos, atol=1e-5)
    # if ee_candidate is None:
    #     jp_ee = JointPoint(r=jp_pos, w=jaxis, name='jEE', attach_endeffector=True)
    #     # graph.add_node(jp_ee)
    #     # graph.add_edge(jp_ee, jp_dict[link_name])
    #     ee_welded_jps = jp_groups[weld_groups[link_name]] #TODO mb add eejp to this group as well
    #     add_branch(graph, [jp_ee, ee_welded_jps])
    # else:
    #     print('candidate ',ee_candidate.name)
    #     # ee_c = graph.nodes[ee_candidate]
    #     ee_c = None
    #     for n in graph.nodes():
    #         if n.name == ee_candidate.name:
    #             ee_c = n
    #             break
    #     print(ee_c)
    #     print(graph.has_node(ee_c))
    #     # nx.set_node_attributes(graph, {ee_candidate.name:True}, 'attach_endeffector')
    #     # print(ee_candidate.attach_endeffector)
    #     ee_c.attach_endeffector = True

    #     # print(graph.nodes[ee_candidate])
    #     # graph.nodes[ee_candidate]['attach_endeffector'] = True
    #     print(graph.has_node(ee_c))
    #     # ee_candidate.name = 'jEE'
    #     # pass

    # connect different weld groups between each other
    for link_name,v in link_mounts.items():
        is_jp_active = v[3]
        if is_jp_active is None:
            continue
        link_parent_name = v[2]
        if is_ground_id(weld_groups[link_parent_name]):
            continue
        graph.add_edge(jp_dict[link_name], jp_dict[weld_groups[link_parent_name]])

    # ee_candidate = check_overlap_jp(graph, ee_pos, atol=1e-5) 
    # if ee_candidate is not None:
    #     ee_candidate.attach_endeffector = jp_ee.attach_endeffector
    #     graph = nx.contracted_nodes(graph, ee_candidate, jp_ee, self_loops=False)



    # print('jp_group_ext')
    # for k,v in jp_groups_extended.items():
    #     print(f'{k}:')
    #     for j in v:
    #         print(f'{j.name} {j.r} a={j.active}, g={j.attach_ground}, ee={j.attach_endeffector}')
    



    # paths = [(k,v[2]) for k,v in link_mounts.items()]
    # helper_graph_kin = nx.Graph()
    # for path in paths:
    #     nx.add_path(helper_graph_kin, path)

    # welded_lpairs = []

    # # jpmount_dict = {}
    # for k,v in link_mounts.items():
    #     link_name = k
    #     link_parent_name = v[2]
    #     jpos = v[0]
    #     assert len(jpos) == 2
    #     jp_pos = np.array([jpos[0], 0, jpos[1]])
    #     is_jp_active = v[3]

    #     jp_name = str(link_name)+'-'+str(link_parent_name)
    #     if is_jp_active is None:
    #         welded_lpairs.append((link_name, link_parent_name))
    #     attach_ground = is_ground_id(link_parent_name)
    #     current_jp = JointPoint(r=jp_pos, w=jaxis, name=jp_name,
    #                             attach_ground=attach_ground, active=is_jp_active)
    #     # jpmount_dict[k] = current_jp
    #     helper_graph_kin.add_edge(link_name, link_parent_name, jp=current_jp)
    #     print(current_jp)

    # # from equalities
    # for eq in equalities:
    #     link1_name = eq[1]
    #     link2_name = eq[2]
    #     jpos = eq[0]
    #     assert len(jpos) == 2
    #     jp_pos = np.array([jpos[0], 0, jpos[1]])
    #     attach_ground = is_ground_id(link1_name) or is_ground_id(link2_name)
    #     current_jp = JointPoint(r=jp_pos, w=jaxis, name=str(link1_name)+'-'+str(link2_name),
    #                             attach_ground=attach_ground)
    #     helper_graph_kin.add_edge(link1_name, link2_name, jp=current_jp)

    # # from trackpoints
    # tp = trackpoints[0] # only the first is used
    # jpos = tp[0]
    # link_name = tp[1]
    # assert len(jpos) == 2
    # jp_pos = np.array([jpos[0], 0, jpos[1]])
    # jp_ee = JointPoint(r=jp_pos, w=jaxis, name='jEE', attach_endeffector=True)
    # helper_graph_kin.add_edge('jpEE', link_name, jp=jp_ee)

    # # for j in helper_graph_kin.nodes():
    # #     print(j)
    # for names_to_merge in list_groups:
    #     names_sorted = sorted(names_to_merge)
    #     for name in names_sorted[1:]:
    #         helper_graph_kin = nx.contracted_nodes(helper_graph_kin, names_sorted[0], name, self_loops=False)
    # for j in helper_graph_kin.nodes():
    #     print(j)
    # for u,v in helper_graph_kin.edges():
    #     print(u,v)

    # # mark all neighbour jps of G link as grounded

    # #delete ground link

    # # for link1,link2 in welded_lpairs:
    # #     helper_graph_kin.remove_edge(link1,link2)
    # #     # helper_graph_kin.remove_node(link1)
    
    # # jp_candidates = helper_graph_kin.edges()
    # helper_graph_jp = nx.line_graph(helper_graph_kin)
    # for j in helper_graph_jp.nodes():
    #     print(j)
    # for u,v in helper_graph_jp.edges():
    #     print(u,v)
    # helper_graph_jp.add_nodes_from((node, helper_graph_kin.edges[node]) for node in helper_graph_jp) #copy jp attribute

    


    # graph_jp = nx.Graph()
    # # Create a mapping from original nodes to new nodes based on the 'jp' attribute
    # for node in helper_graph_jp.nodes(data=True):
    #     jp_value = node[1]['jp']
    #     graph_jp.add_node(jp_value)
    # # Add edges based on the original graph's structure but using 'jp' values
    # for u, v in helper_graph_jp.edges():
    #     jp_u = helper_graph_jp.nodes[u]['jp']
    #     jp_v = helper_graph_jp.nodes[v]['jp']
    #     graph_jp.add_edge(jp_u, jp_v)


    return graph, is_open_chain

def build_jp_old(link_mounts, equalities, trackpoints):
    # constrain_dict = {}  # should be updated after creating each joint
    current_main_branch = []
    graph = nx.Graph()

    jaxis = np.array([0, 1, 0])

    mnth = MntHandler(link_mounts)
    weld_groups = mnth.welded_bar_groups
    
    seen_groups = {}
    for name, group in weld_groups.items():
        if group not in seen_groups:
            seen_groups[group] = []
        seen_groups[group].append(name)
    list_groups = list(seen_groups.values())


    # # remove parenting to welded links
    # for i in range(len(link_mounts)):
    #     for k,v in link_mounts.items():
    #         parent_id = v[2]
            
    #         if not is_ground_id(parent_id):
    #             grandparent_id = link_mounts[parent_id][2]
    #             parent_joint = link_mounts[parent_id][3]
    #             if parent_joint is None: # if parent is welded to grandparent
    #                 link_mounts[k][2] = grandparent_id
    #                 link_mounts[k][0] = link_mounts[parent_id][0]  # move starting point of a bar to 

    paths = [(k,v[2]) for k,v in link_mounts.items()]
    helper_graph_kin = nx.Graph()
    for path in paths:
        nx.add_path(helper_graph_kin, path)

    welded_lpairs = []

    # jpmount_dict = {}
    for k,v in link_mounts.items():
        link_name = k
        link_parent_name = v[2]
        jpos = v[0]
        assert len(jpos) == 2
        jp_pos = np.array([jpos[0], 0, jpos[1]])
        is_jp_active = v[3]

        jp_name = str(link_name)+'-'+str(link_parent_name)
        if is_jp_active is None:
            welded_lpairs.append((link_name, link_parent_name))
        attach_ground = is_ground_id(link_parent_name)
        current_jp = JointPoint(r=jp_pos, w=jaxis, name=jp_name,
                                attach_ground=attach_ground, active=is_jp_active)
        # jpmount_dict[k] = current_jp
        helper_graph_kin.add_edge(link_name, link_parent_name, jp=current_jp)
        print(current_jp)

    # from equalities
    for eq in equalities:
        link1_name = eq[1]
        link2_name = eq[2]
        jpos = eq[0]
        assert len(jpos) == 2
        jp_pos = np.array([jpos[0], 0, jpos[1]])
        attach_ground = is_ground_id(link1_name) or is_ground_id(link2_name)
        current_jp = JointPoint(r=jp_pos, w=jaxis, name=str(link1_name)+'-'+str(link2_name),
                                attach_ground=attach_ground)
        helper_graph_kin.add_edge(link1_name, link2_name, jp=current_jp)

    # from trackpoints
    tp = trackpoints[0] # only the first is used
    jpos = tp[0]
    link_name = tp[1]
    assert len(jpos) == 2
    jp_pos = np.array([jpos[0], 0, jpos[1]])
    jp_ee = JointPoint(r=jp_pos, w=jaxis, name='jEE', attach_endeffector=True)
    helper_graph_kin.add_edge('jpEE', link_name, jp=jp_ee)

    # for j in helper_graph_kin.nodes():
    #     print(j)
    for names_to_merge in list_groups:
        names_sorted = sorted(names_to_merge)
        for name in names_sorted[1:]:
            helper_graph_kin = nx.contracted_nodes(helper_graph_kin, names_sorted[0], name, self_loops=False)
    for j in helper_graph_kin.nodes():
        print(j)
    for u,v in helper_graph_kin.edges():
        print(u,v)

    # mark all neighbour jps of G link as grounded

    #delete ground link

    # for link1,link2 in welded_lpairs:
    #     helper_graph_kin.remove_edge(link1,link2)
    #     # helper_graph_kin.remove_node(link1)
    
    # jp_candidates = helper_graph_kin.edges()
    helper_graph_jp = nx.line_graph(helper_graph_kin)
    for j in helper_graph_jp.nodes():
        print(j)
    for u,v in helper_graph_jp.edges():
        print(u,v)
    helper_graph_jp.add_nodes_from((node, helper_graph_kin.edges[node]) for node in helper_graph_jp) #copy jp attribute

    


    graph_jp = nx.Graph()
    # Create a mapping from original nodes to new nodes based on the 'jp' attribute
    for node in helper_graph_jp.nodes(data=True):
        jp_value = node[1]['jp']
        graph_jp.add_node(jp_value)
    # Add edges based on the original graph's structure but using 'jp' values
    for u, v in helper_graph_jp.edges():
        jp_u = helper_graph_jp.nodes[u]['jp']
        jp_v = helper_graph_jp.nodes[v]['jp']
        graph_jp.add_edge(jp_u, jp_v)

    # ee_overlappers = []
    # for j in graph_jp.adj[jp_ee]:
    #     if (j.r == jp_ee.r).all():
    #         print('ASKLJDFLKASFHJ:ADSHLHKJAS')
    #         ee_overlappers.append(j)

    # overlapping_pairs = []
    # for edge in graph_jp.edges():
    #     if (edge[0].r == edge[1].r).all():
    #         print(f'0 length edge detected: {edge[0].name} - {edge[1].name}')
    #         if edge[1] == jp_ee: # so that ee is preserved after the merge
    #             overlapping_pairs.append((edge[1],edge[0]))
    #             continue
    #         # jp_u = edge[0].copy()
    #         # jp_v = edge[1].copy()
    #         if edge[1].attach_ground: # inherit ground to remaining node
    #             edge[0].attach_ground = True
    #         if edge[1].active: # inherit active to remaining node
    #             edge[0].active = True
                
    #         overlapping_pairs.append((edge[0], edge[1]))
    # for edge in overlapping_pairs:
    #     # graph_jp = nx.contracted_edge(graph_jp, edge, self_loops=False)
    #     graph_jp = nx.contracted_nodes(graph_jp, edge[0], edge[1], self_loops=False)


        
    # add_branch(graph, current_jp_pair)


    # ground_joint = JointPoint(
    #     r=np.zeros(3),
    #     w=jaxis,
    #     attach_ground=True,
    #     active=True,
    #     name="Main_ground"
    # )

    # paths = [(k,v[2]) for k,v in link_mounts.items()]

    # # # graph_dict = {"TL_ground": ground_joint}
    # # self.constrain_dict[ground_joint.name] = {'optim': False,
    #                                             # 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}
    
    # knee_joint_pos = np.array([right_shift, 0, knee_pos])

    # knee_joint = JointPoint(
    #     r=knee_joint_pos, w=jaxis, name="Main_knee")
    # # self.constrain_dict[knee_joint.name] = {
    #     # 'optim': True, 'x_range': (-0.1, 0.1), 'z_range': (-0.1, 0.1)}
    
    # ee = JointPoint(
    #     r=np.array([0, 0, -nominal_length]),
    #     w=jaxis,
    #     attach_endeffector=True,
    #     name="Main_ee"
    # )

    # current_main_branch.append(ground_joint)
    # current_main_branch.append(knee_joint)
    # current_main_branch.append(ee)
    # # self.constrain_dict[ee.name] = {
    # #     'optim': False, 'x_range': (-0.2, 0.2), 'z_range': (-0.2, 0.2)}

    # add_branch(graph, current_main_branch)
    return graph_jp, helper_graph_jp,  helper_graph_kin, jp_ee