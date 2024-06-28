from box_utils import prepare_boxes_plot, mul_intervals, pow_interval, neg_interval, subtr_intervals, gain_interval, atan2_intervals, atan_intervals
from boxapprox import box_parallel, box_serial
from boxapprox import process_box_1shrink, process_box_1shrink_splitselected, process_box_splitselected, process_box_maxshrink, AsyncBoxerManaged, AsyncBoxerManagedWaves
import modern_robotics as mr
# from typing import Union
import numpy as np
from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopInverseKinematicsProximal
from auto_robot_design.pinokla.robot_utils import freezeJoints, freezeJointsWithoutVis
import pinocchio as pin


def angle_wrap(arr):
    # return (arr + np.pi) % (2 * np.pi) - np.pi
    return ((-arr + np.pi) % (2 * np.pi) - np.pi) * -1

def ForwardKCustom(
    model,
    constraint_model,
    actuation_model,
    q_prec=None,
    max_it=100,
    alpha = 0.7,
    eps=1e-12,
    rho=1e-10,
    mu=1e-4,
    connectivity=1e-12
):
    """
    q=proximalSolver(model,data,constraint_model,constraint_data,max_it=100,eps=1e-12,rho=1e-10,mu=1e-4)

    Build the robot in respect to the constraints using a proximal solver.

    Args:
        model (pinocchio.Model): Pinocchio model.
        data (pinocchio.Data): Pinocchio data.
        constraint_model (list): List of constraint models.
        constraint_data (list): List of constraint data.
        actuation_model (ActuationModelFreeFlyer): Actuation model.
        q_prec (list or np.array, optional): Initial guess for joint positions. Defaults to [].
        max_it (int, optional): Maximum number of iterations. Defaults to 100.
        eps (float, optional): Convergence threshold for primal and dual feasibility. Defaults to 1e-12.
        rho (float, optional): Scaling factor for the identity matrix. Defaults to 1e-10.
        mu (float, optional): Penalty parameter. Defaults to 1e-4.

    Returns:
        np.array: Joint positions of the robot respecting the constraints.

    raw here (L84-126):https://gitlab.inria.fr/jucarpen/pinocchio/-/blob/pinocchio-3x/examples/simulation-closed-kinematic-chains.py
    """

    Lid = actuation_model.idMotJoints
    Lid_q = actuation_model.idqmot

    q_previous = np.delete(q_prec.copy(), Lid_q, axis=0)

    (reduced_model, reduced_constraint_models, reduced_actuation_model) = freezeJointsWithoutVis(
        model, constraint_model, None, Lid, q_prec
    )

    reduced_data = reduced_model.createData()
    reduced_constraint_data = [c.createData() for c in reduced_constraint_models]

    q = np.delete(q_prec, Lid_q, axis=0)
    constraint_dim = 0
    for cm in reduced_constraint_models:
        constraint_dim += cm.size()

    y = np.ones((constraint_dim))
    reduced_data.M = np.eye(reduced_model.nv) * rho
    kkt_constraint = pin.ContactCholeskyDecomposition(
        reduced_model, reduced_constraint_models
    )

    for k in range(max_it):
        pin.computeJointJacobians(reduced_model, reduced_data, q)
        kkt_constraint.compute(
            reduced_model,
            reduced_data,
            reduced_constraint_models,
            reduced_constraint_data,
            mu,
        )

        constraint_value = np.concatenate(
            [
                (pin.log(cd.c1Mc2).np[: cm.size()])
                for (cd, cm) in zip(reduced_constraint_data, reduced_constraint_models)
            ]
        )

        # LJ = []
        # for cm, cd in zip(reduced_constraint_models, reduced_constraint_data):
        #     Jc = pin.getConstraintJacobian(reduced_model, reduced_data, cm, cd)
        #     LJ.append(Jc)
        # J = np.concatenate(LJ)

        primal_feas = np.linalg.norm(constraint_value, np.inf)
        # print(f'pf={primal_feas}, qL2={np.linalg.norm(q-q_previous)}')
        # dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
        if primal_feas < eps:
            # print("Convergence achieved")
            break
        # print("constraint_value:", np.linalg.norm(constraint_value))
        rhs = np.concatenate([-constraint_value - y * mu, np.zeros(reduced_model.nv)])

        dz = kkt_constraint.solve(rhs)
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]


        q = pin.integrate(reduced_model, q, -alpha * dq)
        y -= alpha * (-dy + y)

    # print('mod',constraint_model)
    # # print('dim',constraint_dim)
    # print('redmod',reduced_constraint_models)
    # print('reddata',reduced_constraint_data)
    # print('val',constraint_value)
    q_final = q_prec
    free_q_dict = zip(actuation_model.idqfree, q)
    for index, value in free_q_dict:
        q_final[index] = value
    return q_final, primal_feas


def search_workspace_FK_nojacs(
    model,
    data,
    effector_frame_name: str,
    q_space: np.ndarray,
    actuation_model,
    constraint_models,
    viz=None, is_using_history=True
):
    """Iterate forward kinematics over q_space and try to minimize constrain value.

    Args:
        model (_type_): _description_
        data (_type_): _description_
        effector_frame_name (str): _description_
        base_frame_name (str): _description_
        q_space (np.ndarray): _description_
        actuation_model (_type_): _description_
        constraint_models (_type_): _description_
        viz (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    c = 0    
    q_start = pin.neutral(model)
    
    workspace_xyz = np.empty((len(q_space), 3))
    available_q = np.empty((len(q_space), len(q_start)))

    for q_sample in q_space:
        q_start = angle_wrap(q_start)
        q_sample = angle_wrap(q_sample)

        q_dict_mot = zip(actuation_model.idqmot, q_sample)
        for key, value in q_dict_mot:
            q_start[key] = value
        q3, error = ForwardKCustom(
            model,
            constraint_models,
            actuation_model,
            q_start,
            50,
        )
        q3 = angle_wrap(q3)
        q_start = angle_wrap(q_start)

        if error < 1e-11:
            if viz:
                viz.display(q3)
                # time.sleep(0.005)
            
            if is_using_history:
                q_start = q3

            pin.framesForwardKinematics(model, data, q3)
            id_effector = model.getFrameId(effector_frame_name)
            effector_pos = data.oMf[id_effector].translation
            transformed_pos = effector_pos
            
            # pin.computeJointJacobians(model, data, q3)  # precomputes all jacobians

            workspace_xyz[c] = transformed_pos
            available_q[c] = q3
            c += 1
    return (workspace_xyz[0:c], available_q[0:c])



def search_workspace_IK_nojacs(
    model,
    data,
    effector_frame_name: str,
    ee_space: np.ndarray,
    constraint_models,
    constraint_data,
    viz=None, is_using_history=True
):
    """Iterate forward kinematics over q_space and try to minimize constrain value.

    Args:
        model (_type_): _description_
        data (_type_): _description_
        effector_frame_name (str): _description_
        base_frame_name (str): _description_
        q_space (np.ndarray): _description_
        actuation_model (_type_): _description_
        constraint_models (_type_): _description_
        viz (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    c = 0
    q_start = pin.neutral(model)

    id_effector = model.getFrameId(effector_frame_name)

    jointspace_q = np.empty((len(ee_space), len(q_start)))
    available_xyz = np.empty((len(ee_space), 3))

    for ee_sample in ee_space:
        q, min_feas, is_reach = closedLoopInverseKinematicsProximal(
            model,
            data,
            constraint_models,
            constraint_data,
            ee_sample,
            id_effector,
            onlytranslation=True,
            q_start=q_start, max_it=300,
        )
        q = angle_wrap(q)

        if is_reach:
            if viz:
                viz.display(q)
                # time.sleep(0.005)
            if is_using_history:
                q_start = q

            pin.framesForwardKinematics(model, data, q)
            
            effector_pos = data.oMf[id_effector].translation
            transformed_pos = effector_pos

            # pin.computeJointJacobians(model, data, q) #TODO TRY COMMENT

            available_xyz[c] = transformed_pos
            jointspace_q[c] = q
            c += 1
    return (jointspace_q[0:c], available_xyz[0:c])


def box_workspace(kinematic_graph, sigma: float, 
                  init_safety_factor: float=10.,
                #   x_b: Union[list, tuple, np.ndarray], 
                #   y_b: Union[list, tuple, np.ndarray]
                is_2linker_based = True, return_lorder=False, split_selected=True
                  ):
    """
    Approximates linkage workspace with boxes. Parametrize every movable link with 6 vars: 
    (rx, ry, cos, sin, cos^2, sin^2) and the last 2 vars are x and z of EE.

    Works for mechs with only 1 joint being connected to EE. 
    
    Args:
        kinematic_graph (KinematicGraph): kinematic graph with defined EE, G fields and frames for each link.
        sigma (float): max length of a box along EE coordinates. Set it to 2 (or just below the safety factor) 
        to just shrink the initial box without subdivisions.
        init_safety_factor (float): coefficient for initial bbox that relates range of motion of any link to EE's range. 
        Needed to not miss any configurations, can be set large without any drawbacks (may crop some solutions if too low).
        is_2linker_based (bool): flag to normalize distances with max 2linker's EE distance from (0,0)

    Returns:
        list[np.ndarray]: list of solution boxes, each box contains 6*n_links bounds for intermediate 
        vars and last two bounds for two EE coordinates (x,z). n_links is len(kinematic_graph.nodes())-2 -- excluding G and EE.
    """
    # l_order = ['L6','L5','L3','L4'] #initial
    # l_order = ['L4','L6','L3','L5']
    l_order = [l.name for l in (kinematic_graph.nodes()-{kinematic_graph.EE, kinematic_graph.G})]
    G = kinematic_graph.G.name

    j_order = []
    for j in kinematic_graph.joint_graph.nodes():
        if not kinematic_graph.EE in j.links:
            j_order.append(j.jp.name)
            # print([l.name for l in list(j.links)])
        else:
            jee = j
            ljee = list(j.links - {kinematic_graph.EE})[0]
    # nj = len(j_order)
    # nl = len(l_order)
    j_order.append(jee.jp.name)
    # print(l_order)
    # print(j_order)

    # ljee_ord = 4
    # j_order = ['Main_ground', '2L_ground','Main_knee','2L_knee','2L_bot','Main_ee']
    # #real mechanical values
    nl = len(kinematic_graph.nodes())-2 # -ee -G   #4
    nj = len(kinematic_graph.edges())-1 # -ee      #5

    jname2ord = {n: i for i,n in enumerate(j_order)}
    lname2ord = {n: j+1 for j,n in enumerate(l_order)}
    # print(lname2ord)
    lname2ord[G] = 0

    ljee_ord = lname2ord[ljee.name]

    # # print(len(kinematic_graph))
    # # print(len(main_branch))

    # # for l in main_branch:
    # #     print(l.name)

    links_dict = kinematic_graph.name2link
    # # jp_dict = kinematic_graph.name2jp
    # j_dict = kinematic_graph.name2joint

    # # for n in l_order:
    # #     print(len(links_dict[n].joints))

    # # for n in j_order:
    # #     print(j_dict[n].jp.name)

    # jname2lname = {}
    jord2lord = {}
    lord2jord = {}
    for j,e in kinematic_graph.joint2edge.items():
        # try:
        #     jname2lname[j.jp.name] = [l.name for l in e]
        # except(KeyError):
        #     pass
        try:
            jord2lord[jname2ord[j.jp.name]] = [lname2ord[l.name] for l in e]
            lord2jord[frozenset([lname2ord[l.name] for l in e])] = jname2ord[j.jp.name]
        except(KeyError):
            pass
        # print(j.jp.name,[l.name for l in e])
    
    

    # take max distance from (0,0) to EE of 2linker as a reference length
    if is_2linker_based:
        ee_b = calc_ee_range_of_2linker(kinematic_graph)
        # norm_factor = 1./np.linalg.norm(kinematic_graph.name2joint["Main_ee"].jp.r)
        norm_factor = 1./ee_b[1]
        
    else:
        raise NotImplementedError('The only supported base structure is two-linker, \
                                  which is used to calculate normalization factor. \
                                  Implement new structures or just set factor to 1.')
    
    # Make sure that bounds contain ENTIRE range of locations for all the links' frames.
    # x_b (Union[list, tuple, np.ndarray]): bounds for 1st coord of each link's frame.
    # y_b (Union[list, tuple, np.ndarray]): bounds for 2st coord of each link's frame.
    x_b = ee_b *init_safety_factor
    y_b = ee_b *init_safety_factor

    rxb = np.asarray(x_b) *norm_factor
    ryb = np.asarray(y_b) *norm_factor

    un_b = np.array([-1., 1.])

    # rb_j = np.full((nl+1,2,2),None) #+g-ee
    p_i = np.full((nl+1,nj+1,2),None) #+g-ee, joints +ee, ncoords

    for ind_l, ln in enumerate([G,*l_order]):
        js = list(links_dict[ln].joints)
        Rot,pos = mr.TransToRp(links_dict[ln].frame)
        pos = pos *norm_factor
        for j in js:
            ind = jname2ord[j.jp.name]
            Pi = j.jp.r *norm_factor
            # print(Pi) 
            loc = Rot.T@(Pi-pos)
            loc[abs(loc)<1e-16] = 0.
            p_i[ind_l,ind,:] = loc[(0,2),]
        
    #     rb_j[ind_l,0,:] = rxb
    #     rb_j[ind_l,1,:] = ryb
    # rb_j[0,0,:] = np.zeros(2)
    # rb_j[0,1,:] = np.zeros(2)


    w = 6
    B = np.zeros((nl*w+2,2))
    # B = np.full((len(l_order)*w+2,2), None)

    for j, ln in enumerate(l_order):
        B[j*w,:] = rxb
        B[j*w+1,:] = ryb
        B[j*w+2,:] = un_b
        B[j*w+3,:] = un_b
        B[j*w+4,:] = pow_interval(un_b,2)
        B[j*w+5,:] = pow_interval(un_b,2)

    B[(-2),:] = rxb
    B[(-1),:] = ryb

    #links which frames do not translate
    gr_links = {lname2ord[l.name]: p_i[0, lord2jord[frozenset((0,lname2ord[l.name]))], :] 
                for l in kinematic_graph.neighbors(kinematic_graph.G) 
                if not len(p_i[lname2ord[l.name], lord2jord[frozenset((0,lname2ord[l.name]))], :].nonzero()[0])}
    # print(gr_links)
    # gr_links = {3: (0.,0.), 1: (-.1,0.)}
    for j,(x,y) in gr_links.items():
        B[(j-1)*w,:] = (x,x)
        B[(j-1)*w+1,:] = (y,y)

    # print(B.T)

    nvar_1 = B.shape[0]

    A_eq = np.zeros((nj*2+nl+2,nvar_1))
    b_eq = np.zeros(nj*2+nl+2)

    for i in range(nj):
        j1,j2 = jord2lord[i] # 2 links' ids connected by i-th joint
        if j1 > 0: # if not ground, hence can move
            # x
            A_eq[2*i,w*(j1-1)] = 1 #rx
            A_eq[2*i,w*(j1-1)+2] = p_i[j1,i,0] # cos
            A_eq[2*i,w*(j1-1)+3] = -p_i[j1,i,1] # sin
            # y
            A_eq[2*i+1,w*(j1-1)+1] = 1 #ry
            A_eq[2*i+1,w*(j1-1)+2] = p_i[j1,i,1] # cos
            A_eq[2*i+1,w*(j1-1)+3] = p_i[j1,i,0] # sin
        else:
            b_eq[(2*i,2*i+1),] = -p_i[j1,i,:]

        if j2 > 0: # if not ground, hence can move
            # x
            A_eq[2*i,w*(j2-1)] = -1 #rx
            A_eq[2*i,w*(j2-1)+2] = -p_i[j2,i,0] # cos
            A_eq[2*i,w*(j2-1)+3] = p_i[j2,i,1] # sin
            # y
            A_eq[2*i+1,w*(j2-1)+1] = -1 #ry
            A_eq[2*i+1,w*(j2-1)+2] = -p_i[j2,i,1] # cos
            A_eq[2*i+1,w*(j2-1)+3] = -p_i[j2,i,0] # sin
        else:
            b_eq[(2*i,2*i+1),] = p_i[j2,i,:]

    sqr_pairs = []

    # quadratic equations
    for j in range(nl):
        A_eq[nj*2+j,w*j+4] = 1
        A_eq[nj*2+j,w*j+5] = 1
        b_eq[nj*2+j] = 1
        sqr_pairs.append((w*j+2,w*j+4))
        sqr_pairs.append((w*j+3,w*j+5))

    A_eq[-2,w*(ljee_ord-1):w*(ljee_ord-1)+4] = (1,0,p_i[ljee_ord,-1,0],-p_i[ljee_ord,-1,1])
    A_eq[-2,-2] = -1
    A_eq[-1,w*(ljee_ord-1):w*(ljee_ord-1)+4] = (0,1,p_i[ljee_ord,-1,1],p_i[ljee_ord,-1,0])
    A_eq[-1,-1] = -1

    # print(A_eq)
    # print(b_eq)
    # print(sqr_pairs)

    # cross_triplets = []

    # threshold_s = {-2: 0.9/15, -1: 0.5/15}
    threshold_s = {-2: sigma, -1: sigma}
    P=[B]
    # sols = box_serial(P, A_eq, b_eq, sqr_pairs, threshold_s=threshold_s)
    sols = box_parallel(P, A_eq, b_eq, sqr_pairs, threshold_s=threshold_s, 
                        process_func=process_box_1shrink_splitselected if split_selected else process_box_1shrink)
    # ab = AsyncBoxerManagedWaves()
    # sols = ab.box_parallel_async(P, A_eq, b_eq, sqr_pairs, threshold_s=threshold_s, process_func=process_box_1shrink)

    # denormalization
    for k in range(len(sols)):
        for j in range(nl+1): # rx ry for all links and +1 for eex eey
            sols[k][w*j,:] = sols[k][w*j,:] /norm_factor
            sols[k][w*j+1,:] = sols[k][w*j+1,:] /norm_factor
    if return_lorder:
        return sols, {n: j for j,n in enumerate(l_order)}
    return sols

def calc_ee_range_of_2linker(kinematic_graph):
    l1 = np.linalg.norm(kinematic_graph.name2joint["Main_knee"].jp.r)
    l2 = np.linalg.norm(kinematic_graph.name2joint["Main_ee"].jp.r - kinematic_graph.name2joint["Main_knee"].jp.r)

    rng = np.asarray([-l1-l2, l1+l2])
    return rng

def filter_points2d_by_boxes(boxes, x_ind, y_ind, points):
    boxes2d = np.hstack(boxes)
    boxes2d = boxes2d[(x_ind,y_ind),:]

    inside_rectangles = np.zeros(points.shape[0], dtype=bool)
    # print(inside_rectangles)
    for i in range(boxes2d.shape[1]//2):
        rect= boxes2d[:,(2*i,2*i+1)]
        inside_rectangles |= (points[:, 0] >= rect[0,0]) & (points[:, 0] <= rect[0,1]) & (points[:, 1] >= rect[1,0]) & (points[:, 1] <= rect[1,1])
    return points[inside_rectangles]

