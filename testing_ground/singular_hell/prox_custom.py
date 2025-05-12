from itertools import product
import time

import numpy as np
import pinocchio as pin

from auto_robot_design.pinokla.closed_loop_jacobian import jacobian_constraint

def select_max_from_repeaters(colors, repeaters):
    #вернет индексы с наиб значением colors
    group_indices = []
    current_idx = 0
    for rep in repeaters:
        group_indices.append((current_idx, current_idx + rep))
        current_idx += rep

    # Для каждой группы находим индекс точки с максимальным colors
    result = []
    for start, end in group_indices:
        # print(start, end, colors[start:end])
        max_idx_in_group = start + np.argmax(colors[start:end])
        result.append(max_idx_in_group)
    return result

def angle_wrap(arr):
    # return (arr + np.pi) % (2 * np.pi) - np.pi
    return ((-arr + np.pi) % (2 * np.pi) - np.pi) * -1

def calc_dqdmot(Jmot, Jfree, actuation_model):
    #init of constant
    Lidmot=actuation_model.idvmot
    Lidfree=actuation_model.idvfree
    nv_mot=len(Lidmot)

    # computation of dq/dqmot
    pinvJfree=np.linalg.pinv(Jfree)
    dq_dmot_no=np.concatenate((np.identity(nv_mot),-pinvJfree@Jmot)) 
    
    #re order dq/dqmot
    dq_dmot=dq_dmot_no.copy()
    dq_dmot[Lidmot]=dq_dmot_no[:nv_mot,:]
    dq_dmot[Lidfree]=dq_dmot_no[nv_mot:,:]
    return dq_dmot

def jacobian_constraint_custom(model,data,constraint_model,constraint_data,actuation_model,q0):
    #update of the jacobian an constraint model
    # pin.computeJointJacobians(model,data,q0)
    cdatas = pin.StdVec_RigidConstraintData()
    cmodels = pin.StdVec_RigidConstraintModel()
    for cm,cd in zip(constraint_model,constraint_data):
        cdatas.append(cd)
        cmodels.append(cm)
    Jright = pin.getConstraintsJacobian(model,data,cmodels,cdatas)
    

    LJ=[np.array(())]*len(constraint_model)
    for (cm,cd,i) in zip(constraint_model,constraint_data,range(len(LJ))):
        LJ[i]=pin.getConstraintJacobian(model,data,cm,cd)
        

    #init of constant
    Lidmot=actuation_model.idvmot
    Lidfree=actuation_model.idvfree
    nv=model.nv
    nv_mot=len(Lidmot)
    nv_free=len(Lidfree)
    Lnc=[J.shape[0] for J in LJ]
    nc=int(np.sum(Lnc))
    
    
    Jmot=np.zeros((nc,nv_mot))
    Jfree=np.zeros((nc,nv_free))
    
    #separation between Jmot and Jfree
    nprec=0
    for J,n in zip(LJ,Lnc):
        Smot=np.zeros((nv,nv_mot))
        Smot[Lidmot,range(nv_mot)]=1
        Sfree=np.zeros((nv,nv_free))
        Sfree[Lidfree,range(nv_free)]=1

        Jmot[nprec:nprec+n,:]=J@Smot
        Jfree[nprec:nprec+n,:]=J@Sfree

        nprec=nprec+n
    
    #act2pass?
    # E_tau = np.zeros((nv, nv))
    # E_tau[range(nv_mot), Lidmot] = 1
    # E_tau[range(nv_mot,nv), Lidfree] = 1
    # # computation of dq/dqmot
    # pinvJfree=np.linalg.pinv(Jfree)
    # dq_dmot_no=np.concatenate((np.identity(nv_mot),-pinvJfree@Jmot))
    # #re order dq/dqmot
    # dq_dmot=dq_dmot_no.copy()
    # dq_dmot[Lidmot]=dq_dmot_no[:nv_mot,:]
    # dq_dmot[Lidfree]=dq_dmot_no[nv_mot:,:]
    # return dq_dmot_no, E_tau
    
    return (Jmot, Jfree), Jright

def jacobian_closed(model,data,constraint_model,constraint_data,actuation_model,q0, ideff):
    Jmot, Jfree = jacobian_constraint(model,data,constraint_model,constraint_data,actuation_model,q0) # q0 does nothing here

    #init of constant
    Lidmot=actuation_model.idvmot
    Lidfree=actuation_model.idvfree
    nv_mot=len(Lidmot)

    # computation of dq/dqmot
    pinvJfree=np.linalg.pinv(Jfree)
    dq_dmot_no=np.concatenate((np.identity(nv_mot),-pinvJfree@Jmot)) 
    # DOES NOT CHANGE DURING SIM
    
    #re order dq/dqmot
    dq_dmot=dq_dmot_no.copy()
    dq_dmot[Lidmot]=dq_dmot_no[:nv_mot,:]
    dq_dmot[Lidfree]=dq_dmot_no[nv_mot:,:]
    
    #computation of the closed-loop jacobian
    # Jf_closed = pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL)@dq_dmot
    Jf_closed = pin.computeFrameJacobian(model,data,q0,ideff,pin.LOCAL_WORLD_ALIGNED)@dq_dmot
    return Jf_closed

from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopInverseKinematicsProximal#, closedLoopProximalMount
from auto_robot_design.pinokla.robot_utils import freezeJoints, freezeJointsWithoutVis

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

    if constraint_dim > 0:
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
    else:
        pin.computeJointJacobians(reduced_model, reduced_data, q)
        return q, 0
    
from auto_robot_design.pinokla.closed_loop_jacobian import dq_dqmot
from testing_ground.singular_hell.mj2jp import calc_effective_inertia


def search_workspace_FK(
    model,
    data,
    effector_frame_name: str,
    base_frame_name: str,
    q_space: np.ndarray,
    actuation_model,
    constraint_models,
    constraint_data,
    viz=None, link_names=None, q_default=None, free_jiggle=[0.]
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
    q_last_feas = q_start
    # if q_default is not None:
    #     q_start = q_default

    # free_jiggle = [0,np.pi]
    q_free_jiggle_space = list(product(*([free_jiggle]*len(actuation_model.idqfree))))
    repeaters = []
    
    workspace_xyz = np.empty((len(q_space) * len(q_free_jiggle_space), 3))
    available_q = np.empty((len(q_space) * len(q_free_jiggle_space), len(q_start)))
    Jcls = []
    Jcs = []
    link_ids = []
    Ldsdq = []
    Lambdas = np.empty([len(q_space) * len(q_free_jiggle_space),2,2])
    # if link_names is not None:
    #     link_ids = [model.getFrameId(n) for n in link_names]
    # link_ids.append(model.getFrameId(effector_frame_name))
    for q_sample in q_space:
        # q_dict_free = zip(actuation_model.idqfree, q_start)
    
        solutions = np.full([len(q_free_jiggle_space),len(q_start)], None)
        c_with_dupl = 0

        for qfree_sample in q_free_jiggle_space:
            

            q_dict_free = zip(actuation_model.idqfree, qfree_sample) #TODO remake smarter
            for key, value in q_dict_free:
                q_start[key] += value

            q_dict_mot = zip(actuation_model.idqmot, q_sample)
            for key, value in q_dict_mot:
                q_start[key] = value
            # q_start = angle_wrap(q_start)

            # print(q_start, q_sample, actuation_model.idqfree)
            q_start = angle_wrap(q_start)
            q_sample = angle_wrap(q_sample)
            # print(q_start[actuation_model.idqmot], q_sample, q_sample - q_start[actuation_model.idqmot])
            qmot_diff = q_sample - q_start[actuation_model.idqmot]
            # qmot_diff = q_sample - q_last_feas[actuation_model.idqmot]

            assert len(constraint_models) > 0
            q3, error = ForwardKCustom(
                model,
                constraint_models,
                actuation_model,
                q_start,
                21,
            )
            q3 = angle_wrap(q3)
            q_start = angle_wrap(q_start)

            q_diff = q_last_feas-q3
            # q_diff = q_start-q3

            #q_diff[actuation_model.idqmot]
            qfree_diff = np.delete(q_diff, actuation_model.idqmot, axis=0)

            if error < 1e-11 :#and np.linalg.norm(qfree_diff) < 20*np.linalg.norm(qmot_diff) + 1e-12:
                # print(f'pf={error}, qmL2={np.linalg.norm(qmot_diff)}, qfL2={np.linalg.norm(qfree_diff)}')
                # print(qmot_diff, qfree_diff)
                if viz:
                    print(q3)
                    viz.display(q3)
                    # time.sleep(0.005)
                q_start = q3 #TODO remember the first sample in jiggle
                solutions[c_with_dupl] = q3
                c_with_dupl += 1

        # solutions = solutions[solutions != np.array(None)]
        mask = (solutions[:, 0] != None)
        solutions_found = solutions[mask, :].astype(float)
        # print(solutions_found.shape[0])
        solutions = solutions_found#.astype(float)
        if solutions_found.shape[0] > 1:
            solutions = np.unique(solutions_found.round(4), axis=0)
        # print(solutions_found)
        # print(solutions)
        n_sols = solutions.shape[0]
        if n_sols > 0:
            repeaters.append(n_sols)
        for sol_idx in range(n_sols):
            q3 = solutions[sol_idx,:]
            # if q_default is not None:
            #     q_start = q_default
            # q_start = pin.neutral(model)
            pin.framesForwardKinematics(model, data, q3)
            id_effector = model.getFrameId(effector_frame_name)
            id_base = model.getFrameId(base_frame_name)
            effector_pos = data.oMf[id_effector].translation
            # base_pos = data.oMf[id_base].translation
            # print('pos',effector_pos)
            # transformed_pos = effector_pos - base_pos
            transformed_pos = effector_pos

            # SOMETHING WRONG with oJf, ONLY 1 q column for linear vel so rank is always 1
            
            pin.computeJointJacobians(model, data, q3)  # precomputes all jacobians
            dsdq = []
            # link_ids = []
            # link_ids.append(id_effector)
            for lid in link_ids:
                dsdq.append(pin.getFrameJacobian(model, data, lid, pin.WORLD)) #TODO try diff coords LOCAL_WORLD_ALIGNED
            Ldsdq.append(dsdq)

            # oJf = pin.getFrameJacobian(model, data, id_effector, pin.WORLD)
            # oJf = oJf[(0,2),:]
            # _, singvals, _ = np.linalg.svd(oJf)
            # print('sv ',singvals)
            # print('jac',oJf)
            # print('rank=',np.linalg.matrix_rank(oJf))

            # all singular values for Jconstr are normal even near singularities
            Jconstraint = jacobian_constraint_custom(model,data,constraint_models,constraint_data,actuation_model,q3)
            Jclosed = jacobian_closed(model,data,constraint_models,constraint_data,actuation_model,q3, id_effector)

            M=pin.crba(model,data,q3)

            LJ=[np.array(())]*len(constraint_models)
            for (cm,cd,i) in zip(constraint_models,constraint_data,range(len(LJ))):
                LJ[i]=pin.getConstraintJacobian(model,data,cm,cd)

            dqdmot=dq_dqmot(model,actuation_model,LJ)
            Lambda = calc_effective_inertia(M, dqdmot, Jclosed)
            # print(Lambda)

            Jcs.append(Jconstraint)
            Jcls.append(Jclosed)
            # Lambdas.append(Lambda[2,2])
            Lambdas[c] = Lambda
            workspace_xyz[c] = transformed_pos
            available_q[c] = q3
            # q_last_feas = q3
            c += 1


    return (workspace_xyz[0:c], available_q[0:c], Jcls, Jcs, Ldsdq, Lambdas[0:c], repeaters)

def compute_total_inertia_at_point(model, data, target_point):
    total_inertia = pin.Inertia.Zero()
    # print('init iner', total_inertia)
    
    for joint_id in range(1, model.njoints):
        # Получить инерцию тела в локальных координатах
        body_inertia = model.inertias[joint_id]
        # print(f'm on jnt{joint_id}:',body_inertia.matrix()[0,0])
        # Преобразовать в мировые координаты
        body_inertia_world = body_inertia.se3Action(data.oMi[joint_id])
        # Сместить инерцию в целевую точку
        translation = data.oMi[joint_id].translation - target_point
        transform = pin.SE3.Identity()
        transform.translation = translation
        body_inertia_at_point = body_inertia_world.se3Action(transform)
        # Суммировать
        total_inertia += body_inertia_at_point
    
    return total_inertia.matrix()#[:3, :3]  # Возвращаем тензор инерции (3x3)

def calculate_Iyy_and_COMdist_at_point(model, data, q_0, target_point):
    # Инициализация
    v_0 = np.zeros(model.nv)
    # Вычисление кинематики и динамических параметров
    pin.computeAllTerms(model, data, q_0, v_0)
    # Вызов функции для сбора инерций
    inertia_tensor = compute_total_inertia_at_point(model, data, target_point)
    com_dist = np.linalg.norm(pin.centerOfMass(model, data)-target_point)
    # print('MASSSSSSSS:',inertia_tensor[0,0])
    return inertia_tensor[4,4], com_dist

def closedLoopProximalMount_open_support(
    model,
    data,
    constraint_model,
    constraint_data,
    #actuation_model,
    q_prec=None,
    max_it=100,
    eps=1e-12,
    rho=1e-10,
    mu=1e-4,
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

    #Lid = actuation_model.idqmot
    if q_prec is None:
        q_prec = pin.neutral(model)
    q = q_prec

    constraint_dim = 0
    for cm in constraint_model:
        constraint_dim += cm.size()

    if constraint_dim > -1:
        y = np.ones((constraint_dim))
        data.M = np.eye(model.nv) * rho
        kkt_constraint = pin.ContactCholeskyDecomposition(model, constraint_model)

        for k in range(max_it):
            pin.computeJointJacobians(model, data, q)
            kkt_constraint.compute(model, data, constraint_model, constraint_data, mu)

            constraint_value = np.concatenate(
                [
                    (pin.log(cd.c1Mc2).np[: cm.size()])
                    for (cd, cm) in zip(constraint_data, constraint_model)
                ]
            )

            LJ = []
            for cm, cd in zip(constraint_model, constraint_data):
                Jc = pin.getConstraintJacobian(model, data, cm, cd)
                LJ.append(Jc)
            J = np.concatenate(LJ)

            primal_feas = np.linalg.norm(constraint_value, np.inf)
            dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
            if primal_feas < eps and dual_feas < eps:
                #print("Convergence achieved")
                break
            #print("constraint_value:", np.linalg.norm(constraint_value))
            rhs = np.concatenate([-constraint_value - y * mu, np.zeros(model.nv)])

            dz = kkt_constraint.solve(rhs)
            dy = dz[:constraint_dim]
            dq = dz[constraint_dim:]

            alpha = 1.0
            q = pin.integrate(model, q, -alpha * dq)
            y -= alpha * (-dy + y)
    else:
        pin.computeJointJacobians(model, data, q)
    return q

def folow_traj_by_proximal_inv_k(
    model,
    data,
    constraint_models,
    constraint_data,
    end_effector_frame: str,
    traj_6d: np.ndarray,
    viz=None,
    q_start: np.ndarray = None,
    actuation_model=None, is_open_chain=False, pos_inertia=np.zeros(3)
):
    """Solve the inverse kinematic problem

    Args:
        model (_type_): _description_
        data (_type_): _description_
        constraint_models (_type_): _description_
        constraint_data (_type_): _description_
        end_effector_frame (str): _description_
        traj_6d (np.ndarray): _description_
        viz (_type_, optional): _description_. Defaults to None.
        q_start (np.ndarray, optional): _description_. Defaults to None.

    Returns:
        np.array: end-effector positions in final state
        np.array: joint coordinates in final state
        np.array: deviations from the desired position

    """
    if q_start:
        q = q_start
    else:
        q = pin.neutral(model)

    ee_frame_id = model.getFrameId(end_effector_frame)
    poses = np.zeros((len(traj_6d), 3))
    q_array = np.zeros((len(traj_6d), len(q)))
    constraint_errors = np.zeros((len(traj_6d), 1))

    #-----ADDED
    M_list = []
    Iyy_list = []
    coms_list = []
    Jacs_closed = []
    Jacs_dqdmot = []
    has_equalities = len(constraint_models) > 0
    assert has_equalities != is_open_chain # check that input corresponds to reality

    for num, i_pos in enumerate(traj_6d):
        q, min_feas, is_reach = closedLoopInverseKinematicsProximal(
            model,
            data,
            constraint_models,
            constraint_data,
            i_pos,
            ee_frame_id,
            onlytranslation=True,
            q_start=q,
        )
        if not is_reach and not is_open_chain:
            q = closedLoopProximalMount_open_support(
                model, data, constraint_models, constraint_data, q
            )
        if viz:
            viz.display(q)
            time.sleep(0.1)

        pin.framesForwardKinematics(model, data, q)
        poses[num] = data.oMf[ee_frame_id].translation
        q_array[num] = q
        constraint_errors[num] = min_feas

        #--------------------------------ADDED
        pin.computeJointJacobians(model, data, q)  # precomputes all jacobians
        # all singular values for Jconstr are normal even near singularities
        Jconstraint = jacobian_constraint_custom(model,data,constraint_models,constraint_data,actuation_model,q)
        if not is_open_chain:
            Jclosed = jacobian_closed(model,data,constraint_models,constraint_data,actuation_model,q, ee_frame_id)
        else: # не имеет смысла для двузвенника, потому что обе ветки дают одинаковый результат
            Jac = pin.computeFrameJacobian(model,data,q,ee_frame_id,pin.LOCAL_WORLD_ALIGNED)
            Jclosed = Jac#[(0,2),:]

        M=pin.crba(model,data,q)
        Iyy, comdist = calculate_Iyy_and_COMdist_at_point(model, data, q, pos_inertia)

        (Jmot, Jfree), _ = Jconstraint
        dqdmot = calc_dqdmot(Jmot, Jfree, actuation_model)

        # LJ=[np.array(())]*len(constraint_models)
        # for (cm,cd,i) in zip(constraint_models,constraint_data,range(len(LJ))):
        #     LJ[i]=pin.getConstraintJacobian(model,data,cm,cd)
        # dqdmot=dq_dqmot(model,actuation_model,LJ)
        
        # Lambda = calc_effective_inertia(M, dqdmot, Jclosed)

        M_list.append(M)
        Iyy_list.append(Iyy)
        coms_list.append(comdist)
        # Jacs_con.append(Jconstraint)
        Jacs_closed.append(Jclosed)
        Jacs_dqdmot.append(dqdmot)

    return poses, q_array, constraint_errors, M_list, Jacs_closed, Jacs_dqdmot, Iyy_list, coms_list
