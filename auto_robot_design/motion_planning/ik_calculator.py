import pinocchio as pin
import numpy as np
from numpy.linalg import solve, norm, pinv


def open_loop_ik(rmodel, target_pos, ideff, q_start=None, eps=1e-5, max_it=30):
    """This function is hitting nails with a microscope

        It actually works only for 2 link robot. For more links it will return some result, 
        since we do not set any additional conditions.    

    Args:
        rmodel (_type_): pinocchio model
        target_pos (_type_): 6d position of the target
        ideff (_type_): end-effector id
        q_start (_type_, optional): starting position for ik search. Defaults to None.
        eps (_type_, optional): precision. Defaults to 1e-5.
        max_it (int, optional): max iterations for ik search algorithm. Defaults to 30.

    Returns:
        _type_: configuration, error and reachability state
    """
    # create copy of the model and corresponding data
    model = pin.Model(rmodel)
    data = model.createData()
    if q_start is None:
        q_start = pin.neutral(model)
        q = q_start
    else:
        q = q_start

    # set the SE3 representation of the final position (useless)
    target_SE3 = pin.SE3.Identity()
    target_SE3.translation = np.array(target_pos[0:3])
    is_reach = False
    DT = 1e-1  # Optimization step
    for _ in range(max_it):
        pin.framesForwardKinematics(model, data, q)
        err = data.oMf[ideff].translation-target_SE3.translation
        if norm(err) < eps:
            is_reach = True
            break
        J = pin.computeFrameJacobian(
            model, data, q, ideff, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        v = - pinv(J) @ err
        q = pin.integrate(model, q, v * DT)
    return q, norm(err), is_reach


def closed_loop_ik_pseudo_inverse(rmodel, 
                                  rconstraint_model, 
                                  target_pos, ideff, 
                                  q_start=None, 
                                  onlytranslation:bool=True, 
                                  eps:float=1e-5, 
                                  max_it:int=30, 
                                  alpha:float=0.5, 
                                  l:float=1e-5, 
                                  q_delta_threshold:float=0.5):
    """Finds the IK solution using constraint Jacobian. 

        The target position is added to the list of constraints and treated as a constraint violated in the starting position.

    Args:
        rmodel (_type_): _description_
        rconstraint_model (_type_): _description_
        target_pos (_type_): _description_
        ideff (_type_): _description_
        q_start (_type_, optional): _description_. Defaults to None.
        onlytranslation (bool, optional): _description_. Defaults to False.
        eps (float, optional): _description_. Defaults to 1e-5.
        max_it (int, optional): _description_. Defaults to 30.
        alpha (float, optional): _description_. Defaults to 0.5.
        l (float, optional): _description_. Defaults to 1e-5.
        q_delta_threshold (float, optional): _description_. Defaults to 0.5.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    # create copy of the model, constraints and corresponding data
    model = pin.Model(rmodel)
    constraint_model = [pin.RigidConstraintModel(x) for x in rconstraint_model]
    # set the SE3 representation of the final position. Here we add final position as a constraint.
    target_SE3 = pin.SE3.Identity()
    target_SE3.translation = np.array(target_pos[0:3])
    frame_constraint = model.frames[ideff]
    parent_joint = frame_constraint.parentJoint # ee parent joint is in the same position as the frame?
    placement = frame_constraint.placement # placement is calculated relative to parent joint?
    # constraint can include orientation or not
    if onlytranslation:
        final_constraint = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,
                                                    model, parent_joint,
                                                    placement, 0, target_SE3,
                                                    pin.ReferenceFrame.LOCAL)
    else:
        final_constraint = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D, model, parent_joint, placement,
            model.getJointId("universel"), target_pos,
            pin.ReferenceFrame.LOCAL)
        raise Exception("Not implemented")

    final_constraint.name = "TrajCons"
    constraint_model.append(final_constraint)

    data = model.createData()
    constraint_data = [cm.createData() for cm in constraint_model]
    if q_start is None:
        q_start = pin.neutral(model)
        q = q_start
    else:
        q = q_start
    #calculates pin joints and frames corresponding to the q. Neutral position is the position in URDF and it has all q=0
    pin.framesForwardKinematics(model, data, q)
    constraint_dim = 0
    for cm in constraint_model:
        constraint_dim += cm.size()

    is_reach = False
    # I dont know why we need kkt here, but it seems to fail without it
    kkt_constraint = pin.ContactCholeskyDecomposition(model, constraint_model)
    primal_feas_array = np.zeros(max_it)
    real_feas_array = np.zeros(max_it)
    dim = len(q)
    q_array = np.zeros((max_it, dim))
    constr_array = []
    # IK search iteration loop
    for k in range(max_it):
        pin.computeJointJacobians(model, data, q)
        kkt_constraint.compute(model, data, constraint_model, constraint_data)# mb here we actually update constraint_data and we can find more direct way to do it
        constraint_value = np.concatenate([
            (pin.log(cd.c1Mc2).np[:cm.size()])
            for (cd, cm) in zip(constraint_data, constraint_model)
        ])
        # calculate total constraint Jacobian
        LJ = []
        for cm, cd in zip(constraint_model, constraint_data):
            Jc = pin.getConstraintJacobian(model, data, cm, cd)
            LJ.append(Jc)
        J = np.concatenate(LJ)
        constr_array.append(constraint_value)
        primal_feas = np.linalg.norm(constraint_value, np.inf)
        real_constrain_feas = np.linalg.norm(constraint_value[:-3])
        real_feas_array[k] = real_constrain_feas
        primal_feas_array[k] = primal_feas
        q_array[k] = q

        if primal_feas < eps:
            is_reach = True
            break
        # here we use pseudo inverse with additional regularization and l is parameter of the regularization
        dq = (J.T@(np.linalg.inv(J@J.T-l*np.eye(len(constraint_value))))
              ).dot(constraint_value)

        # Jacobian methods use linearization, so any solution works only in small area. 
        # If the solution step is large, it means the direction is close to singular one. 
        # TODO: may be we should add some kind of normalization, the solution for dq is actually a direction and it is proportional to current constraints violation 
        if np.linalg.norm(dq, np.inf) > q_delta_threshold:
            break

        q = pin.integrate(model, q, alpha * dq)
        # total_delta_q = q-q_start - alternatively we can use total error instead of a step error

    min_feas = primal_feas
    min_real_feas = real_constrain_feas
    # if the required position is unreachable we choose the position closest to the required point
    if not is_reach:
        for_sort = np.column_stack(
            (primal_feas_array[0:k+1], real_feas_array[0:k+1], q_array[0:k+1, :]))

        def key_sort(x): return x[1]
        for_sort = sorted(for_sort, key=key_sort)
        finish_q = for_sort[0][2:]
        q = finish_q
        min_feas = for_sort[0][0]
        min_real_feas = for_sort[0][1]
        pin.framesForwardKinematics(model, data, q)

    return q, min_feas, is_reach


def closed_loop_ik_grad(rmodel, rconstraint_model, target_pos, ideff, q_start=None, onlytranslation=False, eps=1e-5, step=1e-1, max_it=1000000):

    model = pin.Model(rmodel)
    constraint_model = [pin.RigidConstraintModel(x) for x in rconstraint_model]
    open_loop = False
    target_SE3 = pin.SE3.Identity()
    target_SE3.translation = np.array(target_pos[0:3])
    frame_constraint = model.frames[ideff]
    parent_joint = frame_constraint.parentJoint
    placement = frame_constraint.placement
    if onlytranslation:
        final_constraint = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,
                                                    model, parent_joint,
                                                    placement, 0, target_SE3,
                                                    pin.ReferenceFrame.LOCAL)
    else:
        final_constraint = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D, model, parent_joint, placement,
            model.getJointId("universel"), target_pos,
            pin.ReferenceFrame.LOCAL)
        raise Exception("Not implemented")

    final_constraint.name = "TrajCons"
    constraint_model.append(final_constraint)

    data = model.createData()
    constraint_data = [cm.createData() for cm in constraint_model]

    if q_start is None:
        q_start = pin.neutral(model)
        q = q_start
    else:
        q = q_start
    pin.framesForwardKinematics(model, data, q)
    constraint_dim = 0
    for cm in constraint_model:
        constraint_dim += cm.size()
    is_reach = False
    kkt_constraint = pin.ContactCholeskyDecomposition(model, constraint_model)
    primal_feas_array = np.zeros(max_it)
    real_feas_array = np.zeros(max_it)
    dim = len(q)
    q_array = np.zeros((max_it, dim))
    constr_array = []
    for k in range(max_it):
        pin.computeJointJacobians(model, data, q)
        kkt_constraint.compute(model, data, constraint_model, constraint_data)
        constraint_value = np.concatenate([
            (pin.log(cd.c1Mc2).np[:cm.size()])
            for (cd, cm) in zip(constraint_data, constraint_model)
        ])

        LJ = []
        for cm, cd in zip(constraint_model, constraint_data):
            Jc = pin.getConstraintJacobian(model, data, cm, cd)
            LJ.append(Jc)
        J = np.concatenate(LJ)
        constr_array.append(constraint_value)
        primal_feas = np.linalg.norm(constraint_value, np.inf)
        real_constrain_feas = np.linalg.norm(constraint_value[:-3])
        real_feas_array[k] = real_constrain_feas
        primal_feas_array[k] = primal_feas
        q_array[k] = q

        if primal_feas < eps:
            is_reach = True
            break

        grad = 2*J.T.dot(constraint_value)/dim
        target = constraint_value.dot(constraint_value)/dim
        # grad = np.sign(sum(constraint_value)) * np.sum(J,axis=0)

        # grad = J.T.dot(np.sign(constraint_value))/dim
        # target = np.sum(np.abs(constraint_value))/dim
        step = 1e-4/np.linalg.norm(grad, np.inf)
        q = pin.integrate(model, q, step * grad)
        pin.framesForwardKinematics(model, data, q)
        total_delta_q = q-q_start
        if np.linalg.norm(total_delta_q, np.inf) > 0.2:
            break

    pin.framesForwardKinematics(model, data, q)
    min_feas = primal_feas
    min_real_feas = real_constrain_feas
    if not is_reach:
        for_sort = np.column_stack(
            (primal_feas_array[0:k+1], real_feas_array[0:k+1], q_array[0:k+1, :]))

        def key_sort(x): return x[0]
        for_sort = sorted(for_sort, key=key_sort)
        finish_q = for_sort[0][2:]
        q = finish_q
        min_feas = for_sort[0][0]
        min_real_feas = for_sort[0][1]
        pin.framesForwardKinematics(model, data, q)

    return q, min_feas, is_reach
