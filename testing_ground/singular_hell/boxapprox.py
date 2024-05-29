import numpy as np
from scipy.optimize import linprog
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
import collections
from multiprocessing import Pool, freeze_support, cpu_count
import time
from functools import partial
# from mpl_toolkits.mplot3d import Axes3D

from testing_ground.singular_hell.box_utils import pow_interval

def calc_parab_ineq(m_bounds, m_ind: int, p_ind: int, vector_size: int):
    """Calculate halfplane inequalities (bounding p) for p=m**2"""
    g = m_bounds[0]
    h = m_bounds[1]
    g2 = g**2
    h2 = h**2
    A_ub = np.zeros((3,vector_size))
    A_ub[0,m_ind] = g2-h2#g**2-h**2 #np.square(g)-np.square(h)
    A_ub[0,p_ind] = h-g
    A_ub[1,m_ind] = 2*g
    A_ub[1,p_ind] = -1
    A_ub[2,m_ind] = 2*h
    A_ub[2,p_ind] = -1

    b_ub = np.array([g2*(h-g)-g*(h2-g2), g2, h2])
    # b_ub = np.array([g**2*(h-g)-g*(h**2-g**2), g**2, h**2])
    # b_ub = np.array([np.square(g)*(h-g)-g*(np.square(h)-np.square(g)), np.square(g), np.square(h)])
    return A_ub, b_ub

def calc_plane_coeffs(p1,p2,p3):
    """Calculate the coefficients of a plane A*x+B*y+C*z=Dn which is defined by three points."""
    x1,y1,z1 = p1[0],p1[1],p1[2]
    x2,y2,z2 = p2[0],p2[1],p2[2]
    x3,y3,z3 = p3[0],p3[1],p3[2]

    A = y1*(z2 - z3) + y2*(z3 - z1) + y3*(z1 - z2)
    B = z1*(x2 - x3) + z2*(x3 - x1) + z3*(x1 - x2)
    C = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
    Dn = x1*(y2*z3 - y3*z2) + x2*(y3*z1 - y1*z3) + x3*(y1*z2 - y2*z1)
    return np.array([A,B,C,Dn])

def calc_hyberb_ineq(m_bounds, n_bounds, m_ind: int, n_ind: int, p_ind: int, vector_size: int):
    """Calculate halfspace inequalities (bounding p) for p=m*n"""
    gm = m_bounds[0]
    hm = m_bounds[1]
    gn = n_bounds[0]
    hn = n_bounds[1]

    p1 = (gm,gn,gm*gn)
    p2 = (gm,hn,gm*hn)
    p3 = (hm,hn,hm*hn)
    p4 = (hm,gn,hm*gn)
    # print(p1,p2,p3,p4)

    pl1 = calc_plane_coeffs(p1,p2,p3) #< Dn
    pl2 = calc_plane_coeffs(p1,p3,p4) #< Dn
    pl3 = calc_plane_coeffs(p1,p2,p4) #> 
    pl4 = calc_plane_coeffs(p2,p3,p4) #> 
    # print(pl1,pl2,pl3,pl4)
    
    A_ub = np.zeros((4,vector_size))
    A_ub[0,[m_ind, n_ind, p_ind]] = -pl1[:3]
    A_ub[1,[m_ind, n_ind, p_ind]] = -pl2[:3]
    A_ub[2,[m_ind, n_ind, p_ind]] = pl3[:3]
    A_ub[3,[m_ind, n_ind, p_ind]] = pl4[:3]

    b_ub = np.asarray([-pl1[3], -pl2[3], pl3[3], pl4[3]])
    return A_ub, b_ub


#TODO fix overlapping boxes issue, not critical
def approximate_boxes(B, A_eq, b_eq, sqr_pairs=[], cross_triplets=[], threshold_s=[0.1], threshold_v=0.95):
    """
    Initial single-threaded version of box approximation (all-in-one function). Works faster than box_serial().

    Args:
        B (np.ndarray): Initial bounding box.
        A_eq (np.ndarray): Matrix of linear equations.
        b_eq (np.ndarray): Vector rhs of linear equations.
        sqr_pairs (list, optional): list of pairs of indexes, binding variable m with its square variable m**2. Defaults to [].
        cross_triplets (list, optional): list of triplets of indexes, binding two variables m,n with its product variable m*n. Defaults to [].
        threshold_s (list|np.ndarray, optional): Vector of 1 or len(B) elements, defining min allowed length of the 
        longest (or corresponding to vector) side of the boxes. Defines which box is considered ready solution. Defaults to [0.1].
        threshold_v (float, optional): max allowed ratio of box volume after shrinking to volume before. Defines when to stop shrinking a particular box. Defaults to 0.95.

    Returns:
        list: all solutions' bounding boxes.
    """
    n = np.shape(B)[0]
    is_sigma_simple = len(threshold_s) == 1

    # print(B)
    # A_ub = np.zeros((3*len(sqr_pairs), n))
    # b_ub = np.zeros(3*len(sqr_pairs))
    A_ub = np.full((3*len(sqr_pairs)+4*len(cross_triplets), n),None)
    b_ub = np.full(3*len(sqr_pairs)+4*len(cross_triplets),None)

    # Blist = [(B[i,0],B[i,1]) for i in range(n)]
    # print(Blist)
    # c = np.zeros(n)
    # c[0] = 1
    # minim = linprog(c,A_eq=A_eq,b_eq=b_eq,bounds=B)
    # print(minim.fun)
    # print(f'suc{minim.success}, stat{minim.status}, nit{minim.nit}\nmsg: {minim.message}')

    # minim = linprog(-c,A_eq=A_eq,b_eq=b_eq,bounds=B)
    # print(-minim.fun)
    # print(f'suc{minim.success}, stat{minim.status}, nit{minim.nit}\nmsg: {minim.message}')
    print(f'Starting single-threaded box approximation.')
    sols = []
    P = [B]
    while len(P) > 0:
        print(f'lenP {len(P)}')
        B_c = P.pop() #0

        lengths_c = B_c[:,1] - B_c[:,0]
        # valid_inds = (lengths_c > 0).nonzero()
        # print('valind',valid_inds)
        # print('valleng',lengths_c[valid_inds])
        
        is_empty = False
        is_shrinkable = True
        if is_sigma_simple:
            is_small_enough = np.max(lengths_c) <= threshold_s[0]
        else:
            is_small_enough = (lengths_c <= threshold_s).all()
        
        while not is_empty and is_shrinkable and not is_small_enough:  #uncomment for better timings
            lengths_p = lengths_c
            # valid_inds_p = np.transpose((lengths_p > 0).nonzero())
            valid_inds_p = (lengths_p > 0).nonzero()[0]

            #shrink
            for i in valid_inds_p: #range(n):
                c = np.zeros(n)
                c[i] = 1

                #inequalities for current bounds
                shift = 0
                for j, (m_ind, p_ind) in enumerate(sqr_pairs):
                    A, b = calc_parab_ineq(B_c[m_ind,:], m_ind, p_ind, n)
                    A_ub[3*j:3*j+3,:] = A
                    b_ub[3*j:3*j+3] = b
                    shift += 3
                for j, (m_ind, n_ind, p_ind) in enumerate(cross_triplets):
                    A, b = calc_hyberb_ineq(B_c[m_ind,:], B_c[n_ind,:], m_ind, n_ind, p_ind, n)
                    A_ub[4*j+shift:4*j+4+shift,:] = A
                    b_ub[4*j+shift:4*j+4+shift] = b

                minim = linprog(c,A_eq=A_eq,b_eq=b_eq,bounds=B_c, A_ub=A_ub, b_ub=b_ub,options={'presolve':False}) #method='highs-ipm' #slower but more boxes with presolve 
                if not minim.success:
                    is_empty = True
                    print(minim.message)
                    # if minim.status != 2:
                    #     print(minim.message)
                    break

                B_c[i,0] = minim.fun

                #comment for better timings
                #inequalities for updated bounds
                shift = 0
                for j, (m_ind, p_ind) in enumerate(sqr_pairs):
                    A, b = calc_parab_ineq(B_c[m_ind,:], m_ind, p_ind, n)
                    A_ub[3*j:3*j+3,:] = A
                    b_ub[3*j:3*j+3] = b
                    shift += 3
                for j, (m_ind, n_ind, p_ind) in enumerate(cross_triplets):
                    A, b = calc_hyberb_ineq(B_c[m_ind,:], B_c[n_ind,:], m_ind, n_ind, p_ind, n)
                    A_ub[4*j+shift:4*j+4+shift,:] = A
                    b_ub[4*j+shift:4*j+4+shift] = b

                maxim = linprog(-c,A_eq=A_eq,b_eq=b_eq,bounds=B_c, A_ub=A_ub, b_ub=b_ub,options={'presolve':False}) #method='highs-ipm'
                if not maxim.success:
                    is_empty = True
                    print(maxim.message)
                    # if maxim.status != 2:
                    #     print(maxim.message)
                    break

                # B_c[i,0] = minim.fun
                B_c[i,1] = -maxim.fun

                # if is_small_enough: #worse than without checking, but better time
                #     break

            lengths_c = B_c[:,1] - B_c[:,0]
            longest_dim_ind = np.argmax(lengths_c)
            if is_sigma_simple:
                is_small_enough = lengths_c[longest_dim_ind] <= threshold_s[0]
            else:
                is_small_enough = (lengths_c <= threshold_s).all()

            valid_inds = (lengths_c > 0).nonzero()
            print('valid indexes',valid_inds[0])
            # print('valleng',lengths_c[valid_inds])
            V_p = np.prod(lengths_p[valid_inds])
            V_c = np.prod(lengths_c[valid_inds])
            
            # print(V_c,V_p)
            is_shrinkable = V_c/V_p <= threshold_v
            # print(is_shrinkable)

        if not is_empty:
            if is_small_enough:
                sols.append(B_c)
            else:
                #split
                half_length = lengths_c[longest_dim_ind]/2.
                B1 = B_c.copy()
                B1[longest_dim_ind,1] -= half_length
                B2 = B_c.copy()
                B2[longest_dim_ind,0] += half_length
                P.append(B1)
                P.append(B2)
    return sols


def process_box(B_c, A_eq, b_eq, 
                sqr_pairs=[], cross_triplets=[], 
                threshold_s=[0.1], threshold_v=0.95):
    
    is_sigma_simple = len(threshold_s) == 1
    n = A_eq.shape[1]
    # if len(P) == 0:
    #     return
    # try:
    #     B_c = P.pop() #0
    # except(IndexError):
    #     return
    
    # print('B',B_c)
    # print(P)

    # print(B)
    # A_ub = np.zeros((3*len(sqr_pairs), n))
    # b_ub = np.zeros(3*len(sqr_pairs))
    A_ub = np.full((3*len(sqr_pairs)+4*len(cross_triplets), n),None)
    b_ub = np.full(3*len(sqr_pairs)+4*len(cross_triplets),None)

    # Blist = [(B[i,0],B[i,1]) for i in range(n)]
    # print(Blist)
    # c = np.zeros(n)
    # c[0] = 1
    # minim = linprog(c,A_eq=A_eq,b_eq=b_eq,bounds=B)
    # print(minim.fun)
    # print(f'suc{minim.success}, stat{minim.status}, nit{minim.nit}\nmsg: {minim.message}')

    # minim = linprog(-c,A_eq=A_eq,b_eq=b_eq,bounds=B)
    # print(-minim.fun)
    # print(f'suc{minim.success}, stat{minim.status}, nit{minim.nit}\nmsg: {minim.message}')


    lengths_c = B_c[:,1] - B_c[:,0]
    # valid_inds = (lengths_c > 0).nonzero()
    # print('valind',valid_inds)
    # print('valleng',lengths_c[valid_inds])
    
    is_empty = False
    is_shrinkable = True
    if is_sigma_simple:
        is_small_enough = np.max(lengths_c) <= threshold_s[0]
    else:
        is_small_enough = (lengths_c <= threshold_s).all()
    
    while not is_empty and is_shrinkable and not is_small_enough:  #uncomment for better timings
        lengths_p = lengths_c
        # valid_inds_p = np.transpose((lengths_p > 0).nonzero())
        valid_inds_p = (lengths_p > 0).nonzero()[0]

        #shrink
        for i in valid_inds_p: #range(n):
            c = np.zeros(n)
            c[i] = 1

            #inequalities for current bounds
            shift = 0
            for j, (m_ind, p_ind) in enumerate(sqr_pairs):
                A, b = calc_parab_ineq(B_c[m_ind,:], m_ind, p_ind, n)
                A_ub[3*j:3*j+3,:] = A
                b_ub[3*j:3*j+3] = b
                shift += 3
            for j, (m_ind, n_ind, p_ind) in enumerate(cross_triplets):
                A, b = calc_hyberb_ineq(B_c[m_ind,:], B_c[n_ind,:], m_ind, n_ind, p_ind, n)
                A_ub[4*j+shift:4*j+4+shift,:] = A
                b_ub[4*j+shift:4*j+4+shift] = b

            minim = linprog(c,A_eq=A_eq,b_eq=b_eq,bounds=B_c, A_ub=A_ub, b_ub=b_ub,options={'presolve':False}) #method='highs-ipm' #slower but more boxes with presolve 
            if not minim.success:
                is_empty = True
                print('min',minim.message,flush=True)
                # if minim.status != 2:
                #     print(minim.message)
                break

            B_c[i,0] = minim.fun

            #comment for better timings
            #inequalities for updated bounds
            shift = 0
            for j, (m_ind, p_ind) in enumerate(sqr_pairs):
                A, b = calc_parab_ineq(B_c[m_ind,:], m_ind, p_ind, n)
                A_ub[3*j:3*j+3,:] = A
                b_ub[3*j:3*j+3] = b
                shift += 3
            for j, (m_ind, n_ind, p_ind) in enumerate(cross_triplets):
                A, b = calc_hyberb_ineq(B_c[m_ind,:], B_c[n_ind,:], m_ind, n_ind, p_ind, n)
                A_ub[4*j+shift:4*j+4+shift,:] = A
                b_ub[4*j+shift:4*j+4+shift] = b

            maxim = linprog(-c,A_eq=A_eq,b_eq=b_eq,bounds=B_c, A_ub=A_ub, b_ub=b_ub,options={'presolve':False}) #method='highs-ipm'
            if not maxim.success:
                is_empty = True
                print('max',maxim.message,flush=True)
                # if maxim.status != 2:
                #     print(maxim.message)
                break

            # B_c[i,0] = minim.fun
            B_c[i,1] = -maxim.fun

            # if is_small_enough: #worse than without checking, but better time
            #     break

        lengths_c = B_c[:,1] - B_c[:,0]
        longest_dim_ind = np.argmax(lengths_c)
        if is_sigma_simple:
            is_small_enough = lengths_c[longest_dim_ind] <= threshold_s[0]
        else:
            is_small_enough = (lengths_c <= threshold_s).all()

        valid_inds = (lengths_c > 0).nonzero()
        print('valid indexes',valid_inds[0])
        # print('valleng',lengths_c[valid_inds])
        V_p = np.prod(lengths_p[valid_inds])
        V_c = np.prod(lengths_c[valid_inds])
        
        # print(V_c,V_p)
        is_shrinkable = V_c/V_p <= threshold_v
        # print(is_shrinkable)

    if not is_empty:
        if is_small_enough:
            return (1,B_c)
            # sols.append(B_c)
        else:
            #split
            half_length = lengths_c[longest_dim_ind]/2.
            B1 = B_c.copy()
            B1[longest_dim_ind,1] -= half_length
            B2 = B_c.copy()
            B2[longest_dim_ind,0] += half_length
            return (2,B1,B2)
        
            # half_length = lengths_c[longest_dim_ind]/3.
            # B1 = B_c.copy()
            # B1[longest_dim_ind,1] -= 2*half_length
            # B2 = B_c.copy()
            # B2[longest_dim_ind,1] -= half_length
            # B2[longest_dim_ind,0] += half_length
            # B3 = B_c.copy()
            # B3[longest_dim_ind,0] += 2*half_length
            # return (3,B1,B2,B3)
    return (0,)

# def append_split(res, P):
#     if res is not None:
#         P.append(res[0])
#         P.append(res[1])
#         print('aloha')

# def approximate_boxes_parallel(B, A_eq, b_eq, 
#                                sqr_pairs=[], cross_triplets=[], 
#                                threshold_s=[0.1], threshold_v=0.95):
#     timeout = 10
#     queue_cap = 16
#     n_processes = 2

#     n = np.shape(B)[0]
#     # sols = []
#     # P = [B]

#     freeze_support()
#     manager=Manager()
#     pool = Pool(n_processes)
#     sharedP=manager.list()
#     sharedP.append(B)
#     # print('B',B)
#     # print(sharedP)
#     sharedsols=manager.list()

#     # print(sharedP)
#     print('-------------------')
#     process_box_ = partial(process_box, sols=sharedsols, n=n, 
#                            A_eq=A_eq, b_eq=b_eq, 
#                            sqr_pairs=sqr_pairs, cross_triplets=cross_triplets, 
#                            threshold_s=threshold_s, threshold_v=threshold_v)
#     # append_split_ = partial(append_split, P=sharedP)

#     t0 = time.time()
#     q = collections.deque()
#     while True:
#         # while len(sharedP) > 0 and len(sharedP)<5:
#         print(f'lenP {len(sharedP)}')
#         # mB_c = []
#         # slicer = 0
#         # while len(sharedP) > 0 and slicer<3:
#         #     mB_c.append(sharedP.pop())
#         #     slicer += 1
#             # B_c = sharedP.pop() #0
#         # print('mB len',len(mB_c))
#         batch = 4
#         mapres = pool.map(process_box_,sharedP[:np.min((len(sharedP),batch))])
#         if mapres:
#             print(mapres)
#             for r in mapres:
#                 if r is not None:
#                     sharedP.append(r[0])
#                     sharedP.append(r[1])
#                     print('aloha')
#                     print('shP',len(sharedP))
#         else:
#             break
    

#         # q.append(pool.apply_async(process_box_, (B_c,))) # MUST USE (arg,), but callback isnt necessary , callback=log_solution
#         # if len(q) >= queue_cap:
#         #     res = q.popleft().get(timeout=timeout)
#         #     if res is not None:
#         #         sharedP.append(res[0])
#         #         sharedP.append(res[1])
#         #         print('aloha')
#         # while len(q):
#         #     print('lq',len(q))
#         #     res=q.popleft().get(timeout=timeout)
#         #     if res is not None:
#         #         sharedP.append(res[0])
#         #         sharedP.append(res[1])
#         #         print('aloha2')
#         # if len(sharedP) < 1:
#         #     break
#     pool.close()
#     pool.join()
#     print('Finished at',time.time()-t0)

#     print('-------------------')
#     # print(sharedP)
#     # print(sharedsols)

#     return sharedsols

def box_parallel(boxes, A_eq, b_eq, 
                 sqr_pairs=[], cross_triplets=[], 
                 threshold_s=[0.1], threshold_v=0.95,
                 n_processes=cpu_count()-1,batch=10**4):
    """
    Multi-threaded version of box approximation.

    Args:
        boxes (list): Initial bounding boxes.
        A_eq (np.ndarray): Matrix of linear equations.
        b_eq (np.ndarray): Vector rhs of linear equations.
        sqr_pairs (list, optional): list of pairs of indexes, binding variable m with its square variable m**2. Defaults to [].
        cross_triplets (list, optional): list of triplets of indexes, binding two variables m,n with its product variable m*n. Defaults to [].
        threshold_s (list|np.ndarray, optional): Vector of 1 or len(B) elements, defining min allowed length of the 
        longest (or corresponding to vector) side of the boxes. Defines which box is considered ready solution. Defaults to [0.1].
        threshold_v (float, optional): max allowed ratio of box volume after shrinking to volume before. Defines when to stop shrinking a particular box. Defaults to 0.95.
        n_processes (int): number of processes to run for the solver. Defaults to cpu_count()-1.
        batch (int): max number of boxes to add to a processing queue in one go. More->better, may be auto-selected in the future. Defaults to 10 000.

    Returns:
        list: all solutions' bounding boxes.
    """
    freeze_support()
    # manager=Manager()
    
    # data = [i for i in range(3,11)]

    # sols=[]
    # P = [i for i in range(20,200,20)]

    # n_processes = cpu_count()//2
    pool = Pool(n_processes)
    print(f'Starting box approximation on {n_processes} processes.')
    # with Pool(4) as pool: #cpu_count()
    # P = [B]
    # P = boxes.copy()
    P = collections.deque(boxes)
    sols = []
    # sharedP=manager.list()
    # sharedsols=manager.list()
    # sharedP.append(B)
    # for i in range(20,400,20):
    #     sharedP.append(i)
    # print(sharedP)
    print('-------------------')
    # mockup_ = partial(process_box, P=sharedP, sols=sharedsols, threshold_s=[0.5])
    process_box_ = partial(process_box, A_eq=A_eq, b_eq=b_eq, 
                           sqr_pairs=sqr_pairs, cross_triplets=cross_triplets, 
                           threshold_s=threshold_s, threshold_v=threshold_v)
    # add_solution_ = partial(add_solution, sols=sharedsols)
    # log_solution_ = partial(log_solution, sols=sharedsols)
    # mockup_ = partial(mockup, sols=sharedsols, t1=True, t2=0.4)
    # inputs = [(sharedP, sharedsols) for i in range(4)]
    # inputs = [sharedP for i in range(8)]
    t2 = time.time()

    # batch = 128*n_processes
    while True:
        print('unexplored:',len(P))
        mapres = pool.map(process_box_, [P.popleft() for _ in range(np.min((batch,len(P))))])
        if mapres:
            for r in mapres:
                # result.extend(g2)
                if r[0] == 0:
                    pass
                elif r[0] == 1:
                    sols.append(r[1])
                    # print('sol')
                else:
                    for i in range(1,len(r)):
                        P.append(r[i])
                    # print('split')
            # time.sleep(1)
        else:
            break

    # resultsA = []
    # q = collections.deque()
    # for i in range(16):
    #     q.append(pool.apply_async(mockup_)) # MUST USE (arg,), but callback isnt necessary , callback=log_solution
    #     if len(q) >= 16:
    #         q.popleft().get(timeout=10)
        # time.sleep(1)
    # while len(q):
    #     q.popleft().get(timeout=10)
    # resultsA = pool.map(mockup_, inputs)  
    # resultsA = pool.starmap(mockup_, inputs)
    pool.close()
    pool.join()
    print(f'Search took {time.time()-t2:.3f} seconds')

    print('-------------------')
    # print(sharedP)
    # print(len(sharedP))
    # print(sharedsols)
    print('unexplored:',len(P))
    print('solutions:',len(sols))
    return sols

def box_serial(boxes, A_eq, b_eq, 
                sqr_pairs=[], cross_triplets=[], 
                threshold_s=[0.1], threshold_v=0.95):
    """
    Single-threaded version of box approximation. Works slower than approximate_boxes(), but is more fair for comparison with box_parallel().

    Args:
        boxes (list): Initial bounding boxes.
        A_eq (np.ndarray): Matrix of linear equations.
        b_eq (np.ndarray): Vector rhs of linear equations.
        sqr_pairs (list, optional): list of pairs of indexes, binding variable m with its square variable m**2. Defaults to [].
        cross_triplets (list, optional): list of triplets of indexes, binding two variables m,n with its product variable m*n. Defaults to [].
        threshold_s (list|np.ndarray, optional): Vector of 1 or len(B) elements, defining min allowed length of the 
        longest (or corresponding to vector) side of the boxes. Defines which box is considered ready solution. Defaults to [0.1].
        threshold_v (float, optional): max allowed ratio of box volume after shrinking to volume before. Defines when to stop shrinking a particular box. Defaults to 0.95.

    Returns:
        list: all solutions' bounding boxes.
    """
    P = collections.deque(boxes)
    sols = []
    print(f'Starting single-threaded box approximation.')
    print('-------------------') 
    process_box_ = partial(process_box, A_eq=A_eq, b_eq=b_eq, 
                           sqr_pairs=sqr_pairs, cross_triplets=cross_triplets, 
                           threshold_s=threshold_s, threshold_v=threshold_v)
    t2 = time.time()

    while len(P) > 0:
        print('unexplored:',len(P))
        B_c = P.popleft()

        r = process_box_(B_c)

        # result.extend(g2)
        if r[0] == 0:
            pass
        elif r[0] == 1:
            sols.append(r[1])
            # print('sol')
        else:
            for i in range(1,len(r)):
                P.append(r[i])
            # print('split')    

    print(f'Search took {time.time()-t2:.3f} seconds')

    print('-------------------')
    # print(sharedP)
    # print(len(sharedP))
    # print(sharedsols)
    print('unexplored:',len(P))
    print('solutions:',len(sols))
    return sols

if __name__ == '__main__':
    xb = np.array([-0.75,0.75])
    yb = np.array([-1.25,1.25])
    # xb = np.array([-.2,.2])
    # yb = np.array([-.2,.2])

    B = np.asarray([xb,yb,pow_interval(xb,2),pow_interval(yb,2),pow_interval(yb,4)])

    A_eq = np.array([[0,0,1,-1,1]]) #колво равенств
    b_eq = np.array([0]) # правые части равенств

    sqr_pairs = [(0,2), (1,3), (3,4)]
    cross_triplets = []


    threshold_s = np.array([0.005, 0.005, 1, 1, 1])
    # threshold_s = [0.1, 0.1, 1, 1, 1]
    P = [B]

    sols = box_parallel(P, A_eq, b_eq, sqr_pairs, cross_triplets, threshold_s,
                        n_processes=4,batch=10096) #4 proc a bit better than 2, and 10k batch better 4k or 256 

    # for s in sols:
    #     print(s)
    # print(len(sols))

    delta = 0.025
    xrange = np.arange(-5.0, 20.0, delta)
    yrange = np.arange(-5.0, 20.0, delta)
    X, Y = np.meshgrid(xrange,yrange)

    F = Y**4
    G = Y**2 - X**2

    fig = plt.figure(dpi=150)
    # plt.contour(X, Y, (F - G), [0])
    ax = plt.gca()
    # ax = fig.axes[0]

    for b in sols:#[:4]:
        # print(b)
        xlb, xub = b[0,0], b[0,1]
        ylb, yub = b[1,0], b[1,1]
        ax.add_patch(Rectangle((xlb,ylb),xub-xlb,yub-ylb,linewidth=1/2,edgecolor='r',facecolor='none'))

    plt.xlim([-0.75,0.75])
    plt.ylim([-1.25,1.25])
    # plt.xlim([-.2,.2])
    # plt.ylim([-.2,.2])

    ax.set_aspect(1)
    plt.tight_layout()

    plt.show()