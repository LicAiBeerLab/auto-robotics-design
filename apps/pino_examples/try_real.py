import collections
from multiprocessing import Pool, freeze_support, Manager, cpu_count
# import multiprocessing
import time
from functools import partial

import numpy as np

def process_box(B_c,threshold_s=[0.1]):
    # if len(P) < 1:
    #     print('Pempty!!!!!!!!!!!!!')
    #     return
    # B_c = P.pop()

    is_sigma_simple = len(threshold_s) == 1
    
    lengths_c = B_c[:,1] - B_c[:,0]
    longest_dim_ind = np.argmax(lengths_c)
    if is_sigma_simple:
        is_small_enough = lengths_c[longest_dim_ind] <= threshold_s[0]
    else:
        is_small_enough = (lengths_c <= threshold_s).all()

    # c=0
    # for i in range(0, 99999):
    #     c += 1

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
        # P.append(B1)
        # P.append(B2)

    return (0,)
    
def box_serial(boxes,threshold_s=[0.1]):
    # P = [B]
    # P = boxes.copy()
    P = collections.deque(boxes)
    sols = []

    while len(P) > 0:
        print('unexplored:',len(P))
        B_c = P.popleft() #0

        # print('P leng',len(P))

        r = process_box(B_c, threshold_s)

        # result.extend(g2)
        if r[0] == 0:
            pass
        elif r[0] == 1:
            sols.append(r[1])
            # print('sol')
        else:
            P.append(r[1])
            P.append(r[2])
            # print('split')
    
    return sols

def box_parallel(boxes,threshold_s=[0.1],n_processes=2,batch=256):
    freeze_support()
    # manager=Manager()
    
    # data = [i for i in range(3,11)]

    # sols=[]
    # P = [i for i in range(20,200,20)]

    # n_processes = cpu_count()//2
    pool = Pool(n_processes)
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
    boxes_ = partial(process_box,threshold_s=threshold_s)
    # add_solution_ = partial(add_solution, sols=sharedsols)
    # log_solution_ = partial(log_solution, sols=sharedsols)
    # mockup_ = partial(mockup, sols=sharedsols, t1=True, t2=0.4)
    # inputs = [(sharedP, sharedsols) for i in range(4)]
    # inputs = [sharedP for i in range(8)]
    t2 = time.time()

    # batch = 128*n_processes
    while True:
        print('unexplored:',len(P))
        mapres = pool.map(boxes_, [P.popleft() for _ in range(np.min((batch,len(P))))])
        if mapres:
            for r in mapres:
                # result.extend(g2)
                if r[0] == 0:
                    pass
                elif r[0] == 1:
                    sols.append(r[1])
                    # print('sol')
                else:
                    P.append(r[1])
                    P.append(r[2])
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

def log_solution(s):
    # print(s,'sols',sols)
    print(s)
    # sols.append(s)

if __name__ == '__main__':
    bnds = np.array([-5.,5.])
    B = np.asarray([bnds]*2)
    P = [B]

    # print('shapeB',B.shape)

    res = box_parallel(P,[0.1], batch=256) #2(no more) 256(no less) is the best, can be 2s vs 11.5 in serial (WITHOUT DELAY)


    # results = []
    # answers = []
    # for d in data:
    #     result1 = pool.apply_async(solve1, [d])
    #     results.append(result1)
    # # freeze_support()
    # t0 = time.time()
    # for r in results:
    #     answer1 = r.get(timeout=1)
    #     answers.append(answer1)    
    # print(time.time()-t0)
    # print(answers)

    # locP = []
    # # for i in range(20,400,20):
    # #     locP.append(i)
    # locP.append(B)
    # t1 = time.time()
    # res = []
    # for i in range(16):
    #     aa = process_box(P=locP, sols=res, threshold_s=[0.5])
    #     # res.append(aa)
    # print(time.time()-t1)
    # print(res)

    # t1 = time.time()
    # res = box_serial(P,[0.1])
    # print(time.time()-t1)
    # print(len(res))
