import collections
from multiprocessing import Pool, freeze_support, Manager, cpu_count
# import multiprocessing
import time
from functools import partial
def solve1(a):
    c = 0
    for i in range(0, 9999999):
        c = c + 1
    return a+a
def mockup(P, sols, t1, t2):
    Bc = P.pop()
    c=0
    for i in range(0, 9999999):
        c += 1
    sols.append(Bc)
    return Bc
    
    # return
def log_solution(s):
    # print(s,'sols',sols)
    print(s)
    # sols.append(s)

if __name__ == '__main__':
    locP = []
    for i in range(20,400,20):
        locP.append(i)
    freeze_support()
    manager=Manager()
    
    # data = [i for i in range(3,11)]

    # sols=[]
    # P = [i for i in range(20,200,20)]

    pool = Pool(4)
    # with Pool(4) as pool: #cpu_count()
    sharedP=manager.list()
    sharedsols=manager.list()
    for i in range(20,400,20):
        sharedP.append(i)
    print(sharedP)
    print('-------------------')
    mockup_ = partial(mockup, P=sharedP, sols=sharedsols, t1=True, t2=0.4)
    # add_solution_ = partial(add_solution, sols=sharedsols)
    # log_solution_ = partial(log_solution, sols=sharedsols)
    # mockup_ = partial(mockup, sols=sharedsols, t1=True, t2=0.4)
    # inputs = [(sharedP, sharedsols) for i in range(4)]
    # inputs = [sharedP for i in range(8)]
    t2 = time.time()
    resultsA = []
    q = collections.deque()
    for i in range(16):
        q.append(pool.apply_async(mockup_)) # MUST USE (arg,), but callback isnt necessary , callback=log_solution
        if len(q) >= 16:
            q.popleft().get(timeout=10)
    # while len(q):
    #     q.popleft().get(timeout=10)
    # resultsA = pool.map(mockup_, inputs)  
    # resultsA = pool.starmap(mockup_, inputs)
    pool.close()
    pool.join()
    print(time.time()-t2)


    print('-------------------')
    print(sharedP)
    print(sharedsols)
    # print(resultsA)

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

    # t1 = time.time()
    # res = []
    # for i in range(16):
    #     aa = mockup(P=locP, sols=res, t1=True, t2=0.4)
    #     # res.append(aa)
    # print(time.time()-t1)
    # print(res)