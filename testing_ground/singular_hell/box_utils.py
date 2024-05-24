import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def box_vertices(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plot_boxes(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["b"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( box_vertices(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6), **kwargs)

def mul_intervals(interval1, interval2):
    a1,a2 = interval1[0], interval1[1]
    b1,b2 = interval2[0], interval2[1]
    return np.asarray((np.min([a1*b1,a1*b2,a2*b1,a2*b2]), np.max([a1*b1,a1*b2,a2*b1,a2*b2])))

def subtr_intervals(interval1, interval2):
    a1,a2 = interval1[0], interval1[1]
    b1,b2 = interval2[0], interval2[1]
    return np.asarray((a1-b2,a2-b1))

def neg_interval(interval):
    return np.asarray((-interval[1],-interval[0]))

def pow_interval(interval, p: int):
    is_odd = p%2>0
    if is_odd:
        return (interval[0]**p, interval[1]**p)
    else:
        is_zerocross = interval[0]*interval[1]<=0
        return np.asarray((0 if is_zerocross else np.min(abs(interval))**p, 
                           np.max(abs(interval))**p))
    