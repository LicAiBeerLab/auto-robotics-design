import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def box_vertices(o, size=(1,1,1)):
    """
     Returns 6 lists of vertices (for each face of a box)
     
     Args:
     	 o (tuple|list|np.ndarray): Coordinates of the box's corner in x y z
     	 size (tuple|list|np.ndarray): Size of the box in x y z positive directions from the corner
     
     Returns: 
     	 (np.ndarray) coordinates of each face of a box
    """
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

def prepare_boxes_plot(positions,sizes=None,colors=None, **kwargs):
    """
     Prepare a 3D collection of boxes for plotting
     
     Args:
     	 positions (list|np.ndarray): positions of each box's corner in x y z
     	 sizes (list|np.ndarray): sizes of each box in x y z positive directions from the corner
     	 colors (list|np.ndarray): colors to use for each box
     
     Returns: 
     	 (Poly3DCollection) plottable bunch of boxes
    """
    if not isinstance(colors,(list,np.ndarray)): colors=["b"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( box_vertices(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6), **kwargs)

def mul_intervals(interval1, interval2):
    """
     Multiply two intervals.
     
     Args:
     	 interval1 (tuple|list|np.ndarray): First interval to be multiplied
     	 interval2 (tuple|list|np.ndarray): Second interval to be multiplied
     
     Returns: 
     	 (np.ndarray) resulting interval.
    """
    a1,a2 = interval1[0], interval1[1]
    b1,b2 = interval2[0], interval2[1]
    return np.asarray((np.min([a1*b1,a1*b2,a2*b1,a2*b2]), np.max([a1*b1,a1*b2,a2*b1,a2*b2])))

def subtr_intervals(interval1, interval2):
    """
     Subtract two intervals.
     
     Args:
     	 interval1 (tuple|list|np.ndarray): Interval to subtride from.
     	 interval2 (tuple|list|np.ndarray): Interval to subtride from the first.
     
     Returns: 
     	 (np.ndarray) resulting interval.
    """
    a1,a2 = interval1[0], interval1[1]
    b1,b2 = interval2[0], interval2[1]
    return np.asarray((a1-b2,a2-b1))

def neg_interval(interval):
    """
     Negates an interval.
     
     Args:
     	 interval (tuple|list|np.ndarray): tuple, list or np.ndarray of two floats.
     
     Returns: 
     	 (np.ndarray) resulting interval.
    """
    return np.asarray((-interval[1],-interval[0]))

def pow_interval(interval, p: int):
    """
     Returns the p-th power of the interval.
     
     Args:
     	 interval (tuple|list|np.ndarray): A 2-tuple of floats that is the interval to be raised to the power of p.
     	 p (int): A positive integer - the power the interval is raised to.
     
     Returns: 
     	 (np.ndarray) resulting interval.
    """
    is_odd = p%2>0
    if is_odd:
        return (interval[0]**p, interval[1]**p)
    else:
        is_zerocross = interval[0]*interval[1]<=0
        return np.asarray((0 if is_zerocross else np.min(abs(interval))**p, 
                           np.max(abs(interval))**p))
    