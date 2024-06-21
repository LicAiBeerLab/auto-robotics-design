import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def rotation_matrix(th):
    return np.array([[np.cos(th), -np.sin(th)],
                     [np.sin(th), np.cos(th)]])

class Ellipse:
    def __init__(self, p_center:np.ndarray, angle:float, axis:np.ndarray) -> None:
        self.p_center: np.ndarray = p_center
        self.angle: float = angle
        self.axis: np.ndarray = axis
    
    def get_points(self, step=0.1):
        E = np.linalg.inv(np.diag(self.axis)**2)
        R = rotation_matrix(-self.angle)
        En = R.T @ E @ R
        t = np.arange(0, 2*np.pi, step)
        y = np.vstack([np.cos(t), np.sin(t)])
        x = sp.linalg.sqrtm(np.linalg.inv(En)) @ y
        x[0,:] = x[0,:] + self.p_center[0]
        x[1,:] = x[1,:] + self.p_center[1]
        return x
    



def check_points_in_ellips(points: np.ndarray, ellipse: Ellipse, tolerance = 0.2):
    # https://en.wikipedia.org/wiki/Ellipse
    a = ellipse.axis[0] * (1 + tolerance)
    b = ellipse.axis[1] * (1 + tolerance)
    ang = ellipse.angle
    x0 = ellipse.p_center[0]
    y0 = ellipse.p_center[1]
    
    A = a**2 * np.sin(ang)**2 + b**2 * np.cos(ang)**2
    B = 2*(b**2 - a**2)*np.sin(ang)*np.cos(ang)
    C = a**2 * np.cos(ang)**2 + b**2 * np.sin(ang)**2
    D = -2 * A * x0 - B * y0
    E = -B * x0 - 2*C*y0
    F = A*x0**2 + B*x0*y0 + C*y0**2 - a**2*b**2
    
    ellps_impct_func = lambda point: A*point[0]**2 + C*point[1]**2 + B*np.prod(point) + D*point[0] + E*point[1] + F
    
    if points.size == 2:
        check = np.zeros(1, dtype="bool")
        check[0] = True if ellps_impct_func(points) < 0 else False
    else:
        check = np.zeros(points.shape[1], dtype="bool")
        for i in range(points.shape[1]):
            check[i] = True if ellps_impct_func(points[:,i]) < 0 else False
    return check

if __name__=="__main__":
# def plot_ellipse(ellipse):
    ellipse = Ellipse(np.array([-4,2]), np.deg2rad(45), np.array([1, 2]))
    point_ellipse = ellipse.get_points()
    
    points_x = np.linspace(-5, 5, 50)
    points_y = np.linspace(-5, 5, 50)
    xv, yv = np.meshgrid(points_x, points_y)
    points = np.vstack([xv.flatten(), yv.flatten()])
    mask = check_points_in_ellips(points, ellipse, 0.2)
    rev_mask = np.array(1-mask, dtype="bool")
    plt.figure(figsize=(10,10))
    plt.plot(point_ellipse[0,:], point_ellipse[1,:], "g", linewidth=3)
    plt.scatter(points[:,rev_mask][0],points[:,rev_mask][1])
    plt.scatter(points[:,mask][0],points[:,mask][1])
    plt.show()
    