import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def convert_x_y_to_6d_traj(x: np.ndarray, y: np.ndarray):
    traj_6d = np.zeros((len(x), 6), dtype=np.float64)
    traj_6d[:, 0] = x
    traj_6d[:, 1] = y
    return traj_6d


def convert_x_y_to_6d_traj_xz(x: np.ndarray, y: np.ndarray):
    traj_6d = np.zeros((len(x), 6), dtype=np.float64)
    traj_6d[:, 0] = x
    traj_6d[:, 2] = y
    return traj_6d


def simple_traj_derivative(traj_6d: np.ndarray, dt: float = 0.001):
    traj_6d_v = np.zeros(traj_6d.shape)
    traj_6d_v[1:, :] = np.diff(traj_6d, axis=0)/dt #(traj_6d[1:, :] - traj_6d[:-1, :])/dt
    return traj_6d_v


def get_simple_spline():
    # Sample data points
    x = np.array([-0.5, 0, 0.5])
    y = np.array([-1.02, -0.8, -1.02])
    #y = y - 0.5
    #x = x + 0.4
    # Create the cubic spline interpolator
    cs = CubicSpline(x, y)

    # Create a dense set of points where we evaluate the spline
    x_traj_spline = np.linspace(x.min(), x.max(), 75)
    y_traj_spline = cs(x_traj_spline)

    # Plot the original data points
    # plt.plot(x, y, 'o', label='data points')

    # Plot the spline interpolation
    # plt.plot(x_traj_spline, y_traj_spline, label='cubic spline')

    # plt.legend()
    # plt.show()
    return (x_traj_spline, y_traj_spline)

def get_vertical_trajectory(n_points = 50):
    max_height = -1.1
    min_height = -0.6
    x_trajectory = np.zeros(n_points)
    x_trajectory+=-0.2
    y_trajectory = np.linspace(max_height, min_height, n_points)
    return (x_trajectory, y_trajectory)

if __name__ =="__main__":
    print(get_vertical_trajectory())