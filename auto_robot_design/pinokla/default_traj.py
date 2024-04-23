import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy.linalg as la


def calculate_quantic_polynom_by3point(t0, t1, tf, rhs):
    A = np.array(
        [
            [t0**i for i in range(6)],
            [i * t0 ** np.abs(i - 1) for i in range(6)],
            [t1**i for i in range(6)],
            [tf**i for i in range(6)],
            [i * tf ** np.abs(i - 1) for i in range(6)],
            [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
        ]
    )
    x = la.solve(A, rhs)
    if np.isclose(A@x, rhs).all():
        return x
    else:
        raise ValueError("Failed to solve the system for quantic polynom")


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
    traj_6d_v[1:, :] = (
        np.diff(traj_6d, axis=0) / dt
    )  # (traj_6d[1:, :] - traj_6d[:-1, :])/dt
    return traj_6d_v


def get_simple_spline(num_points = 75):
    # Sample data points
    x = np.array([-0.5, 0, 0.25]) * 0.5
    y = np.array([-0.4, -0.1, -0.4]) * 0.5
    y = y - 0.7
    # x = x + 0.4
    # Create the cubic spline interpolator
    cs = CubicSpline(x, y)

    # Create a dense set of points where we evaluate the spline
    x_traj_spline = np.linspace(x.min(), x.max(), num_points)
    y_traj_spline = cs(x_traj_spline)

    # Plot the original data points
    # plt.plot(x, y, 'o', label='data points')

    # Plot the spline interpolation
    # plt.plot(x_traj_spline, y_traj_spline, label='cubic spline')

    # plt.legend()
    # plt.show()
    return (x_traj_spline, y_traj_spline)


def get_trajectory(q_via_points, time_end: float = 1.0, dt: float = 0.1):
    
    q_traj = lambda t, b: np.sum(np.array([t**i for i in range(6)]).T * b, axis=1)
    dq_traj = lambda t, b: np.sum(np.array([i * t ** np.abs(i - 1) for i in range(6)]).T * b, axis=1)
    ddq_traj = lambda t, b: np.sum(np.array([np.zeros_like(t), np.zeros_like(t), np.ones_like(t)*2, 6 * t, 12 * t**2, 20 * t**3]).T * b, axis=1)
    
    v0 = 0
    vf = 0
    alpf = 0
    polynom_coefs = []
    # Sample data points
    for q in q_via_points:
        assert len(q) == 3
        b = np.array([q[0], v0, q[1], q[2], vf, alpf])
        polynom_coefs.append(calculate_quantic_polynom_by3point(0, time_end / 2, time_end, b))

    # Create a dense set of points where we evaluate the spline
    time_arr = np.arange(0, time_end + dt, dt)
    q_traj_arr = np.zeros((time_arr.size, np.array(q_via_points).shape[0]))
    dq_traj_arr = np.zeros((time_arr.size, np.array(q_via_points).shape[0]))
    ddq_traj_arr = np.zeros((time_arr.size, np.array(q_via_points).shape[0]))
    
    for i, b in enumerate(polynom_coefs):
        q_traj_arr[:, i] = q_traj(time_arr, b)
        dq_traj_arr[:, i] = dq_traj(time_arr, b)
        ddq_traj_arr[:, i] = ddq_traj(time_arr, b)
    # Plot the original data points
    
    # Plot the spline interpolation
    plt.figure()
    for i in range(np.array(q_via_points).shape[0]):
        plt.subplot(np.array(q_via_points).shape[1], 3, np.array(q_via_points).shape[1]*i+1)
        plt.plot(time_arr, q_traj_arr[:, i])
        plt.ylabel(f"q{i}")
        plt.grid()
        plt.xlim([0, time_end])
        plt.subplot(np.array(q_via_points).shape[1], 3, np.array(q_via_points).shape[1]*i+2)
        plt.plot(time_arr, dq_traj_arr[:, i])
        plt.ylabel(f"dq{i}")
        plt.grid()
        plt.xlim([0, time_end])
        plt.subplot(np.array(q_via_points).shape[1], 3, np.array(q_via_points).shape[1]*i+3)
        plt.plot(time_arr, ddq_traj_arr[:, i])
        plt.ylabel(f"ddq{i}")
        plt.grid()
        plt.xlim([0, time_end])
    plt.show()
    return (time_arr, q_traj_arr, dq_traj_arr, ddq_traj_arr)


def get_vertical_trajectory(n_points=50):
    max_height = -1.0
    min_height = -0.6
    x_trajectory = np.zeros(n_points)
    y_trajectory = np.linspace(max_height, min_height, n_points)
    return (x_trajectory, y_trajectory)


if __name__ == "__main__":
    q_via_points = [[0, 0.1, 0.2], [0.5, 0.3, 1], [0.5, 0.7, 0.2]]
    get_trajectory(q_via_points, dt = 0.01)