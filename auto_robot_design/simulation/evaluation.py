
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern Serif",
})

def power_quality(time: np.ndarray, power: np.ndarray, plot=False):
    """
    Evaluate the power quality of the robot
    Args:
        time (np.ndarray): time (s)
        power (np.ndarray): power consumption (W)
        plot (bool): power plot in power space and power over time
    Returns:
        float: power quality
    """
    
    PQ = np.zeros((power.shape[0], 1))
    
    for i in range(power.shape[0]):
        PQ[i] = np.sum(power[i])**2 - np.sum(power[i]**2)
        
    if plot:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(time, power[:, 0], label='$P_1$', linewidth=2)
        plt.plot(time, power[:, 1], label='$P_2$')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(time, PQ)
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Power Quality')
        plt.grid()
        plt.figure()

        plt.plot(power[:, 0], power[:, 1])
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.xlabel('P_1')
        plt.ylabel('P_2')
        plt.grid()
        plt.axis('equal')
        plt.show()
        
    return np.mean(PQ)


def compare_power_quality(time: np.ndarray, power_arrs, plot=False, path=None):
    """
    Evaluate the power quality of the robot
    Args:
        time (np.ndarray): time (s)
        power (np.ndarray): power consumption (W)
        plot (bool): power plot in power space and power over time
    Returns:
        float: power quality
    """
    
    PQ = np.zeros((len(power_arrs), power_arrs[0].shape[0]))
    
    
    
    for j in range(len(power_arrs)):
        for i in range(power_arrs[0].shape[0]):
            PQ[j,i] = np.sum(power_arrs[j][i])**2 - np.sum(power_arrs[j][i]**2)
        
    if plot:
        plot_power_name = ["Design " + str(i) for i in range(1, 1+len(power_arrs))]
        fig = plt.figure(figsize=(len(power_arrs)*6, 6))
        axs = fig.subplot_mosaic([plot_power_name])
        for name, power in zip(plot_power_name, power_arrs):
            axs[name].plot(time, power[:, 0], label='$P_1$', linewidth=2)
            axs[name].plot(time, power[:, 1], label='$P_2$')
            axs[name].set_xlim([time[0], time[-1]])
            axs[name].set_title(name)
            axs[name].set_xlabel('Time (s)')
            axs[name].set_ylabel('Power (W)')
            axs[name].legend()
            axs[name].grid()
        if path is not None:
            plt.savefig(path + "power.pdf")
        plt.figure(figsize=(6,6))
        for i in range(len(power_arrs)):
            plt.plot(time, PQ[i, :], label='Design ' + str(i+1))
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Power Quality')
        plt.legend()
        plt.grid()
        if path is not None:
            plt.savefig(path + "power_quality.pdf")

        plt.figure()
        for i in range(len(power_arrs)):
            plt.plot(power_arrs[i][:, 0], power_arrs[i][:, 1], label='Design ' + str(i+1))
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.xlabel('$P_1$')
        plt.ylabel('$P_2$')
        plt.grid()
        plt.axis('equal')
        plt.legend()
        if path is not None:
            plt.savefig(path + "power_space.pdf")
        else:  
            plt.show()
        
    sum_powers = np.sum(power_arrs, axis=2)
    sum_abs_powers = np.sum(np.abs(power_arrs), axis=2)
    return np.mean(sum_abs_powers, axis=1), np.sum(sum_powers, axis=1), np.sum(sum_abs_powers,axis=1)

def movments_in_xz_plane(time: np.ndarray, x: np.ndarray, des_x: np.ndarray, plot=False):
    """
    Evaluate the movements in the xz plane
    Args:
        time (np.ndarray): time (s)
        x (np.ndarray): posiotion from simulation
        des_x (np.ndarray): desired position
        plot (bool): plot the movements
    Returns:
        float: error tracking trajectory in the xz plane
    """
    
    error = np.zeros((x.shape[0], 1))
    
    for i in range(x.shape[0]):
        error[i] = np.linalg.norm((x[i] - des_x[i]))
        
    if plot:
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(time, des_x[:, 0], label='ref', linestyle='--', linewidth=3)
        plt.plot(time, x[:, 0], label='actual')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('X (m)')
        plt.grid()
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(time, des_x[:, 2], label='ref', linestyle='--', linewidth=3)
        plt.plot(time, x[:, 2], label='actual')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Z (m)')
        plt.grid()
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(time, error)
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.grid()
        plt.figure()
        plt.plot(des_x[:, 0], des_x[:, 2], label='ref', linestyle='--', linewidth=3)
        plt.plot(x[:, 0], x[:, 2], label="actual")
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.grid()
        plt.legend()
        plt.axis('equal')
        plt.show()
        
    return np.mean(error)


def compare_movments_in_xz_plane(time: np.ndarray, x, des_x: np.ndarray, plot=False, path=None):
    """
    Evaluate the movements in the xz plane
    Args:
        time (np.ndarray): time (s)
        x (np.ndarray): posiotion from simulation
        des_x (np.ndarray): desired position
        plot (bool): plot the movements
    Returns:
        float: error tracking trajectory in the xz plane
    """
    
    error_arrs = np.zeros((len(x), x[0].shape[0]))
    
    for j in range(len(x)):
        for i in range(x[0].shape[0]):
            error_arrs[j, i] = np.linalg.norm((x[j][i] - des_x[i]))
        
    if plot:
        plot_name = ["Design " + str(i) for i in range(1, 1+len(x))]
        fig = plt.figure(figsize=(len(x)*8, 10))
        axs = fig.subplot_mosaic([[name + "_X" for name in plot_name],
                                  [name + "_Z" for name in plot_name]])
        for name, x_arr in zip(plot_name, x):
            axs[name + "_X"].set_title(name)
            axs[name + "_X"].plot(time, des_x[:, 0], label='ref', linestyle='--', linewidth=3)
            axs[name + "_X"].plot(time, x_arr[:, 0], label='actual')
            axs[name + "_X"].set_xlim([time[0], time[-1]])
            axs[name + "_X"].set_xlabel('Time (s)')
            axs[name + "_X"].set_ylabel('X (m)')
            axs[name + "_X"].legend()
            axs[name + "_X"].grid()
            axs[name + "_Z"].plot(time, des_x[:, 2], label='ref', linestyle='--', linewidth=3)
            axs[name + "_Z"].plot(time, x_arr[:, 2], label='actual')
            axs[name + "_Z"].set_xlim([time[0], time[-1]])
            axs[name + "_Z"].set_xlabel('Time (s)')
            axs[name + "_Z"].set_ylabel('Z (m)')
            axs[name + "_Z"].legend()
            axs[name + "_Z"].grid()
        if path is not None:
            plt.savefig(path + "trajectory.pdf")
        plt.figure(figsize=(6,6))
        for i in range(len(x)):
            plt.plot(time, error_arrs[i, :], label='Design ' + str(1+i))
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.legend()
        plt.grid()
        
        if path is not None:
            plt.savefig(path + "error.pdf")

        fig = plt.figure(figsize=(len(x)*6, 6))
        axs = fig.subplot_mosaic([[name + "_xz" for name in plot_name]])
        for name, x_arr in zip(plot_name, x):
            axs[name + "_xz"].set_title(name)
            axs[name + "_xz"].plot(des_x[:, 0], des_x[:, 2], label='ref', linestyle='--', linewidth=3)
            axs[name + "_xz"].plot(x_arr[:, 0], x_arr[:, 2], label="actual")
            axs[name + "_xz"].set_xlabel('X (m)')
            axs[name + "_xz"].set_ylabel('Z (m)')
            axs[name + "_xz"].legend()
            axs[name + "_xz"].grid()
            axs[name + "_xz"].axis('equal')
        
        if path is not None:
            plt.savefig(path + "xz_plane.pdf")
        else:
            plt.show()
        
    return np.mean(error_arrs, axis=1)

def torque_evaluation(time: np.ndarray, torque: np.ndarray, plot = False):
    """
    Evaluate the torque
    Args:
        time (np.ndarray): time (s)
        torque (np.ndarray): torque
        plot (bool): plot the torque
    Returns:
        float: torque evaluation
    """
    if plot:
        plt.figure()
        for i in range(torque.shape[1]):
            plt.plot(time, torque[:, i], label=r'$\tau_' + str(i) + r'$')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.grid()
        plt.legend()
        plt.show()
    
    return np.max(np.abs(torque), axis=0)


def compare_torque_evaluation(time: np.ndarray, torque_arrs, plot = False, path=None):
    """
    Evaluate the torque
    Args:
        time (np.ndarray): time (s)
        torque (np.ndarray): torque
        plot (bool): plot the torque
    Returns:
        float: torque evaluation
    """
    
    if plot:
        plot_name = ["Design " + str(i) for i in range(1, 1+len(torque_arrs))]
        fig = plt.figure(figsize=(len(torque_arrs)*6, 6))
        
        axs = fig.subplot_mosaic([[name + "_tau" for name in plot_name]])
        
        for name, torque in zip(plot_name, torque_arrs):
            axs[name + "_tau"].set_title(name)
            for i in range(torque.shape[1]):
                axs[name + "_tau"].plot(time, torque[:, i], label=r'$\tau_' + str(i+1) + r'$')
            axs[name + "_tau"].set_xlim(time[0], time[-1])
            axs[name + "_tau"].set_xlabel('Time (s)')
            axs[name + "_tau"].set_ylabel('Torque (Nm)')
            axs[name + "_tau"].grid()
            axs[name + "_tau"].legend()
        if path is not None:
            plt.savefig(path + "torque.pdf")
        else:
            plt.show()
    
    return np.max(np.abs(torque_arrs), axis=1), np.mean(np.abs(torque_arrs), axis=1)