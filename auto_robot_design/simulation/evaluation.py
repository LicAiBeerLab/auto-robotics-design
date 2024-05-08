
import numpy as np
import matplotlib.pyplot as plt

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
        plt.plot(time, power[:, 0], label='P_1', linewidth=2)
        plt.plot(time, power[:, 1], label='P_2')
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


def compare_power_quality(time: np.ndarray, power_arrs, plot=False):
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
        plot_power_name = ["Design " + str(i) for i in range(len(power_arrs))]
        fig = plt.figure(figsize=(len(power_arrs)*6, 10))
        axs = fig.subplot_mosaic([plot_power_name,
                                ["PQ" for i in range(len(power_arrs))]])
        for name, power in zip(plot_power_name, power_arrs):
            axs[name].plot(time, power[:, 0], label='P_1', linewidth=2)
            axs[name].plot(time, power[:, 1], label='P_2')
            axs[name].set_xlim([time[0], time[-1]])
            axs[name].set_title(name)
            axs[name].set_xlabel('Time (s)')
            axs[name].set_ylabel('Power (W)')
            axs[name].legend()
            axs[name].grid()
        for i in range(len(power_arrs)):
            axs["PQ"].plot(time, PQ[i, :], label='PQ_' + str(i))
        axs["PQ"].set_xlim([time[0], time[-1]])
        axs["PQ"].set_xlabel('Time (s)')
        axs["PQ"].set_ylabel('Power Quality')
        axs["PQ"].legend()
        axs["PQ"].grid()
        
        plt.figure()
        for i in range(len(power_arrs)):
            plt.plot(power_arrs[i][:, 0], power_arrs[i][:, 1], label='Design ' + str(i))
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.xlabel('P_1')
        plt.ylabel('P_2')
        plt.grid()
        plt.axis('equal')
        plt.legend()
        plt.show()
        
    return np.mean(PQ, axis=1)

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
        plt.plot(time, des_x[:, 0], label='des', linestyle='--', linewidth=3)
        plt.plot(time, x[:, 0], label='real')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('X (m)')
        plt.grid()
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(time, des_x[:, 2], label='des', linestyle='--', linewidth=3)
        plt.plot(time, x[:, 2], label='real')
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
        plt.plot(des_x[:, 0], des_x[:, 2], label='des', linestyle='--', linewidth=3)
        plt.plot(x[:, 0], x[:, 2], label="real")
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.grid()
        plt.legend()
        plt.axis('equal')
        plt.show()
        
    return np.mean(error)


def compare_movments_in_xz_plane(time: np.ndarray, x, des_x: np.ndarray, plot=False):
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
        plot_name = ["Design " + str(i) for i in range(len(x))]
        fig = plt.figure(figsize=(len(x)*6, 10))
        axs = fig.subplot_mosaic([[name + "_X" for name in plot_name],
                                  [name + "_Z" for name in plot_name],
                                ["error" for i in range(len(x))]])
        for name, x_arr in zip(plot_name, x):
            axs[name + "_X"].plot(time, des_x[:, 0], label='des', linestyle='--', linewidth=3)
            axs[name + "_X"].plot(time, x_arr[:, 0], label='real')
            axs[name + "_X"].set_xlim([time[0], time[-1]])
            axs[name + "_X"].set_xlabel('Time (s)')
            axs[name + "_X"].set_ylabel('X (m)')
            axs[name + "_X"].legend()
            axs[name + "_X"].grid()
            axs[name + "_Z"].plot(time, des_x[:, 2], label='des', linestyle='--', linewidth=3)
            axs[name + "_Z"].plot(time, x_arr[:, 2], label='real')
            axs[name + "_Z"].set_xlim([time[0], time[-1]])
            axs[name + "_Z"].set_xlabel('Time (s)')
            axs[name + "_Z"].set_ylabel('Z (m)')
            axs[name + "_Z"].legend()
            axs[name + "_Z"].grid()

        for i in range(len(x)):
            axs["error"].plot(time, error_arrs[i, :], label='error_' + str(i))
        axs["error"].set_xlim([time[0], time[-1]])
        axs["error"].set_xlabel('Time (s)')
        axs["error"].set_ylabel('Error')
        axs["error"].legend()
        axs["error"].grid()
        

        plt.figure(figsize=(len(x)*6, 6))
        for i in range(len(x)):
            plt.plot(des_x[:, 0], des_x[:, 2], label='des', linestyle='--', linewidth=3)
            plt.plot(x[i][:, 0], x[i][:, 2], label="real")
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.grid()
        plt.legend()
        plt.axis('equal')
        
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
            plt.plot(time, torque[:, i], label='tau_' + str(i))
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.grid()
        plt.legend()
        plt.show()
    
    return np.max(np.abs(torque), axis=0)


def compare_torque_evaluation(time: np.ndarray, torque_arrs, plot = False):
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
        plot_name = ["Design " + str(i) for i in range(len(torque_arrs))]
        fig = plt.figure(figsize=(len(torque_arrs)*6, 6))
        
        axs = fig.subplot_mosaic([name + "_tau" for name in plot_name])
        
        for name, torque in zip(plot_name, torque_arrs):
            for i in range(torque.shape[1]):
                axs[name + "_tau"].plot(time, torque[:, i], label='tau_' + str(i))
            axs[name + "_tau"].set_xlim(time[0], time[-1])
            axs[name + "_tau"].set_xlabel('Time (s)')
            axs[name + "_tau"].set_ylabel('Torque (Nm)')
            axs[name + "_tau"].grid()
            axs[name + "_tau"].legend()
        plt.show()
    
    return np.max(np.abs(torque_arrs), axis=1), np.mean(np.abs(torque_arrs), axis=1)