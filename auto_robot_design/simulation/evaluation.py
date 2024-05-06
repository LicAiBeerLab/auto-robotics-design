
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


def compare_power_quality(time: np.ndarray, power_init: np.ndarray, power_opt: np.ndarray, plot=False):
    """
    Evaluate the power quality of the robot
    Args:
        time (np.ndarray): time (s)
        power (np.ndarray): power consumption (W)
        plot (bool): power plot in power space and power over time
    Returns:
        float: power quality
    """
    
    PQ_old = np.zeros((power_init.shape[0], 1))
    PQ_new = np.zeros((power_opt.shape[0], 1))
    
    for i in range(power_init.shape[0]):
        PQ_old[i] = np.sum(power_init[i])**2 - np.sum(power_init[i]**2)
        PQ_new[i] = np.sum(power_opt[i])**2 - np.sum(power_opt[i]**2)
        
    if plot:
        fig = plt.figure(figsize=(7, 7), layout='constrained')
        axs = fig.subplot_mosaic([["power_old", "power_new"],
                                ["PQ", "PQ"]])
        axs["power_old"].plot(time, power_init[:, 0], label='P_1', linewidth=2)
        axs["power_old"].plot(time, power_init[:, 1], label='P_2')
        axs["power_old"].set_xlim([time[0], time[-1]])
        axs["power_old"].set_title('Initial')
        axs["power_old"].set_xlabel('Time (s)')
        axs["power_old"].set_ylabel('Power (W)')
        axs["power_old"].legend()
        axs["power_old"].grid()
        axs["power_new"].plot(time, power_opt[:, 0], label='P_1', linewidth=2)
        axs["power_new"].plot(time, power_opt[:, 1], label='P_2')
        axs["power_new"].set_xlim([time[0], time[-1]])
        axs["power_new"].set_title('Optimized')
        axs["power_new"].set_xlabel('Time (s)')
        axs["power_new"].set_ylabel('Power (W)')
        axs["power_new"].legend()
        axs["power_new"].grid()
    
        axs["PQ"].plot(time, PQ_old, label='Initial', linewidth=2)
        axs["PQ"].plot(time, PQ_new, label='Optimized')
        axs["PQ"].set_xlim([time[0], time[-1]])
        axs["PQ"].set_xlabel('Time (s)')
        axs["PQ"].set_ylabel('Power Quality')
        axs["PQ"].legend()
        axs["PQ"].grid()
        
        plt.figure()
        plt.plot(power_init[:, 0], power_init[:, 1], label='Initial')
        plt.plot(power_opt[:, 0], power_opt[:, 1], label='Optimized')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.xlabel('P_1')
        plt.ylabel('P_2')
        plt.grid()
        plt.axis('equal')
        plt.legend()
        plt.show()
        
    return np.mean(PQ_old), np.mean(PQ_new)

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


def compare_movments_in_xz_plane(time: np.ndarray, x_init: np.ndarray, x_opt: np.ndarray, des_x: np.ndarray, plot=False):
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
    
    error_old = np.zeros((x_init.shape[0], 1))
    error_new = np.zeros((x_opt.shape[0], 1))
    
    for i in range(x_init.shape[0]):
        error_old[i] = np.linalg.norm((x_init[i] - des_x[i]))
        error_new[i] = np.linalg.norm((x_opt[i] - des_x[i]))
        
    if plot:
        plt.figure(figsize=(10, 10))
        plt.subplot(3, 2, 1)
        plt.plot(time, des_x[:, 0], label='des', linestyle='--', linewidth=3)
        plt.plot(time, x_init[:, 0], label='real')
        plt.title('Initial')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('X (m)')
        plt.grid()
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(time, des_x[:, 0], label='des', linestyle='--', linewidth=3)
        plt.plot(time, x_opt[:, 0], label='real')
        plt.xlim([time[0], time[-1]])
        plt.title('Optimized')
        plt.xlabel('Time (s)')
        plt.ylabel('X (m)')
        plt.grid()
        plt.legend()
    
        plt.subplot(3, 2, 3)
        plt.plot(time, des_x[:, 2], label='des', linestyle='--', linewidth=3)
        plt.plot(time, x_init[:, 2], label='real')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Z (m)')
        plt.grid()
        plt.legend()
        
        plt.subplot(3, 2, 4)
        plt.plot(time, des_x[:, 2], label='des', linestyle='--', linewidth=3)
        plt.plot(time, x_opt[:, 2], label='real')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Z (m)')
        plt.grid()
        plt.legend()
        
        plt.subplot(3, 2, (5, 6))
        plt.plot(time, error_old, label='error_init')
        plt.plot(time, error_new, label='error_opt')
        plt.xlim([time[0], time[-1]])
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.legend()
        plt.grid()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(des_x[:, 0], des_x[:, 2], label='des', linestyle='--', linewidth=3)
        plt.plot(x_init[:, 0], x_init[:, 2], label="real")
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.grid()
        plt.legend()
        plt.axis('equal')
        plt.subplot(1, 2, 2)
        plt.plot(des_x[:, 0], des_x[:, 2], label='des', linestyle='--', linewidth=3)
        plt.plot(x_opt[:, 0], x_opt[:, 2], label="real")
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.grid()
        plt.legend()
        plt.axis('equal')
        plt.show()
        
    return np.mean(error_old), np.mean(error_new)

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


def compare_torque_evaluation(time: np.ndarray, torque_init: np.ndarray, torque_opt, plot = False):
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
        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        for i in range(torque_init.shape[1]):
            plt.plot(time, torque_init[:, i], label='tau_' + str(i))
        plt.xlim([time[0], time[-1]])
        plt.title('Initial')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.grid()
        plt.legend()
        plt.subplot(1,2,2)
        for i in range(torque_opt.shape[1]):
            plt.plot(time, torque_opt[:, i], label='tau_' + str(i))
        plt.xlim([time[0], time[-1]])
        plt.title('Optimized')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.grid()
        plt.legend()
        plt.show()
    
    return np.max(np.abs(torque_init), axis=0), np.max(np.abs(torque_opt), axis=0)