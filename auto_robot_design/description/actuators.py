


from dataclasses import dataclass, field

import numpy as np


@dataclass
class Actuator:
    mass: float = 0.0
    peak_effort: float = 0.0
    peak_velocity: float = 0.0
    size: list[float] = field(default_factory=list)
    reduction_ratio: float = 0.0
    
    def get_max_effort(self):
        return self.peak_effort * 0.6
    
    def get_max_vel(self):
        max_vel_rads = self.peak_velocity * 2 * np.pi / 60
        return max_vel_rads * 0.6
    
    def torque_weight_ratio(self):
        return self.get_max_effort() / self.mass
    
@dataclass
class RevoluteActuator:
    mass: float = 0.0
    peak_effort: float = 0.0
    peak_velocity: float = 0.0
    size: list[float] = field(default_factory=list)
    reduction_ratio: float = 0.0
    
    def get_max_effort(self):
        return self.peak_effort * 0.6
    
    def get_max_vel(self):
        max_vel_rads = self.peak_velocity * 2 * np.pi / 60
        return max_vel_rads * 0.6
    
    def torque_weight_ratio(self):
        return self.get_max_effort() / self.mass
    
    def calculate_inertia(self):
        Izz = 1 / 2 * self.mass * self.size[0]
        Iyy = 1 / 12 * self.mass * self.size[1] + 1 / 4 * self.mass * self.size[0]
        Ixx = Iyy
        return np.diag([Ixx, Iyy, Izz])

@dataclass
class RevoluteUnit(RevoluteActuator):
    def __init__(self) -> None:
        self.mass = 0.1
        self.peak_effort = 1000
        self.peak_velocity = 100
        self.size = []

@dataclass
class CustomActuators_KG3(RevoluteActuator):
    def __init__(self):
            self.mass = 3
            self.peak_effort = 420
            self.peak_velocity = 320
            self.size = [0.06, 0.08]
            self.reduction_ratio = 1/9

@dataclass
class CustomActuators_KG2(RevoluteActuator):
    def __init__(self):
            self.mass = 2
            self.peak_effort = 180
            self.peak_velocity = 300
            self.size = [0.045, 0.06]
            self.reduction_ratio = 1/6

@dataclass
class CustomActuators_KG1(RevoluteActuator):
    def __init__(self):
            self.mass = 1
            self.peak_effort = 80
            self.peak_velocity = 220
            self.size = [0.048, 0.06]
            self.reduction_ratio = 1/10

@dataclass
class TMotor_AK10_9(RevoluteActuator):
    def __init__(self):
            self.mass = 0.960
            self.peak_effort = 48
            self.peak_velocity = 297.5
            self.size = [0.045, 0.062]
            self.reduction_ratio = 1/9

@dataclass
class TMotor_AK70_10(RevoluteActuator):
    def __init__(self):
            self.mass = 0.521
            self.peak_effort = 24.8
            self.peak_velocity = 382.5
            self.size = [0.0415, 0.05]
            self.reduction_ratio = 1/10

@dataclass
class TMotor_AK60_6(RevoluteActuator):
    def __init__(self):
            self.mass = 0.368
            self.peak_effort = 9
            self.peak_velocity = 285
            self.size = [0.034, 0.0395]
            self.reduction_ratio = 1/6

@dataclass
class TMotor_AK80_64(RevoluteActuator):
    def __init__(self):
            self.mass = 0.850
            self.peak_effort = 120
            self.peak_velocity = 54.6
            self.size = [0.0445, 0.062]
            self.reduction_ratio = 1/64

@dataclass
class TMotor_AK80_9(RevoluteActuator):
    def __init__(self):
            self.mass = 0.485
            self.peak_effort = 18
            self.peak_velocity = 475
            self.size = [0.0425, 0.0385]
            self.reduction_ratio = 1/9
            

@dataclass
class Unitree_GO_Motor(RevoluteActuator):
    def __init__(self):
            self.mass = 0.530
            self.peak_effort = 23.7
            self.peak_velocity = 30 / 2 / np.pi * 60
            self.size = [0.0478, 0.041]
            self.reduction_ratio = 1/6.33

@dataclass
class Unitree_B1_Motor(RevoluteActuator):
    def __init__(self):
            self.mass = 1.740
            self.peak_effort = 140
            self.peak_velocity = 297.5
            self.size = [0.0535, 0.074]
            self.reduction_ratio = 1/10

@dataclass
class Unitree_A1_Motor(RevoluteActuator):
    def __init__(self):
            self.mass = 0.605
            self.peak_effort = 33.5
            self.peak_velocity = 21 / 2 / np.pi * 60
            self.size = [0.0459, 0.044]
            self.reduction_ratio = 1/6.33

@dataclass
class MyActuator_RMD_MT_RH_17_100_N(RevoluteActuator):
    def __init__(self):
            self.mass = 0.640
            self.peak_effort = 25
            self.peak_velocity = 20
            self.size = [0.0405, 0.063]
            self.reduction_ratio = 1/100

t_motor_actuators = [
    TMotor_AK10_9(),
    TMotor_AK60_6(),
    TMotor_AK70_10(),
    TMotor_AK80_64(),
    TMotor_AK80_9(),
]

unitree_actuators = [
    Unitree_A1_Motor(),
    Unitree_B1_Motor(),
    Unitree_GO_Motor()
]

all_actuators = t_motor_actuators + unitree_actuators + [MyActuator_RMD_MT_RH_17_100_N()]