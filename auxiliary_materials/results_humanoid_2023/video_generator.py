import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import meshcat
from pinocchio.visualize import MeshcatVisualizer
import matplotlib.pyplot as plt
import os
from numpy.linalg import inv, norm
import sys
from scipy.spatial import ConvexHull

np.random.seed(1)


script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, "../utils")
sys.path.append(mymodule_dir)
from loader_tools import *

from closed_loop_kinematics import *

from closed_loop_jacobian import *


from matplotlib.ticker import MultipleLocator, FormatStrFormatter

majorFormatter = FormatStrFormatter('%1.1f')

pin.SE3.__repr__ = pin.SE3.__str__
np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)


# Import and init of the robot 
path = os.getcwd() + "/robots/talos_like"
img_path=os.getcwd()+"/results_humanoid_2023"+"/video_wl16" + "/"


model,constraint_model,actuation_model,visual_model= completeRobotLoader(path)
constraint_data = [c.createData() for c in constraint_model]
data = model.createData()

viz = MeshcatVisualizer(model, visual_model, visual_model)
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.clean()
viz.loadViewerModel(rootNodeName="number 1")


q0=closedLoopProximalMount(model,data,constraint_model,constraint_data,actuation_model)
viz.display(q0)

##Init of robot
DT=1e-3
P6D_Walk=np.load("results_humanoid_2023/trajectory/walk_trajectory2.npy")
P6D_Squat=np.load("results_humanoid_2023/trajectory/squat_trajectory2.npy")
P6D_Stair=np.load("results_humanoid_2023/trajectory/climb_trajectory2.npy")

V6D_Walk=np.zeros(P6D_Walk.shape)
V6D_Walk[:,1:]=(P6D_Walk[:,1:]-P6D_Walk[:,:-1])/DT


V6D_Squat=np.zeros(P6D_Squat.shape)
V6D_Squat[:,1:]=(P6D_Squat[:,1:]-P6D_Squat[:,:-1])/DT

V6D_Stair=np.zeros(P6D_Stair.shape)
V6D_Stair[:,1:]=(P6D_Stair[:,1:]-P6D_Stair[:,:-1])/DT


idpied=model.getFrameId('bout_pied_frame')


q=q0



Kp=50
Kd=np.sqrt(Kp)*2

idcentre= model.getFrameId("centre_hanche")


pin.framesForwardKinematics(model,data,q0)
hanche=data.oMf[idcentre].translation
I4=pin.SE3.Identity()
I4.translation=np.array([1,0,0.7-hanche[2]])
viz.viewer.set_transform(I4.np)
n=0
for run in range(3):

    if run == 0:
        P6D=P6D_Walk
        V6D=V6D_Walk
    if run == 1:
        P6D=P6D_Squat
        V6D=V6D_Squat
    if run == 2:
        P6D=P6D_Stair
        V6D=V6D_Stair


    ##init of the start position 
    P_ini=P6D[:,0]
    pin.framesForwardKinematics(model,data,q0)

    pos=np.concatenate([data.oMf[idpied].translation-hanche,pin.log(data.oMf[idpied]).angular])
    oldpos=pos.copy()
    it=0
    q=q0
    Lxtest=[]
    while norm(pos-P_ini)>1e-6 and it < 100:
        pos=np.concatenate([data.oMf[idpied].translation-hanche,pin.log(data.oMf[idpied]).angular])
        Lxtest.append(norm(pos-P_ini))
        vreel=(pos-oldpos)
        va=Kp*(P_ini-pos)*0.8
        vq,Jf36_closed=inverseConstraintKinematicsSpeed(model,data,constraint_model,constraint_data,actuation_model,q,idpied,data.oMf[idpied].action@va)
        q = pin.integrate(model, q, vq * DT)
        viz.display(q)
        pin.framesForwardKinematics(model,data,q)
        it=it+1
        pos=np.concatenate([data.oMf[idpied].translation-hanche,pin.log(data.oMf[idpied]).angular])
        err=np.sum([norm(cd.contact_placement_error.np) for cd in constraint_data])
        if err>1e-1:
            break
    err=np.sum([norm(cd.contact_placement_error.np) for cd in constraint_data])
    if err>1e-1:
        break # check the start position 



    oldpos=pos.copy()
    for i in range(len(P6D[0,:])):
        goal=P6D[:,i]
        vgoal=V6D[:,i]*DT
        
        pos=np.concatenate([data.oMf[idpied].translation-hanche,pin.log(data.oMf[idpied]).angular])
        vreel=(pos-oldpos)
        oldpos=pos.copy()

    
        va=Kp*(goal-pos)+Kd*(vgoal-vreel)


        vq,Jf36_closed=inverseConstraintKinematicsSpeed(model,data,constraint_model,constraint_data,actuation_model,q,idpied,data.oMf[idpied].action@va)
        print(vreel)

        q = pin.integrate(model, q, vq * DT)



        viz.display(q)
        pin.framesForwardKinematics(model,data,q)
        n=n+1
        # img=viz.captureImage()
        # plt.imsave(img_path+str(n)+".png",img)
        


 








print('done')
