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
from auto_robot_design.pinokla.loader_tools import *

from auto_robot_design.pinokla.closed_loop_kinematics import *

from auto_robot_design.pinokla.closed_loop_jacobian import *


from matplotlib.ticker import MultipleLocator, FormatStrFormatter

majorFormatter = FormatStrFormatter('%1.1f')

pin.SE3.__repr__ = pin.SE3.__str__
np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

# Import and init of the robot 
Lpath=[os.getcwd() + "/robots/talos_like"]
Lname=["Talos"]
Lcolor=["skyblue"]
num_robot=-1


max_imf=0
min_imf=np.inf
max_zimf=0
min_zimf=np.inf

delta=0
deltazimf=0
for path,name,ncolor in zip(Lpath,Lname,Lcolor):
    num_robot=num_robot+1
    model,constraint_model,actuation_model,visual_model= completeRobotLoader(path)
    constraint_data = [c.createData() for c in constraint_model]
    data = model.createData()
    #model.armature[actuation_model.idvmot[:]]=np.array([100,100,100,100,100,100])*1e-4

        
    model_free,constraint_model_free,actuation_model_free,visual_model= completeRobotLoader(path,fixed=False)
    constraint_data_free = [c.createData() for c in constraint_model]
    data_free = model_free.createData()
    #model_free.armature[actuation_model_free.idvmot[6:]]=np.array([100,100,100,100,100,100])*1e-4

    viz = MeshcatVisualizer(model_free, visual_model, visual_model)
    viz.viewer = meshcat.Visualizer().open()
    viz.clean()
    viz.loadViewerModel(rootNodeName="number 1")


    q0=closedLoopProximalMount(model,data,constraint_model,constraint_data,actuation_model)

    q0_freeflyer=np.concatenate([np.array([0,0,0,0,0,0,1]),q0])
    viz.display(q0_freeflyer)






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

    idpied=model.getFrameId('bout_pied')


    q=q0
    pin.framesForwardKinematics(model,data,q0)


    Kp=50
    Kd=np.sqrt(Kp)*2

    idcentre= model.getFrameId("centre_hanche")
    hanche=data.oMf[idcentre].translation

    mean_IMF=[]
    err_IMF=[]

    mean_zimf=[]
    err_zimf=[]
    hanche=data.oMf[idcentre].translation
    transforme=pin.SE3.Identity()
    transforme.translation=np.array([1,0,0.7-hanche[2]])
    viz.viewer.set_transform(transforme.np)

    for run in range(3):

        if run == 0:
            P6D=P6D_Walk
            V6D=V6D_Walk
            move="walk"
        if run == 1:
            P6D=P6D_Squat
            V6D=V6D_Squat
            move="squat"
        if run == 2:
            P6D=P6D_Stair
            V6D=V6D_Stair
            move="stair"


        ##init of the start position 
        P_ini=P6D[:,0]
        pin.framesForwardKinematics(model,data,q0)
        pos=np.concatenate([data.oMf[idpied].translation-hanche,pin.log(data.oMf[idpied]).angular])
        oldpos=pos.copy()
        it=0
        q=q0
        Lxtest=[]

        while norm(pos-P_ini)>1e-6 and it < 100:
            print(norm(pos[0:3]-P_ini[0:3]))
            pos=np.concatenate([data.oMf[idpied].translation-hanche,pin.log(data.oMf[idpied]).angular])
            Lxtest.append(norm(pos-P_ini))
            vreel=(pos-oldpos)
            va=Kp*(P_ini-pos)*0.8
            vq,Jf36_closed=inverseConstraintKinematicsSpeed(model,data,constraint_model,constraint_data,actuation_model,q,idpied,data.oMf[idpied].action@va)
            q = pin.integrate(model, q, vq * DT)
            pin.framesForwardKinematics(model,data,q)
            it=it+1
            pos=np.concatenate([data.oMf[idpied].translation-hanche,pin.log(data.oMf[idpied]).angular])
            err=np.sum([norm(cd.contact_placement_error.np) for cd in constraint_data])
            if err>1e-1:
                break # check the start position 

        err=np.sum([norm(cd.contact_placement_error.np) for cd in constraint_data]) 
        if err>1e-1:
            break # check the start position 
        # affichage sur le free flyer
        q_free=np.concatenate([np.array([0,0,0,0,0,0,1]),q])
        viz.display(q_free)


        constrainte1=norm(pos-P_ini) 

        #init of criterion 


        Limf=[]
        Lzimf=[]

        Li=[]

        oldpos=pos.copy()
        for i in range(len(P6D[0,:])):
            goal=P6D[:,i]
            vgoal=V6D[:,i]*DT
            
            pos=np.concatenate([data.oMf[idpied].translation-hanche,pin.log(data.oMf[idpied]).angular])
            vreel=(pos-oldpos)
            oldpos=pos.copy()

        
            va=Kp*(goal-pos)+Kd*(vgoal-vreel)

            vq_free,Jfree=inverseConstraintKinematicsSpeed(model_free,data_free,constraint_model_free,constraint_data_free,actuation_model_free,q_free,idpied,data.oMf[idpied].action@va)
            vq,Jf36_closed=inverseConstraintKinematicsSpeed(model    ,data     ,constraint_model     ,constraint_data     ,actuation_model     ,q     ,idpied,data.oMf[idpied].action@va)
            print(vreel)




            q = pin.integrate(model, q, vq * DT)
            q_free=np.concatenate([np.array([0,0,0,0,0,0,1]),q])
            viz.display(q_free)
            pin.framesForwardKinematics(model,data,q)
            pin.computeAllTerms(model,data,q,vq)

            #compute criterion
            if norm(goal-pos) < 1e1:
                
            

                pin.computeJointJacobians(model,data,q)
                LJ=[]
                for (cm,cd) in zip(constraint_model,constraint_data):
                    Jc=pin.getConstraintJacobian(model,data,cm,cd)
                    LJ.append(Jc)


                
                M=pin.crba(model,data,q)
                dq=dq_dqmot(model,actuation_model,LJ)
                Mmot=dq.T@M@dq
                inv=np.linalg.inv
                z=np.array([0,0,1,0,0,0])
                Lambda=inv(Jf36_closed@inv(Mmot)@Jf36_closed.T) # foot inertia 

                LJ_free=[]
                for (cm,cd) in zip(constraint_model_free,constraint_data_free):
                    Jc=pin.getConstraintJacobian(model_free,data_free,cm,cd)
                    LJ_free.append(Jc)

                M_free=pin.crba(model_free,data_free,q_free)
                dq_free=dq_dqmot(model_free,actuation_model_free,LJ_free)
                Mmot_free=dq_free.T@M_free@dq_free

                Lambda_free=inv(Jfree@inv(Mmot_free)@Jfree.T) # foot inertia 
                Lambda_free_lock=inv(Jfree[:6,:6]@inv(Mmot_free[:6,:6])@Jfree[:6,:6].T)


                IMF=np.linalg.det(np.identity(6)-Lambda_free@inv(Lambda_free_lock))
                zimf=1-(z.T@Lambda_free@z)/(z.T@Lambda_free_lock@z)
                Limf.append(IMF)
                Lzimf.append(zimf)
            

                Li.append(i)


        Limf=np.array(Limf)
        mean_IMF.append(Limf.mean())
        err_IMF.append(Limf.std())

        Lzimf=np.array(Lzimf)
        mean_zimf.append(Lzimf.mean())
        err_zimf.append(Lzimf.std())

        if move=="walk":
            np.save("results_humanoid_2023/results/"+name+"walk_imf",Limf)
            np.save("results_humanoid_2023/results/"+name+"walk_zimf",Lzimf)

        
        if move=="stair":
            np.save("results_humanoid_2023/results/"+name+"stair_imf.npy",Limf)
            np.save("results_humanoid_2023/results/"+name+"stair_zimf.npy",Lzimf)



        if move=="squat":
            np.save("results_humanoid_2023/results/"+name+"squat_imf.npy",Limf)
            np.save("results_humanoid_2023/results/"+name+"squat_zimf.npy",Lzimf)

        ####### ONLY PLOTTING BELOW ##########################
        ######################################################


    if min_imf>min(mean_IMF):
        min_imf=min(mean_IMF)
        delta=max(err_IMF)
    if max_imf<max(mean_IMF):
        max_imf=max(mean_IMF)
        delta=max(err_IMF)
    ax=plt.figure("imf")
    # ax.suptitle("translation isotropy")
    ax=plt.subplot(231)
    ax.errorbar([1+num_robot/10],mean_IMF[0],err_IMF[0],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.set_ylim((min_imf-delta,max_imf+delta))
    ax.set_ylabel("IMF")
    ax.set_title("Walk")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax=plt.subplot(232)
    ax.errorbar([2+num_robot/10],mean_IMF[1],err_IMF[1],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_imf-delta,max_imf+delta))
    ax.set_title("Squat")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax = plt.subplot(233)
    ax.errorbar([3+num_robot/10],mean_IMF[2],err_IMF[2],label=  name,fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_imf-delta,max_imf+delta))
    ax.set_title("Stair")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
 
    if min_zimf>min(mean_zimf):
        min_zimf=min(mean_zimf)
        deltazimf=max(err_zimf)
    if max_zimf<max(mean_zimf):
        max_zimf=max(mean_zimf)
        deltazimf=max(err_zimf)
    ax=plt.figure("imf")
    ax=plt.subplot(234)
    ax.errorbar([1+num_robot/10],mean_zimf[0],err_zimf[0],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.set_ylim((min_zimf-deltazimf,max_zimf+deltazimf))
    ax.set_ylabel("IMF projeted on z")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax=plt.subplot(235)
    ax.errorbar([2+num_robot/10],mean_zimf[1],err_zimf[1],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_zimf-deltazimf,max_zimf+deltazimf))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax = plt.subplot(236)
    ax.errorbar([3+num_robot/10],mean_zimf[2],err_zimf[2],label=  name,fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_zimf-deltazimf,max_zimf+deltazimf))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(-0.7, -0.1),ncols=5)
    plt.savefig('results_humanoid_2023/img/glob_IMF.png',bbox_inches='tight')





print('done')
