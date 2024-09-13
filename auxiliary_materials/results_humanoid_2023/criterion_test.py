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
Lpath=[os.getcwd() + "/robots/talos_like",os.getcwd() + "/robots/digit_like",os.getcwd() + "/robots/kangaroo_like",os.getcwd() + "/robots/disney_like",os.getcwd() + "/robots/wl16_like"]
Lname=["Talos","Digit","Kangaroo","Disney bipedal robot","WL16"]
Lcolor=["skyblue","black","royalblue","orange","firebrick"]
num_robot=-1


max_iso_trans=0
min_iso_trans=np.inf
max_manip_trans=0
min_manip_trans=np.inf

max_iso_rot=0
min_iso_rot=np.inf
max_manip_rot=0
min_manip_rot=np.inf


max_zrr=0
min_zrr=np.inf
max_ziner=0
min_ziner=np.inf
max_v=0
min_v=np.inf
max_y=0
min_y=np.inf

min_nzrr=np.inf
max_nzrr=0
fig_manip=plt.subplots(3,3,gridspec_kw={'height_ratios': [1, 1, 2]})
fig_zrr=plt.subplots(3,1)


for path,name,ncolor in zip(Lpath,Lname,Lcolor):
    num_robot=num_robot+1


    model,constraint_model,actuation_model,visual_model= completeRobotLoader(path)
    constraint_data = [c.createData() for c in constraint_model]
    data = model.createData()

    viz = MeshcatVisualizer(model, visual_model, visual_model)
    viz.viewer = meshcat.Visualizer().open()
    viz.viewer["/Background"].set_property("visible", False)
    viz.viewer["/Grid"].set_property("visible", False)
    viz.viewer["/Axes"].set_property("visible", False)
    viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", [0,0,0.5])
    viz.clean()
    viz.loadViewerModel(rootNodeName="number 1")


    q0=closedLoopProximalMount(model,data,constraint_model,constraint_data)
    viz.display(q0)

    ##Init of robot
    DT=1e-3
    P6D_Walk=np.load("apps/results_humanoid_2023/trajectory/walk_trajectory2.npy")
    P6D_Squat=np.load("apps/results_humanoid_2023/trajectory/squat_trajectory2.npy")
    P6D_Stair=np.load("apps/results_humanoid_2023/trajectory/climb_trajectory2.npy")

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

    err_couple=[]
    mean_couple=[]
    mean_maniptrans=[]
    mean_maniprot=[]
    err_maniptrans=[]
    err_maniprot=[]
    mean_Lv=[]
    err_Lv=[]
    mean_Ldy=[]
    err_Ldy=[]
    mean_isotropytrans=[]
    mean_isotropyrot=[]
    err_isotropyrot=[]
    err_isotropytrans=[]
    mean_zinertia=[]
    err_zinertia=[]


    pin.framesForwardKinematics(model,data,q0)
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

        Lq=[]
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
        err=np.sum([norm(cd.contact_placement_error.np) for cd in constraint_data])
        if err>1e-1:
            break # check the start position 



        constrainte1=norm(pos-P_ini) 

        #init of criterion 
        Lisotropytrans=[]
        Lisotropyrot=[]

        Lcoupleactionneur=[]
        Lzinertia=[]

        Lmaniptrans=[]
        Lmaniprot=[]

        Lv=[]
        Ldy=[]

        Li=[]

        oldpos=pos.copy()
        for i in range(len(P6D[0,:])):
            goal=P6D[:,i]+2*(np.random.rand(6)-0.5)*0.025 #noise on position
            vgoal=V6D[:,i]*DT                             #filter
            
            pos=np.concatenate([data.oMf[idpied].translation-hanche,pin.log(data.oMf[idpied]).angular])
            vreel=(pos-oldpos)
            oldpos=pos.copy()

        
            va=Kp*(goal-pos)+Kd*(vgoal-vreel)


            vq,Jf36_closed=inverseConstraintKinematicsSpeed(model,data,constraint_model,constraint_data,actuation_model,q,idpied,data.oMf[idpied].action@va)
            

            q = pin.integrate(model, q, vq * DT)
            err=np.sum([norm(cd.contact_placement_error.np) for cd in constraint_data])
            if err>1e-1:
                break # check the start position 


            viz.display(q)
            Lq.append(q)
            pin.framesForwardKinematics(model,data,q)
            pin.computeAllTerms(model,data,q,vq)
            if norm(goal-pos) < 1e1:
                
                

                T=np.zeros([len(data.oMi),3])
                T2D=np.zeros([len(data.oMi),2])
                for id,j in enumerate(data.oMi):
                    xyz=j.translation
                    T[id,:]=xyz
                    T2D[id,:]=xyz[0],xyz[2]
                hull3D=ConvexHull(T)
                hull=ConvexHull(T2D)
                volume=hull3D.volume
                dy=hull.volume
                Lv.append(volume)
                Ldy.append(dy)

                D_rot=1/np.linalg.svd(Jf36_closed[3:,:])[1].mean()*np.identity(6)
                D_trans=1/np.linalg.svd(Jf36_closed[:3,:])[1].mean()*np.identity(6)


                manip=np.linalg.svd(Jf36_closed@Jf36_closed.T)[1]



                if "Kangaroo" in name or "WL16" in name: #kangaroo and wl16 use cylinder
                    manip_rot=np.linalg.svd(Jf36_closed[3:,:]@D_rot@D_rot@Jf36_closed[3:,:].T)[1]
                    manip_trans=np.linalg.svd(Jf36_closed[:3,:]@Jf36_closed[:3,:].T)[1]
                    couple_transmis=norm(abs(Jf36_closed.T[:,:3]@np.array([0,0,1])))
                else:
                    manip_rot=np.linalg.svd(Jf36_closed[3:,:]@Jf36_closed[3:,:].T)[1]
                    manip_trans=np.linalg.svd(Jf36_closed[:3,:]@D_trans@D_trans@Jf36_closed[:3,:].T)[1]
                    couple_transmis=norm(abs((Jf36_closed@D_trans).T[:,:3]@np.array([0,0,1])))

                if norm(manip)<1e10:
                    Lmaniptrans.append(np.product(manip_trans)**0.5)
                    Lmaniprot.append(np.product(manip_rot)**0.5)


                
                Lcoupleactionneur.append(couple_transmis)

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
                J=pin.computeFrameJacobian(model,data,q,idpied,pin.LOCAL)
                Lambda_open=inv(J@inv(M)@J.T)

                Lzinertia.append(Lambda[2,2])

                Lisotropytrans.append(min(manip_trans)/max(manip_trans))
                Lisotropyrot.append(min(manip_rot)/max(manip_rot))

                Li.append(i)


        Lcoupleactionneur=np.array(Lcoupleactionneur)
        mean_couple.append(Lcoupleactionneur.mean())
        err_couple.append(Lcoupleactionneur.std())

        Lzinertia=np.array(Lzinertia)
        mean_zinertia.append(Lzinertia.mean())
        err_zinertia.append(Lzinertia.std())

        Lmaniptrans=np.array(Lmaniptrans)
        Lmaniprot=np.array(Lmaniprot)
        
        mean_maniptrans.append(Lmaniptrans.mean())
        mean_maniprot.append(Lmaniprot.mean())

        err_maniptrans.append(Lmaniptrans.std())
        err_maniprot.append(Lmaniprot.std())

        if move=="walk":
            np.save("results_humanoid_2023/results/"+name+"walk_Lq",Lq)
            np.save("results_humanoid_2023/results/"+name+"walk_zrr",Lcoupleactionneur)
            np.save("results_humanoid_2023/results/"+name+"walk_zai.npy",Lzinertia)
            np.save("results_humanoid_2023/results/"+name+"walk_TM.npy",Lmaniptrans)
            np.save("results_humanoid_2023/results/"+name+"walk_RM.npy",Lmaniprot)
            np.save("results_humanoid_2023/results/"+name+"walk_vomlume.npy",Lv)
            np.save("results_humanoid_2023/results/"+name+"walk_surface.npy",Ldy)
            np.save("results_humanoid_2023/results/walk_pos.npy",(np.array(Li)+1-50)/50*20*1e-3)
        
        if move=="stair":
            np.save("results_humanoid_2023/results/"+name+"stair_Lq",Lq)
            np.save("results_humanoid_2023/results/"+name+"stair_zrr.npy",Lcoupleactionneur)
            np.save("results_humanoid_2023/results/"+name+"stair_zai.npy",Lzinertia)
            np.save("results_humanoid_2023/results/"+name+"stair_TM.npy",Lmaniptrans)
            np.save("results_humanoid_2023/results/"+name+"stair_RM.npy",Lmaniprot)
            np.save("results_humanoid_2023/results/"+name+"stair_vomlume.npy",Lv)
            np.save("results_humanoid_2023/results/"+name+"stair_surface.npy",Ldy)
            np.save("results_humanoid_2023/results/stair_pos.npy",(np.array(Li)+1)/100*17*1e-3)

        if move=="squat":
            np.save("results_humanoid_2023/results/"+name+"squat_Lq",Lq)
            np.save("results_humanoid_2023/results/"+name+"squat_zrr.npy",Lcoupleactionneur)
            np.save("results_humanoid_2023/results/"+name+"squat_zai.npy",Lzinertia)
            np.save("results_humanoid_2023/results/"+name+"squat_TM.npy",Lmaniptrans)
            np.save("results_humanoid_2023/results/"+name+"squat_RM.npy",Lmaniprot)
            np.save("results_humanoid_2023/results/"+name+"squat_vomlume.npy",Lv)
            np.save("results_humanoid_2023/results/"+name+"squat_surface.npy",Ldy)    
            np.save("results_humanoid_2023/results/squat_pos.npy",(np.array(Li)+1)/100*20*1e-3)
        ######################################################
        ####### ONLY PLOTTING BELOW ##########################
        ######################################################



        ax=plt.figure('walk ZRR')
        ax=plt.subplot(111)
        if move=="walk":
                ax.plot((np.array(Li)+1-50)/50*20*1e-3,Lcoupleactionneur,color=ncolor,label=name)
                ax.set_xlabel("y position (m)")
                ax.set_title("Walk")
                ax.set_ylabel('ZRR')

        ax.set_yscale('log')
        ax.xaxis.tick_bottom()
        d=0.015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])
        plt.savefig('results_humanoid_2023/img/walk_ZRR.png',bbox_inches='tight')


        if min_nzrr>min(Lcoupleactionneur):
            min_nzrr=min(Lcoupleactionneur)
        
        if max_nzrr<max(Lcoupleactionneur):
            max_nzrr=max(Lcoupleactionneur)

        ax=plt.figure('ZRR_new',figsize=(6.4,3.7))
        ax=plt.subplot(131)
        ax.set_ylim((min_nzrr,max_nzrr))

        if move=="walk":
            ax.plot((np.array(Li)+1-50)/50*20*1e-3,Lcoupleactionneur,color=ncolor,label=name)
            ax.set_xlabel("y position (m)")
            ax.set_title("Walk")
            ax.set_ylabel('ZRR')    

        ax.set_yscale('log')               
        
        ax.yaxis.set_minor_formatter(majorFormatter)    
        ax.yaxis.set_major_formatter(majorFormatter)    

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])

        plt.savefig('results_humanoid_2023/img/nzrr.png',bbox_inches='tight')


        ax=plt.figure('squat translation manipulability')
        ax=plt.subplot(111)
        if move=="squat":
                ax.plot((np.array(Li)+1)/100*20*1e-3,Lcoupleactionneur,color=ncolor,label=name)
                ax.set_xlabel("z position (m)")
                ax.set_title("Squat")
                ax.set_ylabel('ZRR')
                

        ax.set_yscale('log')
        ax.xaxis.tick_bottom()
        d=0.015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])
        plt.savefig('results_humanoid_2023/img/squat_ZRR.png',bbox_inches='tight')



        ax=plt.figure('ZRR_new',figsize=(6.4,3.7))
        ax=plt.subplot(132)
        ax.set_ylim((min_nzrr,max_nzrr))
        if move=="squat":
            ax.plot((np.array(Li)+1)/100*20*1e-3,Lcoupleactionneur,color=ncolor,label=name)
            ax.set_xlabel("z position (m)")
            ax.set_title("Squat")
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_yscale('log')
    
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])
        plt.savefig('results_humanoid_2023/img/nzrr.png',bbox_inches='tight')

       
        ax=plt.figure('stair translation manipulability')
        ax=plt.subplot(111)
        if move=="stair":
                ax.plot((np.array(Li)+1)/100*17*1e-3,Lcoupleactionneur,color=ncolor,label=name)
                ax.set_xlabel("z position (m)")
                ax.set_title("Stair")
                ax.set_ylabel('ZRR')
                
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(majorFormatter) 
        ax.xaxis.tick_bottom()
        d=0.015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])
        plt.savefig('results_humanoid_2023/img/stair_ZRR.png',bbox_inches='tight')
       

        ax=plt.figure('ZRR_new',figsize=(6.4,3.7))
        ax=plt.subplot(133)
        ax.set_ylim((min_nzrr,max_nzrr))
        if move=="stair":
            ax.plot((np.array(Li)+1)/100*17*1e-3,Lcoupleactionneur,color=ncolor,label=name)
            ax.set_xlabel("z position (m)")
            ax.set_title("Stair")

        ax.set_yscale('log')
        ax.yaxis.set_tick_params(labelleft=False)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])

        ax.legend(loc='upper center', bbox_to_anchor=(-0.7, -0.2),ncols=5)
        plt.savefig('results_humanoid_2023/img/nzrr.png',bbox_inches='tight')





        Lv=np.array(Lv)
        Ldy=np.array(Ldy)

        mean_Lv.append(Lv.mean())
        err_Lv.append(Lv.std())
        mean_Ldy.append(Ldy.mean())
        err_Ldy.append(Ldy.std())


        
        Lisotropyrot=np.array(Lisotropyrot)
        Lisotropytrans=np.array(Lisotropytrans)
        mean_isotropytrans.append(Lisotropytrans.mean())
        mean_isotropyrot.append(Lisotropyrot.mean())
        err_isotropyrot.append(Lisotropyrot.std())
        err_isotropytrans.append(Lisotropytrans.std())


    

    if min_zrr>min(mean_couple):
        min_zrr=min(mean_couple)
        deltaec=max(err_couple)
    if max_zrr<max(mean_couple):
        max_zrr=max(mean_couple)
        deltaec=max(err_couple)
    ax=plt.figure("ZRR")
    ax=plt.subplot(131)
    ax.errorbar([1+num_robot/10],mean_couple[0],err_couple[0],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.set_ylim((min_zrr-deltaec,max_zrr+deltaec))
    ax.set_title("Walk")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax=plt.subplot(132)
    ax.errorbar([2+num_robot/10],mean_couple[1],err_couple[1],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_zrr-deltaec,max_zrr+deltaec))
    ax.set_title("Squat")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax = plt.subplot(133)
    ax.errorbar([3+num_robot/10],mean_couple[2],err_couple[2],label=  name,fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_zrr-deltaec,max_zrr+deltaec))
    ax.set_title("Stair")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(-0.7, -0.1),ncols=5)
    plt.savefig('results_humanoid_2023/img/ZRR.png',bbox_inches='tight')


    if min_ziner>min(mean_zinertia):
        min_ziner=min(mean_zinertia)
        deltazi=max(err_zinertia)
    if max_ziner<max(mean_zinertia):
        max_ziner=max(mean_zinertia)
        deltazi=max(err_zinertia)
    ax=plt.figure("Z inertia",figsize=(6.4,3.7))
    ax=plt.subplot(131)
    ax.errorbar([1+num_robot/10],mean_zinertia[0],err_zinertia[0],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.set_ylim((min_ziner-deltazi,max_ziner+deltazi))
    ax.set_title("Walk")
    ax.set_ylabel("ZAI")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax=plt.subplot(132)
    ax.errorbar([2+num_robot/10],mean_zinertia[1],err_zinertia[1],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_ziner-deltazi,max_ziner+deltazi))
    ax.set_title("Squat")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax = plt.subplot(133)
    ax.errorbar([3+num_robot/10],mean_zinertia[2],err_zinertia[2],label=  name,fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_ziner-deltazi,max_ziner+deltazi))
    ax.set_title("Stair")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])

    ax.legend(loc='upper center', bbox_to_anchor=(-0.7, -0.1),ncols=5)  
    plt.savefig('results_humanoid_2023/img/Zinertia.png',bbox_inches='tight')

    if min_v>min(mean_Lv):
        min_v=min(mean_Lv)
        deltav=max(err_Lv)
    if max_v<max(mean_Lv):
        max_v=max(mean_Lv)
        deltav=max(err_Lv)
    ax=plt.figure("Convex hull")
    ax=plt.subplot(231)
    ax.errorbar([1+num_robot/10],mean_Lv[0],err_Lv[0],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.set_ylim((min_v-deltav,max_v+deltav))
    ax.set_title("Walk")
    ax.set_ylabel("pcomplete convex hull")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax=plt.subplot(232)
    ax.errorbar([2+num_robot/10],mean_Lv[1],err_Lv[1],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_v-deltav,max_v+deltav))
    ax.set_title("Squat")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax = plt.subplot(233)
    ax.errorbar([3+num_robot/10],mean_Lv[2],err_Lv[2],label=  name,fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_v-deltav,max_v+deltav))
    ax.set_title("Stair")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])



    if min_y>min(mean_Ldy):
        min_y=min(mean_Ldy)
        delta=max(err_Ldy)
    if max_y<max(mean_Ldy):
        max_y=max(mean_Ldy)
        delta=max(err_Ldy)
    ax=plt.figure("Convex hull")
    ax=plt.subplot(234)
    ax.errorbar([1+num_robot/10],mean_Ldy[0],err_Ldy[0],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.set_ylim((min_y-delta,max_y+delta))
    ax.set_ylabel("projeted convex hull")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax=plt.subplot(235)
    ax.errorbar([2+num_robot/10],mean_Ldy[1],err_Ldy[1],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_y-delta,max_y+delta))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax = plt.subplot(236)
    ax.errorbar([3+num_robot/10],mean_Ldy[2],err_Ldy[2],label=  name,fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_y-delta,max_y+delta))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(-0.7, -0.1),ncols=5)
    plt.savefig('results_humanoid_2023/img/Convex_hull.png',bbox_inches='tight')

    if name!="Kangaroo":
        if min_manip_trans>min(mean_maniptrans):
            min_manip_trans=min(mean_maniptrans)
        if max_manip_trans<max(mean_maniptrans):
            max_manip_trans=max(mean_maniptrans)
            deltamaniptrans=max(err_maniptrans)
        ax=fig_manip[1][1,0]
        ax.errorbar([1+num_robot/10],mean_maniptrans[0],err_maniptrans[0],fmt='o',color=ncolor)
        
        ax.set_ylabel('translation')
        ax.yaxis.set_label_coords(0.07, 0.7, transform=fig_manip[0].transFigure)
        ax.xaxis.set_visible(False)
        ax.set_ylim((min_manip_trans-deltamaniptrans,max_manip_trans+deltamaniptrans))
        
        ax.spines['top'].set_visible(False)
        ax.xaxis.tick_bottom()
        d=0.015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])
        ax=fig_manip[1][1,1]
        ax.errorbar([2+num_robot/10],mean_maniptrans[1],err_maniptrans[1],fmt='o',color=ncolor)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_ylim((min_manip_trans-deltamaniptrans,max_manip_trans+deltamaniptrans))
        
        ax.spines['top'].set_visible(False)
        ax.xaxis.tick_bottom()
        d=0.015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])
        ax=fig_manip[1][1,2]
        ax.errorbar([3+num_robot/10],mean_maniptrans[2],err_maniptrans[2],label=  name,fmt='o',color=ncolor)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_ylim((min_manip_trans-deltamaniptrans,max_manip_trans+deltamaniptrans))
        
        ax.spines['top'].set_visible(False)
        ax.xaxis.tick_bottom()
        d=0.015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    else :
        
        ax=fig_manip[1][0,0]
        ax.errorbar([1+num_robot/10],mean_maniptrans[0],err_maniptrans[0],fmt='o',color=ncolor)
        ax.xaxis.set_visible(False)
        ax.set_ylim((min(mean_maniptrans)-max(err_maniptrans),max(mean_maniptrans)+max(err_maniptrans)))
        ax.set_title("Walk")
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.tick_top()
        d=0.015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        ax.tick_params(labeltop=False)  # don't put tick labels at the top
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])
        ax=fig_manip[1][0,1]
        ax.errorbar([2+num_robot/10],mean_maniptrans[1],err_maniptrans[1],fmt='o',color=ncolor)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_ylim((min(mean_maniptrans)-max(err_maniptrans),max(mean_maniptrans)+max(err_maniptrans)))
        ax.set_title("Squat")
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.tick_top()
        d=0.015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        ax.tick_params(labeltop=False)  # don't put tick labels at the top
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height])
        ax=fig_manip[1][0,2]
        ax.errorbar([3+num_robot/10],mean_maniptrans[2],err_maniptrans[2],label=  name,fmt='o',color=ncolor)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_ylim((min(mean_maniptrans)-max(err_maniptrans),max(mean_maniptrans)+max(err_maniptrans)))
        ax.set_title("Stair")
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.tick_top()
        d=0.015
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        ax.tick_params(labeltop=False)  # don't put tick labels at the top



    if min_manip_rot>min(mean_maniprot):
        min_manip_rot=min(mean_maniprot)
        # deltamaniprot=max(err_maniprot)
    if max_manip_rot<max(mean_maniprot):
        deltamaniprot=max(err_maniprot)
        max_manip_rot=max(mean_maniprot)
    ax=fig_manip[1][2,0]
    ax.errorbar([1+num_robot/10],mean_maniprot[0],err_maniprot[0],fmt='o',color=ncolor)
    ax.set_ylabel('rotation')
    ax.xaxis.set_visible(False)
    ax.set_ylim((min_manip_rot-deltamaniprot,max_manip_rot+deltamaniprot))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax=fig_manip[1][2,1]
    ax.errorbar([2+num_robot/10],mean_maniprot[1],err_maniprot[1],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_manip_rot-deltamaniprot,max_manip_rot+deltamaniprot))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax=fig_manip[1][2,2]
    ax.errorbar([3+num_robot/10],mean_maniprot[2],err_maniprot[2],label=  name,fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_manip_rot-deltamaniprot,max_manip_rot+deltamaniprot))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(-0.7, -0.1),ncols=5)
    fig_manip[0].savefig('results_humanoid_2023/img/mean_manipulability.png',bbox_inches='tight')



    if min_iso_trans>min(mean_isotropytrans):
        min_iso_trans=min(mean_isotropytrans)
        # deltaisotrans=max(err_isotropytrans)
    if max_iso_trans<max(mean_isotropytrans):
        max_iso_trans=max(mean_isotropytrans)
        deltaisotrans=max(err_isotropytrans)
    ax=plt.figure("mean isotropy")
    ax=plt.subplot(131)
    ax.errorbar([1+num_robot/10],mean_isotropytrans[0],err_isotropytrans[0],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.set_ylim((min_iso_trans-deltaisotrans,max_iso_trans+deltaisotrans))
    ax.set_title("Walk")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax=plt.subplot(132)
    ax.errorbar([2+num_robot/10],mean_isotropytrans[1],err_isotropytrans[1],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_iso_trans-deltaisotrans,max_iso_trans+deltaisotrans))
    ax.set_title("Squat")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax = plt.subplot(133)
    ax.errorbar([3+num_robot/10],mean_isotropytrans[2],err_isotropytrans[2],label=  name,fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_iso_trans-deltaisotrans,max_iso_trans+deltaisotrans))
    ax.set_title("Stair")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(-0.7, -0.1),ncols=5)
    plt.savefig('results_humanoid_2023/img/mean_translation_isotropy.png',bbox_inches='tight')
    

    if min_iso_rot>min(mean_isotropyrot):
        min_iso_rot=min(mean_isotropyrot)
        deltaiso=max(err_isotropyrot)
    if max_iso_rot<max(mean_isotropyrot):
        max_iso_rot=max(mean_isotropyrot)    
        deltaiso=max(err_isotropyrot)
    ax=plt.figure("mean isotropy rotation")
    ax=plt.subplot(131)
    ax.errorbar([1+num_robot/10],mean_isotropyrot[0],err_isotropyrot[0],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.set_ylim((min_iso_rot-deltaiso,max_iso_rot+deltaiso))
    ax.set_title("Walk")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax=plt.subplot(132)
    ax.errorbar([2+num_robot/10],mean_isotropyrot[1],err_isotropyrot[1],fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_iso_rot-deltaiso,max_iso_rot+deltaiso))
    ax.set_title("Squat")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax = plt.subplot(133)
    ax.errorbar([3+num_robot/10],mean_isotropyrot[2],err_isotropyrot[2],label=  name,fmt='o',color=ncolor)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim((min_iso_rot-deltaiso,max_iso_rot+deltaiso))
    ax.set_title("Stair")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(-0.7, -0.1),ncols=5)
    plt.savefig('results_humanoid_2023/img/mean_rotation_manipulability.png',bbox_inches='tight')


    print("done")