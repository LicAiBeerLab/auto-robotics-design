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
import cv2
from moviepy.editor import *
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, "../../utils")
sys.path.append(mymodule_dir)
from loader_tools import *

from closed_loop_kinematics import *

from closed_loop_jacobian import *


from matplotlib.ticker import MultipleLocator, FormatStrFormatter

majorFormatter = FormatStrFormatter('%1.1f')

pin.SE3.__repr__ = pin.SE3.__str__
np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

Lpath=[os.getcwd() + "/robots/talos_like",os.getcwd() + "/robots/digit_like",os.getcwd() + "/robots/kangaroo_new_model",os.getcwd() + "/robots/disney_like",os.getcwd() + "/robots/wl16_like"]
Lpath=[os.getcwd() + "/robots/disney_like"]
Lmovement=["walk","squat","stair"]
Lrobot=["Talos","Digit","Kangaroo","Disney bipedal robot","WL16"]
Lrobot=["Disney bipedal robot"]
for path,name in zip(Lpath,Lrobot):
    model,constraint_model,actuation_model,visual_model= completeRobotLoader(path)
    constraint_data = [c.createData() for c in constraint_model]
    data = model.createData()
    idcentre= model.getFrameId("centre_hanche")
    viz = MeshcatVisualizer(model, visual_model, visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="number 1")
    pin.framesForwardKinematics(model,data,np.zeros(model.nq))
    hanche=data.oMf[idcentre].translation
    transforme=pin.SE3.Identity()
    transforme.translation=np.array([-hanche[0],-hanche[1],0.7-hanche[2]])
    viz.viewer.set_transform(transforme.np)
    for movement in Lmovement:
        Lq=np.load("results_humanoid_2023/results/"+name+movement+"_Lq.npy")
        frames=[]
        os.system("rm img_vid/*")
        i=0
        for q in Lq:
            viz.display(q)
            img=viz.viewer.get_image()
            frames.append(img)
            img.save('img_vid/'+str(i)+'.png')
            i=i+1
        os.system("ffmpeg -r 24 -i img_vid/%01d.png -vcodec mpeg4 -y "+'robots/'+name[:5]+movement+"robotmovement.mp4")


        # frame_one = frames[0]
        # frame_one.save('robots/'+name+movement+"robotmovement.gif", format="GIF",interlace=False, append_images=frames,save_all=True, duration=100, loop=0, optimize=False, lossless=True)
        
