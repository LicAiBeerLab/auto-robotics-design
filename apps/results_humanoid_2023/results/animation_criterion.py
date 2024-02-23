import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

Lmovement=["walk","squat","stair"]
Lrobot=["Talos","Digit","Disney bipedal robot","WL16","Kangaroo"]
Lcritere=["zrr","RM","TM","zai","vomlume","surface","zimf"]
Lcolor=["skyblue","black","royalblue","orange","firebrick"]


squat_pos=np.load("results_humanoid_2023/results/squat_pos.npy")
walk_pos=np.load("results_humanoid_2023/results/squat_pos.npy")
stair_pos=np.load("results_humanoid_2023/results/squat_pos.npy")

Lfig_zrr=[]
Lfig_TM=[]
Lfig_RM=[]


Lfig_zrr_squat=[]
Lfig_zrr_walk=[]
Lfig_zrr_stair=[]

Lfig_volume_squat=[]
Lfig_volume_walk=[]
Lfig_volume_stair=[]

Lfig_surface_squat=[]
Lfig_surface_walk=[]
Lfig_surface_stair=[]

Lfig_zai_squat=[]
Lfig_zai_walk=[]
Lfig_zai_stair=[]

Lfig_TM_squat=[]
Lfig_TM_walk=[]
Lfig_TM_stair=[]

Lfig_RM_squat=[]
Lfig_RM_walk=[]
Lfig_RM_stair=[]


for movement in Lmovement:
    if movement=="walk":
        xpos=walk_pos
        xlegend="y position (m)"

    if movement=="squat":
        xpos=squat_pos
        xlegend="z position (m)"

    if movement=="stair":
        xpos=stair_pos
        xlegend="z position (m)"

xlim=(xpos[0],xpos[-1])
for robot,ncolor in zip(Lrobot,Lcolor):
    for critere in Lcritere:
        for movement in Lmovement:
            if movement=="walk":
                xpos=walk_pos
                xlegend="y position (m)"

            if movement=="squat":
                xpos=squat_pos
                xlegend="z position (m)"

            if movement=="stair":
                xpos=stair_pos
                xlegend="z position (m)"

            xlim=(xpos[0],xpos[-1])
            result=np.load("results_humanoid_2023/results/"+robot+movement+"_"+critere +".npy")
            Limg=[]
            # if robot != "Kangaroo":
            #     fig=plt.figure(critere+movement)
            #     ax=plt.subplot(111)
            #     ax.plot(xpos,result,color=ncolor,label=robot)
            #     ax.set_xlabel(xlegend)
            #     # ax.set_title("Stair")
            #     ax.set_ylabel(critere)
                
            #     # ax.set_ylim((min(result),max(result)))
            #     ax.legend()
            # else : 

            for i in range(len(result)):
                
                fig=plt.figure(critere+movement)
                ax=plt.subplot(111)
                if i==0:
                    ax.plot(xpos[:i+1],result[:i+1],color=ncolor,label=robot)
                else:
                    ax.plot(xpos[:i+1],result[:i+1],color=ncolor)

                ax.set_xlabel(xlegend)
                # ax.set_title("Stair")
                ax.set_ylabel(critere)
                box = ax.get_position()

                ax.set_position([box.x0, box.y0, box.width , box.height])
                ax.legend(loc='upper center', bbox_to_anchor=(0.53, 1.15),ncols=3)

                # ax.set_ylim((min(result),max(result)))
            
                Limg.append(fig2img(fig))
            frame_one = Limg[0]
            frame_one.save(robot+movement+critere+".gif", format="GIF", append_images=Limg,save_all=True, duration=42, loop=0)




                

# for movement in Lmovement:
#     if movement=="walk":
#         xpos=walk_pos
#         xlegend="y position (m)"

#     if movement=="squat":
#         xpos=squat_pos
#         xlegend="z position (m)"

#     if movement=="stair":
#         xpos=stair_pos
#         xlegend="z position (m)"

#     xlim=(xpos[0],xpos[-1])
#     for robot,color in zip(Lrobot,Lcolor):
#         for critere in Lcritere:
#             result=np.load("results_humanoid_2023/results/"+robot+movement+"_"+critere +".npy")
#             Limg=[]
#             for i in range(len(result)):
#                 fig=plt.figure()
#                 ax=plt.subplot(111)
#                 ax.plot(xpos[:i+1],result[:i+1])
#                 ax.set_xlabel(xlegend)
#                 # ax.set_title("Stair")
#                 ax.set_ylabel(critere)
#                 ax.set_xlim(xlim)
#                 ax.set_ylim((min(result),max(result)))

                # Limg.append(fig2img(fig))


                # if movement=="walk":
                #     if critere=="vomlume":
                #         fig=plt.figure("walk_volume")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)
                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.006,0.069))
                #         Lfig_volume_walk.append(fig2img(fig))
                #     if critere=="surface":
                #         fig=plt.figure("walk_surface")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)
                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.090,0.24))
                #         Lfig_surface_walk.append(fig2img(fig))
                #     if critere=="zai":
                #         fig=plt.figure("walk_zai")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)
                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.62,12.58))
                #         Lfig_zai_walk.append(fig2img(fig))
                #     if critere=="zrr":
                #         fig=plt.figure("walk_zrr")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.3,5.74))
                #         Lfig_zrr_walk.append(fig2img(fig))
                #     if critere=="TM":
                #         fig=plt.figure("walk_TM")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.2,2.06))
                #         Lfig_TM_walk.append(fig2img(fig))
                #     if critere=="RM":
                #         fig=plt.figure("walk_RM")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.0011,9))
                #         Lfig_RM_walk.append(fig2img(fig))

                # if movement=="stair":
                #     if critere=="vomlume":
                #         fig=plt.figure("stair_volume")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.006,0.069))
                #         Lfig_volume_stair.append(fig2img(fig))
                #     if critere=="surface":
                #         fig=plt.figure("stair_surface")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.090,0.24))
                #         Lfig_surface_stair.append(fig2img(fig))
                #     if critere=="zai":
                #         fig=plt.figure("stair_zai")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.62,12.58))
                #         Lfig_zai_stair.append(fig2img(fig))
                #     if critere=="zrr":
                #         fig=plt.figure("stair_zrr")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.3,5.74))
                #         Lfig_zrr_stair.append(fig2img(fig))
                #     if critere=="TM":
                #         fig=plt.figure("stair_TM")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.2,2.06))
                #         Lfig_TM_stair.append(fig2img(fig))
                #     if critere=="RM":
                #         fig=plt.figure("stair_RM")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.0011,9))
                #         Lfig_RM_stair.append(fig2img(fig))

                # if movement=="squat":
                #     if critere=="vomlume":
                #         fig=plt.figure("squat_volume")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.006,0.069))
                #         Lfig_volume_squat.append(fig2img(fig))
                #     if critere=="surface":
                #         fig=plt.figure("squat_surface")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.090,0.24))
                #         Lfig_surface_squat.append(fig2img(fig))
                #     if critere=="zai":
                #         fig=plt.figure("squat_zai")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.62,12.58))
                #         Lfig_zai_squat.append(fig2img(fig))
                #     if critere=="zrr":
                #         fig=plt.figure("squat_zrr")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.3,5.74))
                #         Lfig_zrr_squat.append(fig2img(fig))
                #     if critere=="TM":
                #         fig=plt.figure("squat_TM")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.2,2.06))
                #         Lfig_TM_squat.append(fig2img(fig))
                #     if critere=="RM":
                #         fig=plt.figure("squat_RM")
                #         ax=plt.subplot(111)
                #         ax.plot(xpos[:i+1],result[:i+1],label=robot)

                #         ax.set_xlabel(xlegend)
                #         ax.set_title("walk")
                #         ax.set_ylabel(critere)
                #         ax.set_xlim(xlim)
                #         ax.set_ylim((0.0011,9))
                #         Lfig_RM_squat.append(fig2img(fig))
                    



            # frame_one = Limg[0]
            # frame_one.save("results_humanoid_2023/results/gif/"+robot+critere+movement+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)


# frame_one = Lfig_volume_walk[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_walk_volume"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_volume_stair[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_stair_volume"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_volume_squat[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_squat_volume"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)

# frame_one = Lfig_surface_walk[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_walk_surface"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_surface_stair[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_stair_surface"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_surface_squat[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_squat_surface"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)

# frame_one = Lfig_zai_walk[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_walk_zai"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_zai_stair[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_stair_zai"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_zai_squat[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_squat_zai"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)

# frame_one = Lfig_zrr_walk[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_walk_zrr"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_zrr_stair[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_stair_zrr"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_zrr_squat[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_squat_zrr"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)

# frame_one = Lfig_TM_walk[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_walk_TM"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_TM_stair[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_stair_TM"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_TM_squat[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_squat_TM"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)

# frame_one = Lfig_RM_walk[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_walk_RM"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_RM_stair[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_stair_RM"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
# frame_one = Lfig_RM_squat[0]
# frame_one.save("results_humanoid_2023/results/gif/all_robot_squat_RM"+".gif", format="GIF", append_images=Limg,save_all=True, duration=100, loop=0)
