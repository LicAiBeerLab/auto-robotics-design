#%%
import time
from auto_robot_design.pinokla.closed_loop_kinematics import closedLoopProximalMount

from auto_robot_design.pinokla.calc_criterion import search_workspace
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

import pinocchio as pin
from auto_robot_design.pinokla.closed_loop_kinematics import ForwardK

from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE, jps_graph2pinocchio_robot
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator


def search_workspace_smart(
    model,
    data,
    effector_frame_name: str,
    base_frame_name: str,
    q_space: np.ndarray,
    actuation_model,
    constraint_models,
    viz=None,
):
    """Iterate forward kinematics over q_space and try to minimize constrain value.

    Args:
        model (_type_): _description_
        data (_type_): _description_
        effector_frame_name (str): _description_
        base_frame_name (str): _description_
        q_space (np.ndarray): _description_
        actuation_model (_type_): _description_
        constraint_models (_type_): _description_
        viz (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    c = 0
    q_start = pin.neutral(model)
    workspace_xyz = np.empty((len(q_space), 3))
    available_q = np.empty((len(q_space), len(q_start)))
    for q_sample in q_space:

        q_dict_mot = zip(actuation_model.idqmot, q_sample)
        for key, value in q_dict_mot:
            q_start[key] = value
        q3, error = ForwardK(
            model,
            constraint_models,
            actuation_model,
            q_start,
            150,
        )

        if error < 1e-11:
            if viz:
                viz.display(q3)
                time.sleep(0.005)
            q_start = q3
            pin.framesForwardKinematics(model, data, q3)
            id_effector = model.getFrameId(effector_frame_name)
            id_base = model.getFrameId(base_frame_name)
            effector_pos = data.oMf[id_effector].translation
            base_pos = data.oMf[id_base].translation
            print('pos',effector_pos)
            transformed_pos = effector_pos - base_pos

            workspace_xyz[c] = transformed_pos
            available_q[c] = q3
            c += 1
    return (workspace_xyz[0:c], available_q[0:c])


gen = TwoLinkGenerator()
builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE)
graphs_and_cons = gen.get_standard_set()
np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)

graph_jp, __ = graphs_and_cons[0]
robo, robo_free = jps_graph2pinocchio_robot(graph_jp, builder)

q0 = closedLoopProximalMount(
    robo.model,
    robo.data,
    robo.constraint_models,
    robo.constraint_data,
    max_it=100,
)


viz = MeshcatVisualizer(robo.model, robo.visual_model, robo.visual_model)
viz.viewer = meshcat.Visualizer().open()
viz.clean()
viz.loadViewerModel()
viz.display(q0)


EFFECTOR_NAME = "EE"
BASE_FRAME = "G"
#%%
# q_space_mot_1 = np.linspace(-np.pi, np.pi, 3)
# q_space_mot_2 = np.linspace(-np.pi, np.pi, 3)
q_space_mot_1 = np.asarray([0])
q_space_mot_2 = np.linspace(-np.pi, np.pi, 3)
q_mot_double_space = list(product(q_space_mot_1, q_space_mot_2))

workspace_xyz, available_q = search_workspace_smart(robo.model, robo.data, EFFECTOR_NAME, BASE_FRAME, np.array(
    q_mot_double_space), robo.actuation_model, robo.constraint_models, viz)


print("Coverage q " + str(len(available_q)/(len(q_mot_double_space))))


plt.figure()
plt.scatter(workspace_xyz[:, 0],  workspace_xyz[:, 2])
plt.title("WorkScape")
plt.xlabel("X")
plt.ylabel("Ya popravil")


plt.show()

# %%
