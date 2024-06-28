from itertools import product
import time
import dill
import copy
import os

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import modern_robotics as mr
import numpy as np

from auto_robot_design.description.mechanism import JointPoint2KinematicGraph, KinematicGraph
from auto_robot_design.description.actuators import TMotor_AK80_9
from auto_robot_design.description.utils import draw_links
from auto_robot_design.pino_adapter.pino_adapter import get_pino_description
from auto_robot_design.pinokla.loader_tools import build_model_with_extensions
from testing_ground.singular_hell.workspace_search import box_workspace, calc_ee_range_of_2linker, filter_points2d_by_boxes, search_workspace_FK_nojacs, search_workspace_IK_nojacs
from auto_robot_design.description.builder import ParametrizedBuilder, DetailedURDFCreatorFixedEE#, jps_graph2pinocchio_robot

def kin_graph2pinocchio_robot(
    kinematic_graph: KinematicGraph,
    builder: ParametrizedBuilder
):
    """
    Converts a kinematic graph to a Pinocchio robot model.

    Args:
        kinematic_graph (KinematicGraph): The kinametic graph representing the robot's kinematic structure. It has to have defined main branch, span tree and link frames.
        builder (ParametrizedBuilder): The builder object used to create the kinematic graph.

    Returns:
        tuple: A tuple containing the robot model with fixed base and free base.
    """
    
    robot, ative_joints, constraints = builder.create_kinematic_graph(kinematic_graph)

    act_description, constraints_descriptions = get_pino_description(
        ative_joints, constraints
    )

    fixed_robot = build_model_with_extensions(robot.urdf(),
                                joint_description=act_description,
                                loop_description=constraints_descriptions,
                                actuator_context=kinematic_graph,
                                fixed=True)

    free_robot = build_model_with_extensions(robot.urdf(),
                                joint_description=act_description,
                                loop_description=constraints_descriptions,
                                actuator_context=kinematic_graph,
                                fixed=False)
    
    return fixed_robot, free_robot

class JPGraphHandler:
    def __init__(self, jp_graph) -> None:
        self.graph_initial = jp_graph
        self.graph = copy.deepcopy(jp_graph)
        self.name2jp = dict(map(lambda x: (x.name, x), self.graph.nodes()))
        self.name2jp_init = dict(map(lambda x: (x.name, x), self.graph_initial.nodes()))

    def mutate_graph(self, variant: dict):
        for k, v in variant.items():
            self.name2jp[k].r = np.asarray(v)

    def prepare_kinematic_graph(self, is_showed=True):
        kinematic_graph = JointPoint2KinematicGraph(self.graph)
        self.EFFECTOR_NAME = kinematic_graph.EE.name

        if is_showed:
            draw_links(kinematic_graph, self.graph)
            plt.show()
        # # draw_kinematic_graph(kinematic_graph)
        # main_branch = kinematic_graph.define_main_branch()
        # # draw_kinematic_graph(main_branch)
        # kin_tree = kinematic_graph.define_span_tree()

        # thickness = 0.04
        # density = 2700 / 2.8
        # for n in kinematic_graph.nodes():
        #     n.thickness = thickness
        #     n.density = density

        # for j in kinematic_graph.joint_graph.nodes():
        #     j.pos_limits = (-np.pi, np.pi)
        #     if j.jp.active:
        #         j.actuator = TMotor_AK80_9()
        #     j.damphing_friction = (0.05, 0)
            
        # kinematic_graph.define_link_frames()

        kinematic_graph.define_main_branch()
        kinematic_graph.define_span_tree()

        kinematic_graph.define_link_frames()
        return kinematic_graph

class WorkspaceRunner(JPGraphHandler):
    def __init__(self, jp_graph, constrain_dict,
                 j_divisions=2, is_using_FK=False,
                 path='', is_graph_saved=True, fname='g', is_using_date=True, is_saving_extra=False) -> None:
        super().__init__(jp_graph)
        self.path = path
        date = "_" + time.strftime("%Y-%m-%d_%H-%M-%S") if is_using_date else ""
        self.fname = fname + date
        self.is_using_FK = is_using_FK

        self.urdf_builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE, thickness=0.015)

        # filter varying joints
        self.varjp_dict = dict(
            filter(lambda x: x[1]["optim"] and x[0] in self.name2jp, constrain_dict.items()))
        # self.jp_list = list(self.graph.nodes())
        # self.jp_name2id = {n: i for i,n in enumerate(self.jp_list)}
        self.varjp_names = list(self.varjp_dict.keys())
        self.n_varjp = len(self.varjp_names)
        self.varjp_name2id = {n: i for i,n in enumerate(self.varjp_names)}

        self.is_saving_extra = is_saving_extra
        if is_graph_saved:
            self._save_graph_and_varjp()
        
        self.variants = self._get_all_variants(j_divisions)

    def run(self, box_sigma=2/4, prox_divisions=20, is_using_prox_history=True, 
            is_showing_graphs=False, is_showing_points=False, show_every_nth=10):
        print('Starting workspace search for graph variants...')
        t0 = time.time()

        if self.is_saving_extra:
            with open(os.path.join(self.path,self.fname + '.npy'), 'wb') as f, open(os.path.join(self.path,self.fname + '.npye'), 'wb') as f_extra:
                var_counter = 0
                for v in self.variants:
                    print('Current variant:',v)

                    self.mutate_graph(v)
                    self.kinematic_graph = self.prepare_kinematic_graph(is_showing_graphs and var_counter % show_every_nth == 0)
                    self._prepare_urdf()

                    boxes, extra_boxes = self._run_boxes(box_sigma)
                    ee_bounds = calc_ee_range_of_2linker(self.kinematic_graph)

                    points = self._prepare_points(boxes, ee_bounds, prox_divisions)
                    xyz_feas, q_feas = self._run_proximal(points, is_using_prox_history)
                    
                    if is_showing_points and var_counter % show_every_nth == 0:
                        plot_boxes_and_points(extra_boxes if self.is_using_FK else boxes, ee_bounds, ee_bounds, 
                                            None if self.is_using_FK else points[:,(0,2)], 
                                            xyz_feas[:,(0,2)])
                    var_counter += 1

                    self._save_for_variant(v, xyz_feas, q_feas, f)
                    self._save_for_variant_extra(v, points, f_extra)
        else:
            with open(os.path.join(self.path,self.fname + '.npy'), 'wb') as f:
                var_counter = 0
                for v in self.variants:
                    print('Current variant:',v)

                    self.mutate_graph(v)
                    self.kinematic_graph = self.prepare_kinematic_graph(is_showing_graphs and var_counter % show_every_nth == 0)
                    self._prepare_urdf()

                    boxes, extra_boxes = self._run_boxes(box_sigma)
                    ee_bounds = calc_ee_range_of_2linker(self.kinematic_graph)

                    points = self._prepare_points(boxes, ee_bounds, prox_divisions)
                    xyz_feas, q_feas = self._run_proximal(points, is_using_prox_history)
                    
                    if is_showing_points and var_counter % show_every_nth == 0:
                        plot_boxes_and_points(extra_boxes if self.is_using_FK else boxes, ee_bounds, ee_bounds, 
                                            None if self.is_using_FK else points[:,(0,2)], 
                                            xyz_feas[:,(0,2)])
                    var_counter += 1

                    self._save_for_variant(v, xyz_feas, q_feas, f)
                
        print(f'Search for {var_counter} variants took {time.time()-t0:.1f} seconds')

    def _save_graph_and_varjp(self):
        with open(os.path.join(self.path, self.fname + '.pkl'), 'wb') as f:
            dill.dump(self.graph, f)
            dill.dump(self.varjp_names, f)

    def _get_all_variants(self, divisions): # 16k for 2, 5M for 3
        xyz_spaces_dict = {}
        for k,v in self.varjp_dict.items():
            joint_pos = self.name2jp_init[k].r
            x_from = v['x_range'][0] + joint_pos[0]
            x_to = v['x_range'][1] + joint_pos[0]
            z_from = v['z_range'][0] + joint_pos[2]
            z_to = v['z_range'][1] + joint_pos[2]
            x_space = np.linspace(x_from, x_to, divisions if v['x_range'][1]-v['x_range'][0] else 1) #TODO test if need this check
            z_space = np.linspace(z_from, z_to, divisions if v['z_range'][1]-v['z_range'][0] else 1)

            xyz_space = [(x,0,z) for x,z in list(product(x_space, z_space))]
            xyz_spaces_dict[k] = xyz_space

        all_variants = (dict(zip(xyz_spaces_dict.keys(), values)) for values in product(*xyz_spaces_dict.values())) #TODO make without keys and y zeros
        return all_variants

    # def _mutate_graph(self, variant):
    #     for n in self.varjp_names:
    #         self.name2jp[n].r = np.asarray(variant[n])

    def _run_boxes(self, sigma):
        boxes, lname2id = box_workspace(self.kinematic_graph, sigma, return_lorder=True, 
                                        split_selected=not self.is_using_FK)

        if self.is_using_FK:
            lid_pairs = self._calc_initial_qmot(lname2id)
            if len(lid_pairs) != 2:
                raise NotImplementedError('Currently calculating Rq0 is supported for 2 motors only.')
            boxes_qmot = boxes2qmot(boxes, lid_pairs[0][0], lid_pairs[1][0], Rq01=lid_pairs[0][1], Rq02=lid_pairs[1][1])
            return boxes_qmot, boxes

        return boxes, None

    def _prepare_urdf(self):
        robo, _ = kin_graph2pinocchio_robot(self.kinematic_graph, self.urdf_builder)
        self.robo = robo
    
    def _prepare_points(self, boxes, ee_bounds, divisions):
        
        if self.is_using_FK:
            boxes_qmot = boxes

            qmot_space_1 = np.linspace(-2*np.pi, 2*np.pi, divisions*2, endpoint=False)
            qmot_space_2 = np.linspace(-2*np.pi, 2*np.pi, divisions*2, endpoint=False)
            # qm = []
            # for i, m1 in enumerate(qmot_space_1): 
            #     for j, m2 in enumerate(qmot_space_2[::-1] if i%2 else qmot_space_2):
            #         qm.append((m1, m2))
            # qmot_space = qm
            qmot_space = list(product(qmot_space_1, qmot_space_2))

            qpoints = np.asarray(qmot_space)
            inliers = filter_points2d_by_boxes(boxes_qmot, x_ind=0, y_ind=1, points=qpoints)

            qmot_space_filtered = np.column_stack((inliers[:,0], inliers[:,1]))
            print(f'{qmot_space_filtered.shape[0]} points remained after filtering out of {np.asarray(qmot_space).shape[0] // 4} points')
            return qmot_space_filtered

        else:
            ee_b = ee_bounds

            x_space = np.linspace(ee_b[0], ee_b[1], divisions)
            z_space = np.linspace(ee_b[0], ee_b[1], divisions)
            xyz_space = [(x,0,z) for x,z in list(product(x_space, z_space))]

            points = np.asarray(xyz_space)[:,(0,2)]
            inliers = filter_points2d_by_boxes(boxes, x_ind=-2, y_ind=-1, points=points)

            xyz_space_filtered = np.column_stack((inliers[:,0], np.zeros(inliers.shape[0]), inliers[:,1]))
            print(f'{xyz_space_filtered.shape[0]} points remained after filtering out of {np.asarray(xyz_space).shape[0]} points')
            return xyz_space_filtered

    def _calc_initial_qmot(self, lname2id):
        jname2Rq0 = {}
        for k,v in self.robo.actuation_model.motname2id_q.items():
            j = self.kinematic_graph.name2joint[k]
            ln1, ln2 = j.link_in.name, j.link_out.name #jname2lnames[k]
            R1, _ = mr.TransToRp(self.kinematic_graph.name2link[ln1].frame)
            R2, _ = mr.TransToRp(self.kinematic_graph.name2link[ln2].frame)
            Rq0 = R1 @ R2.T
            jname2Rq0[k] = ((lname2id.get(ln1), lname2id.get(ln2)), (Rq0[0,0], Rq0[0,2])) #cos, sin from Ry

        # print(jname2Rq0)
        lid_pairs = list(jname2Rq0.values()) #TODO add sorting as in q vector
        # print(lid_pairs)
        return lid_pairs

    def _run_proximal(self, points, is_using_history):
        t0 = time.time()

        if self.is_using_FK:
            print('Starting FK proximal...')
            qmot_space_filtered = points
            # workspace_xyz, available_q, jacs6d, jacsC6d, Ldsdq_6d = search_workspace_FK(self.robo.model, self.robo.data, self.EFFECTOR_NAME, BASE_FRAME, np.array(
            #     qmot_space_filtered), self.robo.actuation_model, self.robo.constraint_models, self.robo.constraint_data, viz, None)
            workspace_xyz, available_q = search_workspace_FK_nojacs(self.robo.model, self.robo.data, self.EFFECTOR_NAME, np.array(
                qmot_space_filtered), self.robo.actuation_model, self.robo.constraint_models, viz=None, is_using_history=is_using_history)
            print(f'Proximal took {time.time()-t0:.1f} seconds')
            print(f'Coverage of filtered q points: {len(available_q)/(len(qmot_space_filtered)):.3f}') #TODO removejacs

            return workspace_xyz, available_q

        else:
            print('Starting IK proximal...')
            xyz_space_filtered = points
            # jointspace_q, available_xyz, jacsik6d, jacsCik6d = search_workspace_IK(self.robo.model, self.robo.data, EFFECTOR_NAME, BASE_FRAME, np.asarray(
            #     xyz_space_filtered), self.robo.actuation_model, self.robo.constraint_models, self.robo.constraint_data, viz)
            jointspace_q, available_xyz = search_workspace_IK_nojacs(self.robo.model, self.robo.data, self.EFFECTOR_NAME, np.asarray(
                xyz_space_filtered), self.robo.constraint_models, self.robo.constraint_data, viz=None, is_using_history=is_using_history)
            print(f'Proximal took {time.time()-t0:.1f} seconds')
            print(f"Coverage of filtered xyz points: {len(available_xyz)/len(xyz_space_filtered):.3f}")

            return available_xyz, jointspace_q

    def _save_for_variant(self, variant, xyz_points, q_points, file):
        ordered_variant = np.zeros(2 * self.n_varjp)
        for i,n in enumerate(self.varjp_names):
            ordered_variant[2*i] = variant[n][0]
            ordered_variant[2*i+1] = variant[n][2]
        # print(ordered_variant)
        np.save(file, ordered_variant)
        np.save(file, np.hstack((xyz_points[:,(0,2)], q_points)))

    def _save_for_variant_extra(self, variant, points, file):
        # ordered_variant = np.zeros(2 * self.n_varjp)
        # for i,n in enumerate(self.varjp_names):
        #     ordered_variant[2*i] = variant[n][0]
        #     ordered_variant[2*i+1] = variant[n][2]
        # print(ordered_variant)
        # np.save(file, ordered_variant)
        np.save(file, points[:,(0,-1)])

def reduce_constr_dict(constr_dict, n_jp_to_leave):
    new_dict = copy.deepcopy(constr_dict)
    counter = 0
    for k,v in new_dict.items():
        if v['optim']:
            if counter >= n_jp_to_leave:
                new_dict[k]['optim'] = False
            counter += 1

    return new_dict, counter >= n_jp_to_leave

def plot_boxes_and_points(boxes, xlim, ylim, points2d=None, points2d_final=None, x_ind=-2, y_ind=-1):
    fig = plt.figure(dpi=150)
    ax = plt.gca()

    for b in boxes:#[::10]:
        xlb, xub = b[x_ind,0], b[x_ind,1]
        ylb, yub = b[y_ind,0], b[y_ind,1]
        ax.add_patch(Rectangle((xlb,ylb),xub-xlb,yub-ylb,linewidth=1/2,edgecolor='none',facecolor='k',alpha=.1))
    if points2d is not None:
        plt.scatter(points2d[:,0], points2d[:,1],c='r',linewidths=0,s=2)
    if points2d_final is not None:
        plt.scatter(points2d_final[:,0], points2d_final[:,1],c='b',linewidths=0,s=5)

    plt.xlim(xlim)
    plt.ylim(ylim)

    ax.set_aspect(1)
    plt.tight_layout()
    plt.show()

from testing_ground.singular_hell.box_utils import atan2_intervals, mul_intervals, subtr_intervals, gain_interval


def boxes2qmot(boxes, lid_pair1, lid_pair2, Rq01, Rq02):
    inp = boxes.copy()

    w = 6
    def get_cos(box, lid):
        return box[w*lid+2, :] if lid is not None else np.array((1.,1.))

    def get_sin(box, lid):
        return box[w*lid+3, :] if lid is not None else np.array((0.,0.))

    lid1 = lid_pair1[0] 
    lid2 = lid_pair1[1]
    lid3 = lid_pair2[0]
    lid4 = lid_pair2[1]

    boxes_qmot = []
    q10 = np.arctan2(Rq01[1], Rq01[0])
    q20 = np.arctan2(Rq02[1], Rq02[0])
    # print(q10/np.pi*180, q20/np.pi*180)

    for b in inp:
        # atang1 = atan2_intervals(b[sin_ind1,:],b[cos_ind1,:])
        # atang2 = atan2_intervals(b[sin_ind2,:],b[cos_ind2,:])           
        # xlb, xub = atang1# - (np.sign(atang1)-np.ones(2))*np.pi
        # ylb, yub = atang2# - (np.sign(atang2)-np.ones(2))*np.pi

        cosq1 = mul_intervals(get_cos(b,lid1), get_cos(b,lid2)) + mul_intervals(get_sin(b,lid1), get_sin(b,lid2))
        sinq1 = subtr_intervals(mul_intervals(get_sin(b,lid1), get_cos(b,lid2)), mul_intervals(get_cos(b,lid1), get_sin(b,lid2)))
        cosq2 = mul_intervals(get_cos(b,lid3), get_cos(b,lid4)) + mul_intervals(get_sin(b,lid3), get_sin(b,lid4))
        sinq2 = subtr_intervals(mul_intervals(get_sin(b,lid3), get_cos(b,lid4)), mul_intervals(get_cos(b,lid3), get_sin(b,lid4)))

        # cosq1_shift = gain_interval(cosq1,Rq01[0]) + gain_interval(sinq1,Rq01[1])
        # sinq1_shift = subtr_intervals(gain_interval(sinq1,Rq01[0]), gain_interval(cosq1,Rq01[1]))
        # cosq2_shift = gain_interval(cosq2,Rq02[0]) + gain_interval(sinq2,Rq02[1])
        # sinq2_shift = subtr_intervals(gain_interval(sinq2,Rq02[0]), gain_interval(cosq2,Rq02[1]))

        atang1 = atan2_intervals(sinq1,cosq1)  +q10
        atang2 = atan2_intervals(sinq2,cosq2)  +q20     
        boxes_qmot.append(np.vstack((atang1, atang2)))

        # xlb, xub = atang1# - (np.sign(atang1)-np.ones(2))*np.pi
        # ylb, yub = atang2# - (np.sign(atang2)-np.ones(2))*np.pi
    return boxes_qmot


class WorkspaceLoader(JPGraphHandler):
    def __init__(self, path='', fname='g') -> None:
        self.path = path
        self.fname = fname

        with open(os.path.join(self.path, self.fname + '.pkl'), 'rb') as f:
            graph_jp = dill.load(f)
            self.varjp_names = dill.load(f)
        self.n_varjp = len(self.varjp_names)
        super().__init__(graph_jp)

        self.data = self.load_generator(os.path.join(self.path,self.fname + '.npy'))
        fname_extra = os.path.join(self.path,self.fname + '.npye')
        if os.path.isfile(fname_extra):
            self.extra_data = self.load_generator(fname_extra)
        else:
            self.extra_data = None
    
    @staticmethod
    def load_generator(filename):
        with open(filename, "rb") as f:
            while True:
                try:
                    yield np.load(f)
                except EOFError:
                    break

    def restore_variant(self, arr):
        variant = {}
        for i in range(self.n_varjp):
            variant[self.varjp_names[i]] = (arr[2*i],0.,arr[2*i+1])
        return variant
