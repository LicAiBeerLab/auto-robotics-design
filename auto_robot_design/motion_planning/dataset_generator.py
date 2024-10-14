import time
import dill
import pathlib
import csv

from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from joblib import Parallel, cpu_count, delayed
from joblib import cpu_count
import concurrent.futures

import pinocchio as pin
from tqdm import tqdm

from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.description.builder import (
    jps_graph2pinocchio_robot_3d_constraints,
    MIT_CHEETAH_PARAMS_DICT,
    ParametrizedBuilder,
    URDFLinkCreater3DConstraints,
)

from auto_robot_design.user_interface.check_in_ellips import Ellipse, check_points_in_ellips
from auto_robot_design.motion_planning.utils import Workspace
from auto_robot_design.motion_planning.bfs_ws import BreadthFirstSearchPlanner
from auto_robot_design.utils.bruteforce import get_n_dim_linspace
from auto_robot_design.utils.append_saver import chunk_list

WORKSPACE_ARGS_NAMES = ["bounds", "resolution", "dexterous_tolerance", "grid_shape"]
class DatasetGenerator:

    def __init__(self, graph_manager, path, workspace_args):

        self.ws_args = workspace_args
        self.graph_manager = graph_manager
        self.path = pathlib.Path(path)
        workspace = Workspace(None, *self.ws_args[:-1])

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
        draw_joint_point(
            self.graph_manager.get_graph(
                self.graph_manager.generate_central_from_mutation_range()
            )
        )
        plt.savefig(self.path / "graph.png")
        with open(self.path / "graph.pkl", "wb") as file:
            dill.dump(self.graph_manager, file)

        wrt_lines = []
        arguments = self.ws_args + (workspace.mask_shape,)
        for nm, vls in zip(WORKSPACE_ARGS_NAMES, arguments):
            wrt_lines.append(nm + ": " + str(np.round(vls, 3)) + "\n")

        np.savez(
            self.path / "workspace_arguments.npz",
            bounds=arguments[0],
            resolution=arguments[1],
            dexterous_tolerance=arguments[2],
            grid_shape=arguments[3],
        )

        with open(self.path / "info.txt", "w") as file:
            file.writelines(wrt_lines)

        self.builder = ParametrizedBuilder(URDFLinkCreater3DConstraints)

        self.params_size = len(self.graph_manager.generate_random_from_mutation_range())
        self.ws_grid_size = np.prod(workspace.mask_shape)

        dataset_fields_names = ["jp_" + str(i) for i in range(self.params_size)]
        dataset_fields_names += ["ws_" + str(i) for i in range(self.ws_grid_size)]
        self.field_names = dataset_fields_names
        with open(self.path / "dataset.csv", "a", newline="") as f_object:
            # Pass the file object and a list of column names to DictWriter()
            dict_writer_object = csv.DictWriter(f_object, fieldnames=self.field_names)
            # If the file is empty or you are adding the first row, write the header
            if f_object.tell() == 0:
                dict_writer_object.writeheader()

    def _find_workspace(self, joint_positions: np.ndarray):

        graph = self.graph_manager.get_graph(joint_positions)
        robot, __ = jps_graph2pinocchio_robot_3d_constraints(graph, self.builder)
        workspace = Workspace(robot, *self.ws_args[:-1])
        ws_search = BreadthFirstSearchPlanner(workspace, 0, self.ws_args[-1])

        q = pin.neutral(robot.model)
        pin.framesForwardKinematics(robot.model, robot.data, q)
        id_ee = robot.model.getFrameId(robot.ee_name)
        start_pos = robot.data.oMf[id_ee].translation[[0, 2]]

        workspace = ws_search.find_workspace(start_pos, q)

        return joint_positions, workspace.reachabilty_mask.flatten()

    def save_batch_to_dataset(self, batch):

        joints_pos_batch = np.zeros((len(batch), self.params_size))
        ws_grid_batch = np.zeros((len(batch), self.ws_grid_size))
        for k, el in enumerate(batch):
            joints_pos_batch[k, :] = el[0]
            ws_grid_batch[k, :] = el[1]
        sorted_batch = np.hstack((joints_pos_batch, ws_grid_batch)).round(3)

        with open(self.path / "dataset.csv", "a", newline="") as f_object:
            # Pass the file object and a list of column names to DictWriter()
            writer = csv.writer(f_object)
            writer.writerows(sorted_batch)

    # def _calculate_batch(self, joint_poses_batch: np.ndarray):
    #     batch_results = []
    #     cpus = cpu_count() - 1
    #     batch_results = Parallel(
    #         cpus, backend="multiprocessing", verbose=100, timeout=60 * 1000
    #     )(delayed(self._find_workspace)(i) for i in joint_poses_batch)

    #     return batch_results

    def _calculate_batch(self, joint_poses_batch: np.ndarray):
        bathch_result = []
        cpus = cpu_count() - 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
            futures = [executor.submit(self._find_workspace, i) for i in joint_poses_batch]
            for future in concurrent.futures.as_completed(futures):
                bathch_result.append(future.result())
        return bathch_result

    def start(self, num_points, size_batch):
        central_jp = self.graph_manager.generate_central_from_mutation_range()
        low_bnds = [
            value[0] for key, value in self.graph_manager.mutation_ranges.items()
        ]
        up_bnds = [
            value[1] for key, value in self.graph_manager.mutation_ranges.items()
        ]

        vecs = get_n_dim_linspace(up_bnds, low_bnds, num_points)
        batches = list(chunk_list(vecs, size_batch))
        start_time = time.time()
        for num, batch in tqdm(enumerate(batches)):
            try:
                batch_results = self._calculate_batch(batch)
                self.save_batch_to_dataset(batch_results)
            except Exception as e:
                print(e)
            print(f"Tested chunk {num} / {len(batches)}")
            ellip = (time.time() - start_time) / 60
            print(f"Spending time {ellip}")

class Dataset:
    def __init__(self, path_to_dir):
        self.path = pathlib.Path(path_to_dir)
        
        self.df = pd.read_csv(self.path / "dataset.csv")
        self.dict_ws_args = np.load(self.path / "workspace_arguments.npz")
        self.ws_args = [self.dict_ws_args[name] for name in WORKSPACE_ARGS_NAMES[:-1]]
        self.workspace = Workspace(None, *self.ws_args[:-1])
        
        with open(self.path / "graph.pkl", "rb") as f:
            self.graph_manager = dill.load(f)
            
        self.params_size = len(self.graph_manager.generate_random_from_mutation_range())
        self.ws_grid_size = np.prod(self.workspace.mask_shape)
        
        self.builder = ParametrizedBuilder(URDFLinkCreater3DConstraints)

    def get_workspace_by_sample(self, sample):
        
        joint_poses = sample[:self.params_size].to_numpy()
        ws_mask = sample[self.params_size:].to_numpy().reshape(self.dict_ws_args["grid_shape"])
        
        graph = self.graph_manager.get_graph(joint_poses)
        robot, __ = jps_graph2pinocchio_robot_3d_constraints(graph, self.builder)
        reach_indexes = {self.workspace.calc_grid_index_with_index(ind): ind for ind in np.argwhere(ws_mask==1).tolist()}
        
        ws_out = deepcopy(self.workspace)
        
        ws_out.robot = robot
        ws_out.reachable_index = reach_indexes
        
        return ws_out
    
    def get_all_design_indexes_cover_ellipse(self, ellipse: Ellipse):
        points_on_ellps = ellipse.get_points(0.1).T
        
        for pt in points_on_ellps:
            if not self.workspace.point_in_bound(pt):
                raise Exception("Input ellipse out of workspace bounds")
        
        ws_points = self.workspace.points
        mask_ws_n_ellps = check_points_in_ellips(ws_points, ellipse, 0.1)
        ellips_mask = np.zeros(self.workspace.mask_shape, dtype=bool)
        for point in ws_points[mask_ws_n_ellps, :]:
            index = self.workspace.calc_index(point)
            ellips_mask[tuple(index)] = True
        ws_bool_flatten = np.asarray(self.df.values[:,self.params_size:], dtype=bool)
        ell_mask_2d = ellips_mask.flatten()[np.newaxis :]
        indexes = np.argwhere(np.sum(ell_mask_2d * ws_bool_flatten, axis=1) ==np.sum(ell_mask_2d))
        return indexes
    
    def get_df_calc_reward_manager(self, indexes, reward_manager):

        graph = gm.get_graph(gm.generate_random_from_mutation_range())
        robot, free_robot = jps_graph2pinocchio_robot(graph, builder)
        reward_manager.calculate_total(robot, free_robot, builder.actuator["default"])

if __name__ == "__main__":

    from auto_robot_design.generator.topologies.bounds_preset import (
        get_preset_by_index_with_bounds,
    )

    gm = get_preset_by_index_with_bounds(8)
    ws_agrs = (
        np.array([[-0.4, 0.4], [-0.42, -0.1]]),
        np.array([0.01, 0.01]),
        np.array([0, np.inf]),
    )
    dataset_generator = DatasetGenerator(gm, "test_top_8", ws_agrs)

    # jp_batch = []
    # for __ in range(10):
    #     jp_batch.append(gm.generate_random_from_mutation_range())
    # res = dataset_generator._calculate_batch(jp_batch)
    # dataset_generator.save_batch_to_dataset(res)
    # print(res)

    dataset_generator.start(3, 230)
    