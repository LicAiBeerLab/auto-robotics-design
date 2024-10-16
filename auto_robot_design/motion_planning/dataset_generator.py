import os
import csv
import glob
import pathlib
import time
from copy import deepcopy
import concurrent.futures
import dill
from joblib import cpu_count
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pinocchio as pin


from auto_robot_design.description.builder import (
    MIT_CHEETAH_PARAMS_DICT,
    ParametrizedBuilder,
    URDFLinkCreater3DConstraints,
    jps_graph2pinocchio_robot_3d_constraints,
)


from auto_robot_design.description.utils import draw_joint_point
from auto_robot_design.motion_planning.bfs_ws import BreadthFirstSearchPlanner
from auto_robot_design.motion_planning.utils import Workspace, build_graphs
from auto_robot_design.user_interface.check_in_ellips import (
    Ellipse,
    check_points_in_ellips,
)
from auto_robot_design.utils.append_saver import chunk_list
from auto_robot_design.utils.bruteforce import get_n_dim_linspace


WORKSPACE_ARGS_NAMES = ["bounds", "resolution", "dexterous_tolerance", "grid_shape"]


class DatasetGenerator:
    def __init__(self, graph_manager, path, workspace_args):
        """
        Initializes the DatasetGenerator.
        Args:
            graph_manager (GraphManager): The manager responsible for handling the graph operations.
            path (str): The directory path where the dataset and related files will be saved.
            workspace_args (tuple): Arguments required to initialize the workspace.
        Attributes:
            ws_args (tuple): Stored workspace arguments.
            graph_manager (GraphManager): Stored graph manager.
            path (pathlib.Path): Path object for the directory where files will be saved.
            builder (ParametrizedBuilder): Builder for creating URDF links with 3D constraints.
            params_size (int): Size of the parameters generated from the mutation range.
            ws_grid_size (int): Size of the workspace grid.
            field_names (list): List of field names for the dataset CSV file.
        Operations:
            - Creates the directory if it does not exist.
            - Draws and saves a graph image.
            - Serializes the graph manager to a pickle file.
            - Saves workspace arguments to a .npz file and writes them to an info.txt file.
            - Initializes the dataset CSV file with appropriate headers.
        """
        
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
        robot, _ = jps_graph2pinocchio_robot_3d_constraints(graph, self.builder)
        workspace = Workspace(robot, *self.ws_args[:-1])
        ws_search = BreadthFirstSearchPlanner(workspace, 0, self.ws_args[-1])

        q = pin.neutral(robot.model)
        pin.framesForwardKinematics(robot.model, robot.data, q)
        id_ee = robot.model.getFrameId(robot.ee_name)
        start_pos = robot.data.oMf[id_ee].translation[[0, 2]]

        workspace = ws_search.find_workspace(start_pos, q)

        return joint_positions, workspace.reachabilty_mask.flatten()

    def save_batch_to_dataset(self, batch, postfix=""):
        """
        Save a batch of data to the dataset file.
        This method processes a batch of data, combining joint positions and workspace grid data,
        and saves it to a CSV file. The data is rounded to three decimal places before saving.
        Args:
            batch (list): A list of tuples, where each tuple contains joint positions and workspace grid data.
            postfix (str, optional): A string to append to the dataset filename. Defaults to "".
        Returns:
            None
        """
        
        joints_pos_batch = np.zeros((len(batch), self.params_size))
        ws_grid_batch = np.zeros((len(batch), self.ws_grid_size))
        for k, el in enumerate(batch):
            joints_pos_batch[k, :] = el[0]
            ws_grid_batch[k, :] = el[1]
        sorted_batch = np.hstack((joints_pos_batch, ws_grid_batch)).round(3)
        file_dataset = self.path / ("dataset" + postfix + ".csv")
        with open(file_dataset, "a", newline="") as f_object:
            # Pass the file object and a list of column names to DictWriter()

            dict_writer_object = csv.DictWriter(f_object, fieldnames=self.field_names)
            # If the file is empty or you are adding the first row, write the header

            if f_object.tell() == 0:
                dict_writer_object.writeheader()

            writer = csv.writer(f_object)
            writer.writerows(sorted_batch)

    def _parallel_calculate_batch(self, joint_poses_batch: np.ndarray):
        bathch_result = []
        cpus = cpu_count() - 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
            futures = [
                executor.submit(self._find_workspace, i) for i in joint_poses_batch
            ]
            for future in concurrent.futures.as_completed(futures):
                bathch_result.append(future.result())
        return bathch_result

    def _calculate_batches(self, batches: np.ndarray, postfix=""):
        for batch in batches:
            bathch_result = []
            for i in batch:
                bathch_result.append(self._find_workspace(i))
            self.save_batch_to_dataset(bathch_result, postfix)

    def start(self, num_points, size_batch):
        """
        Generates a dataset by creating points within specified mutation ranges and processes them in batches.
        Args:
            num_points (int): The number of points to generate.
            size_batch (int): The size of each batch.
        Raises:
            Exception: If an error occurs during batch processing.
        Writes:
            A file named "info.txt" containing the number of points generated.
            A file named "dataset.csv" containing the concatenated results of all processed batches.
        """
        
        self.graph_manager.generate_central_from_mutation_range()
        low_bnds = [value[0] for value in self.graph_manager.mutation_ranges.values()]
        up_bnds = [value[1] for value in self.graph_manager.mutation_ranges.values()]
        vecs = get_n_dim_linspace(up_bnds, low_bnds, num_points)
        batches = list(chunk_list(vecs, size_batch))

        with open(self.path / "info.txt", "a") as file:
            file.writelines("Number of points: " + str(num_points) + "\n")

        cpus = cpu_count() - 1 if cpu_count() - 1 < len(batches) else len(batches)
        batches_chunks = list(chunk_list(batches, (len(batches) // cpus) + 1))
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=cpus
            ) as executor:
                futures = [
                    executor.submit(self._calculate_batches, batches, "_" + str(m // cpus ))
                    for m, batches in enumerate(batches_chunks)
                ]
        except Exception as e:
            print(e)
        finally:
            all_files = glob.glob(os.path.join(self.path, "*.csv"))
            df = pd.concat(
                (pd.read_csv(f, low_memory=False) for f in all_files),
                ignore_index=True,
            )

        for file in all_files:
            os.remove(file)

        pd.DataFrame(df).to_csv(self.path / "dataset.csv", index=False)

        # for num, batch in tqdm(enumerate(batches)):
        #     try:
        #         batch_results = self._parallel_calculate_batch(batch)
        #         self.save_batch_to_dataset(batch_results)
        #     except Exception as e:
        #         print(e)


class Dataset:
    def __init__(self, path_to_dir):
        """
        Initializes the DatasetGenerator with the specified directory path.
        Args:
            path_to_dir (str): The path to the directory containing the dataset and other necessary files.
        Attributes:
            path (pathlib.Path): The path to the directory as a pathlib.Path object.
            df (pd.DataFrame): The dataset loaded from 'dataset.csv'.
            dict_ws_args (dict): The workspace arguments loaded from 'workspace_arguments.npz'.
            ws_args (list): The list of workspace arguments.
            workspace (Workspace): The Workspace object initialized with the workspace arguments.
            graph_manager (GraphManager): The graph manager loaded from 'graph.pkl'.
            params_size (int): The size of the parameters generated by the graph manager.
            ws_grid_size (int): The size of the workspace grid.
            builder (ParametrizedBuilder): The builder object initialized with URDFLinkCreater3DConstraints.
        """
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

    def get_workspace_by_indexes(self, indexes):
        """
        Generates a list of workspace objects based on the provided indexes.
        Args:
            indexes (list): A list of indexes to retrieve workspace data.
        Returns:
            list: A list of workspace objects with updated robot and reachable index information.
        The function performs the following steps:
        1. Initializes an empty list to store reachable indexes.
        2. Iterates over the provided indexes to extract workspace masks and calculates reachable indexes.
        3. Retrieves graphs corresponding to the provided indexes.
        4. Builds robot configurations from the graphs.
        5. Creates a deep copy of the workspace for each index.
        6. Updates each workspace copy with the corresponding robot configuration and reachable indexes.
        """
        arr_reach_indexes = []
        for k in indexes:
            ws_mask = (
                self.df.loc[k]
                .values[self.params_size : self.params_size + self.ws_grid_size]
                .reshape(self.dict_ws_args["grid_shape"])
            )
            arr_reach_indexes.append(
                {
                    self.workspace.calc_grid_index_with_index(ind): ind
                    for ind in np.argwhere(ws_mask == 1).tolist()
                }
            )
        graphs = self.get_graphs_by_indexes(indexes)
        robot_list = list(
            build_graphs(graphs, self.builder, jps_graph2pinocchio_robot_3d_constraints)
        )
        arr_ws_outs = [deepcopy(self.workspace) for _ in range(len(indexes))]

        for k, ws_out in enumerate(arr_ws_outs):
            ws_out.robot = robot_list[k][0]
            ws_out.reachable_index = arr_reach_indexes[k]
        return arr_ws_outs

    def get_all_design_indexes_cover_ellipse(self, ellipse: Ellipse):
        """
        Get all design indexes that cover the given ellipse.
        This method calculates the indexes of designs that cover the specified ellipse
        within the workspace. It first verifies that all points on the ellipse are within
        the workspace bounds. Then, it creates a mask for the workspace points that fall
        within the ellipse and uses this mask to find the relevant design indexes.
        Args:
            ellipse (Ellipse): The ellipse object for which to find covering design indexes.
        Returns:
            numpy.ndarray: An array of indexes corresponding to designs that cover the given ellipse.
        Raises:
            Exception: If any point on the ellipse is out of the workspace bounds.
        """
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
        ws_bool_flatten = np.asarray(self.df.values[:, self.params_size :], dtype=bool)
        ell_mask_2_d = ellips_mask.flatten()[np.newaxis :]
        indexes = np.argwhere(
            np.sum(ell_mask_2_d * ws_bool_flatten, axis=1) == np.sum(ell_mask_2_d)
        )
        return indexes.flatten()

    def get_design_parameters_by_indexes(self, indexes):
        """
        Retrieve design parameters based on provided indexes.
        Args:
            indexes (list or array-like): The indexes of the rows to retrieve from the dataframe.
        Returns:
            numpy.ndarray: A 2D array containing the design parameters for the specified indexes.
        """
        return self.df.loc[indexes].values[:, : self.params_size]

    def get_graphs_by_indexes(self, indexes):
        """
        Retrieve graphs based on the provided indexes.
        Args:
            indexes (list): A list of indexes to retrieve the corresponding design parameters.
        Returns:
            list: A list of graphs corresponding to the design parameters obtained from the provided indexes.
        """
        desigm_parameters = self.get_design_parameters_by_indexes(indexes)
        return [
            self.graph_manager.get_graph(des_param) for des_param in desigm_parameters
        ]


def calc_criteria(id_design, joint_poses, graph_manager, builder, reward_manager):
    """
    Calculate the criteria for a given design based on joint poses and reward management.
    Args:
        id_design (int): Identifier for the design.
        joint_poses (list): List of joint poses.
        graph_manager (GraphManager): Instance of GraphManager to handle graph operations.
        builder (Builder): Instance of Builder to construct robots.
        reward_manager (RewardManager): Instance of RewardManager to calculate rewards.
    Returns:
        tuple: A tuple containing the design identifier and partial rewards.
    """
    graph = graph_manager.get_graph(joint_poses)
    fixed_robot, free_robot = jps_graph2pinocchio_robot_3d_constraints(graph, builder)
    reward_manager.precalculated_trajectories = None
    _, partial_rewards, _ = reward_manager.calculate_total(
        fixed_robot, free_robot, builder.actuator["default"]
    )

    return id_design, partial_rewards


def parallel_calculation_rew_manager(indexes, dataset, reward_manager):
    """
    Perform parallel calculations on a subset of a dataset using a reward manager.
    This function utilizes a process pool executor to parallelize the computation
    of criteria for a subset of the dataset. The results are then aggregated into
    a new DataFrame with updated reward values.
    Args:
        indexes (list): List of indexes to select the subset of the dataset.
        dataset (object): The dataset object containing the data and associated parameters.
        reward_manager (object): The reward manager object used for calculating rewards.
    Returns:
        pd.DataFrame: A new DataFrame containing the subset of the dataset with updated reward values.
    """
    rwd_mgrs = [reward_manager] * len(indexes)
    sub_df = dataset.df.loc[indexes]
    designs = sub_df.values[:, : dataset.params_size].round(4)
    grph_mngrs = [dataset.graph_manager] * len(indexes)
    bldrs = [dataset.builder] * len(indexes)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                calc_criteria, list(indexes), designs, grph_mngrs, bldrs, rwd_mgrs
            )
        )
    new_df = pd.DataFrame(columns=dataset.df.columns)
    for k, res in results:
        new_df.loc[k] = sub_df.loc[k]
        new_df.at[k, "reward"] = np.sum(res[1])
    new_df = new_df.dropna()
    return new_df


class ManyDatasetAPI:

    def __init__(self, path_to_dirs):
        """
        Initializes the DatasetGenerator with a list of directories.
        Args:
            path_to_dirs (list of str): A list of directory paths where datasets are located.
        Attributes:
            datasets (list of Dataset): A list of Dataset objects created from the provided directory paths.
        """
        self.datasets = [] + [Dataset(path) for path in path_to_dirs]

    def get_indexes_cover_ellipse(self, ellipse: Ellipse):
        """
        Get the indexes of all designs that cover the given ellipse.
        Args:
            ellipse (Ellipse): The ellipse object for which to find covering design indexes.
        Returns:
            list: A list of indexes from all datasets that cover the given ellipse.
        """

        return [
            dataset.get_all_design_indexes_cover_ellipse(ellipse)
            for dataset in self.datasets
        ]

    def sorted_indexes_by_reward(self, indexes, num_samples, reward_manager):
        """
        Sorts and returns indexes based on rewards for each dataset.
        Args:
            indexes (list of np.ndarray): A list of numpy arrays where each array contains indexes for corresponding datasets.
            num_samples (int): The number of samples to randomly choose from each dataset.
            reward_manager (RewardManager): An instance of RewardManager to calculate rewards.
        Returns:
            list of pd.Index: A list of pandas Index objects, each containing sorted indexes based on rewards for the corresponding dataset.
        """

        sorted_indexes = []

        for k, dataset in enumerate(self.datasets):

            sample_indexes = np.random.choice(indexes[k].flatten(), num_samples)
            df = parallel_calculation_rew_manager(
                sample_indexes, dataset, reward_manager
            )

            df.sort_values(["reward"], ascending=False, inplace=True)

            sorted_indexes.append(df.index)

        return sorted_indexes


def set_up_reward_manager(traj_6d):
    from auto_robot_design.optimization.rewards.jacobian_and_inertia_rewards import (
        HeavyLiftingReward,
        MinAccelerationCapability,
    )

    from auto_robot_design.optimization.rewards.reward_base import RewardManager

    from auto_robot_design.pinokla.calc_criterion import (
        ActuatedMass,
        EffectiveInertiaCompute,
        ManipJacobian,
        MovmentSurface,
        NeutralPoseMass,
    )

    from auto_robot_design.pinokla.criterion_agregator import CriteriaAggregator

    dict_trajectory_criteria = {
        "MASS": NeutralPoseMass(),
    }
    # criteria calculated for each point on the trajectory

    dict_point_criteria = {
        "Effective_Inertia": EffectiveInertiaCompute(),
        "Actuated_Mass": ActuatedMass(),
        "Manip_Jacobian": ManipJacobian(MovmentSurface.XZ),
    }
    # special object that calculates the criteria for a robot and a trajectory

    crag = CriteriaAggregator(dict_point_criteria, dict_trajectory_criteria)

    # set the rewards and weights for the optimization task

    acceleration_capability = MinAccelerationCapability(
        manipulability_key="Manip_Jacobian",
        trajectory_key="traj_6d",
        error_key="error",
        actuated_mass_key="Actuated_Mass",
    )

    heavy_lifting = HeavyLiftingReward(
        manipulability_key="Manip_Jacobian",
        trajectory_key="traj_6d",
        error_key="error",
        mass_key="MASS",
    )

    reward_manager = RewardManager(crag=crag)
    reward_manager.add_trajectory(traj_6d, 0)

    reward_manager.add_reward(acceleration_capability, 0, 1)
    reward_manager.add_reward(heavy_lifting, 0, 1)
    
    return reward_manager


def test_dataset_generator(name_path):
    from auto_robot_design.generator.topologies.bounds_preset import (
        get_preset_by_index_with_bounds,
    )

    gm = get_preset_by_index_with_bounds(0)
    ws_agrs = (
        np.array([[-0.05, 0.05], [-0.4, -0.3]]),
        np.array([0.01, 0.01]),
        np.array([0, np.inf]),
    )
    dataset_generator = DatasetGenerator(gm, name_path, ws_agrs)

    # jp_batch = []
    # for __ in range(10):
    #     jp_batch.append(gm.generate_random_from_mutation_range())
    # res = dataset_generator._calculate_batch(jp_batch)
    # dataset_generator.save_batch_to_dataset(res)

    dataset_generator.start(3, 50)

def test_dataset_functionality(path_to_dir):

    dataset = Dataset(path_to_dir)

    thickness = MIT_CHEETAH_PARAMS_DICT["thickness"]
    actuator = MIT_CHEETAH_PARAMS_DICT["actuator"]
    density = MIT_CHEETAH_PARAMS_DICT["density"]
    body_density = MIT_CHEETAH_PARAMS_DICT["body_density"]

    ParametrizedBuilder(
        URDFLinkCreater3DConstraints,
        density={"default": density, "G": body_density},
        thickness={"default": thickness, "EE": 0.033},
        actuator={"default": actuator},
        size_ground=np.array(MIT_CHEETAH_PARAMS_DICT["size_ground"]),
        offset_ground=MIT_CHEETAH_PARAMS_DICT["offset_ground_rl"],
    )

    df_upd = dataset.df.assign(
        total_ws=lambda x: np.sum(x.values[:, dataset.params_size :], axis=1)
        / dataset.ws_grid_size
    )

    df_upd = df_upd[df_upd["total_ws"] > 100 / dataset.ws_grid_size]
    df_upd = df_upd.sort_values(["total_ws"], ascending=False)
    from auto_robot_design.pinokla.default_traj import add_auxilary_points_to_trajectory

    des_point = np.array([-0.1, -0.35])
    traj = np.array(
        add_auxilary_points_to_trajectory(([des_point[0]], [des_point[1]]))
    ).T
    test_ws = dataset.get_workspace_by_indexes([0])[0]
    traj_6d = test_ws.robot.motion_space.get_6d_traj(traj)

    reward_manager = set_up_reward_manager(traj_6d)
    time_start = time.perf_counter()
    parallel_calculation_rew_manager(df_upd.head(200).index, dataset, reward_manager)
    time_end = time.perf_counter()

    print(f"Time spent {time_end - time_start}")

def test_many_dataset_api(list_paths):
    
    many_dataset = ManyDatasetAPI(
            list_paths
    )

    cover_design_indexes = many_dataset.get_indexes_cover_ellipse(
        Ellipse(np.array([0.04, -0.31]), 0, np.array([0.04, 0.01]))
    )
    from auto_robot_design.pinokla.default_traj import add_auxilary_points_to_trajectory

    des_point = np.array([-0.1, -0.35])
    traj = np.array(
        add_auxilary_points_to_trajectory(([des_point[0]], [des_point[1]]))
    ).T
    test_ws = many_dataset.datasets[0].get_workspace_by_indexes([0])[0]
    traj_6d = test_ws.robot.motion_space.get_6d_traj(traj)

    reward_manager = set_up_reward_manager(traj_6d)

    many_dataset.sorted_indexes_by_reward(cover_design_indexes, 10, reward_manager)

if __name__ == "__main__":

    pass