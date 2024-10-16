import csv


import pathlib


import time


from copy import deepcopy


import concurrent.futures


import dill


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


from joblib import cpu_count


from tqdm import tqdm


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
        robot, _ = jps_graph2pinocchio_robot_3d_constraints(graph, self.builder)
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

    def _calculate_batch(self, joint_poses_batch: np.ndarray):
        bathch_result = []
        cpus = cpu_count() - 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
            futures = [
                executor.submit(self._find_workspace, i) for i in joint_poses_batch
            ]
            for future in concurrent.futures.as_completed(futures):
                bathch_result.append(future.result())
        return bathch_result

    def start(self, num_points, size_batch):
        self.graph_manager.generate_central_from_mutation_range()
        low_bnds = [value[0] for value in self.graph_manager.mutation_ranges.values()]
        up_bnds = [value[1] for value in self.graph_manager.mutation_ranges.values()]
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

    def get_workspace_by_indexes(self, indexes):
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
        return self.df.loc[indexes].values[:, : self.params_size]

    def get_graphs_by_indexes(self, indexes):
        desigm_parameters = self.get_design_parameters_by_indexes(indexes)
        return [
            self.graph_manager.get_graph(des_param) for des_param in desigm_parameters
        ]


def calc_criteria(id_design, joint_poses, graph_manager, builder, reward_manager):

    graph = graph_manager.get_graph(joint_poses)

    fixed_robot, free_robot = jps_graph2pinocchio_robot_3d_constraints(graph, builder)

    reward_manager.precalculated_trajectories = None

    _, partial_rewards, _ = reward_manager.calculate_total(
        fixed_robot, free_robot, builder.actuator["default"]
    )

    return id_design, partial_rewards


def parallel_calculation_rew_manager(indexes, dataset, reward_manager):
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

        self.datasets = [] + [Dataset(path) for path in path_to_dirs]

    def get_indexes_cover_ellipse(self, ellipse: Ellipse):

        return [
            dataset.get_all_design_indexes_cover_ellipse(ellipse)
            for dataset in self.datasets
        ]

    def sorted_indexes_by_reward(self, indexes, num_samples, reward_manager):

        sorted_indexes = []

        for k, dataset in enumerate(self.datasets):

            sample_indexes = np.random.choice(indexes[k].flatten(), num_samples)
            df = parallel_calculation_rew_manager(
                sample_indexes, dataset, reward_manager
            )

            df.sort_values(["reward"], ascending=False, inplace=True)

            sorted_indexes.append(df.index)

        return sorted_indexes


if __name__ == "__main__":

    dataset = Dataset("D:\\Files\\Working\\auto-robotics-design\\test_top_8")

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

    time_start = time.perf_counter()
    parallel_calculation_rew_manager(df_upd.head(200).index, dataset, reward_manager)
    time_end = time.perf_counter()

    print(f"Time spent {time_end - time_start}")

    many_dataset = ManyDatasetAPI(
        [
            "D:\\Files\\Working\\auto-robotics-design\\test_top_8",
            "D:\\Files\\Working\\auto-robotics-design\\test",
        ]
    )

    cover_design_indexes = many_dataset.get_indexes_cover_ellipse(
        Ellipse(np.array([0.04, -0.31]), 0, np.array([0.04, 0.01]))
    )
    many_dataset.sorted_indexes_by_reward(cover_design_indexes, 10, reward_manager)
