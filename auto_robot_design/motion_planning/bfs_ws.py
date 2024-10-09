from itertools import product
from collections import deque

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import meshcat
from pinocchio.visualize import MeshcatVisualizer
    
from auto_robot_design.pinokla.closed_loop_jacobian import (
    constraint_jacobian_active_to_passive,
)
from auto_robot_design.pinokla.closed_loop_kinematics import (
    closedLoopProximalMount,
)
from auto_robot_design.motion_planning.ik_calculator import closed_loop_ik_pseudo_inverse
from auto_robot_design.pinokla.default_traj import add_auxilary_points_to_trajectory

from auto_robot_design.motion_planning.trajectory_ik_manager import (
    IK_METHODS, TrajectoryIKManager)

from auto_robot_design.motion_planning.utils import (
    Workspace
)


class BreadthFirstSearchPlanner:

    class Node:
        def __init__(
            self, pos, parent_index, cost=None, q_arr=None, parent=None, is_reach=None
        ):
            self.pos = pos  # Положение ноды в рабочем пространстве
            self.cost = cost  # Стоимость ноды = 1 / число обусловленности Якобиана
            self.q_arr = q_arr  # Координаты в конфигурационном пространстве
            self.parent_index = parent_index  # Индекс предыдущей ноды для bfs
            self.parent = parent  # Предыдущая нода, необязательное поле
            self.is_reach = is_reach  # Флаг, что до положение ноды можно достигнуть

        def transit_to_node(self, parent, q_arr, cost, is_reach):
            # Обновляет параметры ноды
            self.q_arr = q_arr
            self.cost = cost
            self.parent = parent
            self.is_reach = is_reach

        def __str__(self):
            return (
                str(self.pos)
                + ", "
                + str(self.q_arr)
                + ", "
                + str(self.cost)
                + ", "
                + str(self.parent_index)
            )

    def __init__(
        self,
        workspace: Workspace,
        verbose: int = 0
    ) -> None:

        self.workspace = workspace
        self.verbose = verbose
        self.num_indexes = self.workspace.mask_shape

        # Варианты движения при обходе сетки (8-связности)
        self.motion = self.get_motion_model()

    def find_workspace(self, start_pos, prev_q):
        """Поиск рабочее пространство с помощью алгоритма BFS. Алгоритм работает по сетке, которая задается в `workspace`. Обход выполняется по 4-связности.
        Перед основным циклом ищется ближайшая стартовая точка на сетке от `start_pos`. Найденная точка считается начальной и добавляется в очередь.

        Общие описание алгоритма: Из очереди достается нода и находяться для нее соседи, которые не выходят за бонды. После отбора идет проверка, что сосед это не проверенная точка (не находится в `closed_set`),
        не нахдоится в очереди (`open_set`). Если со соседом все хорошо, то запускатеся алгоритм ОК из рассматриваемой точки `current` в соседа `node`. Если решение есть, то соседнию точку считаем достижимой и добавлем ее в очередь.
        Иначе, нода заносится в `bad_nodes`, множество в которые алгоритм ОК не смог дойти. Алгоритм заканчивает работу, когда очередь опустеет
        Args:
            start_pos (np.ndarray): Стартовая точка алгоритма
            prev_q (np.ndarray): Предыдущее значение в конфигурационном пространстве

        Returns:
            Workspace: обновляет переменную `workspace` и возвращает её.
        """
        robot = self.workspace.robot

        pin.framesForwardKinematics(robot.model, robot.data, prev_q)
        if self.verbose > 1:
            pos_6d = self.workspace.robot.motion_space.get_6d_point(start_pos)
            pin.framesForwardKinematics(
                robot.model, robot.data, np.zeros(robot.model.nq))

            q = prev_q
            pin.forwardKinematics(robot.model, robot.data, q)
            ballID = "world/ball" + "_start"
            material = meshcat.geometry.MeshPhongMaterial()
            material.color = int(0x00FF00)

            viz = MeshcatVisualizer(
                robot.model, robot.visual_model, robot.visual_model)
            viz.viewer = meshcat.Visualizer().open()
            viz.viewer["/Background"].set_property("visible", False)
            viz.viewer["/Grid"].set_property("visible", False)
            viz.viewer["/Axes"].set_property("visible", False)
            viz.viewer["/Cameras/default/rotated/<object>"].set_property("position", [
                                                                         0, -0.1, 0.5])
            viz.clean()
            viz.loadViewerModel()

            material.opacity = 1
            viz.viewer[ballID].set_object(
                meshcat.geometry.Sphere(0.001), material)
            T = np.r_[np.c_[np.eye(3), pos_6d[:3]], np.array([[0, 0, 0, 1]])]
            viz.viewer[ballID].set_transform(T)
            pin.framesForwardKinematics(robot.model, robot.data, q)
            viz.display(q)

            bound_pos = product(self.workspace.bounds[0, :], self.workspace.bounds[1, :])

            for k, pos in enumerate(bound_pos):
                ballID = "world/ball" + "_bound_" + str(k)
                material = meshcat.geometry.MeshPhongMaterial()
                material.color = int(0x0000FF)
                material.opacity = 1
                pos_3d = np.array([pos[0], 0, pos[1]])
                viz.viewer[ballID].set_object(
                    meshcat.geometry.Sphere(0.003), material)
                T = np.r_[np.c_[np.eye(3), pos_3d], np.array([[0, 0, 0, 1]])]
                viz.viewer[ballID].set_transform(T)

            pin.framesForwardKinematics(robot.model, robot.data, q)
            viz.display(q)

        # Функция для заполнения сетки нодами и обхода их BFS
        # Псевдо первая нода, определяется по стартовым положению, может не лежать на сетки
        pseudo_start_node = self.Node(start_pos, -1, q_arr=prev_q)

        start_index_on_grid = self.workspace.calc_index(start_pos)
        start_pos_on_grid = self.workspace.calc_grid_position(
            start_index_on_grid)
        # Настоящая стартовая нода, которая лежит на сетки. Не имеет предков
        start_n = self.Node(start_pos_on_grid, -1)
        # Проверка достижимости стартовой ноды из псевдо ноды
        self.transition_function(pseudo_start_node, start_n)
        start_n.parent = None
        
        if not start_n.is_reach:
            raise Exception("Start position of workspace is not reachable")

        del pseudo_start_node, start_index_on_grid, start_pos_on_grid
        # Словари для обхода bfs
        open_set, closed_set, bad_nodes = dict(), dict(), dict()
        queue = deque()
        open_set[self.calc_grid_index(start_n)] = start_n

        if self.verbose > 0:
            points = []
            point = self.workspace.bounds[:, 0]
            k, m = 0, 0
            while point[1] <= self.workspace.bounds[1, 1]:

                while point[0] <= self.workspace.bounds[0, 1]:
                    points.append(point)
                    m += 1
                    point = self.workspace.bounds[:, 0] + np.array(
                        self.workspace.resolution) * np.array([m, k])
                k += 1
                m = 0
                point = self.workspace.bounds[:, 0] + np.array(
                    self.workspace.resolution) * np.array([m, k])

            points = np.array(points)
            plt.plot(points[:, 0], points[:, 1], "xy")

        queue.append(self.calc_grid_index(start_n))
        while len(open_set) != 0:
            # Вытаскиваем первую из очереди ноду
            c_id = queue.popleft()
            current = open_set.pop(c_id)

            closed_set[c_id] = current

            if self.verbose > 1:
                viz.display(current.q_arr)
                boxID = "world/box" + "_ws_" + str(c_id)
                material = meshcat.geometry.MeshPhongMaterial()
                material.opacity = 0.5
                if current.is_reach:
                    plt.plot(current.pos[0], current.pos[1], "xc")
                    # time.sleep(0.5)
                    material.color = int(0x00FF00)
                else:
                    material.color = int(0xFF0000)
                    plt.plot(current.pos[0], current.pos[1], "xr")
                pos_3d = np.array([current.pos[0], 0, current.pos[1]])
                size_box = np.array(
                    [self.workspace.resolution[0], 0.001, self.workspace.resolution[1]])
                viz.viewer[boxID].set_object(
                    meshcat.geometry.Box(size_box), material)
                T = np.r_[np.c_[np.eye(3), pos_3d], np.array([[0, 0, 0, 1]])]
                viz.viewer[boxID].set_transform(T)
            # time.sleep(2)

            if self.verbose > 0:
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    "key_release_event",
                    lambda event: [exit(0) if event.key == "escape" else None],
                )
                if len(closed_set.keys()) % 1 == 0:
                    plt.pause(0.001)

            # Соседние ноды
            neigb_node = {}
            for i, moving in enumerate(self.motion):
                new_pos = current.pos + moving[:-1] * self.workspace.resolution
                node = self.Node(new_pos, c_id)

                # Проверка что ноды не вышли за бонды
                if self.verify_node(node):
                    n_id = self.calc_grid_index(node)
                    neigb_node[n_id] = node
                else:
                    continue

            for n_id, node in neigb_node.items():
                if (n_id not in closed_set) and (n_id not in open_set) and (n_id not in bad_nodes):
                    self.transition_function(current, node)
                    if node.is_reach:
                        open_set[n_id] = node
                        queue.append(n_id)
                    else:
                        bad_nodes[n_id] = node
                        if self.verbose > 0:
                            plt.plot(node.pos[0], node.pos[1], "xr")

                    if self.verbose > 1:
                        line_id = "world/line" + "_from_" + \
                            str(c_id) + "_to_" + str(n_id)
                        verteces = np.zeros((2, 3))
                        verteces[0, :] = self.workspace.robot.motion_space.get_6d_point(current.pos)[
                            :3]
                        verteces[1, :] = self.workspace.robot.motion_space.get_6d_point(node.pos)[
                            :3]

                        material = meshcat.geometry.LineBasicMaterial()
                        material.opacity = 1
                        material.linewidth = 50
                        if bool(node.is_reach):
                            material.color = 0x66FFFF
                        else:
                            material.color = 0x990099
                        pts_meshcat = meshcat.geometry.PointsGeometry(
                            verteces.astype(np.float32).T)
                        viz.viewer[line_id].set_object(
                            meshcat.geometry.Line(pts_meshcat, material))

            if self.verbose > 0:
                if not bool(current.is_reach):
                    plt.plot(current.pos[0], current.pos[1], "xr")
                else:
                    plt.plot(current.pos[0], current.pos[1], "xc")

        self.workspace.updated_by_bfs(closed_set)
        return self.workspace

    def transition_function(self, from_node: Node, to_node: Node):
        # Функция для перехода от одной ноды в другую.
        # По сути рассчитывает IK, где стартовая точка `from_node` (известны кушки)
        # в `to_node`
        robot = self.workspace.robot
        ee_id = robot.model.getFrameId(robot.ee_name)
        robot_ms = robot.motion_space
        q, min_feas, is_reach = closed_loop_ik_pseudo_inverse(
            robot.model,
            robot.constraint_models,
            robot_ms.get_6d_point(to_node.pos),
            ee_id,
            onlytranslation=True,
            q_start=from_node.q_arr,
        )

        if is_reach:

            dq_dqmot, __ = constraint_jacobian_active_to_passive(
                robot.model,
                robot.data,
                robot.constraint_models,
                robot.constraint_data,
                robot.actuation_model,
                q,
            )

            pin.framesForwardKinematics(robot.model, robot.data, q)
            Jfclosed = (
                pin.computeFrameJacobian(
                    robot.model, robot.data, q, ee_id, pin.LOCAL_WORLD_ALIGNED
                )
                @ dq_dqmot
            )
            # Подсчет числа обусловленности Якобиана или индекса маневренности
            __, S, __ = np.linalg.svd(
                Jfclosed[robot.motion_space.indexes, :], hermitian=True
            )

            dext_index = np.abs(S).max() / np.abs(S).min()
            to_node.transit_to_node(
                from_node, q, 1 / dext_index, bool(is_reach)
            )
        else:
            dext_index = np.inf

    def calc_grid_index(self, node):
        idx = self.workspace.calc_index(node.pos)

        grid_index = 0
        for k, id in enumerate(idx):
            grid_index += id * np.prod(self.num_indexes[:k])

        return grid_index

    def verify_node(self, node):
        pos = node.pos
        self.workspace.mask_shape
        if np.all(pos >= self.workspace.bounds[:, 0] - self.workspace.resolution*0.9) and np.all(pos <= self.workspace.bounds[:, 1] + self.workspace.resolution*0.9):
            return True
        return False

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [
            # [1, -1, np.sqrt(2)],
            [1, 0, 1],
            # [1, 1, np.sqrt(2)],
            [0, 1, 1],
            # [-1, 1, np.sqrt(2)],
            [-1, 0, 1],
            # [-1, -1, np.sqrt(2)],
            [0, -1, 1],
        ]

        return motion


if __name__ == "__main__":
    from auto_robot_design.generator.topologies.bounds_preset import (
        get_preset_by_index_with_bounds,
    )
    from auto_robot_design.description.builder import (
        ParametrizedBuilder,
        URDFLinkCreater3DConstraints,
        jps_graph2pinocchio_robot_3d_constraints,
    )

    builder = ParametrizedBuilder(URDFLinkCreater3DConstraints)

    gm = get_preset_by_index_with_bounds(1)
    x_centre = gm.generate_central_from_mutation_range()
    graph_jp = gm.get_graph(x_centre)

    robo, __ = jps_graph2pinocchio_robot_3d_constraints(
        graph_jp, builder=builder)

    center_bound = np.array([0, -0.3])
    size_box_bound = np.array([0.3, 0.3])

    start_pos = center_bound
    pos_6d = np.zeros(6)
    pos_6d[[0, 2]] = start_pos

    id_ee = robo.model.getFrameId(robo.ee_name)

    pin.framesForwardKinematics(robo.model, robo.data, np.zeros(robo.model.nq))

    init_pos = robo.data.oMf[id_ee].translation[[0, 2]]
    traj_init_to_center = add_auxilary_points_to_trajectory(
        ([start_pos[0]], [start_pos[1]]), init_pos
    )

    point_6d = robo.motion_space.get_6d_traj(np.array(traj_init_to_center).T)
    ik_manager = TrajectoryIKManager()
    ik_manager.register_model(robo.model, robo.constraint_models)
    ik_manager.set_solver("Closed_Loop_PI")
    poses_6d, q_fixed, constraint_errors,reach_array = ik_manager.follow_trajectory(point_6d)

    # poses_6d, q_fixed, constraint_errors, reach_array = (
    #     closed_loop_pseudo_inverse_follow(
    #         robo.model,
    #         robo.data,
    #         robo.constraint_models,
    #         robo.constraint_data,
    #         robo.ee_name,
    #         point_6d,
    #     )
    # )
    # start_pos = init_pos
    # pos_6d = np.zeros(6)

    # q = np.zeros(robo.model.nq)
    q = q_fixed[-1]

    bounds = np.array(
        [
            [-size_box_bound[0] / 2 - 0.001, size_box_bound[0] / 2],
            [-size_box_bound[1] / 2, size_box_bound[1] / 2],
        ]
    )
    bounds[0, :] += center_bound[0]
    bounds[1, :] += center_bound[1]

    workspace = Workspace(robo, bounds, np.array([0.01, 0.01]))
    ws_bfs = BreadthFirstSearchPlanner(workspace, 1)
    workspace = ws_bfs.find_workspace(start_pos, q)

    # dext_index = [1 / n.cost for n in viewed_nodes.values()]

    # print(np.nanmax(dext_index), np.nanmin(dext_index))

    # workspace.updated_by_bfs(viewed_nodes)

    print(workspace.reachabilty_mask)
