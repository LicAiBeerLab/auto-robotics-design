import numpy as np


def create_bounded_box(center: np.ndarray, bound_range: np.ndarray):
    bounds = np.array(
        [
            [-bound_range[0] / 2 - 0.001, bound_range[0] / 2],
            [-bound_range[1] / 2, bound_range[1] / 2],
        ]
    )
    bounds[0, :] += center[0]
    bounds[1, :] += center[1]

    return bounds


class Workspace:
    def __init__(self, robot, bounds, resolution: np.ndarray):
        ''' Class for working workspace of robot like grid with `resolution` and `bounds`. 
        Grid's indices go from bottom-right to upper-left corner of bounds

        '''
        self.robot = robot
        self.resolution = resolution
        self.bounds = bounds

        # TODO: Need to change pattern. For example first create a workspace and BFS work and update with it.
        num_indexes = (np.max(bounds, 1) - np.min(bounds, 1)) / self.resolution
        self.mask_shape = np.zeros_like(num_indexes)
        self.bounds = np.zeros_like(bounds)
        # Bounds correction for removing ucertainties with indices. Indices was calculated with minimal `bounds` and `resolution`
        for id, idx_value in enumerate(num_indexes):
            residue_div = np.round(idx_value % 1, 6)

            check_bound_size = np.isclose(residue_div, 1.0)
            check_min_bound = np.isclose(
                bounds[id, 0] % self.resolution[id], 0)
            check_max_bound = np.isclose(
                bounds[id, 1] % self.resolution[id], 0)
            if check_bound_size and check_min_bound and check_max_bound:
                self.bounds[id, :] = bounds[id, :]
                self.mask_shape[id] = num_indexes[id]
            else:
                self.bounds[id, 1] = np.round(
                    bounds[id, 1] + bounds[id, 1] % self.resolution[id], 4)
                self.bounds[id, 0] = np.round(
                    bounds[id, 0] - bounds[id, 0] % self.resolution[id], 4)
                self.mask_shape[id] = np.ceil(
                    (self.bounds[id, 1] - self.bounds[id, 0]) /
                    self.resolution[id]
                )
        self.mask_shape = np.asarray(self.mask_shape.round(4), dtype=int) + 1
        self.bounds = self.bounds.round(4)
        self.set_nodes = {}
        # self.grid_nodes = np.zeros(tuple(self.mask_shape), dtype=object)

    def updated_by_bfs(self, set_expl_nodes):

        self.set_nodes = set_expl_nodes

    def calc_grid_position(self, indexes):

        pos = indexes * self.resolution + self.bounds[:, 0]

        return pos

    def calc_index(self, pos):
        return np.round((pos - self.bounds[:, 0]) / self.resolution).astype(int)

    @property
    def reachabilty_mask(self):

        mask = np.zeros(tuple(self.mask_shape), dtype=float)

        for node in self.set_nodes.values():
            index = self.calc_index(node.pos)
            mask[tuple(index)] = node.is_reach

        return mask
