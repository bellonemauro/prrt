import random
from abc import ABCMeta
from pathlib import Path
from typing import List

import matplotlib.image as mpimg
import numpy as np

from prrt.helper import INT_MAX
from prrt.primitive import PoseR2S1, PointR2


class KDPair(object):
    """
    K (PTG index) <=> Distance Pair
    """

    def __init__(self, k: int, d: float):
        self.k = k
        self.d = d


class Grid(metaclass=ABCMeta):
    """
    An ixj cells grid, each cell contains a list of zero or more
    (k,d_min) pairs
    """

    def __init__(self, size: float, resolution: float):
        self._resolution = resolution
        self._size = size
        self._half_cell_count_x = int(size / resolution)
        self._half_cell_count_y = self._half_cell_count_x
        self._cell_count_x = 2 * self._half_cell_count_x + 1
        self._cell_count_y = self._cell_count_x
        self._cells = np.empty((self._cell_count_x, self._cell_count_y), dtype=object)

    def x_to_ix(self, x: float) -> int:
        """
        Maps x position to column index in dmap
        :param x: x position in WS
        :return: column index of cell in dmap
        """
        return int(np.floor(x / self._resolution) + self._half_cell_count_x)

    def y_to_iy(self, y: float) -> int:
        """
        Maps y position to row index in dmap
        :param y: x position in WS
        :return: row index of cell in dmap
        """
        return int(np.floor(y / self._resolution) + self._half_cell_count_y)

    def idx_to_x(self, idx: int) -> float:
        """
        Map grid cell index to x position
        :param idx: column index in dmap
        :return: x position in WS
        """
        return idx * self._resolution - self._size

    def idx_to_y(self, idy: int) -> float:
        """
        Map grid cell index to x position
        :param idy: column index in dmap
        :return: y position in WS
        """
        return idy * self._resolution - self._size

    def cell_by_pos(self, pos: PointR2) -> List[KDPair]:
        ix = self.x_to_ix(pos.x)
        iy = self.y_to_iy(pos.y)
        if ix >= self._cell_count_x or iy >= self._cell_count_y:
            return None
        if ix < 0 or iy < 0:
            return None
        return self._cells[iy][ix]

    @property
    def cell_count_x(self):
        return self._cell_count_x

    @property
    def cell_count_y(self):
        return self._cell_count_y


class ObstacleGrid(Grid):
    def update_cell(self, ix: int, iy: int, k: int, d: float):
        # if this is the first entry create a new pair and return
        if self._cells[ix][iy] is None:
            self._cells[ix][iy] = [KDPair(k, d)]
            return
        # Cell already has an entry or more.
        # check if one exist for the same k
        for k_d_pair in self._cells[ix][iy]:
            if k_d_pair.k == k:
                k_d_pair.d = min(k_d_pair.d, d)
                return
        # The entry is new for the given k
        self._cells[ix][iy].append(KDPair(k, d))


class CPointsGrid(Grid):
    def update_cell(self, ix: int, iy: int, k: int, n: int):
        if self._cells[ix][iy] is None:
            self._cells[ix][iy] = (INT_MAX, INT_MAX, 0, 0)
        k_min, n_min, k_max, n_max = self._cells[ix][iy]
        k_min = min(k_min, k)
        n_min = min(n_min, n)
        k_max = max(k_max, k)
        n_max = max(n_max, n)
        self._cells[ix][iy] = (k_min, n_min, k_max, n_max)


class WorldGrid(object):
    """
    Holds obstacle data for world environment
    """

    def __init__(self, map_file: str, width: float, height: float):
        file_path = Path(map_file)
        assert file_path.exists(), FileExistsError
        map_32bit = mpimg.imread(map_file)
        self.min_ix = 0
        self.min_iy = 0
        self.max_ix = map_32bit.shape[1] - 1
        self.max_iy = map_32bit.shape[0] - 1
        self.iwidth = self.max_ix - self.min_ix + 1
        self.iheight = self.max_iy - self.min_iy + 1
        self.width = width
        self.height = height
        self.x_resolution = self.width / self.iwidth
        self.y_resolution = self.height / self.iheight
        map8bit = (np.dot(map_32bit[..., :3], [1, 1, 1]))
        self.omap = (map8bit < 1.5)
        self._obstacle_buffer = []  # type: List[PointR2]

    def x_to_ix(self, x: float) -> int:
        """
        Maps x position to column index in dmap
        :param x: x position in WS
        :return: column index of cell in dmap
        """
        return int(np.floor(x / self.x_resolution))

    def y_to_iy(self, y: float) -> int:
        """
        Maps y position to row index in dmap
        :param y: x position in WS
        :return: row index of cell in dmap
        """
        return int(np.floor(y / self.y_resolution))

    def idx_to_x(self, idx: int) -> float:
        """
        Map grid cell index to x position
        :param idx: column index in dmap
        :return: x position in WS
        """
        return idx * self.x_resolution

    def idx_to_y(self, idy: int) -> float:
        """
        Map grid cell index to x position
        :param idy: column index in dmap
        :return: y position in WS
        """
        return idy * self.y_resolution

    def get_random_pose(self, bias_pose: PoseR2S1 = None, bias=0.05): #MB: the bias should be defined into the configuration file
        if bias_pose is not None:
            rand = random.uniform(0, 1)
            if rand <= bias:
                return bias_pose.copy()
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        theta = random.uniform(-np.pi, np.pi)
        return PoseR2S1(x, y, theta)

    def build_obstacle_buffer(self):
        for ix in range(self.iwidth):
            for iy in range(self.iheight):
                if self.omap[iy][ix]:
                    x = self.idx_to_x(ix)
                    y = self.idx_to_y(iy)
                    self._obstacle_buffer.append(PointR2(x, y))

    def transform_point_cloud(self, ref_pose: PoseR2S1, max_dist):
        in_region_obstacles = []  # type: List[PointR2]
        inv_pose = -ref_pose  # type: PoseR2S1
        for obstacle in self._obstacle_buffer:
            dx = obstacle.x - ref_pose.x
            dy = obstacle.y - ref_pose.y
            if abs(dx) > max_dist or abs(dy) > max_dist:
                continue
            obs_rel = inv_pose.compose_point(PointR2(obstacle.x, obstacle.y))
            in_region_obstacles.append(obs_rel)
        return in_region_obstacles
