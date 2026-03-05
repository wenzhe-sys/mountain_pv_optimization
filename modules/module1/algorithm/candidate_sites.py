"""
候选箱变站址生成模块
=====================
生成可行的箱变安装候选位置，供 Arc-Flow MILP 和分支定价算法使用。

策略：
1. 逆变器重心候选点
2. K-Means 聚类中心
3. 网格包围盒采样
4. 不可建区域过滤 + 网格对齐
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class CandidateSiteGenerator:
    """候选箱变站址生成器"""

    def __init__(self, instance_data: Dict, inverter_coords: List[Tuple[float, float]]):
        self.grid_size = instance_data["terrain_data"]["grid_size"]
        self.inverter_coords = inverter_coords
        self.n_inverters = len(inverter_coords)
        self.substation_coord = tuple(instance_data["equipment_params"]["substation"]["coord"])

        # 可建区域矩阵
        self.buildable_matrix = np.array(instance_data["terrain_data"].get("buildable_matrix", []))
        self.slope_matrix = np.array(instance_data["terrain_data"].get("slope_matrix", []))

        # 箱变参数
        self.Q_box_options = instance_data["equipment_params"]["transformer"]["Q_box_options"]
        # 3200kVA 最多连10台逆变器，1600kVA 最多连5台
        self.capacity_map = {1600: 5, 3200: 10}

    def _align_to_grid(self, coord: Tuple[float, float]) -> Tuple[float, float]:
        """将坐标对齐到网格"""
        x, y = coord
        return (round(x / self.grid_size) * self.grid_size,
                round(y / self.grid_size) * self.grid_size)

    def _is_buildable(self, coord: Tuple[float, float]) -> bool:
        """检查坐标是否在可建区域内"""
        if self.buildable_matrix.size == 0:
            return True
        gx = int(round(coord[0] / self.grid_size))
        gy = int(round(coord[1] / self.grid_size))
        rows, cols = self.buildable_matrix.shape
        if 0 <= gx < rows and 0 <= gy < cols:
            return bool(self.buildable_matrix[gx][gy])
        return True  # 超出范围默认为可建

    def generate_centroid_sites(self) -> List[Tuple[float, float]]:
        """策略1：逆变器群组重心作为候选点"""
        if self.n_inverters == 0:
            return []
        centroid = (
            np.mean([c[0] for c in self.inverter_coords]),
            np.mean([c[1] for c in self.inverter_coords])
        )
        return [self._align_to_grid(centroid)]

    def generate_kmeans_sites(self, n_clusters: int = None) -> List[Tuple[float, float]]:
        """策略2：K-Means 聚类中心作为候选点"""
        if self.n_inverters < 2:
            return self.generate_centroid_sites()

        if n_clusters is None:
            # 自动计算聚类数：每台箱变最多连接 max_cap 台逆变器
            max_cap = self.capacity_map.get(max(self.Q_box_options), 10)
            n_clusters = max(1, int(np.ceil(self.n_inverters / max_cap)))

        n_clusters = min(n_clusters, self.n_inverters)
        coords_np = np.array(self.inverter_coords)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(coords_np)

        sites = []
        for center in kmeans.cluster_centers_:
            aligned = self._align_to_grid(tuple(center))
            if self._is_buildable(aligned):
                sites.append(aligned)

        return sites

    def generate_grid_sampling_sites(self, margin: float = 1.0) -> List[Tuple[float, float]]:
        """策略3：在逆变器包围盒内进行网格采样"""
        if self.n_inverters == 0:
            return []

        coords_np = np.array(self.inverter_coords)
        x_min, y_min = coords_np.min(axis=0) - margin * self.grid_size
        x_max, y_max = coords_np.max(axis=0) + margin * self.grid_size

        # 对齐到网格
        x_min = np.floor(x_min / self.grid_size) * self.grid_size
        y_min = np.floor(y_min / self.grid_size) * self.grid_size
        x_max = np.ceil(x_max / self.grid_size) * self.grid_size
        y_max = np.ceil(y_max / self.grid_size) * self.grid_size

        sites = []
        # 每隔 2*grid_size 采样一个点（避免候选点过多）
        step = 2 * self.grid_size
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                site = (float(x), float(y))
                if self._is_buildable(site):
                    sites.append(site)
                y += step
            x += step

        return sites

    def generate_all_candidates(self, max_candidates: int = 20) -> List[Tuple[float, float]]:
        """
        综合三种策略生成候选点集，去重并限制数量。

        Parameters
        ----------
        max_candidates : int
            候选点最大数量

        Returns
        -------
        List[Tuple[float, float]]
            候选箱变站址列表（网格对齐后）
        """
        all_sites = set()

        # 策略1：重心
        for s in self.generate_centroid_sites():
            all_sites.add(s)

        # 策略2：KMeans（多种 k 值）
        max_cap = self.capacity_map.get(max(self.Q_box_options), 10)
        n_min = max(1, int(np.ceil(self.n_inverters / max_cap)))
        n_max = min(self.n_inverters, n_min + 3)
        for k in range(n_min, n_max + 1):
            for s in self.generate_kmeans_sites(k):
                all_sites.add(s)

        # 策略3：网格采样（仅在候选点不足时启用）
        if len(all_sites) < max_candidates // 2:
            for s in self.generate_grid_sampling_sites():
                all_sites.add(s)

        # 也将各逆变器位置加入作为候选点
        for coord in self.inverter_coords:
            aligned = self._align_to_grid(coord)
            if self._is_buildable(aligned):
                all_sites.add(aligned)

        # 去重（相同网格位置的点）
        unique_sites = list(all_sites)

        # 如果候选点过多，按到逆变器群重心的距离排序，保留最近的 max_candidates 个
        if len(unique_sites) > max_candidates:
            centroid = np.mean(self.inverter_coords, axis=0)
            unique_sites.sort(key=lambda s: np.sqrt((s[0] - centroid[0])**2 + (s[1] - centroid[1])**2))
            unique_sites = unique_sites[:max_candidates]

        logger.info(f"【候选站址】生成 {len(unique_sites)} 个候选箱变站址")
        return unique_sites

    def compute_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float],
                         use_manhattan: bool = True) -> float:
        """计算两点间距离（支持曼哈顿距离和欧几里得距离）"""
        if use_manhattan:
            return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])
        else:
            return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
