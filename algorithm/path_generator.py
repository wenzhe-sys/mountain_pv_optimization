"""
电缆路径生成模块
=================
生成逆变器→箱变→升压站的电缆路径候选集。

路径策略：
1. 曼哈顿网格路径（L形折线）
2. KNN 剪枝：每个逆变器只考虑最近 K 个候选箱变
3. 地形修正距离计算
4. 路径边集提取（用于管沟共享建模）
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)


class CablePath:
    """一条完整的电缆路径：逆变器 → 箱变 → 升压站"""

    def __init__(self, path_id: int, inverter_idx: int, box_idx: int,
                 inv_coord: Tuple[float, float], box_coord: Tuple[float, float],
                 sub_coord: Tuple[float, float], grid_size: float):
        self.path_id = path_id
        self.inverter_idx = inverter_idx
        self.box_idx = box_idx
        self.inv_coord = inv_coord
        self.box_coord = box_coord
        self.sub_coord = sub_coord
        self.grid_size = grid_size

        # 计算路径信息
        self.inv_to_box_length = self._manhattan_distance(inv_coord, box_coord)
        self.box_to_sub_length = self._manhattan_distance(box_coord, sub_coord)
        self.total_length = self.inv_to_box_length + self.box_to_sub_length

        # 生成边集（曼哈顿 L 形路径上的网格边）
        self.inv_to_box_edges = self._generate_manhattan_edges(inv_coord, box_coord)
        self.box_to_sub_edges = self._generate_manhattan_edges(box_coord, sub_coord)
        self.all_edges = self.inv_to_box_edges | self.box_to_sub_edges

    def _manhattan_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _generate_manhattan_edges(self, start: Tuple[float, float],
                                   end: Tuple[float, float]) -> Set[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        生成从 start 到 end 的曼哈顿 L 形路径的边集。
        先水平移动，再垂直移动（L 形策略）。
        """
        edges = set()
        gs = self.grid_size
        x1, y1 = start
        x2, y2 = end

        # 当前位置
        cx, cy = x1, y1

        # 水平移动（限制最大步数避免非网格对齐导致的无限循环）
        max_steps = max(int(abs(x2 - x1) / gs) + 2, 1)
        dx = gs if x2 > cx else -gs
        step = 0
        while abs(cx - x2) >= gs * 0.5 and step < max_steps:
            nx = cx + dx
            # 防止越过终点
            if (dx > 0 and nx > x2 + gs * 0.5) or (dx < 0 and nx < x2 - gs * 0.5):
                break
            edge = self._make_edge((cx, cy), (nx, cy))
            edges.add(edge)
            cx = nx
            step += 1

        # 垂直移动
        max_steps = max(int(abs(y2 - cy) / gs) + 2, 1)
        dy = gs if y2 > cy else -gs
        step = 0
        while abs(cy - y2) >= gs * 0.5 and step < max_steps:
            ny = cy + dy
            if (dy > 0 and ny > y2 + gs * 0.5) or (dy < 0 and ny < y2 - gs * 0.5):
                break
            edge = self._make_edge((cx, cy), (cx, ny))
            edges.add(edge)
            cy = ny
            step += 1

        return edges

    @staticmethod
    def _make_edge(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """确保边的表示唯一（端点排序）"""
        rp1 = (round(p1[0], 2), round(p1[1], 2))
        rp2 = (round(p2[0], 2), round(p2[1], 2))
        return (min(rp1, rp2), max(rp1, rp2))


class PathGenerator:
    """电缆路径生成器"""

    def __init__(self, instance_data: Dict, inverter_coords: List[Tuple[float, float]],
                 candidate_box_coords: List[Tuple[float, float]]):
        self.grid_size = instance_data["terrain_data"]["grid_size"]
        self.inverter_coords = inverter_coords
        self.candidate_box_coords = candidate_box_coords
        self.substation_coord = tuple(instance_data["equipment_params"]["substation"]["coord"])
        self.n_inverters = len(inverter_coords)
        self.n_boxes = len(candidate_box_coords)

        # 地形数据（用于距离修正）
        self.slope_matrix = np.array(instance_data["terrain_data"].get("slope_matrix", []))

        # 预计算距离矩阵
        self.dist_inv_box = self._compute_distance_matrix(inverter_coords, candidate_box_coords)
        self.dist_box_sub = np.array([
            self._manhattan_distance(bc, self.substation_coord) for bc in candidate_box_coords
        ])

    def _manhattan_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _compute_distance_matrix(self, coords1: List[Tuple[float, float]],
                                  coords2: List[Tuple[float, float]]) -> np.ndarray:
        """计算两组坐标间的曼哈顿距离矩阵"""
        n1, n2 = len(coords1), len(coords2)
        dist = np.zeros((n1, n2))
        for i, c1 in enumerate(coords1):
            for j, c2 in enumerate(coords2):
                dist[i, j] = self._manhattan_distance(c1, c2)
        return dist

    def generate_paths(self, knn_k: int = 5) -> List[CablePath]:
        """
        生成电缆路径候选集。

        Parameters
        ----------
        knn_k : int
            每个逆变器只考虑最近 K 个候选箱变（KNN 剪枝）

        Returns
        -------
        List[CablePath]
            路径候选集
        """
        paths = []
        path_id = 0
        k = min(knn_k, self.n_boxes)

        for inv_idx in range(self.n_inverters):
            # KNN 剪枝：选择最近的 k 个箱变
            distances = self.dist_inv_box[inv_idx]
            nearest_boxes = np.argsort(distances)[:k]

            for box_idx in nearest_boxes:
                path = CablePath(
                    path_id=path_id,
                    inverter_idx=inv_idx,
                    box_idx=box_idx,
                    inv_coord=self.inverter_coords[inv_idx],
                    box_coord=self.candidate_box_coords[box_idx],
                    sub_coord=self.substation_coord,
                    grid_size=self.grid_size
                )
                paths.append(path)
                path_id += 1

        logger.info(f"【路径生成】生成 {len(paths)} 条候选路径"
                     f"（{self.n_inverters}台逆变器 × 最多{k}个箱变）")
        return paths

    def get_all_edges(self, paths: List[CablePath]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """提取所有路径中出现的边集（用于管沟共享建模）"""
        all_edges = set()
        for path in paths:
            all_edges |= path.all_edges
        return list(all_edges)

    def get_edge_to_paths(self, paths: List[CablePath]) -> Dict[Tuple, List[int]]:
        """构建边→路径映射（某条边被哪些路径使用）"""
        edge_to_paths = {}
        for path in paths:
            for edge in path.all_edges:
                if edge not in edge_to_paths:
                    edge_to_paths[edge] = []
                edge_to_paths[edge].append(path.path_id)
        return edge_to_paths

    def compute_path_cable_cost(self, path: CablePath, c2: float) -> float:
        """计算路径的电缆成本"""
        return path.total_length * c2

    def compute_path_trench_cost(self, path: CablePath, c3: float) -> float:
        """计算路径的管沟成本（忽略共享，最坏情况）"""
        return path.total_length * c3
