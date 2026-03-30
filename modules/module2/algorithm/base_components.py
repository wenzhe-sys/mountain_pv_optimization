"""
基础组件模块
============
提供模块二算法的基础数据结构和工具函数：
- 坐标转换与对齐
- 距离计算（曼哈顿、地形修正）
- 路径生成（曼哈顿网格路径）
- 候选箱变站址生成
- 直流电缆成本计算
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CoordinateHelper:
    """坐标转换与距离计算工具"""

    def __init__(self, grid_size: float, slope_matrix: np.ndarray,
                 buildable_matrix: np.ndarray):
        self.grid_size = grid_size
        self.slope_matrix = slope_matrix
        self.buildable_matrix = buildable_matrix
        self.n_rows, self.n_cols = slope_matrix.shape

    def grid_to_xy(self, grid_coord: Tuple[int, int]) -> Tuple[float, float]:
        row, col = grid_coord
        return (float(col * self.grid_size), float(row * self.grid_size))

    def xy_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        row = int(round(y / self.grid_size))
        col = int(round(x / self.grid_size))
        return (row, col)

    def align_to_grid(self, coord: Tuple[float, float]) -> Tuple[float, float]:
        x = round(coord[0] / self.grid_size) * self.grid_size
        y = round(coord[1] / self.grid_size) * self.grid_size
        return (float(x), float(y))

    def is_buildable(self, coord: Tuple[float, float]) -> bool:
        row, col = self.xy_to_grid(coord[0], coord[1])
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return bool(self.buildable_matrix[row, col] == 1)
        return False

    def manhattan_distance(self, c1: Tuple[float, float],
                           c2: Tuple[float, float]) -> float:
        return float(abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]))

    def terrain_corrected_distance(self, c1: Tuple[float, float],
                                    c2: Tuple[float, float]) -> float:
        """地形修正距离 D_uv = d_manhattan × (1 + 0.05 × avg_slope)"""
        base_dist = self.manhattan_distance(c1, c2)
        try:
            r1, c1_idx = self.xy_to_grid(c1[0], c1[1])
            r2, c2_idx = self.xy_to_grid(c2[0], c2[1])
            if (0 <= r1 < self.n_rows and 0 <= c1_idx < self.n_cols and
                    0 <= r2 < self.n_rows and 0 <= c2_idx < self.n_cols):
                avg_slope = (self.slope_matrix[r1, c1_idx] + self.slope_matrix[r2, c2_idx]) / 2.0
                return float(base_dist * (1.0 + 0.05 * avg_slope))
        except Exception:
            pass
        return base_dist


class PathFactory:
    """曼哈顿网格路径生成器"""

    def __init__(self, coord_helper: CoordinateHelper):
        self.coord = coord_helper
        self.edges: Dict[Tuple, Dict] = {}  # edge_key → {length, from, to}
        self._path_counter = 0

    def reset(self):
        self.edges.clear()
        self._path_counter = 0

    @staticmethod
    def get_edge_key(c1: Tuple[float, float],
                     c2: Tuple[float, float]) -> Tuple[float, float, float, float]:
        if c1 < c2:
            return (c1[0], c1[1], c2[0], c2[1])
        return (c2[0], c2[1], c1[0], c1[1])

    def _manhattan_waypoints(self, start: Tuple[float, float],
                             end: Tuple[float, float]) -> List[Tuple[float, float]]:
        gs = self.coord.grid_size
        path = [start]
        cx, cy = start

        target_x = end[0]
        if abs(cx - target_x) > 1e-6:
            step_x = gs if target_x > cx else -gs
            while abs(cx - target_x) > gs / 2:
                cx += step_x
                path.append((cx, cy))

        target_y = end[1]
        if abs(cy - target_y) > 1e-6:
            step_y = gs if target_y > cy else -gs
            while abs(cy - target_y) > gs / 2:
                cy += step_y
                path.append((cx, cy))

        if path[-1] != end:
            path.append(end)
        return path

    def _register_edge(self, edge_key: Tuple) -> None:
        if edge_key not in self.edges:
            c1 = (edge_key[0], edge_key[1])
            c2 = (edge_key[2], edge_key[3])
            self.edges[edge_key] = {
                "length": self.coord.terrain_corrected_distance(c1, c2),
                "from": c1,
                "to": c2,
            }

    def generate_path(self, from_coord: Tuple[float, float],
                      to_coord: Tuple[float, float],
                      path_type: str,
                      inv_idx: int = None,
                      box_idx: int = None) -> Dict:
        coords = self._manhattan_waypoints(from_coord, to_coord)
        edges = []
        for i in range(len(coords) - 1):
            ek = self.get_edge_key(coords[i], coords[i + 1])
            self._register_edge(ek)
            edges.append(ek)

        total_length = sum(self.edges[e]["length"] for e in edges)
        p = {
            "id": self._path_counter,
            "from": from_coord,
            "to": to_coord,
            "coords": coords,
            "edges": edges,
            "length": total_length,
            "type": path_type,
            "inv_idx": inv_idx,
            "box_idx": box_idx,
        }
        self._path_counter += 1
        return p


class CandidateSiteGenerator:
    """候选箱变站址生成（四策略 + 可建性过滤）"""

    def __init__(self, coord_helper: CoordinateHelper,
                 inverter_coords: List[Tuple[float, float]],
                 substation_coord: Tuple[float, float]):
        self.coord = coord_helper
        self.inverter_coords = inverter_coords
        self.substation_coord = substation_coord

    def generate(self) -> List[Tuple[float, float]]:
        candidates: Set[Tuple[float, float]] = set()
        n_inv = len(self.inverter_coords)
        gs = self.coord.grid_size

        # 策略1：逆变器坐标
        for c in self.inverter_coords:
            if self.coord.is_buildable(c):
                candidates.add(c)

        # 策略2：逆变器重心
        if self.inverter_coords:
            avg_x = sum(c[0] for c in self.inverter_coords) / n_inv
            avg_y = sum(c[1] for c in self.inverter_coords) / n_inv
            centroid = self.coord.align_to_grid((avg_x, avg_y))
            if self.coord.is_buildable(centroid):
                candidates.add(centroid)

        # 策略3：包围盒网格采样
        if self.inverter_coords:
            xs = [c[0] for c in self.inverter_coords]
            ys = [c[1] for c in self.inverter_coords]
            step = gs * 4
            x = min(xs)
            while x <= max(xs):
                y = min(ys)
                while y <= max(ys):
                    pt = self.coord.align_to_grid((x, y))
                    if self.coord.is_buildable(pt):
                        candidates.add(pt)
                    y += step
                x += step

        # 策略4：升压站坐标
        candidates.add(self.coord.align_to_grid(self.substation_coord))

        # 安全兜底
        if len(candidates) < max(2, n_inv // 3):
            logger.warning("候选站址过少，放宽可建约束")
            for c in self.inverter_coords:
                candidates.add(c)

        result = sorted(list(candidates))
        logger.info(f"【候选站址】{len(result)} 个候选箱变站址")
        return result


class InitialPathGenerator:
    """基于KNN剪枝的初始路径集生成"""

    def __init__(self, coord_helper: CoordinateHelper,
                 path_factory: PathFactory,
                 inverter_coords: List[Tuple[float, float]],
                 substation_coord: Tuple[float, float]):
        self.coord = coord_helper
        self.factory = path_factory
        self.inverter_coords = inverter_coords
        self.substation_coord = substation_coord

    def generate(self, candidate_sites: List[Tuple[float, float]],
                 knn_k: int = 5) -> List[Dict]:
        self.factory.reset()
        paths = []
        K = min(knn_k, len(candidate_sites))

        for inv_idx, inv_coord in enumerate(self.inverter_coords):
            distances = []
            for box_idx, box_coord in enumerate(candidate_sites):
                d = self.coord.manhattan_distance(inv_coord, box_coord)
                distances.append((d, box_idx, box_coord))
            distances.sort()

            for _, box_idx, box_coord in distances[:K]:
                p = self.factory.generate_path(
                    inv_coord, box_coord, "inv_to_box",
                    inv_idx=inv_idx, box_idx=box_idx)
                paths.append(p)

        for box_idx, box_coord in enumerate(candidate_sites):
            p = self.factory.generate_path(
                box_coord, self.substation_coord, "box_to_sub",
                box_idx=box_idx)
            paths.append(p)

        logger.info(f"【初始路径】{len(paths)} 条路径，{len(self.factory.edges)} 条唯一边")
        return paths


def compute_inverter_coords(zone_summary: List[Dict],
                            partition_result: List[Dict],
                            coord_helper: CoordinateHelper,
                            substation_coord: Tuple[float, float]) -> List[Tuple[float, float]]:
    """从 M1-Output 计算逆变器坐标（分区面板重心）"""
    coords = []
    for zone in zone_summary:
        zid = zone["zone_id"]
        panels = [p for p in partition_result if p["zone_id"] == zid]
        if panels:
            avg_r = sum(p["grid_coord"][0] for p in panels) / len(panels)
            avg_c = sum(p["grid_coord"][1] for p in panels) / len(panels)
            xy = coord_helper.grid_to_xy((avg_r, avg_c))
            coords.append(coord_helper.align_to_grid(xy))
        else:
            coords.append(coord_helper.align_to_grid(substation_coord))
    return coords


def compute_dc_cable_cost(zone_summary: List[Dict],
                          partition_result: List[Dict],
                          inverter_coords: List[Tuple[float, float]],
                          coord_helper: CoordinateHelper,
                          c1: float) -> float:
    """直流电缆成本 c1 × Σ(面板→逆变器距离)，返回万元"""
    total_len = 0.0
    for inv_idx, zone in enumerate(zone_summary):
        zid = zone["zone_id"]
        inv_coord = inverter_coords[inv_idx]
        panels = [p for p in partition_result if p["zone_id"] == zid]
        for panel in panels:
            pc = coord_helper.grid_to_xy(tuple(panel["grid_coord"]))
            total_len += coord_helper.manhattan_distance(pc, inv_coord)
    cost = total_len * c1 / 10000.0
    logger.info(f"【直流电缆】{total_len:.1f}m，{cost:.2f}万元")
    return cost
