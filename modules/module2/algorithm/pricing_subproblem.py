"""
定价子问题求解器
================
利用对偶变量求解定价子问题，寻找负检验数路径。

检验数 = 路径成本 - 对偶价值
       = c2·L_p + Σ_{e∈p}(c3·L_e - μ_e) - π_{k(p)}

支持：
- 枚举所有候选路径（小规模）
- 动态规划最短路径（大规模）
- 多路径生成（每轮返回多条负检验数路径）
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

from modules.module2.algorithm.base_components import (
    CoordinateHelper, PathFactory
)

logger = logging.getLogger(__name__)


class PricingSubproblem:
    """
    定价子问题求解器

    在列生成循环中，利用 RMP 对偶变量，为每个逆变器寻找
    修正成本（reduced cost）为负的新路径。
    """

    def __init__(self, coord_helper: CoordinateHelper,
                 path_factory: PathFactory,
                 inverter_coords: List[Tuple[float, float]],
                 candidate_sites: List[Tuple[float, float]],
                 substation_coord: Tuple[float, float],
                 c2: float, c3: float, knn_k: int = 5):
        self.coord = coord_helper
        self.factory = path_factory
        self.inverter_coords = inverter_coords
        self.candidate_sites = candidate_sites
        self.substation_coord = substation_coord
        self.c2 = c2
        self.c3 = c3
        self.knn_k = knn_k
        self.n_inv = len(inverter_coords)
        self.n_sites = len(candidate_sites)

        # 预计算逆变器到各候选站址的距离
        self._precompute_distances()

        # 缓存已生成的路径 {(inv_idx, box_idx): path_dict}
        self._path_cache: Dict[Tuple[int, int], Dict] = {}

        # 统计信息
        self.stats = {"calls": 0, "paths_found": 0, "total_time": 0.0}

    def _precompute_distances(self):
        """预计算距离矩阵"""
        self.dist_inv_box = np.zeros((self.n_inv, self.n_sites))
        self.dist_box_sub = np.zeros(self.n_sites)

        for i, ic in enumerate(self.inverter_coords):
            for j, bc in enumerate(self.candidate_sites):
                self.dist_inv_box[i, j] = self.coord.manhattan_distance(ic, bc)

        for j, bc in enumerate(self.candidate_sites):
            self.dist_box_sub[j] = self.coord.manhattan_distance(bc, self.substation_coord)

        # KNN 索引：每个逆变器的最近 K 个候选站址
        self.knn_indices = []
        for i in range(self.n_inv):
            sorted_idx = np.argsort(self.dist_inv_box[i])
            self.knn_indices.append(sorted_idx[:self.knn_k].tolist())

    def solve(self, dual_assignment: Dict[int, float],
              dual_trench: Dict[Tuple, float],
              active_path_ids: set,
              all_paths: List[Dict],
              max_new_paths: int = 20) -> List[Dict]:
        """
        求解定价子问题，返回负检验数的新路径。

        Parameters
        ----------
        dual_assignment : Dict[int, float]
            分配约束 (式12) 的对偶变量 π_k
        dual_trench : Dict[Tuple, float]
            共沟约束 (式17) 的对偶变量 μ_e（以 edge_key 为键）
        active_path_ids : set
            当前活跃路径 ID 集合
        all_paths : List[Dict]
            全部候选路径
        max_new_paths : int
            本轮最多添加的新路径数

        Returns
        -------
        List[Dict]
            负检验数路径列表（按检验数升序）
        """
        self.stats["calls"] += 1
        new_paths = []

        # === 策略1：从已有路径池中筛选负检验数路径 ===
        for path in all_paths:
            if path["id"] in active_path_ids:
                continue
            if path["type"] != "inv_to_box":
                continue

            rc = self._compute_reduced_cost(path, dual_assignment, dual_trench)
            if rc < -1e-6:
                new_paths.append((rc, path))

        # === 策略2：动态生成新路径（扩展KNN范围）===
        if len(new_paths) < max_new_paths // 2:
            extended_paths = self._generate_extended_paths(
                dual_assignment, dual_trench, active_path_ids, all_paths)
            new_paths.extend(extended_paths)

        # 按检验数排序，取前 max_new_paths 条
        new_paths.sort(key=lambda x: x[0])
        selected = [p for _, p in new_paths[:max_new_paths]]

        self.stats["paths_found"] += len(selected)
        if selected:
            logger.info(f"【定价子问题】找到 {len(selected)} 条负检验数路径，"
                        f"最小检验数 = {new_paths[0][0]:.4f}")
        return selected

    def _compute_reduced_cost(self, path: Dict,
                               dual_assignment: Dict[int, float],
                               dual_trench: Dict[Tuple, float]) -> float:
        """
        计算路径的检验数（reduced cost）

        rc = c2·L_p + Σ_{e∈p}(c3·L_e - μ_e) - π_{k(p)}
        """
        inv_idx = path["inv_idx"]
        pi_k = dual_assignment.get(inv_idx, 0.0)

        # 电缆成本
        c2_wan = self.c2 / 10000.0
        cable_cost = path["length"] * c2_wan

        # 管沟修正成本
        c3_wan = self.c3 / 10000.0
        trench_component = 0.0
        for ek in path["edges"]:
            edge_len = self.factory.edges.get(ek, {}).get("length", self.coord.grid_size)
            mu_e = dual_trench.get(ek, 0.0)
            trench_component += (c3_wan * edge_len - mu_e)

        return cable_cost + trench_component - pi_k

    def _generate_extended_paths(self, dual_assignment, dual_trench,
                                  active_path_ids, all_paths) -> List[Tuple[float, Dict]]:
        """扩展KNN范围，生成新的候选路径"""
        new_paths = []
        existing_pairs = set()
        for p in all_paths:
            if p["type"] == "inv_to_box":
                existing_pairs.add((p["inv_idx"], p["box_idx"]))

        extended_k = min(self.knn_k + 3, self.n_sites)
        for inv_idx in range(self.n_inv):
            sorted_idx = np.argsort(self.dist_inv_box[inv_idx])
            for box_idx in sorted_idx[:extended_k]:
                box_idx = int(box_idx)
                if (inv_idx, box_idx) in existing_pairs:
                    continue

                # 动态生成路径
                path = self._get_or_create_path(inv_idx, box_idx)
                if path is None:
                    continue

                rc = self._compute_reduced_cost(path, dual_assignment, dual_trench)
                if rc < -1e-6:
                    new_paths.append((rc, path))
                    existing_pairs.add((inv_idx, box_idx))

        return new_paths

    def _get_or_create_path(self, inv_idx: int, box_idx: int) -> Optional[Dict]:
        """获取或动态创建一条路径"""
        key = (inv_idx, box_idx)
        if key in self._path_cache:
            return self._path_cache[key]

        inv_coord = self.inverter_coords[inv_idx]
        box_coord = self.candidate_sites[box_idx]
        path = self.factory.generate_path(
            inv_coord, box_coord, "inv_to_box",
            inv_idx=inv_idx, box_idx=box_idx)
        self._path_cache[key] = path
        return path


class DynamicPricingSolver:
    """
    基于动态规划的定价子问题求解器（大规模算例）

    对于给定的对偶变量，寻找从逆变器k经过箱变b到升压站的
    最小修正成本路径。使用修正成本的 Dijkstra/DP 方法。
    """

    def __init__(self, coord_helper: CoordinateHelper,
                 inverter_coords: List[Tuple[float, float]],
                 candidate_sites: List[Tuple[float, float]],
                 substation_coord: Tuple[float, float],
                 c2: float, c3: float):
        self.coord = coord_helper
        self.inverter_coords = inverter_coords
        self.candidate_sites = candidate_sites
        self.substation_coord = substation_coord
        self.c2 = c2
        self.c3 = c3

    def find_best_path_for_inverter(self, inv_idx: int,
                                     dual_pi: float,
                                     edge_duals: Dict[Tuple, float]) -> Tuple[Optional[int], float]:
        """
        为指定逆变器找到修正成本最小的箱变。

        Returns
        -------
        (best_box_idx, reduced_cost)
        """
        inv_coord = self.inverter_coords[inv_idx]
        best_box = None
        best_rc = 0.0  # 只关注负检验数

        c2_wan = self.c2 / 10000.0
        c3_wan = self.c3 / 10000.0

        for b_idx, box_coord in enumerate(self.candidate_sites):
            d_ib = self.coord.manhattan_distance(inv_coord, box_coord)
            d_bs = self.coord.manhattan_distance(box_coord, self.substation_coord)

            cable_cost = c2_wan * (d_ib + d_bs)

            # 估算管沟修正成本（用近似方式，不逐边计算）
            n_edges_ib = max(1, int(d_ib / self.coord.grid_size))
            n_edges_bs = max(1, int(d_bs / self.coord.grid_size))
            trench_cost = c3_wan * (d_ib + d_bs)

            # 对偶松弛（简化：均匀分配到边）
            total_dual = sum(edge_duals.values()) / max(1, len(edge_duals)) * (n_edges_ib + n_edges_bs) if edge_duals else 0

            rc = cable_cost + trench_cost - total_dual - dual_pi

            if rc < best_rc - 1e-6:
                best_rc = rc
                best_box = b_idx

        return best_box, best_rc
