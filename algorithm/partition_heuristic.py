"""
启发式分区算法

采用 "种子选择 → 贪心扩展 → 局部搜索" 三阶段策略，
为 Benders 分解的子问题提供初始可行分区方案。

同时作为 S2V-DQN 的 baseline 对比方法。

算法流程：
  1. 种子选择：基于面板分布均匀选取 p 个种子节点
  2. 贪心扩展：从种子出发，按优先级依次扩展分区
     优先级 = 距离近 + 保持紧凑 + 保持规整
  3. 局部搜索：在分区边界进行面板交换，最小化总周长
"""

import random
import logging
from typing import List, Dict, Set, Tuple, Optional

import numpy as np
import networkx as nx

from utils.graph_utils import (
    build_coord_index, check_connectivity, calculate_perimeter_fast,
    get_boundary_nodes, get_adjacent_external_nodes, get_zone_bounding_box
)
from model.partition_sub import PartitionValidator, PartitionResult

logger = logging.getLogger(__name__)


class GreedyPartitioner:
    """
    贪心+局部搜索分区器。

    使用方式:
        partitioner = GreedyPartitioner(graph, n_zones=5)
        result = partitioner.solve()
    """

    def __init__(self, graph: nx.Graph, n_zones: int,
                 min_panels: int = 18, max_panels: int = 26,
                 perimeter_lb: float = 60.0, perimeter_ub: float = 90.0,
                 max_panel_diff: int = 2,
                 local_search_iters: int = 100,
                 random_seed: int = 42):
        """
        Args:
            graph: 面板邻接图
            n_zones: 目标分区数（= 逆变器数 p）
            min_panels: 每分区最少面板数
            max_panels: 每分区最多面板数
            perimeter_lb: 周长下界
            perimeter_ub: 周长上界
            max_panel_diff: 负载平衡约束
            local_search_iters: 局部搜索最大迭代次数
            random_seed: 随机种子
        """
        self.graph = graph
        self.n_zones = n_zones
        self.min_panels = min_panels
        self.max_panels = max_panels
        self.perimeter_lb = perimeter_lb
        self.perimeter_ub = perimeter_ub
        self.max_panel_diff = max_panel_diff
        self.local_search_iters = local_search_iters
        self.coord_index = build_coord_index(graph)
        self.all_panels = set(graph.nodes())

        random.seed(random_seed)
        np.random.seed(random_seed)

        # 目标每分区面板数
        self.target_size = len(self.all_panels) // n_zones

    def solve(self) -> PartitionResult:
        """
        执行完整的分区求解。

        Returns:
            PartitionResult 分区结果
        """
        # 阶段 1：种子选择
        seeds = self._select_seeds()

        # 阶段 2：贪心扩展
        zones = self._greedy_expand(seeds)

        # 阶段 2.5：连通性修复
        zones = self._repair_connectivity(zones)

        # 阶段 3：局部搜索优化
        zones = self._local_search(zones)

        # 验证并返回结果
        validator = PartitionValidator(
            self.graph, self.n_zones,
            min_panels=self.min_panels, max_panels=self.max_panels,
            max_panel_diff=self.max_panel_diff,
            perimeter_lb=self.perimeter_lb, perimeter_ub=self.perimeter_ub
        )
        result = validator.validate(zones)
        result.solver_method = "heuristic"
        return result

    def _select_seeds(self) -> List[str]:
        """
        基于 K-means 思想选择种子节点。

        将面板坐标聚类为 n_zones 个簇，每个簇中心最近的面板作为种子。
        """
        nodes = list(self.all_panels)
        coords = np.array([
            [self.graph.nodes[n]["row"], self.graph.nodes[n]["col"]]
            for n in nodes
        ], dtype=float)

        # 简易 K-means 初始化
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_zones, random_state=42, n_init=10)
        kmeans.fit(coords)

        seeds = []
        for center in kmeans.cluster_centers_:
            # 找最近的面板
            dists = np.linalg.norm(coords - center, axis=1)
            # 排除已选的种子
            for idx in np.argsort(dists):
                if nodes[idx] not in seeds:
                    seeds.append(nodes[idx])
                    break

        return seeds

    def _greedy_expand(self, seeds: List[str]) -> List[Set[str]]:
        """
        从种子节点开始贪心扩展分区。

        策略：轮流为每个分区添加一个节点，直到所有面板被分配。
        优先级：距种子近 + 保持连通性 + 面板数不超上限。
        """
        zones = [set([seed]) for seed in seeds]
        assigned = set(seeds)
        remaining = self.all_panels - assigned

        # 轮流扩展，每轮每个分区尝试加一个节点
        max_rounds = len(self.all_panels)  # 安全上限
        for _ in range(max_rounds):
            if not remaining:
                break

            progress_made = False
            for i in range(self.n_zones):
                if not remaining:
                    break
                if len(zones[i]) >= self.max_panels:
                    continue

                # 获取当前分区的外部相邻可扩展节点
                candidates = get_adjacent_external_nodes(self.graph, zones[i])
                candidates = candidates & remaining  # 只考虑未分配的

                if not candidates:
                    continue

                # 按扩展优先级排序
                best = self._pick_best_candidate(zones[i], candidates, seeds[i])
                if best is not None:
                    zones[i].add(best)
                    assigned.add(best)
                    remaining.discard(best)
                    progress_made = True

            if not progress_made:
                break

        # 处理剩余未分配的面板（强制分配到最近的分区）
        if remaining:
            self._assign_remaining(zones, remaining)

        return zones

    def _pick_best_candidate(self, zone: Set[str], candidates: Set[str],
                              seed: str) -> Optional[str]:
        """
        选择最佳扩展候选节点。

        评分 = -距离种子的距离 + 紧凑性加分（与分区重心近）
        """
        seed_row = self.graph.nodes[seed]["row"]
        seed_col = self.graph.nodes[seed]["col"]

        # 计算分区重心
        if zone:
            center_row = np.mean([self.graph.nodes[n]["row"] for n in zone])
            center_col = np.mean([self.graph.nodes[n]["col"] for n in zone])
        else:
            center_row, center_col = seed_row, seed_col

        best_node = None
        best_score = float("-inf")

        for c in candidates:
            c_row = self.graph.nodes[c]["row"]
            c_col = self.graph.nodes[c]["col"]

            # 距种子距离（曼哈顿）
            dist_to_seed = abs(c_row - seed_row) + abs(c_col - seed_col)
            # 距重心距离
            dist_to_center = abs(c_row - center_row) + abs(c_col - center_col)

            # 连通性检查
            new_zone = zone | {c}
            if not check_connectivity(self.graph, new_zone):
                continue

            # 评分：距离越小越好
            score = -(0.5 * dist_to_seed + 0.5 * dist_to_center)

            if score > best_score:
                best_score = score
                best_node = c

        return best_node

    def _assign_remaining(self, zones: List[Set[str]], remaining: Set[str]):
        """将剩余面板强制分配到最近且保持连通的分区。"""
        max_attempts = len(remaining) * self.n_zones
        attempts = 0
        while remaining and attempts < max_attempts:
            attempts += 1
            best_panel = None
            best_zone = -1
            best_dist = float("inf")

            for panel in remaining:
                p_row = self.graph.nodes[panel]["row"]
                p_col = self.graph.nodes[panel]["col"]

                for i, zone in enumerate(zones):
                    if len(zone) >= self.max_panels:
                        continue

                    # 必须与该分区相邻才能保持连通
                    is_adjacent = any(
                        nb in zone for nb in self.graph.neighbors(panel)
                    )
                    if not is_adjacent:
                        continue

                    center_row = np.mean([self.graph.nodes[n]["row"] for n in zone])
                    center_col = np.mean([self.graph.nodes[n]["col"] for n in zone])
                    dist = abs(p_row - center_row) + abs(p_col - center_col)

                    if dist < best_dist:
                        best_dist = dist
                        best_zone = i
                        best_panel = panel

            if best_panel is not None and best_zone >= 0:
                zones[best_zone].add(best_panel)
                remaining.discard(best_panel)
            else:
                # 无法找到相邻分区，强制分配到最近的分区
                panel = next(iter(remaining))
                p_row = self.graph.nodes[panel]["row"]
                p_col = self.graph.nodes[panel]["col"]

                closest_zone = 0
                closest_dist = float("inf")
                for i, zone in enumerate(zones):
                    if len(zone) >= self.max_panels:
                        continue
                    center_row = np.mean([self.graph.nodes[n]["row"] for n in zone])
                    center_col = np.mean([self.graph.nodes[n]["col"] for n in zone])
                    dist = abs(p_row - center_row) + abs(p_col - center_col)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_zone = i

                zones[closest_zone].add(panel)
                remaining.discard(panel)

    def _repair_connectivity(self, zones: List[Set[str]]) -> List[Set[str]]:
        """
        修复分区连通性：将非连通的小分量移到相邻分区。

        对于每个分区中的非连通分量，将较小的分量移到与其相邻的其他分区。
        """
        from utils.graph_utils import get_connected_components

        repaired = True
        max_repair_rounds = 10
        for _ in range(max_repair_rounds):
            if not repaired:
                break
            repaired = False

            for i in range(len(zones)):
                components = get_connected_components(self.graph, zones[i])
                if len(components) <= 1:
                    continue

                # 保留最大连通分量，将其他分量移出
                components.sort(key=len, reverse=True)
                main_component = components[0]
                zones[i] = main_component

                for fragment in components[1:]:
                    # 尝试将碎片加入相邻的其他分区
                    for node in fragment:
                        best_zone = -1
                        for j in range(len(zones)):
                            if j == i:
                                continue
                            if len(zones[j]) >= self.max_panels:
                                continue
                            # 检查是否与目标分区相邻
                            is_adj = any(nb in zones[j] for nb in self.graph.neighbors(node))
                            if is_adj:
                                best_zone = j
                                break

                        if best_zone >= 0:
                            zones[best_zone].add(node)
                        else:
                            # 无相邻分区，强制加入最近的
                            p_row = self.graph.nodes[node]["row"]
                            p_col = self.graph.nodes[node]["col"]
                            closest = 0
                            min_dist = float("inf")
                            for j in range(len(zones)):
                                if j == i or len(zones[j]) >= self.max_panels:
                                    continue
                                c_row = np.mean([self.graph.nodes[n]["row"] for n in zones[j]])
                                c_col = np.mean([self.graph.nodes[n]["col"] for n in zones[j]])
                                d = abs(p_row - c_row) + abs(p_col - c_col)
                                if d < min_dist:
                                    min_dist = d
                                    closest = j
                            zones[closest].add(node)

                    repaired = True

        return zones

    def _local_search(self, zones: List[Set[str]]) -> List[Set[str]]:
        """
        局部搜索：在分区边界进行面板交换，最小化总周长。

        操作：随机选择一对相邻分区的边界面板，尝试交换。
        如果交换后总周长减少且约束仍满足，则接受。
        """
        best_perimeter = self._total_perimeter(zones)

        for iteration in range(self.local_search_iters):
            # 随机选一个分区
            zone_idx = random.randint(0, self.n_zones - 1)
            zone = zones[zone_idx]

            if len(zone) <= self.min_panels:
                continue

            # 获取边界节点
            boundary = get_boundary_nodes(self.graph, zone)
            if not boundary:
                continue

            # 随机选一个边界节点
            node = random.choice(list(boundary))

            # 找该节点的邻居所在的其他分区
            for neighbor in self.graph.neighbors(node):
                target_idx = self._find_zone_of(zones, neighbor)
                if target_idx == zone_idx or target_idx < 0:
                    continue
                if len(zones[target_idx]) >= self.max_panels:
                    continue

                # 尝试交换：将 node 从 zone_idx 移到 target_idx
                new_zones = [set(z) for z in zones]
                new_zones[zone_idx].remove(node)
                new_zones[target_idx].add(node)

                # 检查连通性
                if not check_connectivity(self.graph, new_zones[zone_idx]):
                    continue
                if not check_connectivity(self.graph, new_zones[target_idx]):
                    continue

                # 检查大小约束
                if len(new_zones[zone_idx]) < self.min_panels:
                    continue

                # 计算新周长
                new_perimeter = self._total_perimeter(new_zones)
                if new_perimeter < best_perimeter:
                    zones = new_zones
                    best_perimeter = new_perimeter
                    break

        return zones

    def _total_perimeter(self, zones: List[Set[str]]) -> float:
        """计算所有分区的总周长。"""
        return sum(
            calculate_perimeter_fast(zone, self.graph, self.coord_index)
            for zone in zones
        )

    def _find_zone_of(self, zones: List[Set[str]], node: str) -> int:
        """查找节点所属的分区索引。"""
        for i, zone in enumerate(zones):
            if node in zone:
                return i
        return -1
