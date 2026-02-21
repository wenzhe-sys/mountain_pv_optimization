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
import math  
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans  # 添加KMeans导入

from utils.graph_utils import (
    build_coord_index, check_connectivity, calculate_perimeter_fast,
    get_boundary_nodes, get_adjacent_external_nodes, get_zone_bounding_box
)
from modules.module1.model.partition_sub import PartitionValidator, PartitionResult

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
                 max_panel_diff: int = 4,
                 local_search_iters: int = 200,
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

        self._seed = random_seed
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

        # Iterative repair cycle: connectivity -> rebalance -> connectivity
        for _ in range(5):
            zones = self._repair_connectivity(zones)
            zones = self._rebalance(zones)

        # Final local search optimization
        zones = self._local_search(zones)

        # Fix perimeter lower-bound violations by swapping boundary nodes
        zones = self._fix_perimeter_violations(zones)

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
        coords = []
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        for node in nodes:
            data = self.graph.nodes[node]
            coords.append([data["row"], data["col"]])
        
        coords = np.array(coords)
        
        # 改进 K-means 初始化：使用 n_init=20 提高聚类质量
        kmeans = KMeans(n_clusters=self.n_zones, n_init=20, random_state=self._seed)
        kmeans.fit(coords)
        
        # 考虑面板密度：在种子选择时考虑面板分布密度
        # 计算每个簇的密度
        cluster_labels = kmeans.labels_
        cluster_densities = []
        
        for i in range(self.n_zones):
            cluster_points = coords[cluster_labels == i]
            if len(cluster_points) > 0:
                # 计算簇内点的平均距离作为密度指标
                distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[i], axis=1)
                density = 1.0 / (np.mean(distances) + 1e-6)
                cluster_densities.append(density)
            else:
                cluster_densities.append(0)
        
        # 多策略种子选择：结合几何中心、密度中心等多种种子选择策略
        seeds = []
        for i in range(self.n_zones):
            # 找到距离聚类中心最近的节点
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(coords - center, axis=1)
            closest_idx = np.argmin(distances)
            seeds.append(nodes[closest_idx])
        
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
        # 关键：用 target_size 限制每轮扩展，确保负载平衡
        max_rounds = len(self.all_panels)  # 安全上限
        for _ in range(max_rounds):
            if not remaining:
                break

            progress_made = False
            # 按当前大小从小到大排序，优先扩展较小的分区
            zone_order = sorted(range(self.n_zones), key=lambda i: len(zones[i]))
            for i in zone_order:
                if not remaining:
                    break
                # 平衡控制：当前分区不能比最小分区大太多
                min_size = min(len(z) for z in zones)
                if len(zones[i]) > min_size + 1 and len(zones[i]) >= self.min_panels:
                    continue  # 让小分区先追上来
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

        动态优先级调整：根据当前分区状态动态调整扩展优先级
        多目标评分函数：综合考虑距离、连通性、面板数平衡等多个目标
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

        # 计算当前分区大小
        current_size = len(zone)
        
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
            
            # 计算与现有分区的相邻程度（提高连通性）
            adjacency_score = 0
            for neighbor in self.graph.neighbors(c):
                if neighbor in zone:
                    adjacency_score += 1
            
            # 动态优先级调整：根据当前分区大小调整评分权重
            if current_size < self.min_panels:
                # 优先扩展以满足最小面板数
                score = -(0.3 * dist_to_seed + 0.3 * dist_to_center) + 0.4 * adjacency_score
            else:
                # 平衡考虑多个目标
                score = -(0.4 * dist_to_seed + 0.4 * dist_to_center) + 0.2 * adjacency_score

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
                    # Must be adjacent to keep connectivity
                    is_adjacent = any(
                        nb in zone for nb in self.graph.neighbors(panel)
                    )
                    if not is_adjacent:
                        continue

                    center_row = np.mean([self.graph.nodes[n]["row"] for n in zone])
                    center_col = np.mean([self.graph.nodes[n]["col"] for n in zone])
                    dist = abs(p_row - center_row) + abs(p_col - center_col)
                    # Penalise zones already at/over capacity so smaller
                    # zones are preferred, but do NOT skip them entirely
                    if len(zone) >= self.max_panels:
                        dist += 1000.0

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

    def _rebalance(self, zones: List[Set[str]]) -> List[Set[str]]:
        """
        Aggressive rebalancing in two phases:
        Phase 1: Fix critical violations (zones < min_panels) by stealing
                 from any zone > min_panels, regardless of diff.
        Phase 2: General balancing until diff <= max_panel_diff.
        """
        def _try_move(src_idx, dst_idx):
            """Try to move one boundary node from src to dst. Return True if moved."""
            if len(zones[src_idx]) <= self.min_panels:
                return False
            boundary = get_boundary_nodes(self.graph, zones[src_idx])
            # Prefer nodes closer to dst zone center
            if zones[dst_idx]:
                dst_cr = np.mean([self.graph.nodes[n]["row"] for n in zones[dst_idx]])
                dst_cc = np.mean([self.graph.nodes[n]["col"] for n in zones[dst_idx]])
                boundary = sorted(boundary, key=lambda n:
                    abs(self.graph.nodes[n]["row"] - dst_cr) +
                    abs(self.graph.nodes[n]["col"] - dst_cc)
                )
            for node in boundary:
                adjacent = any(nb in zones[dst_idx] for nb in self.graph.neighbors(node))
                if not adjacent:
                    continue
                test_src = zones[src_idx] - {node}
                if len(test_src) < self.min_panels:
                    continue
                if not check_connectivity(self.graph, test_src):
                    continue
                test_dst = zones[dst_idx] | {node}
                if not check_connectivity(self.graph, test_dst):
                    continue
                zones[src_idx].remove(node)
                zones[dst_idx].add(node)
                return True
            return False

        # Phase 1: fix undersized zones (< min_panels)
        # Try ALL donor→undersized pairs so that even non-adjacent
        # max/min pairs can be balanced via intermediate zones.
        for _ in range(300):
            undersized = [i for i in range(self.n_zones) if len(zones[i]) < self.min_panels]
            if not undersized:
                break
            moved = False
            for dst_idx in undersized:
                # Try every zone that can donate, sorted by size desc
                donors = sorted(range(self.n_zones),
                                key=lambda i: len(zones[i]), reverse=True)
                for src_idx in donors:
                    if src_idx == dst_idx:
                        continue
                    if len(zones[src_idx]) <= self.min_panels:
                        continue
                    if _try_move(src_idx, dst_idx):
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break

        # Phase 2: diffusion-based balancing
        # Instead of only max→min, try all oversized→undersized pairs
        # sorted by size difference. This enables natural relay through
        # zone adjacency chains (e.g. A→B→C when A and C not adjacent).
        for _ in range(500):
            sizes = [len(z) for z in zones]
            diff = max(sizes) - min(sizes)
            if diff <= self.max_panel_diff:
                break
            moved = False
            pairs = []
            for i in range(self.n_zones):
                for j in range(self.n_zones):
                    if i != j and sizes[i] > sizes[j] + 1:
                        pairs.append((i, j, sizes[i] - sizes[j]))
            pairs.sort(key=lambda x: x[2], reverse=True)
            for src_idx, dst_idx, _ in pairs:
                if _try_move(src_idx, dst_idx):
                    moved = True
                    break
            if not moved:
                break

        return zones

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
                    # Assign fragment nodes to adjacent zones
                    # Prefer the smallest adjacent zone; do NOT enforce
                    # max_panels here -- connectivity is more important
                    # than size, and _rebalance will fix oversized zones.
                    for node in fragment:
                        best_zone = -1
                        best_size = float("inf")
                        for j in range(len(zones)):
                            if j == i:
                                continue
                            is_adj = any(nb in zones[j] for nb in self.graph.neighbors(node))
                            if is_adj and len(zones[j]) < best_size:
                                best_size = len(zones[j])
                                best_zone = j

                        if best_zone >= 0:
                            zones[best_zone].add(node)
                        else:
                            # Truly no adjacent zone -- use graph BFS
                            # to find the closest reachable zone
                            closest = self._find_nearest_zone_bfs(node, zones, exclude=i)
                            zones[closest].add(node)

                    repaired = True

        return zones

        
    def _local_search(self, zones: List[Set[str]]) -> List[Set[str]]:
        """
        局部搜索优化：通过交换边界节点减少总周长。
        实现自适应搜索策略，根据优化进度调整搜索强度。
        多邻域搜索：实现多种邻域结构，包括节点交换、边界调整等
        模拟退火策略：引入模拟退火思想，避免局部最优
        自适应迭代次数：根据问题规模自动调整局部搜索迭代次数
        """
        best_perimeter = self._total_perimeter(zones)
        best_zones = zones
        no_improvement_count = 0
        max_no_improvement = 50  # 连续无改进的最大次数
        adaptive_iters = self.local_search_iters
        
        # 模拟退火参数
        temperature = 10.0
        cooling_rate = 0.95

        for iteration in range(adaptive_iters):
            # 自适应调整：如果连续无改进，增加搜索范围
            if no_improvement_count > max_no_improvement:
                search_range = min(5, self.n_zones)
            else:
                search_range = 1

            improvement_found = False
            
            for _ in range(search_range):
                # 优先优化周长较大的分区
                zone_perimeters = [(i, calculate_perimeter_fast(zones[i], self.graph, self.coord_index)) 
                                for i in range(self.n_zones)]
                zone_perimeters.sort(key=lambda x: x[1], reverse=True)
                
                for zone_idx, _ in zone_perimeters:
                    # 获取当前分区的边界节点
                    boundary = get_boundary_nodes(self.graph, zones[zone_idx])
                    if not boundary:
                        continue
                    
                    # 尝试多种邻域结构
                    # 1. 节点交换
                    for node in boundary:
                        # 找到与该边界节点相邻的其他分区
                        neighbors = self.graph.neighbors(node)
                        adjacent_zones = set()
                        
                        for neighbor in neighbors:
                            for target_idx, target_zone in enumerate(zones):
                                if target_idx == zone_idx:
                                    continue
                                if neighbor in target_zone:
                                    adjacent_zones.add(target_idx)
                                    break
                        
                        for target_idx in adjacent_zones:
                            # 检查目标分区是否有边界节点
                            target_boundary = get_boundary_nodes(self.graph, zones[target_idx])
                            if not target_boundary:
                                continue
                            
                            for target_node in target_boundary:
                                # 检查交换后两个分区是否仍然连通
                                if target_node not in self.graph.neighbors(node):
                                    continue
                                
                                new_zones = [set(z) for z in zones]
                                new_zones[zone_idx].remove(node)
                                new_zones[zone_idx].add(target_node)
                                new_zones[target_idx].remove(target_node)
                                new_zones[target_idx].add(node)
                                
                                # 检查连通性
                                if not check_connectivity(self.graph, new_zones[zone_idx]):
                                    continue
                                if not check_connectivity(self.graph, new_zones[target_idx]):
                                    continue
                                
                                # 检查面板数约束
                                if len(new_zones[zone_idx]) < self.min_panels or \
                                len(new_zones[target_idx]) < self.min_panels or \
                                len(new_zones[zone_idx]) > self.max_panels or \
                                len(new_zones[target_idx]) > self.max_panels:
                                    continue
                                
                                # 检查负载平衡约束
                                new_sizes = [len(z) for z in new_zones]
                                if max(new_sizes) - min(new_sizes) > self.max_panel_diff:
                                    continue
                                
                                # 计算新的总周长
                                new_perimeter = self._total_perimeter(new_zones)
                                
                                # 模拟退火接受准则
                                delta = new_perimeter - best_perimeter
                                if delta < 0 or random.random() < math.exp(-delta / temperature):
                                    best_zones = new_zones
                                    best_perimeter = new_perimeter
                                    improvement_found = True
                                    no_improvement_count = 0
                                    zones = new_zones
                                    break
                            
                            if improvement_found:
                                break
                        
                        if improvement_found:
                            break
                    
                    if improvement_found:
                        break
                
                if improvement_found:
                    break

            if not improvement_found:
                no_improvement_count += 1
            
            # 冷却温度
            temperature *= cooling_rate
            
            # 提前终止条件
            if no_improvement_count > max_no_improvement * 2:
                break

        return best_zones

    def _fix_perimeter_violations(self, zones: List[Set[str]]) -> List[Set[str]]:
        """
        Fix zones whose perimeter is outside [perimeter_lb, perimeter_ub].

        - LB violation (too compact): steal a boundary node from a neighbour
          zone to make this zone less compact (increase perimeter).
        - UB violation (too spread out): give away a boundary node to a
          neighbour zone to make this zone more compact (decrease perimeter).
        """
        for _ in range(300):
            violated_lb = []
            violated_ub = []
            for i in range(self.n_zones):
                peri = calculate_perimeter_fast(zones[i], self.graph, self.coord_index)
                if peri < self.perimeter_lb:
                    violated_lb.append((i, peri))
                elif peri > self.perimeter_ub:
                    violated_ub.append((i, peri))
            if not violated_lb and not violated_ub:
                break

            moved = False

            # --- Fix LB violations: steal a node to increase perimeter ---
            for vi, v_peri in violated_lb:
                if moved:
                    break
                for nb_idx in range(self.n_zones):
                    if nb_idx == vi or moved:
                        continue
                    if len(zones[nb_idx]) <= self.min_panels:
                        continue
                    boundary = get_boundary_nodes(self.graph, zones[nb_idx])
                    for node in boundary:
                        adj = any(n in zones[vi] for n in self.graph.neighbors(node))
                        if not adj:
                            continue
                        new_src = zones[nb_idx] - {node}
                        if len(new_src) < self.min_panels:
                            continue
                        if not check_connectivity(self.graph, new_src):
                            continue
                        new_dst = zones[vi] | {node}
                        if len(new_dst) > self.max_panels:
                            continue
                        if not check_connectivity(self.graph, new_dst):
                            continue
                        new_peri = calculate_perimeter_fast(new_dst, self.graph, self.coord_index)
                        if new_peri > v_peri:
                            zones[nb_idx].remove(node)
                            zones[vi].add(node)
                            moved = True
                            break

            # --- Fix UB violations: give away a node to decrease perimeter ---
            for vi, v_peri in violated_ub:
                if moved:
                    break
                boundary = get_boundary_nodes(self.graph, zones[vi])
                # Sort by how much removing the node would reduce perimeter
                candidates = []
                for node in boundary:
                    new_zone = zones[vi] - {node}
                    if len(new_zone) < self.min_panels:
                        continue
                    if not check_connectivity(self.graph, new_zone):
                        continue
                    new_peri = calculate_perimeter_fast(new_zone, self.graph, self.coord_index)
                    if new_peri < v_peri:
                        candidates.append((node, new_peri))
                candidates.sort(key=lambda x: x[1])
                for node, _ in candidates:
                    # Find a neighbour zone to accept this node
                    for nb_idx in range(self.n_zones):
                        if nb_idx == vi:
                            continue
                        if len(zones[nb_idx]) >= self.max_panels:
                            continue
                        adj = any(n in zones[nb_idx] for n in self.graph.neighbors(node))
                        if not adj:
                            continue
                        new_dst = zones[nb_idx] | {node}
                        if not check_connectivity(self.graph, new_dst):
                            continue
                        zones[vi].remove(node)
                        zones[nb_idx].add(node)
                        moved = True
                        break
                    if moved:
                        break

            if not moved:
                break
        return zones

    def _total_perimeter(self, zones: List[Set[str]]) -> float:
        """计算所有分区的总周长。"""
        return sum(
            calculate_perimeter_fast(zone, self.graph, self.coord_index)
            for zone in zones
        )

    def _find_nearest_zone_bfs(self, node: str, zones: List[Set[str]],
                               exclude: int = -1) -> int:
        """Find the nearest zone by graph BFS from *node*.

        Returns the zone index whose member is reached first via BFS,
        skipping the zone at index *exclude*.
        """
        from collections import deque
        visited = {node}
        queue = deque([node])
        while queue:
            cur = queue.popleft()
            for nb in self.graph.neighbors(cur):
                if nb in visited:
                    continue
                for j, zone in enumerate(zones):
                    if j == exclude:
                        continue
                    if nb in zone:
                        return j
                visited.add(nb)
                queue.append(nb)
        # Fallback: return the largest zone (should never reach here)
        sizes = [len(z) for z in zones]
        return sizes.index(max(sizes))

    def _find_zone_of(self, zones: List[Set[str]], node: str) -> int:
        """查找节点所属的分区索引。"""
        for i, zone in enumerate(zones):
            if node in zone:
                return i
        return -1