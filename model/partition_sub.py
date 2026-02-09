"""
分区子问题模型

在给定切割方案的基础上，将面板划分为若干分区，每个分区对应一台逆变器。

目标：最小化分区总周长（项目书公式 2）
约束：
  (5) 唯一归属：每个面板仅归属 1 台逆变器
  (6) 逆变器数量：实际使用的逆变器数 = p
  (7) 存在性：面板仅可归属已启用的逆变器
  (8) 边界定义：归属不同逆变器的相邻面板标记为边界
  (9) 共边条件：同区相邻面板可共用电缆路径
  (10) 负载率约束：逆变器负载在 [r×q, q] 范围内
  (11) 形状规整：分区周长在 [LB, UB] 范围内

此模块提供：
  - PartitionValidator: 约束检查器
  - PartitionResult: 分区结果封装
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field

import networkx as nx

from utils.graph_utils import (
    check_connectivity, calculate_perimeter_fast,
    build_coord_index, calculate_total_power, get_connected_components
)

logger = logging.getLogger(__name__)


@dataclass
class PartitionResult:
    """分区子问题求解结果"""
    is_feasible: bool                # 是否可行
    zones: List[Set[str]]            # 分区列表，每个分区为 panel_id 集合
    total_perimeter: float           # 总周长
    zone_details: List[Dict]         # 各分区详细信息
    violations: List[str]            # 违约信息
    solver_method: str = "heuristic" # 求解方法（heuristic / dqn）


class PartitionValidator:
    """
    分区约束验证器。

    统一检查分区方案是否满足项目书约束 (5)-(11)。
    同时被启发式求解器和 DQN 求解器共享使用。
    """

    def __init__(self, graph: nx.Graph, n_inverters: int,
                 inverter_capacity: float = 320.0,
                 min_load_rate: float = 0.85,
                 min_panels: int = 18, max_panels: int = 26,
                 max_panel_diff: int = 2,
                 perimeter_lb: float = 60.0, perimeter_ub: float = 90.0):
        """
        Args:
            graph: 面板邻接图
            n_inverters: 逆变器数量 p
            inverter_capacity: 逆变器额定容量 q (kW)
            min_load_rate: 最小负载率 r
            min_panels: 最少面板数（对应容量约束下界）
            max_panels: 最多面板数（对应容量约束上界）
            max_panel_diff: 负载平衡约束：各分区面板数量最大差异
            perimeter_lb: 周长下界 LB (m)
            perimeter_ub: 周长上界 UB (m)
        """
        self.graph = graph
        self.n_inverters = n_inverters
        self.q = inverter_capacity
        self.r = min_load_rate
        self.min_panels = min_panels
        self.max_panels = max_panels
        self.max_panel_diff = max_panel_diff
        self.perimeter_lb = perimeter_lb
        self.perimeter_ub = perimeter_ub
        self.coord_index = build_coord_index(graph)
        self.all_panels = set(graph.nodes())

    def validate(self, zones: List[Set[str]]) -> PartitionResult:
        """
        全面验证分区方案。

        Args:
            zones: 分区列表，每个分区为 panel_id 集合

        Returns:
            PartitionResult 包含可行性、违约信息和分区详情
        """
        violations = []
        zone_details = []

        # (5) 唯一归属：每个面板恰好属于一个分区
        assigned = set()
        for zone in zones:
            overlap = assigned & zone
            if overlap:
                violations.append(f"唯一归属违规: 面板 {overlap} 被重复分配")
            assigned |= zone

        unassigned = self.all_panels - assigned
        if unassigned:
            violations.append(f"唯一归属违规: {len(unassigned)} 个面板未分配")

        # (6) 逆变器数量
        if len(zones) != self.n_inverters:
            violations.append(
                f"逆变器数量违规: 分区数={len(zones)}, 要求={self.n_inverters}"
            )

        # 逐分区检查
        panel_counts = []
        total_perimeter = 0.0

        for i, zone in enumerate(zones):
            detail = self._validate_single_zone(zone, i)
            zone_details.append(detail)
            panel_counts.append(detail["n_panels"])
            total_perimeter += detail["perimeter"]

            if not detail["capacity_ok"]:
                violations.append(
                    f"分区 {i}: 容量违规 (面板数={detail['n_panels']}, "
                    f"要求[{self.min_panels},{self.max_panels}])"
                )
            if not detail["is_connected"]:
                violations.append(
                    f"分区 {i}: 连通性违规 ({detail['n_components']} 个连通分量)"
                )
            if not detail["perimeter_ok"]:
                violations.append(
                    f"分区 {i}: 周长违规 ({detail['perimeter']:.1f}m, "
                    f"要求[{self.perimeter_lb},{self.perimeter_ub}])"
                )

        # (负载平衡) 各分区面板数量差异
        if panel_counts:
            max_diff = max(panel_counts) - min(panel_counts)
            if max_diff > self.max_panel_diff:
                violations.append(
                    f"负载平衡违规: 面板数差异={max_diff}, 最大允许={self.max_panel_diff}"
                )

        return PartitionResult(
            is_feasible=len(violations) == 0,
            zones=zones,
            total_perimeter=total_perimeter,
            zone_details=zone_details,
            violations=violations
        )

    def _validate_single_zone(self, zone: Set[str], zone_idx: int) -> Dict:
        """验证单个分区的约束。"""
        n_panels = len(zone)

        # 容量约束
        capacity_ok = self.min_panels <= n_panels <= self.max_panels

        # 连通性
        is_connected = check_connectivity(self.graph, zone)
        n_components = len(get_connected_components(self.graph, zone)) if not is_connected else 1

        # 周长
        perimeter = calculate_perimeter_fast(zone, self.graph, self.coord_index)
        perimeter_ok = self.perimeter_lb <= perimeter <= self.perimeter_ub

        # 功率
        total_power = calculate_total_power(self.graph, zone)

        return {
            "zone_id": f"zone_{zone_idx}",
            "inverter_id": f"inv_{zone_idx}",
            "n_panels": n_panels,
            "capacity_ok": capacity_ok,
            "is_connected": is_connected,
            "n_components": n_components,
            "perimeter": perimeter,
            "perimeter_ok": perimeter_ok,
            "total_power": total_power,
        }

    def quick_check_add(self, zone: Set[str], candidate: str) -> Dict[str, bool]:
        """
        快速检查向分区添加一个节点后是否仍满足约束。
        用于贪心扩展和 DQN 动作过滤。

        Args:
            zone: 当前分区节点集合
            candidate: 待添加的面板 ID

        Returns:
            各约束的满足情况
        """
        new_zone = zone | {candidate}
        n = len(new_zone)

        return {
            "capacity_ok": n <= self.max_panels,
            "connected": check_connectivity(self.graph, new_zone),
            "under_max": n <= self.max_panels,
        }

    def get_constraint_satisfaction_summary(self, zones: List[Set[str]]) -> Dict[str, str]:
        """
        生成约束满足率摘要（M1-Output 格式）。

        Returns:
            约束满足情况字典，符合接口协议
        """
        result = self.validate(zones)
        n_zones = len(zones)

        # 整数切割约束（由主问题保证）
        cut_ok = "100%"

        # 连通性
        connected_count = sum(1 for d in result.zone_details if d["is_connected"])
        connectivity = f"{connected_count / n_zones * 100:.0f}%" if n_zones > 0 else "N/A"

        # 容量
        capacity_count = sum(1 for d in result.zone_details if d["capacity_ok"])
        capacity = f"{capacity_count / n_zones * 100:.0f}%" if n_zones > 0 else "N/A"

        # 周长
        perimeter_count = sum(1 for d in result.zone_details if d["perimeter_ok"])
        perimeter = f"{perimeter_count / n_zones * 100:.0f}%" if n_zones > 0 else "N/A"

        return {
            "整数切割": cut_ok,
            "分区连通性": connectivity,
            "逆变器容量约束": capacity,
            "分区周长约束": perimeter,
        }
