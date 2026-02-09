"""
切割主问题 MIP 模型

基于 PuLP 求解光伏面板切割布局优化问题（项目书公式 1-4）。

主问题目标：最小化原材料使用数量（等价于最小化采购成本）
  min  Σ_{m∈M} y_m

约束：
  (1) Σ_{l∈L} t_l × x_{ml} ≤ D × y_m,  ∀ m ∈ M   -- 原材料长度约束
  (2) Σ_{m∈M} x_{ml} ≥ n_l,             ∀ l ∈ L   -- 需求满足约束
  (3) x_{ml} ∈ Z+                                   -- 切割数量非负整数
  (4) y_m ∈ {0, 1}                                   -- 原材料使用标记

其中：
  D = 标准面板长度（12.0m）
  t_l = 规格 l 的切割长度（2.0, 4.0, ..., 12.0m）
  n_l = 规格 l 的需求量
  x_{ml} = 原材料 m 切出规格 l 的数量
  y_m = 是否使用原材料 m
"""

import pulp
import math
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CuttingResult:
    """切割主问题求解结果"""
    status: str                      # 求解状态: Optimal / Infeasible / ...
    objective_value: float           # 目标值（使用的原材料数量）
    materials_used: int              # 使用的原材料块数
    cut_result: List[Dict]           # 切割方案（M1-Output 格式）
    utilization_rate: float          # 原材料利用率
    demand_satisfaction: Dict        # 各规格需求满足情况
    solver_time: float = 0.0        # 求解时间（秒）


class CuttingMasterProblem:
    """
    切割布局主问题求解器。

    将面板切割建模为一维切割库存问题（1D Cutting Stock Problem），
    使用 PuLP + CBC 求解 MIP。

    使用方式:
        solver = CuttingMasterProblem(D=12.0, t_l_options=[2.0, 4.0, ...])
        result = solver.solve(demand={2.0: 50, 4.0: 30, ...})
    """

    def __init__(self, D: float = 12.0, t_l_options: List[float] = None,
                 benders_cuts: List[Dict] = None):
        """
        Args:
            D: 标准光伏面板长度（米）
            t_l_options: 可切割规格列表（米），需为 2.0 的整数倍
            benders_cuts: Benders 割平面列表（从子问题反馈的约束）
        """
        self.D = D
        self.t_l_options = t_l_options or [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        self.benders_cuts = benders_cuts or []

        # 验证切割规格合法性
        for t_l in self.t_l_options:
            if t_l % 2.0 != 0:
                raise ValueError(f"切割规格 {t_l} 不是 2.0 的整数倍（错误码 E102）")
            if not (0 < t_l <= D):
                raise ValueError(f"切割规格 {t_l} 超出合法范围 (0, {D}]m")

    def solve(self, demand: Dict[float, int],
              max_materials: int = None,
              time_limit: float = 60.0) -> CuttingResult:
        """
        求解切割主问题。

        Args:
            demand: 各规格的需求量 {t_l: n_l}，如 {2.0: 50, 6.0: 30}
            max_materials: 原材料数量上限（None 则自动估算）
            time_limit: 求解器时间限制（秒）

        Returns:
            CuttingResult 求解结果
        """
        import time
        start_time = time.time()

        # 估算原材料数量上限
        if max_materials is None:
            total_length = sum(t_l * n_l for t_l, n_l in demand.items())
            max_materials = math.ceil(total_length / self.D) + 5  # 留余量

        M = range(max_materials)       # 原材料索引集合
        L = list(demand.keys())        # 切割规格集合

        # 创建 MIP 模型
        model = pulp.LpProblem("CuttingMaster", pulp.LpMinimize)

        # 决策变量
        # x[m][l]: 原材料 m 切出规格 l 的数量（非负整数）
        x = {}
        for m in M:
            x[m] = {}
            for l in L:
                x[m][l] = pulp.LpVariable(f"x_{m}_{l}", lowBound=0, cat="Integer")

        # y[m]: 是否使用原材料 m（0-1变量）
        y = {}
        for m in M:
            y[m] = pulp.LpVariable(f"y_{m}", cat="Binary")

        # 目标函数：最小化使用的原材料数量
        model += pulp.lpSum(y[m] for m in M), "MinMaterials"

        # 约束 (1): 原材料长度约束
        for m in M:
            model += (
                pulp.lpSum(l * x[m][l] for l in L) <= self.D * y[m],
                f"MaterialLength_{m}"
            )

        # 约束 (2): 需求满足约束
        for l in L:
            n_l = demand[l]
            model += (
                pulp.lpSum(x[m][l] for m in M) >= n_l,
                f"Demand_{l}"
            )

        # 对称性打破约束（加速求解）：强制原材料按顺序使用
        for m in range(1, max_materials):
            model += y[m] <= y[m - 1], f"Symmetry_{m}"

        # 添加 Benders 割平面
        for i, cut in enumerate(self.benders_cuts):
            self._add_benders_cut(model, cut, x, y, M, L, i)

        # 求解
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
        model.solve(solver)

        solve_time = time.time() - start_time

        # 解析结果
        status = pulp.LpStatus[model.status]

        if status != "Optimal":
            logger.warning(f"切割主问题求解状态: {status}")
            return CuttingResult(
                status=status, objective_value=float("inf"),
                materials_used=0, cut_result=[], utilization_rate=0.0,
                demand_satisfaction={}, solver_time=solve_time
            )

        # 构建切割结果（M1-Output 格式）
        cut_result = []
        materials_used = 0
        total_used_length = 0.0

        for m in M:
            is_used = pulp.value(y[m]) > 0.5
            if is_used:
                materials_used += 1

            cuts = []
            material_used_length = 0.0
            for l in L:
                qty = int(round(pulp.value(x[m][l])))
                if qty > 0:
                    cuts.append({"spec_l": l, "quantity": qty})
                    material_used_length += l * qty

            if is_used:
                total_used_length += material_used_length

            cut_result.append({
                "material_id": f"mat_{m}",
                "is_used": is_used,
                "cuts": cuts,
                "utilization": material_used_length / self.D if is_used else 0.0
            })

        # 需求满足情况
        demand_satisfaction = {}
        for l in L:
            produced = sum(int(round(pulp.value(x[m][l]))) for m in M)
            demand_satisfaction[l] = {
                "demand": demand[l],
                "produced": produced,
                "satisfied": produced >= demand[l]
            }

        # 利用率 = 实际使用的总切割长度 / (使用的原材料数 × D)
        utilization_rate = total_used_length / (materials_used * self.D) if materials_used > 0 else 0.0

        return CuttingResult(
            status=status,
            objective_value=pulp.value(model.objective),
            materials_used=materials_used,
            cut_result=cut_result,
            utilization_rate=utilization_rate,
            demand_satisfaction=demand_satisfaction,
            solver_time=solve_time
        )

    def _add_benders_cut(self, model, cut: Dict, x, y, M, L, index: int):
        """
        向主问题添加 Benders 割平面。

        Feasibility Cut: 排除导致不可行分区的切割方案
        Optimality Cut: 基于子问题目标值收紧上界
        """
        cut_type = cut.get("type", "feasibility")

        if cut_type == "feasibility":
            # 排除特定的切割模式：至少有一个 x[m][l] 与被排除方案不同
            excluded = cut.get("excluded_pattern", {})
            if excluded:
                # Σ |x[m][l] - x̄[m][l]| >= 1 的线性化
                # 等价于：Σ_{方案中 x̄>0 的位置} x[m][l] <= Σ x̄[m][l] - 1
                total = sum(v for v in excluded.values())
                terms = []
                for key, val in excluded.items():
                    m_idx, l_val = key
                    if m_idx in range(len(list(M))) and l_val in L:
                        terms.append(x[m_idx][l_val])
                if terms:
                    model += (
                        pulp.lpSum(terms) <= total - 1,
                        f"FeasibilityCut_{index}"
                    )

        elif cut_type == "optimality":
            # 子问题的目标值下界约束
            # 这里通过限制切割方案的某些特征来收紧
            pass  # 在 Benders 框架中动态添加

    def add_cut(self, cut: Dict):
        """添加一条 Benders 割平面（供迭代时调用）。"""
        self.benders_cuts.append(cut)


def estimate_demand(n_panels: int, t_l_options: List[float],
                     D: float = 12.0,
                     preferred_spec: float = None) -> Dict[float, int]:
    """
    根据面板数量估算各规格的需求量。

    选择能最大化利用率的规格（即 D / t_l 为整数的规格优先），
    确保 MIP 一定有可行解。

    Args:
        n_panels: 面板总数
        t_l_options: 可用切割规格
        D: 标准面板长度
        preferred_spec: 首选规格（None 则自动选择利用率最高的）

    Returns:
        需求字典 {spec_l: quantity}
    """
    if preferred_spec is None:
        # 选择利用率最高的规格（D/t_l 为整数且切割数最多）
        best_spec = None
        best_panels_per_material = 0
        for t_l in t_l_options:
            panels_per_material = int(D // t_l)
            utilization = (panels_per_material * t_l) / D
            if panels_per_material > best_panels_per_material or \
               (panels_per_material == best_panels_per_material and utilization > 0.99):
                best_panels_per_material = panels_per_material
                best_spec = t_l
        preferred_spec = best_spec or t_l_options[0]

    demand = {}
    for t_l in t_l_options:
        if t_l == preferred_spec:
            demand[t_l] = n_panels
        else:
            demand[t_l] = 0

    return demand
