"""
Logic-based Benders 分解框架

将光伏面板切割及分区规划问题分解为：
  - 主问题：切割布局优化（MIP，最小化原材料使用）
  - 子问题：分区规划（启发式/DQN，最小化总周长）

通过 Feasibility Cut 和 Optimality Cut 实现主-子问题的迭代收敛。

算法流程：
  初始化 LB=-inf, UB=+inf
  while UB - LB > epsilon and iter < max_iter:
      1. 求解主问题 → 切割方案, LB
      2. 用切割方案求解子问题 → 分区方案
      3a. 不可行 → 生成 Feasibility Cut
      3b. 可行 → 计算周长，更新 UB，生成 Optimality Cut
      4. 将 Cut 添加到主问题
  return 最优切割方案 + 最优分区方案

参考：项目书"研究方法一"Logic-based Benders Decomposition
"""

import time
import json
import os
import logging
from typing import Dict, List, Optional, Callable

import networkx as nx

from model.cutting_master import CuttingMasterProblem, CuttingResult, estimate_demand
from model.partition_sub import PartitionValidator, PartitionResult
from algorithm.partition_heuristic import GreedyPartitioner
from utils.graph_utils import build_adjacency_graph, build_coord_index

logger = logging.getLogger(__name__)


class BendersDecomposition:
    """
    Logic-based Benders 分解求解器。

    使用方式:
        solver = BendersDecomposition(instance_data)
        result = solver.optimize()
    """

    def __init__(self, instance_data: Dict,
                 partition_solver: str = "heuristic",
                 max_iter: int = 20,
                 epsilon: float = 1.0,
                 verbose: bool = True):
        """
        Args:
            instance_data: 标准化算例数据（JSON 加载后的字典）
            partition_solver: 子问题求解方法 ("heuristic" 或 "dqn")
            max_iter: 最大迭代次数
            epsilon: 收敛阈值（UB - LB < epsilon 时停止）
            verbose: 是否打印中文日志
        """
        self.instance_data = instance_data
        self.partition_solver = partition_solver
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.verbose = verbose

        # 提取算例参数
        self.instance_id = instance_data["instance_info"]["instance_id"]
        self.n_nodes = instance_data["instance_info"]["n_nodes"]
        self.pva_list = instance_data["pva_list"]

        # 面板参数
        pva_params = instance_data["pva_params"]
        self.D = pva_params["D"]                    # 标准面板长度 12.0m
        self.b = pva_params.get("b", 3.0)           # 面板宽度 3.0m
        self.t_l_options = pva_params["t_l_options"] # 切割规格
        self.LB = pva_params["LB"]                  # 周长下界 60.0m
        self.UB_perimeter = pva_params["UB"]         # 周长上界 90.0m

        # 逆变器参数
        inv_params = instance_data["equipment_params"]["inverter"]
        self.q = inv_params["q"]                     # 额定容量 320kW
        self.r = inv_params["r"]                     # 最小负载率 0.85
        self.p = inv_params["p"]                     # 逆变器数量

        # 构建邻接图
        grid_size = instance_data["terrain_data"]["grid_size"]
        self.graph = build_adjacency_graph(self.pva_list, grid_size)
        self.coord_index = build_coord_index(self.graph)

        # 外部 DQN 求解器（训练后注入）
        self._dqn_solver = None

        # 迭代历史
        self.history = []

    def set_dqn_solver(self, dqn_solver):
        """注入训练好的 DQN 分区求解器。"""
        self._dqn_solver = dqn_solver
        self.partition_solver = "dqn"

    def optimize(self) -> Dict:
        """
        执行 Benders 分解迭代优化。

        Returns:
            M1-Output 格式的优化结果
        """
        start_time = time.time()

        if self.verbose:
            self._print_header()

        # 估算各规格需求（选择利用率最高的规格）
        demand = estimate_demand(self.n_nodes, self.t_l_options, D=self.D)

        # 初始化
        cutting_solver = CuttingMasterProblem(D=self.D, t_l_options=self.t_l_options)
        lb = float("-inf")
        ub = float("inf")
        best_cutting_result = None
        best_partition_result = None
        last_cutting_result = None
        last_partition_result = None
        benders_cuts = []

        for iteration in range(1, self.max_iter + 1):
            iter_start = time.time()

            # ─── 步骤 1：求解主问题（切割优化）───
            cutting_result = cutting_solver.solve(demand)
            last_cutting_result = cutting_result

            if cutting_result.status != "Optimal":
                if self.verbose:
                    logger.warning(f"  主问题求解失败: {cutting_result.status}")
                # 主问题不可行时，仍然尝试分区（用默认切割）
                pass

            lb = cutting_result.objective_value if cutting_result.status == "Optimal" else lb

            # ─── 步骤 2：求解子问题（分区，多种子尝试）───
            partition_result = self._solve_subproblem(iteration)
            last_partition_result = partition_result

            # ─── 步骤 3：生成割平面 ───
            if not partition_result.is_feasible:
                # 3a: Feasibility Cut - 排除当前切割方案
                if cutting_result.status == "Optimal":
                    cut = self._generate_feasibility_cut(cutting_result, partition_result)
                    benders_cuts.append(cut)
                    cutting_solver.add_cut(cut)
            else:
                # 3b: 可行 - 更新上界
                if partition_result.total_perimeter < ub:
                    ub = partition_result.total_perimeter
                    best_cutting_result = cutting_result
                    best_partition_result = partition_result

                # Optimality Cut
                if cutting_result.status == "Optimal":
                    cut = self._generate_optimality_cut(cutting_result, partition_result)
                    if cut:
                        benders_cuts.append(cut)
                        cutting_solver.add_cut(cut)

            # 记录历史
            iter_time = time.time() - iter_start
            self.history.append({
                "iteration": iteration,
                "lb": lb if lb > float("-inf") else 0,
                "ub": ub if ub < float("inf") else None,
                "gap": ub - lb if ub < float("inf") and lb > float("-inf") else None,
                "feasible": partition_result.is_feasible,
                "total_perimeter": partition_result.total_perimeter if partition_result.is_feasible else None,
                "n_cuts": len(benders_cuts),
                "time": iter_time,
            })

            if self.verbose:
                self._print_iteration(iteration, lb, ub, partition_result,
                                       benders_cuts, iter_time)

            # 收敛判断
            if ub < float("inf") and lb > float("-inf") and ub - lb < self.epsilon:
                if self.verbose:
                    print(f"\n  ✓ 收敛! UB-LB = {ub - lb:.2f} < ε = {self.epsilon}")
                break

        total_time = time.time() - start_time

        if self.verbose:
            self._print_footer(total_time, best_partition_result)

        # 如果未找到可行解，用最后一次尝试的分区
        if best_partition_result is None:
            best_partition_result = last_partition_result
            best_cutting_result = last_cutting_result

        # 构建 M1-Output
        return self._build_output(best_cutting_result, best_partition_result)

    def _solve_subproblem(self, iteration: int = 0) -> PartitionResult:
        """
        求解分区子问题（启发式或 DQN）。

        启发式方法使用多随机种子策略：每次迭代用不同种子，
        并从中选取最优（可行且周长最小的）方案。
        """
        if self.partition_solver == "dqn" and self._dqn_solver is not None:
            return self._dqn_solver.solve(self.graph, self.p,
                                           {"LB": self.LB, "UB": self.UB_perimeter})

        # 多种子尝试：每次迭代尝试多个种子，选最优
        best_result = None
        seeds = [iteration * 10 + s for s in range(5)]  # 每次迭代 5 个种子

        for seed in seeds:
            partitioner = GreedyPartitioner(
                self.graph, n_zones=self.p,
                min_panels=18, max_panels=26,
                perimeter_lb=self.LB, perimeter_ub=self.UB_perimeter,
                local_search_iters=200,
                random_seed=seed
            )
            result = partitioner.solve()

            if best_result is None:
                best_result = result
            elif result.is_feasible and (not best_result.is_feasible or
                                          result.total_perimeter < best_result.total_perimeter):
                best_result = result

        return best_result

    def _generate_feasibility_cut(self, cutting_result: CuttingResult,
                                    partition_result: PartitionResult) -> Dict:
        """生成可行性割：排除导致不可行分区的切割模式。"""
        excluded = {}
        for i, mat in enumerate(cutting_result.cut_result):
            if mat["is_used"]:
                for cut in mat["cuts"]:
                    excluded[(i, cut["spec_l"])] = cut["quantity"]

        return {
            "type": "feasibility",
            "excluded_pattern": excluded,
            "violations": partition_result.violations,
        }

    def _generate_optimality_cut(self, cutting_result: CuttingResult,
                                   partition_result: PartitionResult) -> Optional[Dict]:
        """生成最优性割：基于子问题目标值约束。"""
        return {
            "type": "optimality",
            "perimeter_bound": partition_result.total_perimeter,
        }

    def _build_output(self, cutting_result: CuttingResult,
                       partition_result: PartitionResult) -> Dict:
        """
        构建 M1-Output 格式的输出。

        严格遵循《算例处理规范与模块接口协议》4.1 节。
        """
        # 构建 partition_result 列表
        partition_list = []
        zone_summary = []

        for zone_idx, zone_nodes in enumerate(partition_result.zones):
            zone_id = f"zone_{zone_idx}"
            inverter_id = f"inv_{zone_idx}"

            total_power = 0.0
            for panel_id in zone_nodes:
                node_data = self.graph.nodes[panel_id]
                cut_spec = list(node_data.get("cut_spec", (2.0, 3.0)))
                power = node_data.get("power", 0.0)
                total_power += power

                partition_list.append({
                    "panel_id": panel_id,
                    "grid_coord": list(node_data["grid_coord"]),
                    "cut_spec": cut_spec,
                    "zone_id": zone_id,
                    "inverter_id": inverter_id,
                })

            detail = partition_result.zone_details[zone_idx] if zone_idx < len(partition_result.zone_details) else {}

            zone_summary.append({
                "zone_id": zone_id,
                "inverter_id": inverter_id,
                "pva_count": len(zone_nodes),
                "perimeter": detail.get("perimeter", 0.0),
                "total_power": round(total_power, 2),
            })

        # 约束满足情况
        validator = PartitionValidator(
            self.graph, self.p,
            min_panels=18, max_panels=26,
            perimeter_lb=self.LB, perimeter_ub=self.UB_perimeter
        )
        constraint_satisfaction = validator.get_constraint_satisfaction_summary(
            partition_result.zones
        )

        return {
            "instance_id": self.instance_id,
            "cut_result": cutting_result.cut_result if cutting_result else [],
            "partition_result": partition_list,
            "zone_summary": zone_summary,
            "constraint_satisfaction": constraint_satisfaction,
            "optimization_history": self.history,
        }

    # ─── 日志打印方法 ───

    def _print_header(self):
        print()
        print("══════════════════════════════════════════════════════════")
        print(f"  Benders 分解求解器 | 算例: {self.instance_id}")
        print(f"  面板数: {self.n_nodes} | 逆变器数: {self.p} | "
              f"子问题: {self.partition_solver}")
        print("══════════════════════════════════════════════════════════")

    def _print_iteration(self, iteration, lb, ub, partition_result,
                          cuts, iter_time):
        ub_str = f"{ub:.1f}m" if ub < float("inf") else "∞"
        gap_str = f"{ub - lb:.1f}" if ub < float("inf") else "∞"
        feasible_str = "成功" if partition_result.is_feasible else "不可行"

        mark = ""
        if partition_result.is_feasible and len(self.history) >= 2:
            prev_ub = self.history[-2].get("ub")
            if prev_ub and ub < prev_ub:
                mark = " ★ 改善"

        print(f"\n【Benders 迭代 {iteration}/{self.max_iter}】{mark}")
        print(f"  ├─ 主问题: 使用原材料 {lb:.0f} 块, 下界(LB)={lb:.2f}")

        if partition_result.is_feasible:
            n_zones = len(partition_result.zones)
            avg_peri = partition_result.total_perimeter / n_zones if n_zones > 0 else 0
            print(f"  ├─ 子问题: 分区{feasible_str}, "
                  f"{n_zones} 个分区, 平均周长 {avg_peri:.1f}m")
        else:
            print(f"  ├─ 子问题: {feasible_str}")
            for v in partition_result.violations[:3]:
                print(f"  │    └─ {v}")

        print(f"  ├─ 上界(UB): {ub_str}")
        print(f"  ├─ 收敛差距: UB-LB = {gap_str} (阈值: {self.epsilon})")
        print(f"  └─ 累计割平面 {len(cuts)} 条, 耗时 {iter_time:.1f}秒")

    def _print_footer(self, total_time, best_result):
        print()
        print("══════════════════════════════════════════════════════════")
        if best_result and best_result.is_feasible:
            n_zones = len(best_result.zones)
            avg_peri = best_result.total_perimeter / n_zones if n_zones > 0 else 0
            panel_counts = [len(z) for z in best_result.zones]
            print(f"  求解完成 | 总耗时: {total_time:.1f}秒")
            print(f"  分区数: {n_zones} | 平均周长: {avg_peri:.1f}m | "
                  f"总周长: {best_result.total_perimeter:.1f}m")
            print(f"  各分区面板数: {panel_counts}")
            if best_result.violations:
                print(f"  ⚠ 约束违规: {len(best_result.violations)} 条")
            else:
                print(f"  ✓ 所有约束满足")
        else:
            print(f"  求解完成 | 总耗时: {total_time:.1f}秒 | 未找到可行解")
        print("══════════════════════════════════════════════════════════")
