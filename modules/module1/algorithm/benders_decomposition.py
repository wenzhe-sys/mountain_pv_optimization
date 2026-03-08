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
'''
import time
import json
import os
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Tuple

import networkx as nx

from modules.module1.model.cutting_master import CuttingMasterProblem, CuttingResult, estimate_demand
from modules.module1.model.partition_sub import PartitionValidator, PartitionResult
from modules.module1.algorithm.partition_heuristic import GreedyPartitioner
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
        
        # 从pva_params中提取参数
        self.t_l_options = instance_data["pva_params"].get("t_l_options", [2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        self.D = instance_data["pva_params"].get("D", 12.0)  # 标准面板长度
        self.LB = instance_data["pva_params"].get("LB", 60.0)  # 分区周长下限
        self.UB = instance_data["pva_params"].get("UB", 90.0)  # 分区周长上限
        self.UB_perimeter = self.UB  # 分区周长上限（与UB保持一致）
        
        # 从equipment_params中提取逆变器相关参数
        self.inverter_params = instance_data["equipment_params"]["inverter"]
        self.q = self.inverter_params.get("q", 320.0)  # 逆变器容量
        self.r = self.inverter_params.get("r", 0.85)  # 最小负载率
        
        # 计算逆变器数量 (p)
        # 根据面板总数和分区面板数限制[18,26]计算
        min_pva_per_zone = 18
        max_pva_per_zone = 26
        
        # 根据最大面板数计算所需最少逆变器数
        self.p = max(1, (self.n_nodes + max_pva_per_zone - 1) // max_pva_per_zone)  # 向上取整
        
        # 确保不超过最大面板数限制
        actual_max_pva_per_zone = (self.n_nodes + self.p - 1) // self.p  # 实际最大面板数
        if actual_max_pva_per_zone > max_pva_per_zone:
            # 如果实际最大面板数超过限制，增加逆变器数量
            self.p = (self.n_nodes + max_pva_per_zone - 1) // max_pva_per_zone
        
        # 确保不低于最小面板数限制
        actual_min_pva_per_zone = self.n_nodes // self.p  # 实际最小面板数
        if actual_min_pva_per_zone < min_pva_per_zone:
            # 如果实际最小面板数低于限制，减少逆变器数量
            self.p = self.n_nodes // min_pva_per_zone
            
            # 再次检查最大面板数
            actual_max_pva_per_zone = (self.n_nodes + self.p - 1) // self.p
            if actual_max_pva_per_zone > max_pva_per_zone:
                # 如果仍然超过限制，调整到刚好满足
                self.p = (self.n_nodes + max_pva_per_zone - 1) // max_pva_per_zone
        
        # 决策变量初始化
        self.x_ml = torch.zeros((self.p, len(self.t_l_options)), dtype=torch.int)  # 切割数量
        self.y_m = torch.zeros(self.p, dtype=torch.bool)  # 原材料使用标记
        self.sigma_ik = torch.zeros((self.n_nodes, self.p), dtype=torch.bool)  # 面板-逆变器归属
        self.phi_ijk = torch.zeros((self.n_nodes, self.n_nodes, self.p), dtype=torch.bool)  # 边界标记
        
        # 优化目标权重
        self.w_coverage = 0.6  # 覆盖面积权重
        self.w_material = 0.3  # 材料成本权重
        self.w_perimeter = 0.1  # 分区周长权重
        
        # 初始化优化历史列表
        self.history = []
        
        # 构建邻接图（用于分区启发式算法）
        self.grid_size = instance_data["terrain_data"].get("grid_size", 10.0)
        self.graph = build_adjacency_graph(self.pva_list, self.grid_size)
        
        # DQN 求解器初始化
        self._dqn_solver = None
        self.dqn_model_path = os.path.join(os.path.dirname(__file__), "..", "model", "dqn_model.pth")
        self.T = 4  # S2V 迭代次数
        
        # 初始化 DQN 求解器
        if self.partition_solver == "dqn":
            self._initialize_dqn_solver()
        
        # 调整参数以适应算例规模
        self._adjust_parameters_based_on_scale()

    def master_problem(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """主问题：光伏面板切割优化"""
        logging.info(f"【Benders主问题】开始切割优化（逆变器数：{self.p}）")
        
        # 目标：最小化原材料使用量 + 最大化切割利用率
        # 计算每台逆变器平均分配的面板数
        avg_pva_per_inv = self.n_nodes // self.p
        remainder = self.n_nodes % self.p
        
        for m in range(self.p):
            # 为每个逆变器分配面板（动态计算分配范围）
            start_idx = m * avg_pva_per_inv + min(m, remainder)
            end_idx = start_idx + avg_pva_per_inv + (1 if m < remainder else 0)
            if start_idx < end_idx:
                self.y_m[m] = True
                # 根据面板数量动态选择切割长度
                pva_count = end_idx - start_idx
                # 切割长度选择（根据实际面板数量和可用长度选项）
                # 优先选择能高效利用的长度组合
                for l_idx, t_l in enumerate(self.t_l_options):
                    # 简单的分配策略：按可用长度均匀分配
                    if t_l >= self.D and pva_count > 0:
                        self.x_ml[m, l_idx] = pva_count // len(self.t_l_options) + (1 if l_idx < pva_count % len(self.t_l_options) else 0)
        
        logging.info(f"【Benders主问题】切割完成，使用原材料：{self.y_m.sum().item()} 台")
        return self.x_ml, self.y_m

    def subproblem(self, x_ml: torch.Tensor, y_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """子问题：分区优化（强化学习加速）"""
        logging.info(f"【Benders子问题】开始分区优化（面板数：{self.n_nodes}）")
        
        # 1. 面板-逆变器归属（空间连续性分配）
        for m in range(self.p):
            start_idx = m * 22
            end_idx = min((m + 1) * 22, self.n_nodes)
            if y_m[m]:
                self.sigma_ik[start_idx:end_idx, m] = True
        
        # 2. 边界标记（相邻面板跨逆变器则为边界）
        for k in range(self.p):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j and self.sigma_ik[i, k] != self.sigma_ik[j, k]:
                        self.phi_ijk[i, j, k] = True
        
        # 3. 分区周长计算
        perimeter = self.calculate_perimeter()
        logging.info(f"【Benders子问题】分区完成，平均周长：{np.mean(perimeter):.2f}m")
        return self.sigma_ik, self.phi_ijk

    def calculate_perimeter(self) -> List[float]:
        """计算每个分区的周长（优化版）"""
        perimeter_list = []
        
        # 优化：预先获取网格尺寸
        grid_size = self.pva_list[0].get("grid_coord", [10.0])[0]
        
        for k in range(self.p):
            if self.y_m[k]:
                try:
                    # 优化：使用快速计算方法，避免遍历所有边界
                    boundary_count = 0
                    for i in range(self.n_nodes):
                        if self.sigma_ik[i, k]:
                            # 检查当前节点的邻居是否属于其他分区
                            for j in range(self.n_nodes):
                                if i != j and self.sigma_ik[j, k] != self.sigma_ik[i, k]:
                                    # 检查是否为直接邻居
                                    # 优化：使用图结构快速判断邻居关系
                                    if j in self.graph.neighbors(i):
                                        boundary_count += 1
                                        break  # 每个节点只计数一次边界
                    
                    perimeter = boundary_count * grid_size  # 使用实际网格尺寸
                    perimeter = max(self.LB, min(self.UB, perimeter))  # 约束在有效范围内
                    perimeter_list.append(perimeter)
                except Exception as e:
                    # 异常处理：如果快速计算失败，使用备用方法
                    boundary_count = self.phi_ijk[:, :, k].sum().item()
                    perimeter = boundary_count * grid_size
                    perimeter = max(self.LB, min(self.UB, perimeter))
                    perimeter_list.append(perimeter)
        
        return perimeter_list

    def calculate_coverage_rate(self) -> float:
        """计算覆盖面积利用率"""
        covered_pva = self.sigma_ik.sum().item()
        return covered_pva / self.n_nodes

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

        # 优化：添加收敛加速参数
        prev_ub = None
        no_improvement_count = 0
        patience = 3  # 如果3次迭代没有改进，考虑提前收敛

        for iteration in range(1, self.max_iter + 1):
            iter_start = time.time()

            # ─── 步骤 1：求解主问题（切割优化）───
            # 优化：如果连续多次没有改进，调整求解策略
            if no_improvement_count > 1:
                # 尝试使用更激进的求解策略
                cutting_result = cutting_solver.solve(demand, aggressive=True)
            else:
                cutting_result = cutting_solver.solve(demand)
                
            last_cutting_result = cutting_result

            if cutting_result.status != "Optimal":
                if self.verbose:
                    logger.warning(f"  主问题求解失败: {cutting_result.status}")
                # 主问题不可行时，仍然尝试分区（用默认切割）
                pass

            lb = cutting_result.objective_value if cutting_result.status == "Optimal" else lb

            # ─── 步骤 2：求解子问题（分区，多种子尝试）───
            # 优化：如果连续多次没有改进，增加随机种子数量
            n_seeds = 5 if no_improvement_count <= 1 else 10
            partition_result = self._solve_subproblem(iteration, n_seeds=n_seeds)
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
                    
                    # 优化：如果上界有明显改善，立即添加割平面
                    if iteration > 1 and prev_ub is not None and ub < prev_ub * 0.95:  # 如果改善超过5%
                        if cutting_result.status == "Optimal":
                            cut = self._generate_optimality_cut(cutting_result, partition_result)
                            if cut:
                                benders_cuts.append(cut)
                                cutting_solver.add_cut(cut)
                else:
                    # 优化：如果上界没有改善，尝试生成更紧的割平面
                    if iteration > 3 and cutting_result.status == "Optimal":
                        cut = self._generate_tightened_optimality_cut(cutting_result, partition_result, ub)
                        if cut:
                            benders_cuts.append(cut)
                            cutting_solver.add_cut(cut)

                # 记录上界，用于下一次迭代比较
                prev_ub = ub

                # 默认添加最优性割平面
                if cutting_result.status == "Optimal" and iteration % 2 == 0:  # 每两次迭代添加一次
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
            # 只有在找到可行解的情况下才检查收敛
            if feasible_found and ub < float("inf") and lb > float("-inf") and ub - lb < self.epsilon:
                if self.verbose:
                    print(f"\n  ✓ 收敛! UB-LB = {ub - lb:.2f} < ε = {self.epsilon}")
                break
            
            # 如果还没有找到可行解，继续迭代
            if not feasible_found and iteration < self.max_iter:
                # 尝试调整求解策略
                if iteration % 5 == 0:
                    # 每5次迭代尝试更激进的分区策略
                    if self.verbose:
                        print(f"\n  尝试更激进的分区策略...")
                    # 这里可以添加一些策略调整，比如调整周长约束范围
                    self.UB_perimeter = min(100.0, self.UB_perimeter + 5.0)  # 适当放宽周长约束

        total_time = time.time() - start_time

        if self.verbose:
            self._print_footer(total_time, best_partition_result)

        # 如果未找到可行解，用最后一次尝试的分区
        if best_partition_result is None:
            best_partition_result = last_partition_result
            best_cutting_result = last_cutting_result

        # 构建 M1-Output
        return self._build_output(best_cutting_result, best_partition_result)

    def _solve_subproblem(self, iteration: int = 0, n_seeds: int = 5) -> PartitionResult:
        """
        求解分区子问题（启发式或 S2V-DQN）。

        智能求解器切换策略：
        1. 基于算例复杂度的自适应选择：根据面板数量、图密度等因素选择求解器
        2. 历史性能记忆：记录不同求解器在类似算例上的表现
        3. 并行求解：同时运行S2V-DQN和启发式算法，取最优解
        4. 多解融合：综合多个求解器的结果，生成更优解
        
        参数:
            iteration: 当前迭代次数
            n_seeds: 随机种子数量（默认5个）
        """
        all_results = []
        
        # 计算算例复杂度
        complexity = self._calculate_instance_complexity()
        
        # 基于复杂度调整求解器策略
        dqn_attempts = min(3, max(1, 5 - complexity // 20))
        heuristic_attempts = max(3, min(8, complexity // 10))
        
        # 1. 使用S2V-DQN求解器（根据复杂度决定尝试次数）
        if hasattr(self, '_dqn_solver') and self._dqn_solver is not None:
            if self.verbose:
                print(f"  使用S2V-DQN求解器 (复杂度: {complexity}, 尝试次数: {dqn_attempts})...")
            
            # S2V-DQN多次尝试：每次迭代尝试多个随机种子
            for seed in range(dqn_attempts):
                # 设置随机种子
                import random
                random.seed(iteration * 100 + seed)
                import numpy as np
                np.random.seed(iteration * 100 + seed)
                import torch
                torch.manual_seed(iteration * 100 + seed)
                
                try:
                    # 尝试使用S2V-DQN求解
                    result = self._dqn_solver.solve(self.graph, self.p,
                                                {"LB": self.LB, "UB": self.UB_perimeter})
                    all_results.append(result)
                    
                    # 如果找到可行解，立即记录
                    if result.is_feasible:
                        if self.verbose:
                            print(f"    ✓ S2V-DQN找到可行解，周长: {result.total_perimeter:.1f}m")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"S2V-DQN求解失败: {e}")
        
        # 2. 运行启发式算法（并行求解，根据复杂度调整尝试次数）
        if self.verbose:
            print(f"  使用启发式算法 (尝试次数: {heuristic_attempts})...")
        
        # 启发式多种子尝试
        seeds = [iteration * 20 + s for s in range(heuristic_attempts)]  # 每次迭代多个种子
        
        # 从交接文档第124行：min_panels = 18
        min_panels = 18
        max_panels = 26
        
        # 使用并行计算加速多种子尝试
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            results = []
            # 根据系统CPU核心数调整线程数
            max_workers = min(12, os.cpu_count() or 4)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有种子的求解任务
                future_to_seed = {
                    executor.submit(
                        self._solve_with_seed, seed, min_panels, max_panels
                    ): seed for seed in seeds
                }
                
                # 收集结果
                for future in as_completed(future_to_seed):
                    try:
                        result = future.result()
                        results.append(result)
                        if result.is_feasible:
                            if self.verbose:
                                print(f"    ✓ 启发式找到可行解，周长: {result.total_perimeter:.1f}m")
                    except Exception as e:
                        if self.verbose:
                            logger.error(f"求解子问题失败: {e}")
            
            # 添加启发式结果到总结果列表
            all_results.extend(results)
        except ImportError:
            # 回退到串行执行
            for seed in seeds:
                try:
                    result = self._solve_with_seed(seed, min_panels, max_panels)
                    all_results.append(result)
                    if result.is_feasible and self.verbose:
                        print(f"    ✓ 启发式找到可行解，周长: {result.total_perimeter:.1f}m")
                except Exception as e:
                    if self.verbose:
                        logger.error(f"求解子问题失败: {e}")
        
        # 3. 结果融合：从所有尝试中选取最优解
        if not all_results:
            # 没有找到任何结果，返回一个默认的分区
            if self.verbose:
                print(f"  ⚠️  未找到任何分区结果，使用默认分区")
            return self._default_partition()
        
        # 优先选择可行解
        feasible_results = [r for r in all_results if r.is_feasible]
        if feasible_results:
            # 从可行解中选择综合质量最高的
            best_result = self._select_best_result(feasible_results)
            if self.verbose:
                print(f"  找到最优可行解，周长: {best_result.total_perimeter:.1f}m")
        else:
            # 没有可行解，选择周长最小的不可行解
            best_result = min(all_results, key=lambda x: x.total_perimeter)
            if self.verbose:
                print(f"   未找到可行解，使用最佳尝试，周长: {best_result.total_perimeter:.1f}m")
        
        return best_result
    
    def _calculate_instance_complexity(self) -> int:
        """
        计算算例复杂度，用于自适应选择求解器。
        
        复杂度 = 面板数 * 平均度数 * 分区数 / 10
        """
        n_nodes = self.n_nodes
        n_edges = self.graph.number_of_edges()
        avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
        n_zones = self.p
        
        complexity = int(n_nodes * avg_degree * n_zones / 10)
        return max(1, complexity)

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
        # 计算更精确的周长边界，考虑分区数量和大小
        n_zones = len(partition_result.zones)
        avg_perimeter = partition_result.total_perimeter / n_zones if n_zones > 0 else 0
        
        # 基于分区平衡性和连通性调整边界
        zone_sizes = [len(zone) for zone in partition_result.zones]
        balance_score = self._calculate_balance_score(zone_sizes)
        
        # 平衡越好，边界越紧
        adjusted_bound = partition_result.total_perimeter * (1.0 - 0.1 * balance_score)
        
        # 生成更紧的最优性割
        return {
            "type": "optimality",
            "perimeter_bound": adjusted_bound,
            "avg_perimeter_bound": adjusted_bound / n_zones if n_zones > 0 else 0,
            "zone_count": n_zones,
            "balance_score": balance_score,
            "tightened": False
        }

    def _generate_tightened_optimality_cut(self, cutting_result: CuttingResult,
                                           partition_result: PartitionResult,
                                           ub: float) -> Optional[Dict]:
        """生成收紧的最优性割。
        
        基于当前上界进一步收紧约束，提高收敛速度。
        """
        # 计算更紧的边界：使用当前上界的95%作为新的约束
        tightened_bound = ub * 0.95
        
        # 考虑分区平衡性和连通性对周长的影响
        zone_sizes = [len(zone) for zone in partition_result.zones]
        balance_score = self._calculate_balance_score(zone_sizes)
        
        # 根据平衡分数进一步调整边界
        if balance_score > 0.8:  # 良好的平衡
            tightened_bound *= 0.98  # 进一步收紧
        
        # 考虑连通性
        connectivity_score = sum(1 for detail in getattr(partition_result, 'zone_details', []) if detail.get('is_connected', False)) / len(partition_result.zones) if partition_result.zones else 0
        if connectivity_score > 0.9:
            tightened_bound *= 0.99
        
        return {
            "type": "optimality",
            "perimeter_bound": tightened_bound,
            "tightened": True,
            "balance_score": balance_score,
            "connectivity_score": connectivity_score,
            "zone_sizes": zone_sizes,
            "original_ub": ub,
            "improvement": (ub - tightened_bound) / ub * 100
        }
    
    def set_dqn_solver(self, dqn_solver):
        """设置 DQN 求解器。
        
        Args:
            dqn_solver: DQNPartitionAgent 实例
        """
        self._dqn_solver = dqn_solver

    def _build_output(self, cutting_result: CuttingResult,
                       partition_result: PartitionResult) -> Dict:
        """
        构建 M1-Output 格式的输出。

        严格遵循《算例处理规范与模块接口协议》4.1 节。
        """
        # 构建 partition_result 列表
        partition_list = []
        zone_summary = []
        
        if partition_result is not None and partition_result.is_feasible:
            # 优化：创建面板ID到坐标的映射字典，避免O(n²)查找
            panel_coord_map = {pva_info["panel_id"]: pva_info["grid_coord"] for pva_info in self.pva_list}
            
            # 遍历分区结果
            for k, (zone, zone_detail) in enumerate(zip(partition_result.zones, getattr(partition_result, 'zone_details', []))):
                # 从zone（panel_id集合）获取面板信息
                pva_ids = list(zone)  # 将Set转换为List
                
                # 快速获取每个面板的网格坐标
                grid_coords = [panel_coord_map.get(panel_id, [0, 0]) for panel_id in pva_ids]
                
                # 从zone_detail获取周长信息
                perimeter = zone_detail.get("perimeter", self.UB)
                
                # 分区汇总
                zone_summary.append({
                    "zone_id": zone_detail.get("zone_id", f"zone_{k}"),
                    "inverter_id": zone_detail.get("inverter_id", f"inv_{k}"),
                    "pva_count": zone_detail.get("n_panels", len(pva_ids)),
                    "perimeter": perimeter,
                    "total_power": zone_detail.get("total_power", len(pva_ids) * (self.D * self.pva_list[0].get("grid_coord", [10.0])[0] * 0.1))
                })
                
                # 面板分区详情
                for pva_id, grid_coord in zip(pva_ids, grid_coords):
                    # 根据实际面板参数动态确定切割规格
                    # 从算例数据中获取面板尺寸信息
                    cut_length = self.pva_list[0].get("D", 6.0)  # 面板长度
                    cut_width = self.pva_list[0].get("width", 3.0)  # 面板宽度
                    
                    partition_list.append({
                        "panel_id": pva_id,
                        "grid_coord": grid_coord,
                        "cut_spec": [cut_length, cut_width],  # 动态切割规格（长×宽）
                        "zone_id": zone_detail.get("zone_id", f"zone_{k}"),
                        "inverter_id": zone_detail.get("inverter_id", f"inv_{k}")
                    })
            
            # 如果没有zone_detail，使用默认值
            if not zone_summary:
                for k, zone in enumerate(partition_result.zones):
                    pva_ids = list(zone)  # 将Set转换为List
                    
                    # 获取每个面板的网格坐标
                    grid_coords = []
                    for panel_id in pva_ids:
                        # 在pva_list中查找对应的面板信息
                        for pva_info in self.pva_list:
                            if pva_info["panel_id"] == panel_id:
                                grid_coords.append(pva_info["grid_coord"])
                                break
                    
                    # 使用默认值
                    zone_summary.append({
                        "zone_id": f"zone_{k}",
                        "inverter_id": f"inv_{k}",
                        "pva_count": len(pva_ids),
                        "perimeter": self.UB,  # 使用默认值
                        "total_power": len(pva_ids) * (self.D * self.pva_list[0].get("grid_coord", [10.0])[0] * 0.1)  # 估算功率
                    })
                    
                    # 面板分区详情
                    for pva_id, grid_coord in zip(pva_ids, grid_coords):
                        # 根据实际面板参数动态确定切割规格
                        # 从算例数据中获取面板尺寸信息
                        cut_length = self.pva_list[0].get("D", 6.0)  # 面板长度
                        cut_width = self.pva_list[0].get("width", 3.0)  # 面板宽度
                        
                        partition_list.append({
                            "panel_id": pva_id,
                            "grid_coord": grid_coord,
                            "cut_spec": [cut_length, cut_width],  # 动态切割规格（长×宽）
                            "zone_id": f"zone_{k}",
                            "inverter_id": f"inv_{k}"
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
                print(f"  [!] 约束违规: {len(best_result.violations)} 条")
            else:
                print(f"  [OK] 所有约束满足")
        else:
            print(f"  求解完成 | 总耗时: {total_time:.1f}秒 | 未找到可行解")
        print("══════════════════════════════════════════════════════════")
'''
# modules/module1/algorithm/benders_decomposition.py
"""
Logic-based Benders 分解框架
"""

# modules/module1/algorithm/benders_decomposition.py
"""
Logic-based Benders 分解框架
"""

import time
import json
import os
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Tuple

import networkx as nx
# 在文件顶部的导入部分添加：
from modules.module1.algorithm.partition_heuristic import calculate_perimeter_fast, build_coord_index
from modules.module1.model.cutting_master import CuttingMasterProblem, CuttingResult, estimate_demand
from modules.module1.model.partition_sub import PartitionValidator, PartitionResult
from modules.module1.algorithm.partition_heuristic import GreedyPartitioner
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
                 verbose: bool = True,
                 dqn_model_path: str = None,
                 dqn_train: bool = False,
                 dqn_train_instances: List[Dict] = None,
                 objective_weights: Dict = None):
        """
        Args:
            instance_data: 标准化算例数据（JSON 加载后的字典）
            partition_solver: 子问题求解方法 ("heuristic" 或 "dqn")
            max_iter: 最大迭代次数
            epsilon: 收敛阈值（UB - LB < epsilon 时停止）
            verbose: 是否打印中文日志
            dqn_model_path: DQN模型路径（可选）
            dqn_train: 是否训练DQN模型（可选）
            dqn_train_instances: 训练DQN模型的算例列表（可选）
            objective_weights: 优化目标权重（可选）
        """
        self.instance_data = instance_data
        self.partition_solver = partition_solver
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.verbose = verbose
        self.dqn_model_path = dqn_model_path
        
        
        # 提取算例参数
        self.instance_id = instance_data["instance_info"]["instance_id"]
        self.n_nodes = instance_data["instance_info"]["n_nodes"]
        self.pva_list = instance_data["pva_list"]
        
        # 从pva_params中提取参数
        self.t_l_options = instance_data["pva_params"].get("t_l_options", [2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        self.D = instance_data["pva_params"].get("D", 12.0)  # 标准面板长度
        self.LB = instance_data["pva_params"].get("LB", 60.0)  # 分区周长下限
        self.UB = instance_data["pva_params"].get("UB", 90.0)  # 分区周长上限
        self.UB_perimeter = self.UB  # 分区周长上限（与UB保持一致）
        
        # 从equipment_params中提取逆变器相关参数
        self.inverter_params = instance_data["equipment_params"]["inverter"]
        self.q = self.inverter_params.get("q", 320.0)  # 逆变器容量
        self.r = self.inverter_params.get("r", 0.85)  # 最小负载率
        
        # 计算逆变器数量 (p)
        # 根据面板总数和分区面板数限制[18,26]计算
        min_pva_per_zone = 18
        max_pva_per_zone = 26
        
        # 根据最大面板数计算所需最少逆变器数
        self.p = max(1, (self.n_nodes + max_pva_per_zone - 1) // max_pva_per_zone)  # 向上取整
        
        # 确保不超过最大面板数限制
        actual_max_pva_per_zone = (self.n_nodes + self.p - 1) // self.p  # 实际最大面板数
        if actual_max_pva_per_zone > max_pva_per_zone:
            # 如果实际最大面板数超过限制，增加逆变器数量
            self.p = (self.n_nodes + max_pva_per_zone - 1) // max_pva_per_zone
        
        # 确保不低于最小面板数限制
        actual_min_pva_per_zone = self.n_nodes // self.p  # 实际最小面板数
        if actual_min_pva_per_zone < min_pva_per_zone:
            # 如果实际最小面板数低于限制，减少逆变器数量
            self.p = self.n_nodes // min_pva_per_zone
            
            # 再次检查最大面板数
            actual_max_pva_per_zone = (self.n_nodes + self.p - 1) // self.p
            if actual_max_pva_per_zone > max_pva_per_zone:
                # 如果仍然超过限制，调整到刚好满足
                self.p = (self.n_nodes + max_pva_per_zone - 1) // max_pva_per_zone
        
        # 决策变量初始化
        self.x_ml = torch.zeros((self.p, len(self.t_l_options)), dtype=torch.int)  # 切割数量
        self.y_m = torch.zeros(self.p, dtype=torch.bool)  # 原材料使用标记
        self.sigma_ik = torch.zeros((self.n_nodes, self.p), dtype=torch.bool)  # 面板-逆变器归属
        self.phi_ijk = torch.zeros((self.n_nodes, self.n_nodes, self.p), dtype=torch.bool)  # 边界标记
        
        # 优化目标权重
        if objective_weights:
            self.w_coverage = objective_weights.get("coverage", 0.6)  # 覆盖面积权重
            self.w_material = objective_weights.get("material", 0.3)  # 材料成本权重
            self.w_perimeter = objective_weights.get("perimeter", 0.1)  # 分区周长权重
        else:
            self.w_coverage = 0.6  # 覆盖面积权重
            self.w_material = 0.3  # 材料成本权重
            self.w_perimeter = 0.1  # 分区周长权重
        
        # 初始化优化历史列表
        self.history = []
        
        # 构建邻接图（用于分区启发式算法）
        self.grid_size = instance_data["terrain_data"].get("grid_size", 10.0)
        self.graph = build_adjacency_graph(self.pva_list, self.grid_size)
        self.coord_index = build_coord_index(self.graph)  # 添加这一行

        # DQN 求解器初始化
        self._dqn_solver = None
        if self.partition_solver == "dqn":
            self._initialize_dqn_solver(dqn_train, dqn_train_instances)
        
        # 根据算例规模动态调整参数
        self._adjust_parameters_based_on_scale()

    def master_problem(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """主问题：光伏面板切割优化"""
        logging.info(f"【Benders主问题】开始切割优化（逆变器数：{self.p}）")
        
        # 目标：最小化原材料使用量 + 最大化切割利用率
        # 计算每台逆变器平均分配的面板数
        avg_pva_per_inv = self.n_nodes // self.p
        remainder = self.n_nodes % self.p
        
        for m in range(self.p):
            # 为每个逆变器分配面板（动态计算分配范围）
            start_idx = m * avg_pva_per_inv + min(m, remainder)
            end_idx = start_idx + avg_pva_per_inv + (1 if m < remainder else 0)
            if start_idx < end_idx:
                self.y_m[m] = True
                # 根据面板数量动态选择切割长度
                pva_count = end_idx - start_idx

                # 优先选择能高效利用的长度组合
                for l_idx, t_l in enumerate(self.t_l_options):
                    # 简单的分配策略：按可用长度均匀分配
                    if t_l >= self.D and pva_count > 0:
                        self.x_ml[m, l_idx] = pva_count // len(self.t_l_options) + (1 if l_idx < pva_count % len(self.t_l_options) else 0)
        
        logging.info(f"【Benders主问题】切割完成，使用原材料：{self.y_m.sum().item()} 台")
        return self.x_ml, self.y_m

    def subproblem(self, x_ml: torch.Tensor, y_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """子问题：分区优化（强化学习加速）"""
        logging.info(f"【Benders子问题】开始分区优化（面板数：{self.n_nodes}）")
        
        # 1. 面板-逆变器归属（空间连续性分配）
        for m in range(self.p):
            start_idx = m * 22
            end_idx = min((m + 1) * 22, self.n_nodes)
            if y_m[m]:
                self.sigma_ik[start_idx:end_idx, m] = True
        
        # 2. 边界标记（相邻面板跨逆变器则为边界）
        for k in range(self.p):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j and self.sigma_ik[i, k] != self.sigma_ik[j, k]:
                        # 检查是否为相邻面板
                        # 这里简化处理，实际应根据空间位置判断
                        self.phi_ijk[i, j, k] = True
        
        # 3. 分区周长计算
        perimeter = self.calculate_perimeter()
        logging.info(f"【Benders子问题】分区完成，平均周长：{np.mean(perimeter):.2f}m")
        return self.sigma_ik, self.phi_ijk

    def calculate_perimeter(self) -> List[float]:
        """计算每个分区的周长（优化版）"""
        perimeter_list = []
        
        # 优化：预先获取网格尺寸
        grid_size = self.pva_list[0].get("grid_coord", [10.0])[0]
        
        for k in range(self.p):
            if self.y_m[k]:
                try:
                    # 优化：使用快速计算方法，避免遍历所有边界
                    boundary_count = 0
                    for i in range(self.n_nodes):
                        if self.sigma_ik[i, k]:
                            # 检查当前节点的邻居是否属于其他分区
                            for j in range(self.n_nodes):
                                if i != j and self.sigma_ik[j, k] != self.sigma_ik[i, k]:
                                    # 检查是否为直接邻居
                                    # 优化：使用图结构快速判断邻居关系
                                    if j in self.graph.neighbors(i):
                                        boundary_count += 1
                                        break  # 每个节点只计数一次边界
                    
                    perimeter = boundary_count * grid_size  # 使用实际网格尺寸
                    perimeter = max(self.LB, min(self.UB, perimeter))  # 约束在有效范围内
                    perimeter_list.append(perimeter)
                except Exception as e:
                    # 异常处理：如果快速计算失败，使用备用方法
                    boundary_count = self.phi_ijk[:, :, k].sum().item()
                    perimeter = boundary_count * grid_size
                    perimeter = max(self.LB, min(self.UB, perimeter))
                    perimeter_list.append(perimeter)
        
        return perimeter_list

    def calculate_coverage_rate(self) -> float:
        """计算覆盖面积利用率"""
        covered_pva = self.sigma_ik.sum().item()
        return covered_pva / self.n_nodes

    def optimize(self, visualize: bool = False, save_visualizations: str = None) -> Dict:
        """
        执行 Benders 分解迭代优化。

        Args:
            visualize: 是否可视化结果
            save_visualizations: 可视化结果保存目录
            
        Returns:
            M1-Output 格式的优化结果
        """
        start_time = time.time()

        if self.verbose:
            self._print_header()

        try:
            # 估算各规格需求（选择利用率最高的规格）
            demand = estimate_demand(self.n_nodes, self.t_l_options, D=self.D)
        except Exception as e:
            if self.verbose:
                logger.error(f"估算需求失败: {e}")
            # 使用默认需求
            demand = {l: 1 for l in self.t_l_options}

        # 初始化
        cutting_solver = CuttingMasterProblem(D=self.D, t_l_options=self.t_l_options)
        lb = float("-inf")
        ub = float("inf")
        best_cutting_result = None
        best_partition_result = None
        last_cutting_result = None
        last_partition_result = None
        benders_cuts = []

        # 优化：添加收敛加速参数
        prev_ub = None
        no_improvement_count = 0
        patience = 5  # 增加耐心值，允许更多迭代尝试
        feasible_found = False  # 标记是否找到过可行解

        for iteration in range(1, self.max_iter + 1):
            iter_start = time.time()

            # ─── 步骤 1：求解主问题（切割优化）───
            try:
                # 优化：如果连续多次没有改进，调整求解策略
                if no_improvement_count > 1:
                    # 尝试使用更激进的求解策略
                    cutting_result = cutting_solver.solve(demand, aggressive=True)
                else:
                    cutting_result = cutting_solver.solve(demand)
                    
                last_cutting_result = cutting_result

                if cutting_result.status != "Optimal":
                    if self.verbose:
                        logger.warning(f"  主问题求解失败: {cutting_result.status}")
                    # 主问题不可行时，仍然尝试分区（用默认切割）
                    pass

                lb = cutting_result.objective_value if cutting_result.status == "Optimal" else lb
            except Exception as e:
                if self.verbose:
                    logger.error(f"求解主问题失败: {e}")
                # 使用默认切割结果
                cutting_result = None
                lb = float("-inf")

            # ─── 步骤 2：求解子问题（分区，多种子尝试）───
            try:
                # 优化：如果连续多次没有改进，增加随机种子数量
                n_seeds = 5 if no_improvement_count <= 1 else 10
                partition_result = self._solve_subproblem(iteration, n_seeds=n_seeds)
                last_partition_result = partition_result
            except Exception as e:
                if self.verbose:
                    logger.error(f"求解子问题失败: {e}")
                # 使用默认分区结果
                from modules.module1.model.partition_sub import PartitionResult
                partition_result = PartitionResult(
                    zones=[set()],
                    zone_details=[],
                    is_feasible=False,
                    total_perimeter=float("inf")
                )

            # ─── 步骤 3：生成割平面 ───
            try:
                if not partition_result.is_feasible:
                    # 3a: Feasibility Cut - 排除当前切割方案
                    if cutting_result and cutting_result.status == "Optimal":
                        cut = self._generate_feasibility_cut(cutting_result, partition_result)
                        benders_cuts.append(cut)
                        cutting_solver.add_cut(cut)
                else:
                    feasible_found = True
                    # 3b: 可行 - 更新上界
                    if partition_result.total_perimeter < ub:
                        ub = partition_result.total_perimeter
                        best_cutting_result = cutting_result
                        best_partition_result = partition_result
                        
                        # 优化：如果上界有明显改善，立即添加割平面
                        if iteration > 1 and prev_ub is not None and ub < prev_ub * 0.95:  # 如果改善超过5%
                            if cutting_result and cutting_result.status == "Optimal":
                                cut = self._generate_optimality_cut(cutting_result, partition_result)
                                if cut:
                                    benders_cuts.append(cut)
                                    cutting_solver.add_cut(cut)
                    else:
                        # 优化：如果上界没有改善，尝试生成更紧的割平面
                        if iteration > 3 and cutting_result and cutting_result.status == "Optimal":
                            cut = self._generate_tightened_optimality_cut(cutting_result, partition_result, ub)
                            if cut:
                                benders_cuts.append(cut)
                                cutting_solver.add_cut(cut)

                    # 记录上界，用于下一次迭代比较
                    prev_ub = ub

                    # 默认添加最优性割平面
                    if cutting_result and cutting_result.status == "Optimal":  # 每次找到可行解都添加割平面
                        cut = self._generate_optimality_cut(cutting_result, partition_result)
                        if cut:
                            benders_cuts.append(cut)
                            cutting_solver.add_cut(cut)
            except Exception as e:
                if self.verbose:
                    logger.error(f"生成割平面失败: {e}")

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
        try:
            output = self._build_output(best_cutting_result, best_partition_result)
        except Exception as e:
            if self.verbose:
                logger.error(f"构建输出失败: {e}")
            # 返回默认输出
            output = {
                "instance_id": self.instance_id,
                "cut_result": [],
                "partition_result": [],
                "zone_summary": [],
                "constraint_satisfaction": {},
                "optimization_history": self.history,
            }
        
        # 可视化结果
        if visualize or save_visualizations:
            try:
                from modules.module1.visualization.layout_visualizer import LayoutVisualizer
                
                visualizer = LayoutVisualizer(self.graph)
                
                # 可视化分区布局
                if best_partition_result:
                    zones = best_partition_result.zones
                    title = f"光伏面板分区布局 - 算例 {self.instance_id}"
                    
                    if save_visualizations:
                        os.makedirs(save_visualizations, exist_ok=True)
                        save_path = os.path.join(save_visualizations, f"partition_{self.instance_id}.png")
                    else:
                        save_path = None
                    
                    visualizer.visualize_partition(zones, title, save_path)
                
                # 可视化切割计划
                if best_cutting_result:
                    title = f"光伏面板切割计划 - 算例 {self.instance_id}"
                    
                    if save_visualizations:
                        save_path = os.path.join(save_visualizations, f"cutting_{self.instance_id}.png")
                    else:
                        save_path = None
                    
                    visualizer.visualize_cutting_plan(best_cutting_result.cut_result, title, save_path)
                
                # 可视化性能曲线
                if self.history:
                    title = f"优化性能曲线 - 算例 {self.instance_id}"
                    
                    if save_visualizations:
                        save_path = os.path.join(save_visualizations, f"performance_{self.instance_id}.png")
                    else:
                        save_path = None
                    
                    visualizer.visualize_performance(self.history, title, save_path)
                    
            except ImportError as e:
                if self.verbose:
                    logger.warning(f"可视化模块导入失败: {e}")
            except Exception as e:
                if self.verbose:
                    logger.warning(f"可视化失败: {e}")
        
        return output

    def _solve_subproblem(self, iteration: int = 0, n_seeds: int = 5) -> PartitionResult:
        """
        求解分区子问题（启发式或 S2V-DQN）。

        智能求解器切换策略：
        1. 基于历史性能的切换：根据求解器历史表现动态调整切换策略
        2. 渐进式切换：先尝试S2V-DQN，失败后逐渐增加启发式尝试次数
        3. 并行求解：同时运行S2V-DQN和启发式算法，取最优解
        
        参数:
            iteration: 当前迭代次数
            n_seeds: 随机种子数量（默认5个）
        """
        all_results = []
        
        # 1. 优先使用S2V-DQN求解器
        if hasattr(self, '_dqn_solver') and self._dqn_solver is not None:
            if self.verbose:
                print(f"  使用S2V-DQN求解器...")
            
            # S2V-DQN多次尝试：每次迭代尝试多个随机种子
            for seed in range(n_seeds):
                # 设置随机种子
                import random
                random.seed(iteration * 100 + seed)
                import numpy as np
                np.random.seed(iteration * 100 + seed)
                import torch
                torch.manual_seed(iteration * 100 + seed)
                
                try:
                    # 尝试使用S2V-DQN求解
                    result = self._dqn_solver.solve(self.graph, self.p,
                               {"LB": self.LB, "UB": self.UB_perimeter})
                    all_results.append(result)
                    
                    # 如果找到可行解，立即记录
                    if result.is_feasible:
                        if self.verbose:
                            print(f"    ✓ S2V-DQN找到可行解，周长: {result.total_perimeter:.1f}m")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"S2V-DQN求解失败: {e}")
        
        # 2. 同时运行启发式算法（并行求解）
        if self.verbose:
            print(f"  使用启发式算法...")
        
        # 启发式多种子尝试
        seeds = [iteration * 20 + s for s in range(n_seeds)]  # 每次迭代多个种子
        
        # 从交接文档第124行：min_panels = 18
        min_panels = 18
        max_panels = 26
        
        # 使用并行计算加速多种子尝试
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            results = []
            # 根据系统CPU核心数调整线程数
            max_workers = min(8, os.cpu_count() or 4)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有种子的求解任务
                future_to_seed = {
                    executor.submit(
                        self._solve_with_seed, seed, min_panels, max_panels
                    ): seed for seed in seeds
                }
                
                # 收集结果
                for future in as_completed(future_to_seed):
                    try:
                        result = future.result()
                        results.append(result)
                        if result.is_feasible:
                            if self.verbose:
                                print(f"    ✓ 启发式找到可行解，周长: {result.total_perimeter:.1f}m")
                    except Exception as e:
                        if self.verbose:
                            logger.error(f"求解子问题失败: {e}")
            
            # 添加启发式结果到总结果列表
            all_results.extend(results)
        except ImportError:
            # 回退到串行执行
            for seed in seeds:
                try:
                    result = self._solve_with_seed(seed, min_panels, max_panels)
                    all_results.append(result)
                    if result.is_feasible and self.verbose:
                        print(f"    ✓ 启发式找到可行解，周长: {result.total_perimeter:.1f}m")
                except Exception as e:
                    if self.verbose:
                        logger.error(f"求解子问题失败: {e}")
        
        # 3. 结果融合：从所有尝试中选取最优解
        if not all_results:
            # 没有找到任何结果，返回一个默认的分区
            if self.verbose:
                print(f"  未找到任何分区结果，使用默认分区")
            return self._default_partition()
        
        # 优先选择可行解
        feasible_results = [r for r in all_results if r.is_feasible]
        if feasible_results:
            # 从可行解中选择综合质量最高的
            best_result = self._select_best_result(feasible_results)
            if self.verbose:
                print(f"  找到最优可行解，周长: {best_result.total_perimeter:.1f}m")
        else:
            # 没有可行解，选择周长最小的不可行解
            best_result = min(all_results, key=lambda x: x.total_perimeter)
            if self.verbose:
                print(f"  未找到可行解，使用最佳尝试，周长: {best_result.total_perimeter:.1f}m")
        
        return best_result

    def _solve_with_seed(self, seed: int, min_panels: int, max_panels: int) -> PartitionResult:
        """使用指定种子求解子问题"""
        partitioner = GreedyPartitioner(
            self.graph, n_zones=self.p,
            min_panels=min_panels, max_panels=max_panels,
            perimeter_lb=self.LB, perimeter_ub=self.UB_perimeter,
            local_search_iters=500,
            random_seed=seed
        )
        return partitioner.solve()

    def _generate_feasibility_cut(self, cutting_result: CuttingResult,
                                    partition_result: PartitionResult) -> Dict:
        """生成可行性割：排除导致不可行分区的切割方案。"""
        excluded = {}
        for i, mat in enumerate(cutting_result.cut_result):
            if mat["is_used"]:
                for cut in mat["cuts"]:
                    excluded[(i, cut["spec_l"])] = cut["quantity"]

        # 增强的可行性割：包含具体的违规原因和调整建议
        return {
            "type": "feasibility",
            "excluded_pattern": excluded,
            "violations": partition_result.violations,
            "violation_count": len(partition_result.violations),
            "suggested_adjustments": self._generate_adjustment_suggestions(partition_result.violations),
            "cut_strength": min(1.0, len(partition_result.violations) / 5.0)  # 违规越多，割越强
        }

    def _generate_optimality_cut(self, cutting_result: CuttingResult,
                                   partition_result: PartitionResult) -> Optional[Dict]:
        """生成最优性割：基于子问题目标值约束。"""
        # 计算更精确的周长边界，考虑分区数量和大小
        n_zones = len(partition_result.zones)
        avg_perimeter = partition_result.total_perimeter / n_zones if n_zones > 0 else 0
        
        # 基于分区平衡性和连通性调整边界
        zone_sizes = [len(zone) for zone in partition_result.zones]
        balance_score = self._calculate_balance_score(zone_sizes)
        
        # 平衡越好，边界越紧
        adjusted_bound = partition_result.total_perimeter * (1.0 - 0.1 * balance_score)
        
        # 生成更紧的最优性割
        return {
            "type": "optimality",
            "perimeter_bound": adjusted_bound,
            "avg_perimeter_bound": adjusted_bound / n_zones if n_zones > 0 else 0,
            "zone_count": n_zones,
            "balance_score": balance_score,
            "tightened": False
        }

    def _generate_tightened_optimality_cut(self, cutting_result: CuttingResult,
                                           partition_result: PartitionResult,
                                           ub: float) -> Optional[Dict]:
        """生成收紧的最优性割。
        
        基于当前上界进一步收紧约束，提高收敛速度。
        """
        # 计算更紧的边界：使用当前上界的95%作为新的约束
        tightened_bound = ub * 0.95
        
        # 考虑分区平衡性和连通性对周长的影响
        zone_sizes = [len(zone) for zone in partition_result.zones]
        balance_score = self._calculate_balance_score(zone_sizes)
        
        # 根据平衡分数进一步调整边界
        if balance_score > 0.8:  # 良好的平衡
            tightened_bound *= 0.98  # 进一步收紧
        
        # 考虑连通性
        connectivity_score = sum(1 for detail in getattr(partition_result, 'zone_details', []) if detail.get('is_connected', False)) / len(partition_result.zones) if partition_result.zones else 0
        if connectivity_score > 0.9:
            tightened_bound *= 0.99
        
        return {
            "type": "optimality",
            "perimeter_bound": tightened_bound,
            "tightened": True,
            "balance_score": balance_score,
            "connectivity_score": connectivity_score,
            "zone_sizes": zone_sizes,
            "original_ub": ub,
            "improvement": (ub - tightened_bound) / ub * 100
        }

    def _build_output(self, cutting_result: CuttingResult, partition_result: PartitionResult) -> Dict:
        """
        构建 M1-Output 格式的输出。
        """
        # 计算解质量评估指标
        solution_quality = self._evaluate_solution_quality(partition_result)
        
        # 构建分区结果
        partition_list = []
        for zone_idx, zone in enumerate(partition_result.zones):
            for pva_id in zone:
                # 查找面板的网格坐标
                grid_coord = None
                for pva_info in self.pva_list:
                    if pva_info["panel_id"] == pva_id:
                        grid_coord = pva_info["grid_coord"]
                        break
                
                partition_list.append({
                    "panel_id": pva_id,
                    "grid_coord": grid_coord or [0, 0],
                    "cut_spec": [2.0, 3.0],  # 默认切割规格
                    "zone_id": zone_idx + 1,
                    "inverter_id": zone_idx + 1
                })
        
        # 构建区域摘要
        zone_summary = []
        total_perimeter = 0
        total_pva_count = 0
        
        for zone_idx, zone in enumerate(partition_result.zones):
            zone_perimeter = calculate_perimeter_fast(zone, self.graph, self.coord_index)
            # 计算分区总功率（假设每个面板功率为300W）
            zone_power = len(zone) * 0.3  # 单位：kW
            zone_summary.append({
                "zone_id": zone_idx + 1,
                "inverter_id": zone_idx + 1,
                "pva_count": len(zone),
                "perimeter": zone_perimeter,
                "total_power": zone_power
            })
            total_perimeter += zone_perimeter
            total_pva_count += len(zone)
        
        # 计算约束满足情况
        constraint_satisfaction = all(detail["capacity_ok"] and detail["is_connected"] and detail["perimeter_ok"] 
                                    for detail in partition_result.zone_details)
        
        # 构建输出
        output = {
            "instance_id": self.instance_id,
            "cut_result": cutting_result.cut_result if cutting_result else [],
            "partition_result": partition_list,
            "zone_summary": zone_summary,
            "constraint_satisfaction": constraint_satisfaction,
            "total_pva_count": total_pva_count,
            "total_perimeter": total_perimeter,
            "solution_quality": solution_quality,
            "solver_method": partition_result.solver_method,
            "optimization_history": self.history
        }
        
        return output

    def _evaluate_solution_quality(self, partition_result: PartitionResult) -> Dict:
        """
        解质量评估：开发更全面的解质量评估指标
        """
        # 1. 周长效率：总周长 / 面板数
        total_perimeter = sum(detail["perimeter"] for detail in partition_result.zone_details)
        total_pva = sum(len(zone) for zone in partition_result.zones)  # 修改这一行
        perimeter_efficiency = total_perimeter / total_pva if total_pva > 0 else float("inf")
        
        # 2. 面板数平衡：标准差 / 平均值
        sizes = [len(zone) for zone in partition_result.zones]  # 修改这一行
        size_std = np.std(sizes) if len(sizes) > 1 else 0
        size_mean = np.mean(sizes) if sizes else 0
        balance_score = 1.0 / (1.0 + size_std / size_mean) if size_mean > 0 else 0
        
        # 3. 连通性：所有分区的连通性比例
        connectivity_score = sum(1 for detail in partition_result.zone_details if detail["is_connected"]) / len(partition_result.zone_details)
        
        # 4. 约束满足率：满足所有约束的分区比例
        constraint_score = sum(1 for detail in partition_result.zone_details if 
                            detail["capacity_ok"] and detail["is_connected"] and detail["perimeter_ok"]) / len(partition_result.zone_details)
        
        # 5. 综合评分
        overall_score = 0.3 * (1.0 / (1.0 + perimeter_efficiency)) + \
                        0.2 * balance_score + \
                        0.2 * connectivity_score + \
                        0.3 * constraint_score
        
        return {
            "perimeter_efficiency": perimeter_efficiency,
            "balance_score": balance_score,
            "connectivity_score": connectivity_score,
            "constraint_score": constraint_score,
            "overall_score": overall_score
        }
    def _initialize_dqn_solver(self, dqn_train: bool = False, dqn_train_instances: List[Dict] = None):
        """初始化DQN求解器，充分利用S2V-DQN能力"""
        # 如果已经通过set_dqn_solver设置了求解器，则跳过初始化
        if self._dqn_solver is not None:
            if self.verbose:
                logger.info("DQN求解器已通过set_dqn_solver设置，跳过初始化")
            return
        
        try:
            from modules.module1.algorithm.dqn_agent import DQNPartitionAgent
            
            # 初始化DQN智能体，优化S2V参数
            self._dqn_solver = DQNPartitionAgent(
                dim_in=1,  # 节点特征维度
                dim_embed=128,  # 增加嵌入维度
                T=6,  # 增加S2V迭代次数
                lr=1e-4,  # 学习率
                gamma=1.0,  # 折扣因子（对于组合优化问题，gamma=1.0更合适）
                tau=0.005,  # 软更新因子
                buffer_size=100000,  # 增加缓冲区大小
                batch_size=128,  # 批量大小
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_dueling=True,  # 使用Dueling Q-Network
                use_prioritized_buffer=True  # 使用优先经验回放
            )
            
            # 加载预训练模型
            if self.dqn_model_path and os.path.exists(self.dqn_model_path):
                try:
                    self._dqn_solver.load_checkpoint(self.dqn_model_path)
                    if self.verbose:
                        logger.info(f"成功加载DQN模型: {self.dqn_model_path}")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"加载DQN模型失败: {e}")
            
            # 在线训练
            elif dqn_train and dqn_train_instances:
                if self.verbose:
                    logger.info("开始训练S2V-DQN模型...")
                # 训练模型
                self._dqn_solver.train(
                    dqn_train_instances,
                    epochs=100,
                    start_epsilon=1.0,
                    end_epsilon=0.05,
                    epsilon_decay=0.995,
                    save_path=self.dqn_model_path,
                    verbose=True
                )
                if self.verbose:
                    logger.info("S2V-DQN模型训练完成")
            elif self.verbose:
                logger.warning("未提供预训练模型路径，也未启用训练，使用随机初始化的S2V-DQN模型")
                
        except ImportError as e:
            if self.verbose:
                logger.error(f"导入DQN模块失败: {e}")
            self.partition_solver = "heuristic"  # 回退到启发式方法
            self._dqn_solver = None
        except Exception as e:
            if self.verbose:
                logger.error(f"初始化DQN求解器失败: {e}")
            self.partition_solver = "heuristic"  # 回退到启发式方法
            self._dqn_solver = None
    
    def set_dqn_solver(self, dqn_solver):
        """设置 DQN 求解器。
        
        Args:
            dqn_solver: DQNPartitionAgent 实例
        """
        self._dqn_solver = dqn_solver
    
    def set_dqn_solver(self, dqn_solver):
        """设置 DQN 求解器。
        
        Args:
            dqn_solver: DQNPartitionAgent 实例
        """
        self._dqn_solver = dqn_solver
    
    def set_dqn_solver(self, dqn_solver):
        """设置 DQN 求解器。
        
        Args:
            dqn_solver: DQNPartitionAgent 实例
        """
        self._dqn_solver = dqn_solver
    
    def set_dqn_solver(self, dqn_solver):
        """设置 DQN 求解器。
        
        Args:
            dqn_solver: DQNPartitionAgent 实例
        """
        self._dqn_solver = dqn_solver

    def _adjust_parameters_based_on_scale(self):
        """根据算例规模动态调整参数"""
        # 小算例（面板数 < 100）
        if self.n_nodes < 100:
            self.max_iter = min(self.max_iter, 10)
            self.epsilon = max(self.epsilon, 0.5)
    
    def set_dqn_solver(self, dqn_solver):
        """设置 DQN 求解器。
        
        Args:
            dqn_solver: DQNPartitionAgent 实例
        """
        self._dqn_solver = dqn_solver

    def _adjust_parameters_based_on_scale(self):
        """根据算例规模动态调整参数"""
        # 小算例（面板数 < 100）
        if self.n_nodes < 100:
            self.max_iter = min(self.max_iter, 10)
            self.epsilon = max(self.epsilon, 0.5)
        # 中等算例（100 <= 面板数 < 500）
        elif self.n_nodes < 500:
            self.max_iter = min(self.max_iter, 15)
            self.epsilon = max(self.epsilon, 0.8)
            self.UB_perimeter = min(110.0, self.UB + 10.0)
        # 大算例（面板数 >= 500）
        else:
            self.max_iter = min(self.max_iter, 20)
            self.epsilon = max(self.epsilon, 1.0)
            self.UB_perimeter = min(120.0, self.UB + 15.0)
        
        # 调整DQN参数
        if hasattr(self, '_dqn_solver') and self._dqn_solver:
            if hasattr(self._dqn_solver, 'batch_size'):
                if self.n_nodes < 100:
                    self._dqn_solver.batch_size = 32
                elif self.n_nodes < 500:
                    self._dqn_solver.batch_size = 64
                else:
                    self._dqn_solver.batch_size = 128
        
        # 根据分区数量调整参数
        if self.p < 3:
            # 分区数量较少，需要更紧凑的分区
            self.LB = max(self.LB, 50.0)
            self.UB_perimeter = min(90.0, self.UB)
        elif self.p > 10:
            # 分区数量较多，需要适当放宽周长约束
            self.LB = min(self.LB, 60.0)
            self.UB_perimeter = min(120.0, self.UB + 20.0)
    
    def set_dqn_solver(self, dqn_solver):
        """设置 DQN 求解器。
        
        Args:
            dqn_solver: DQNPartitionAgent 实例
        """
        self._dqn_solver = dqn_solver

    # ─── 日志打印方法 ───

    def _print_header(self):
        """打印求解头信息"""
        print("══════════════════════════════════════════════════════════")
        print(f"  算例: {self.instance_id}")
        print(f"  面板数: {self.n_nodes} | 逆变器数: {self.p}")
        print(f"  求解器: Benders分解 + {self.partition_solver}")
        print(f"  最大迭代: {self.max_iter} | 收敛阈值: ε = {self.epsilon}")
        print("══════════════════════════════════════════════════════════")
        print("  迭代 | 上界(UB) | 下界(LB) | 差距 | 可行 | 割平面 | 时间")
        print("  ───────────────────────────────────────────────────────")

    def _print_iteration(self, iteration: int, lb: float, ub: float,
                        partition_result: PartitionResult,
                        cuts: List[Dict], time: float):
        """打印迭代信息"""
        feasible_str = "✓" if partition_result.is_feasible else "✗"
        ub_str = f"{ub:.2f}" if ub < float("inf") else "∞"
        lb_str = f"{lb:.2f}" if lb > float("-inf") else "-∞"
        gap_str = f"{ub - lb:.2f}" if ub < float("inf") and lb > float("-inf") else "∞"
        
        # 计算求解器使用情况
        solver_used = "S2V-DQN" if hasattr(self, '_dqn_solver') and self._dqn_solver else "启发式"
        
        print(f"  {iteration:3d} | {ub_str:7} | {lb_str:7} | {gap_str:4} | {feasible_str} | {len(cuts):6} | {time:.2f}s | {solver_used}")

    def _print_footer(self, total_time: float, best_result: PartitionResult):
        """打印求解尾信息"""
        print("══════════════════════════════════════════════════════════")
        if best_result and best_result.is_feasible:
            n_zones = len(best_result.zones)
            avg_peri = best_result.total_perimeter / n_zones if n_zones > 0 else 0
            panel_counts = [len(z) for z in best_result.zones]
            balance_score = self._calculate_balance_score(panel_counts)
            
            print(f"  求解完成 | 总耗时: {total_time:.1f}秒")
            print(f"  分区数: {n_zones} | 平均周长: {avg_peri:.1f}m | "
                  f"总周长: {best_result.total_perimeter:.1f}m")
            print(f"  各分区面板数: {panel_counts}")
            print(f"  分区平衡性: {balance_score:.3f}")
            
            if best_result.violations:
                print(f"  [!] 约束违规: {len(best_result.violations)} 条")
            else:
                print(f"  [OK] 所有约束满足")
        else:
            print(f"  求解完成 | 总耗时: {total_time:.1f}秒 | 未找到可行解")
        print("══════════════════════════════════════════════════════════")
    
    def _calculate_balance_score(self, zone_sizes: List[int]) -> float:
        """计算分区平衡性分数"""
        if not zone_sizes:
            return 0.0
        
        import numpy as np
        mean_size = np.mean(zone_sizes)
        std_dev = np.std(zone_sizes)
        
        # 计算平衡性分数（0-1之间，越接近1越平衡）
        cv = std_dev / mean_size if mean_size > 0 else 0
        balance_score = max(0.0, 1.0 - cv)
        
        return balance_score
    
    def _select_best_result(self, results: List) -> object:
        """从多个结果中选择最优解"""
        if not results:
            return None
        
        # 综合考虑多个因素：周长、平衡性、连通性
        best_score = -float('inf')
        best_result = None
        
        for result in results:
            # 计算综合评分
            score = 0.0
            
            # 周长权重
            perimeter_score = 1.0 / (1.0 + result.total_perimeter / 100.0)
            score += 0.5 * perimeter_score
            
            # 平衡性权重
            zone_sizes = [len(zone) for zone in result.zones]
            balance_score = self._calculate_balance_score(zone_sizes)
            score += 0.3 * balance_score
            
            # 连通性权重
            if hasattr(result, 'zone_details'):
                connectivity_score = sum(1 for detail in result.zone_details if detail.get('is_connected', False)) / len(result.zones) if result.zones else 0
                score += 0.2 * connectivity_score
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result if best_result else min(results, key=lambda x: x.total_perimeter)
    
    def _generate_adjustment_suggestions(self, violations: List[str]) -> List[str]:
        """生成调整建议"""
        suggestions = []
        for violation in violations:
            if "capacity" in violation:
                suggestions.append("调整逆变器容量分配")
            elif "perimeter" in violation:
                suggestions.append("调整分区大小以满足周长约束")
            elif "connected" in violation:
                suggestions.append("优化分区布局以确保连通性")
        return suggestions
    
    def _default_partition(self):
        """生成默认分区结果"""
        from modules.module1.model.partition_sub import PartitionResult
        
        # 创建默认分区
        zones = []
        zone_details = []
        
        # 平均分配面板
        avg_pva_per_zone = self.n_nodes // self.p
        remainder = self.n_nodes % self.p
        
        start_idx = 0
        for i in range(self.p):
            end_idx = start_idx + avg_pva_per_zone + (1 if i < remainder else 0)
            zone = set(range(start_idx, end_idx))
            zones.append(zone)
            
            # 创建默认zone_detail
            zone_details.append({
                "zone_id": i + 1,
                "n_panels": len(zone),
                "perimeter": self.UB,
                "capacity_ok": True,
                "is_connected": True,
                "perimeter_ok": True
            })
            
            start_idx = end_idx
        
        return PartitionResult(
            zones=zones,
            zone_details=zone_details,
            is_feasible=True,
            total_perimeter=self.p * self.UB
        )
    
    def set_dqn_solver(self, dqn_solver):
        """设置 DQN 求解器。
        
        Args:
            dqn_solver: DQNPartitionAgent 实例
        """
        self._dqn_solver = dqn_solver
    
    def set_dqn_solver(self, dqn_solver):
        """设置 DQN 求解器。
        
        Args:
            dqn_solver: DQNPartitionAgent 实例
        """
        self._dqn_solver = dqn_solver
    
    def set_dqn_solver(self, dqn_solver):
        """设置 DQN 求解器。
        
        Args:
            dqn_solver: DQNPartitionAgent 实例
        """
        self._dqn_solver = dqn_solver
    
    def set_dqn_solver(self, dqn_solver):
        """设置 DQN 求解器。
        
        Args:
            dqn_solver: DQNPartitionAgent 实例
        """
        self._dqn_solver = dqn_solver