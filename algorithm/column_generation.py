"""
列生成模块
==========
实现受限主问题（RMP）+ 定价子问题的列生成循环。

核心流程：
1. 从初始路径子集构建 RMP（LP 松弛）
2. 求解 RMP，获取对偶变量
3. 利用对偶变量求解定价子问题，寻找负检验数路径
4. 将新路径加入 RMP
5. 重复直到无负检验数路径（收敛）

定价子问题通过修正成本的最短路算法求解。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import time

from algorithm.path_generator import CablePath, PathGenerator
from algorithm.arc_flow_milp import ArcFlowMILP

logger = logging.getLogger(__name__)


class ColumnGeneration:
    """列生成求解器"""

    def __init__(self, milp_model: ArcFlowMILP, path_generator: PathGenerator,
                 initial_paths: List[CablePath]):
        """
        Parameters
        ----------
        milp_model : ArcFlowMILP
            MILP 模型实例
        path_generator : PathGenerator
            路径生成器
        initial_paths : List[CablePath]
            初始路径集
        """
        self.milp_model = milp_model
        self.path_generator = path_generator
        self.all_paths = list(milp_model.paths)  # 所有可能的路径
        self.n_inverters = milp_model.n_inverters
        self.n_boxes = milp_model.n_boxes

        # 当前活跃路径索引集合
        self.active_path_indices = set()
        for path in initial_paths:
            self.active_path_indices.add(path.path_id)

        # 收敛历史
        self.convergence_history = []

        # 参数
        self.c2 = milp_model.c2
        self.c3 = milp_model.c3
        self.grid_size = milp_model.grid_size

    def solve(self, max_iterations: int = 50, tolerance: float = 1e-4) -> Dict:
        """
        执行列生成循环。

        Parameters
        ----------
        max_iterations : int
            最大迭代次数
        tolerance : float
            收敛容差（检验数阈值）

        Returns
        -------
        Dict
            列生成结果：LP最优值、活跃路径集、收敛历史
        """
        logger.info(f"【列生成】开始（初始路径数: {len(self.active_path_indices)}，"
                     f"最大迭代: {max_iterations}）")

        start_time = time.time()
        prev_obj = float('inf')

        for iteration in range(max_iterations):
            iter_start = time.time()

            # ============ 步骤1: 求解 RMP（LP 松弛） ============
            active_list = sorted(list(self.active_path_indices))
            rmp_result = self.milp_model.build_lp_relaxation(path_indices=active_list)

            if rmp_result is None:
                logger.warning(f"【列生成】第{iteration+1}轮: RMP 不可行")
                # 尝试添加更多路径
                self._add_fallback_paths()
                continue

            current_obj = rmp_result["objective"]
            dual_assignment = rmp_result["dual_assignment"]
            dual_trench_cap = rmp_result.get("dual_trench_cap", {})

            # 记录收敛历史
            self.convergence_history.append({
                "iteration": iteration + 1,
                "objective": current_obj,
                "n_active_paths": len(self.active_path_indices),
                "time": time.time() - iter_start
            })

            logger.info(f"【列生成】第{iteration+1}轮: 目标值={current_obj:.2f}, "
                         f"活跃路径={len(self.active_path_indices)}")

            # ============ 步骤2: 检查收敛 ============
            if abs(prev_obj - current_obj) < tolerance and iteration > 0:
                logger.info(f"【列生成】目标值收敛（变化 < {tolerance}），停止迭代")
                break
            prev_obj = current_obj

            # ============ 步骤3: 求解定价子问题 ============
            new_paths = self._solve_pricing_subproblem(dual_assignment, dual_trench_cap, rmp_result)

            if not new_paths:
                logger.info(f"【列生成】未找到负检验数路径，收敛！")
                break

            # ============ 步骤4: 添加新路径 ============
            for p_idx in new_paths:
                self.active_path_indices.add(p_idx)

            logger.info(f"【列生成】添加 {len(new_paths)} 条新路径")

        total_time = time.time() - start_time
        logger.info(f"【列生成】完成（耗时: {total_time:.1f}s, "
                     f"最终路径数: {len(self.active_path_indices)}）")

        return {
            "objective": current_obj if rmp_result else float('inf'),
            "active_path_indices": sorted(list(self.active_path_indices)),
            "rmp_result": rmp_result,
            "convergence_history": self.convergence_history,
            "total_time": total_time
        }

    def _solve_pricing_subproblem(self, dual_assignment: Dict,
                                   dual_trench_cap: Dict,
                                   rmp_result: Dict) -> List[int]:
        """
        求解定价子问题：寻找检验数为负的新路径。

        检验数 = 路径成本 - 对偶价值
            = c2 * L_p + Σ_{e ∈ p} (c3 * L_e - μ_e) - π_{k(p)}

        如果检验数 < 0，说明该路径能改善目标函数。
        """
        new_paths = []
        edge_list = rmp_result.get("edge_list", [])

        # 构建边 → 对偶价格映射
        edge_dual = {}
        for e_idx, dual_val in dual_trench_cap.items():
            if e_idx < len(edge_list):
                edge_dual[edge_list[e_idx]] = dual_val

        # 对每个逆变器，寻找负检验数路径
        for k in range(self.n_inverters):
            pi_k = dual_assignment.get(k, 0.0)

            best_reduced_cost = 0.0
            best_path_idx = None

            # 遍历该逆变器的所有候选路径
            for p_idx in self.milp_model.paths_for_inverter.get(k, []):
                if p_idx in self.active_path_indices:
                    continue  # 跳过已活跃的路径

                path = self.all_paths[p_idx]

                # 计算检验数
                cable_cost = self.c2 * path.total_length
                trench_component = 0.0
                for edge in path.all_edges:
                    edge_cost = self.c3 * self.grid_size
                    edge_dual_val = edge_dual.get(edge, 0.0)
                    trench_component += (edge_cost - edge_dual_val)

                reduced_cost = cable_cost + trench_component - pi_k

                if reduced_cost < best_reduced_cost - 1e-6:
                    best_reduced_cost = reduced_cost
                    best_path_idx = p_idx

            if best_path_idx is not None:
                new_paths.append(best_path_idx)

        return new_paths

    def _add_fallback_paths(self):
        """当 RMP 不可行时，添加备选路径"""
        for k in range(self.n_inverters):
            inv_paths = self.milp_model.paths_for_inverter.get(k, [])
            has_active = any(p in self.active_path_indices for p in inv_paths)
            if not has_active and inv_paths:
                # 添加该逆变器的最短路径
                best_p = min(inv_paths, key=lambda p: self.all_paths[p].total_length)
                self.active_path_indices.add(best_p)
                logger.info(f"【列生成-回退】为逆变器{k}添加路径{best_p}")


class PricingSubproblem:
    """
    定价子问题求解器 — 基于修正成本最短路。

    对于指定的逆变器 k 和对偶价格，找到从逆变器 k 到升压站的
    最小修正成本路径（经过某个箱变 b）。
    """

    def __init__(self, inverter_coords: List[Tuple[float, float]],
                 box_coords: List[Tuple[float, float]],
                 substation_coord: Tuple[float, float],
                 grid_size: float, c2: float, c3: float):
        self.inverter_coords = inverter_coords
        self.box_coords = box_coords
        self.substation_coord = substation_coord
        self.grid_size = grid_size
        self.c2 = c2
        self.c3 = c3

    def find_best_path(self, inverter_idx: int, dual_pi: float,
                       edge_duals: Dict = None) -> Tuple[Optional[int], float]:
        """
        为指定逆变器找到最小修正成本的箱变。

        Returns
        -------
        Tuple[Optional[int], float]
            (最佳箱变索引, 修正成本)；若修正成本非负则返回 (None, cost)
        """
        if edge_duals is None:
            edge_duals = {}

        inv_coord = self.inverter_coords[inverter_idx]
        best_box = None
        best_cost = float('inf')

        for b_idx, box_coord in enumerate(self.box_coords):
            # 逆变器到箱变的曼哈顿距离
            d_ib = abs(inv_coord[0] - box_coord[0]) + abs(inv_coord[1] - box_coord[1])
            # 箱变到升压站的曼哈顿距离
            d_bs = abs(box_coord[0] - self.substation_coord[0]) + abs(box_coord[1] - self.substation_coord[1])

            total_dist = d_ib + d_bs
            cable_cost = self.c2 * total_dist

            # 简化管沟成本（不考虑边级别的对偶修正）
            trench_cost = self.c3 * total_dist

            # 修正成本
            reduced_cost = cable_cost + trench_cost - dual_pi

            if reduced_cost < best_cost:
                best_cost = reduced_cost
                best_box = b_idx

        return best_box, best_cost
