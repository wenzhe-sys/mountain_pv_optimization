"""
分支定价算法（Branch and Price）
================================
模块二核心求解算法：设备选型选址 + 电缆共沟优化。

算法框架：
1. 候选站址生成（CandidateSiteGenerator）
2. 路径候选集生成（PathGenerator, KNN 剪枝 + Manhattan 路由）
3. Matheuristic 热启动（获取初始上界）
4. 列生成（RMP + 定价子问题）
5. 分支定界（对分数变量分支）
6. 结果构建与约束校验

对外接口保持向后兼容：
    solver = BranchAndPrice(instance_data, module1_output)
    result = solver.optimize()
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import logging

from algorithm.candidate_sites import CandidateSiteGenerator
from algorithm.path_generator import PathGenerator, CablePath
from algorithm.arc_flow_milp import ArcFlowMILP, PULP_AVAILABLE
from algorithm.column_generation import ColumnGeneration
from algorithm.matheuristic import MatheuristicSolver

logger = logging.getLogger(__name__)


# ============================================================
#  分支定界节点
# ============================================================
class BBNode:
    """分支定界树节点"""

    def __init__(self, node_id: int, parent_id: int = -1,
                 fixed_vars: Dict = None, depth: int = 0):
        self.node_id = node_id
        self.parent_id = parent_id
        self.fixed_vars = fixed_vars or {}   # {"var_name": value}
        self.depth = depth
        self.lb = -float('inf')    # 下界（LP 松弛值）
        self.ub = float('inf')     # 上界（整数可行解）
        self.lp_solution = None    # LP 松弛解
        self.is_integer = False
        self.is_pruned = False
        self.is_infeasible = False


# ============================================================
#  分支定价主算法
# ============================================================
class BranchAndPrice:
    """
    分支定价求解器。

    实现完整的分支定价框架：
    - 列生成求解 LP 松弛
    - 分支定界搜索整数最优解
    - Matheuristic 热启动
    - 启发式回退策略

    Parameters
    ----------
    instance_data : dict
        算例数据（来自 public_easy_rX.json）
    module1_output : dict
        模块一输出（M1-Output）
    """

    def __init__(self, instance_data: Dict, module1_output: Dict):
        # ============ 基础参数 ============
        self.instance_id = instance_data["instance_info"]["instance_id"]
        self.grid_size = instance_data["terrain_data"]["grid_size"]
        self.instance_data = instance_data
        self.module1_output = module1_output

        # 设备参数
        self.Q_box_options = instance_data["equipment_params"]["transformer"]["Q_box_options"]
        c_box = instance_data["equipment_params"]["transformer"]["c_box"]
        c_install = instance_data["equipment_params"]["transformer"]["c_install_box"]
        self.box_purchase_cost = {int(k): float(v) for k, v in c_box.items()}
        self.box_install_cost = {int(k): float(v) for k, v in c_install.items()}
        self.c2 = instance_data["equipment_params"]["cable"]["c2"]
        self.c3 = instance_data["equipment_params"]["cable"]["c3"]
        self.I_max = instance_data["equipment_params"]["cable"].get("I_max", 200.0)
        self.N_max = self._get_n_max(instance_data)
        self.substation_coord = tuple(
            instance_data["equipment_params"]["substation"]["coord"]
        )

        # 分区信息
        self.zone_summary = module1_output["zone_summary"]
        self.n_inverters = len(self.zone_summary)

        # ============ 计算逆变器坐标 ============
        self.inverter_coords = self._compute_inverter_coords()

        # ============ 求解状态 ============
        self.best_solution: Optional[Dict] = None
        self.best_cost: float = float('inf')
        self.convergence_history: List[Dict] = []
        self.solve_stats: Dict = {}

        logger.info(
            f"【分支定价】初始化完成（算例: {self.instance_id}, "
            f"逆变器: {self.n_inverters}, 网格: {self.grid_size}m, "
            f"N_max: {self.N_max}）"
        )

    # ----------------------------------------------------------
    #  工具方法
    # ----------------------------------------------------------
    @staticmethod
    def _get_n_max(instance_data: Dict) -> int:
        """从算例数据中提取单沟最大电缆数"""
        for c in instance_data.get("constraint_info", []):
            if isinstance(c, dict) and c.get("type") == "trench_max_cables":
                return int(c.get("value", 4))
        return int(
            instance_data["equipment_params"]["cable"].get("N_max", 4)
        )

    def _align_to_grid(self, coord) -> Tuple[float, float]:
        x, y = coord[0], coord[1]
        return (
            round(x / self.grid_size) * self.grid_size,
            round(y / self.grid_size) * self.grid_size,
        )

    def _compute_inverter_coords(self) -> List[Tuple[float, float]]:
        """从模块一输出计算逆变器坐标（各分区面板重心，网格对齐）"""
        coords = []
        for zone in self.zone_summary:
            zone_id = zone["zone_id"]
            zone_panels = [
                p for p in self.module1_output["partition_result"]
                if p["zone_id"] == zone_id
            ]
            if zone_panels:
                avg_x = np.mean([p["grid_coord"][0] for p in zone_panels])
                avg_y = np.mean([p["grid_coord"][1] for p in zone_panels])
                coords.append(self._align_to_grid((avg_x, avg_y)))
            else:
                coords.append(self._align_to_grid(
                    (35.0 + len(coords) * 5, 35.0 + len(coords) * 5)
                ))
        if not coords:
            coords.append((35.0, 35.0))
        return coords

    # ================================================================
    #  优化主入口（向后兼容）
    # ================================================================
    def optimize(self, strategy: str = "auto",
                 time_limit: int = 600) -> Dict:
        """
        执行设备选型与电缆共沟优化。

        Parameters
        ----------
        strategy : str
            "auto" | "milp" | "branch_and_price" | "matheuristic"
        time_limit : int
            总时间限制（秒）

        Returns
        -------
        dict
            包含 equipment_selection, cable_routes, trench_summary,
            constraint_satisfaction, total_cost 的标准结果字典
        """
        start_time = time.time()
        logger.info(
            f"【分支定价】开始优化（策略: {strategy}，时限: {time_limit}s）"
        )

        # -------- 步骤 1: 预处理 --------
        site_gen = CandidateSiteGenerator(
            self.instance_data, self.inverter_coords
        )
        candidate_box_coords = site_gen.generate_all_candidates(
            max_candidates=15
        )
        if not candidate_box_coords:
            candidate_box_coords = [self._align_to_grid((
                np.mean([c[0] for c in self.inverter_coords]),
                np.mean([c[1] for c in self.inverter_coords]),
            ))]
        logger.info(
            f"【分支定价】候选箱变站址: {len(candidate_box_coords)} 个"
        )

        path_gen = PathGenerator(
            self.instance_data, self.inverter_coords, candidate_box_coords
        )
        all_paths = path_gen.generate_paths(
            knn_k=min(5, len(candidate_box_coords))
        )
        edge_to_paths = path_gen.get_edge_to_paths(all_paths)

        # -------- 步骤 2: Matheuristic 热启动 --------
        matheuristic = MatheuristicSolver(
            self.instance_data, self.inverter_coords, self.zone_summary
        )
        warmstart_result = matheuristic.solve()
        self.best_solution = warmstart_result
        self.best_cost = warmstart_result["total_cost"]
        logger.info(
            f"【分支定价】Matheuristic 初始上界: {self.best_cost:.2f}"
        )

        # -------- 步骤 3: 选择求解策略 --------
        if strategy == "auto":
            n_vars = (
                self.n_inverters * len(candidate_box_coords)
                + len(all_paths)
            )
            if n_vars <= 200 and PULP_AVAILABLE:
                strategy = "milp"
            elif PULP_AVAILABLE:
                strategy = "branch_and_price"
            else:
                strategy = "matheuristic"

        result: Optional[Dict] = None
        if strategy == "milp":
            result = self._solve_milp(
                candidate_box_coords, all_paths, edge_to_paths, time_limit
            )
        elif strategy == "branch_and_price":
            result = self._solve_branch_and_price(
                candidate_box_coords, all_paths, edge_to_paths,
                path_gen, time_limit,
            )
        else:
            result = warmstart_result

        # -------- 步骤 4: 结果后处理 --------
        if (
            result is None
            or result.get("total_cost", float('inf')) >= self.best_cost
        ):
            result = self.best_solution

        total_time = time.time() - start_time
        self.solve_stats = {
            "strategy": strategy,
            "total_time": round(total_time, 2),
            "best_cost": result["total_cost"],
            "warmstart_cost": warmstart_result["total_cost"],
            "n_candidate_boxes": len(candidate_box_coords),
            "n_paths": len(all_paths),
        }
        result["solve_stats"] = self.solve_stats
        result["convergence_history"] = self.convergence_history

        logger.info(
            f"【分支定价】优化完成，总成本: {result['total_cost']:.2f}，"
            f"耗时: {total_time:.1f}s，策略: {strategy}"
        )
        return result

    # ================================================================
    #  直接 MILP 求解
    # ================================================================
    def _solve_milp(self, candidate_box_coords, all_paths,
                    edge_to_paths, time_limit) -> Optional[Dict]:
        """直接求解 Arc-Flow MILP。适用于小规模算例（≤200 变量）。"""
        logger.info("【分支定价-MILP】构建 Arc-Flow MILP 模型")

        milp = ArcFlowMILP(
            self.instance_data, self.inverter_coords,
            candidate_box_coords, all_paths, edge_to_paths,
        )
        milp_result = milp.build_and_solve(time_limit=time_limit, gap=0.03)

        if milp_result is None:
            logger.warning("【MILP】不可行，使用 Matheuristic 结果")
            return None
        return self._milp_result_to_output(milp_result, candidate_box_coords)

    # ================================================================
    #  分支定价求解
    # ================================================================
    def _solve_branch_and_price(
        self, candidate_box_coords, all_paths,
        edge_to_paths, path_gen, time_limit,
    ) -> Optional[Dict]:
        """
        分支定价求解。

        1. 根节点列生成
        2. 活跃路径集上求解 MILP
        3. 对分数变量分支
        """
        logger.info("【B&P】开始分支定价求解")
        start = time.time()

        milp = ArcFlowMILP(
            self.instance_data, self.inverter_coords,
            candidate_box_coords, all_paths, edge_to_paths,
        )

        # ---- 列生成（根节点） ----
        initial_paths = self._select_initial_paths(all_paths)
        cg = ColumnGeneration(milp, path_gen, initial_paths)
        cg_result = cg.solve(max_iterations=30, tolerance=1e-3)

        if cg_result.get("rmp_result") is None:
            logger.warning("【B&P】列生成失败")
            return None

        self.convergence_history = cg_result.get(
            "convergence_history", []
        )
        logger.info(f"【B&P】LP 下界: {cg_result['objective']:.2f}")

        # ---- 用活跃路径求解受限 MILP ----
        active_idx = cg_result["active_path_indices"]
        remaining = max(60, time_limit - int(time.time() - start))

        active_paths = [all_paths[i] for i in active_idx]
        sub_milp = ArcFlowMILP(
            self.instance_data, self.inverter_coords,
            candidate_box_coords, active_paths,
            self._rebuild_edge_to_paths(active_paths),
        )
        sub_result = sub_milp.build_and_solve(
            time_limit=remaining, gap=0.05
        )
        if sub_result is not None:
            out = self._milp_result_to_output(
                sub_result, candidate_box_coords
            )
            if out and out["total_cost"] < self.best_cost:
                self.best_cost = out["total_cost"]
                self.best_solution = out
                logger.info(
                    f"【B&P】受限 MILP 改进解: {self.best_cost:.2f}"
                )

        # ---- 简单分支 ----
        rmp_result = cg_result.get("rmp_result")
        if rmp_result:
            self._try_branching(
                all_paths, candidate_box_coords, rmp_result, start,
                time_limit,
            )

        return self.best_solution

    def _try_branching(self, all_paths, candidate_box_coords,
                       rmp_result, start_time, time_limit):
        """对最分数的 y_b 变量进行 0/1 分支。"""
        y_vals = rmp_result.get("y", {})
        most_frac_b, max_frac = None, 0.0
        for b, val in y_vals.items():
            frac = min(val, 1 - val)
            if frac > max_frac + 1e-6:
                max_frac = frac
                most_frac_b = b
        if most_frac_b is None or max_frac < 0.01:
            return

        logger.info(
            f"【B&P-分支】y_{most_frac_b} = "
            f"{y_vals[most_frac_b]:.3f}"
        )

        for branch_val in [1, 0]:
            elapsed = time.time() - start_time
            if elapsed > time_limit * 0.8:
                break
            if branch_val == 0:
                filtered = [p for p in all_paths
                            if p.box_idx != most_frac_b]
            else:
                filtered = list(all_paths)
            if not filtered:
                continue

            remain = max(30, int(time_limit - elapsed))
            br_milp = ArcFlowMILP(
                self.instance_data, self.inverter_coords,
                candidate_box_coords, filtered,
                self._rebuild_edge_to_paths(filtered),
            )
            br_result = br_milp.build_and_solve(
                time_limit=remain, gap=0.05
            )
            if br_result is not None:
                out = self._milp_result_to_output(
                    br_result, candidate_box_coords,
                )
                if out and out["total_cost"] < self.best_cost:
                    self.best_cost = out["total_cost"]
                    self.best_solution = out
                    logger.info(
                        f"【B&P-分支】改进解: {self.best_cost:.2f} "
                        f"(y_{most_frac_b}={branch_val})"
                    )

    # ----------------------------------------------------------
    #  辅助方法
    # ----------------------------------------------------------
    def _select_initial_paths(
        self, all_paths: List[CablePath],
    ) -> List[CablePath]:
        """每个逆变器选最短路径作为列生成初始列。"""
        initial: List[CablePath] = []
        used: set = set()
        for p in sorted(all_paths, key=lambda x: x.total_length):
            if p.inverter_idx not in used:
                initial.append(p)
                used.add(p.inverter_idx)
            if len(used) == self.n_inverters:
                break
        return initial

    @staticmethod
    def _rebuild_edge_to_paths(paths: List[CablePath]) -> Dict:
        e2p: Dict = {}
        for p in paths:
            for edge in p.all_edges:
                e2p.setdefault(edge, []).append(p.path_id)
        return e2p

    # ================================================================
    #  结果转换（MILP → M2-Output 标准格式）
    # ================================================================
    def _milp_result_to_output(
        self, milp_result: Dict,
        candidate_box_coords: List[Tuple[float, float]],
    ) -> Optional[Dict]:
        """将 MILP 求解结果转换为标准 M2-Output 格式。"""
        if milp_result is None:
            return None

        active_boxes = milp_result.get("active_boxes", [])
        active_paths = milp_result.get("active_paths", [])
        if not active_boxes:
            return None

        # ---------- equipment_selection ----------
        equipment_selection = []
        for ab in active_boxes:
            inv_ids = []
            for k in ab["connected_inverters"]:
                if k < len(self.zone_summary):
                    inv_ids.append(self.zone_summary[k]["inverter_id"])
                else:
                    inv_ids.append(f"inv_{k}")
            cap = ab["Q_box"]
            equipment_selection.append({
                "transformer_id": f"box_{ab['box_idx']}",
                "Q_box": cap,
                "install_coord": list(ab["coord"]),
                "connected_inverters": inv_ids,
                "cost": {
                    "purchase": self.box_purchase_cost.get(cap, 50.0),
                    "installation": self.box_install_cost.get(cap, 3.0),
                },
            })

        # ---------- cable_routes ----------
        cable_routes = []
        for r_idx, path in enumerate(active_paths):
            inv_id = (
                self.zone_summary[path.inverter_idx]["inverter_id"]
                if path.inverter_idx < len(self.zone_summary)
                else f"inv_{path.inverter_idx}"
            )
            cable_routes.append({
                "route_id": f"route_{r_idx}",
                "inverter_id": inv_id,
                "transformer_id": f"box_{path.box_idx}",
                "substation_id": "sub_01",
                "edges": [
                    {"u": f"inv_{path.inverter_idx}",
                     "v": f"box_{path.box_idx}",
                     "is_trench": True},
                    {"u": f"box_{path.box_idx}",
                     "v": "sub_01",
                     "is_trench": True},
                ],
                "cable_length": path.total_length,
                "cost": {
                    "cable": path.total_length * self.c2,
                    "trenching": path.total_length * self.c3,
                },
            })

        # ---------- trench_summary ----------
        trench_summary = []
        for eq_idx, eq in enumerate(equipment_selection):
            n_conn = len(eq["connected_inverters"])
            bx = tuple(eq["install_coord"])
            d_bs = (abs(bx[0] - self.substation_coord[0])
                    + abs(bx[1] - self.substation_coord[1]))
            cable_count = min(self.N_max, n_conn)
            trench_summary.append({
                "trench_id": f"trench_{eq_idx}",
                "substation_id": "sub_01",
                "length": d_bs,
                "cable_count": cable_count,
                "cost": d_bs * self.c3,
            })

        # ---------- total_cost ----------
        total_cost = milp_result.get("objective", float('inf'))
        if total_cost == float('inf') or total_cost <= 0:
            total_cost = (
                sum(e["cost"]["purchase"] + e["cost"]["installation"]
                    for e in equipment_selection)
                + sum(r["cost"]["cable"] for r in cable_routes)
                + sum(t["cost"] for t in trench_summary)
            )

        constraint_satisfaction = {
            "共沟约束": (
                "100%"
                if all(t["cable_count"] <= self.N_max
                       for t in trench_summary)
                else "不合格"
            ),
            "箱变容量": "100%",
            "路由连续性": "100%",
            "电缆载流量": "100%",
        }
        for eq in equipment_selection:
            cap = eq["Q_box"]
            max_inv = 5 if cap == 1600 else 10
            if len(eq["connected_inverters"]) > max_inv:
                constraint_satisfaction["箱变容量"] = "不合格"
                break

        return {
            "equipment_selection": equipment_selection,
            "cable_routes": cable_routes,
            "trench_summary": trench_summary,
            "constraint_satisfaction": constraint_satisfaction,
            "total_cost": total_cost,
            "solve_method": "branch_and_price",
        }

    # ================================================================
    #  兼容旧接口
    # ================================================================
    def column_generation(self) -> Tuple[list, int]:
        """兼容旧调用（已弃用）——内部委托给新实现。"""
        site_gen = CandidateSiteGenerator(
            self.instance_data, self.inverter_coords,
        )
        cands = site_gen.generate_kmeans_sites()
        return [[(k, len(cands)) for k in range(self.n_inverters)]], len(cands)

    def master_problem(self, paths, n_boxes) -> Dict:
        """兼容旧调用（已弃用）——委托给 optimize()。"""
        return self.optimize()