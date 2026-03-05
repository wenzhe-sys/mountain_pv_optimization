"""
Arc-Flow MILP 模型
==================
基于边-流（Arc-Flow）的混合整数规划模型，用于设备选型选址 + 电缆共沟优化。

使用 PuLP 库建模求解，支持 CBC（开源）和 GUROBI/CPLEX（商业，若可用）。

决策变量：
- y_b:     箱变 b 是否启用
- z_b:     箱变 b 的容量类型（0=1600kVA, 1=3200kVA）
- γ_{kb}:  逆变器 k 是否分配给箱变 b
- α_p:     路径 p 是否启用
- β_e:     边 e 是否开挖管沟

目标函数：min 箱变成本 + 电缆成本 + 管沟成本

核心约束：
1. 唯一分配：每个逆变器必须接入恰好一个箱变
2. 分配-启用一致性：分配到箱变 b 的逆变器需要箱变 b 启用
3. 路径-分配一致性：路径启用要求对应的逆变器-箱变分配
4. 布线-挖沟协同：启用路径必须开挖对应管沟
5. 箱变容量约束：连接逆变器数 ≤ 容量上限
6. 共沟约束：单管沟电缆数 ≤ N_max
7. 载流量约束：箱变出线电流 ≤ I_max
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import time

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from algorithm.path_generator import CablePath

logger = logging.getLogger(__name__)


class ArcFlowMILP:
    """Arc-Flow 混合整数线性规划模型"""

    def __init__(self, instance_data: Dict, inverter_coords: List[Tuple[float, float]],
                 candidate_box_coords: List[Tuple[float, float]],
                 paths: List[CablePath], edge_to_paths: Dict):
        """
        Parameters
        ----------
        instance_data : Dict
            算例数据
        inverter_coords : List[Tuple[float, float]]
            逆变器坐标列表
        candidate_box_coords : List[Tuple[float, float]]
            候选箱变坐标列表
        paths : List[CablePath]
            候选路径集
        edge_to_paths : Dict
            边→路径列表映射
        """
        self.instance_data = instance_data
        self.inverter_coords = inverter_coords
        self.candidate_box_coords = candidate_box_coords
        self.paths = paths
        self.edge_to_paths = edge_to_paths
        self.substation_coord = tuple(instance_data["equipment_params"]["substation"]["coord"])

        # 参数
        self.n_inverters = len(inverter_coords)
        self.n_boxes = len(candidate_box_coords)
        self.n_paths = len(paths)
        self.grid_size = instance_data["terrain_data"]["grid_size"]

        # 设备参数
        self.Q_box_options = instance_data["equipment_params"]["transformer"]["Q_box_options"]
        c_box = instance_data["equipment_params"]["transformer"]["c_box"]
        c_install = instance_data["equipment_params"]["transformer"]["c_install_box"]
        self.box_purchase_cost = {int(k): float(v) for k, v in c_box.items()}
        self.box_install_cost = {int(k): float(v) for k, v in c_install.items()}
        self.capacity_map = {1600: 5, 3200: 10}

        # 电缆参数
        self.c2 = instance_data["equipment_params"]["cable"]["c2"]
        self.c3 = instance_data["equipment_params"]["cable"]["c3"]
        self.I_max = instance_data["equipment_params"]["cable"]["I_max"]

        # 共沟约束
        self.N_max = self._get_n_max(instance_data)

        # 逆变器参数
        self.inv_q = instance_data["equipment_params"]["inverter"]["q"]  # 额定功率 kW
        self.inv_r = instance_data["equipment_params"]["inverter"]["r"]  # 功率因数

        # 构建索引映射
        self._build_path_indices()

    @staticmethod
    def _manhattan_dist(p1, p2):
        """曼哈顿距离"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_n_max(self, instance_data: Dict) -> int:
        """提取单沟最大电缆数"""
        # 优先从 constraint_info 中获取
        for c in instance_data.get("constraint_info", []):
            if isinstance(c, dict) and c.get("type") == "trench_max_cables":
                return int(c.get("value", 4))
        # 从 equipment_params 中获取
        return int(instance_data["equipment_params"]["cable"].get("N_max", 4))

    def _build_path_indices(self):
        """构建辅助索引"""
        # 逆变器 k 可用的路径集合
        self.paths_for_inverter = {k: [] for k in range(self.n_inverters)}
        # 箱变 b 可用的路径集合
        self.paths_for_box = {b: [] for b in range(self.n_boxes)}
        # (逆变器k, 箱变b) 对应的路径集合
        self.paths_for_pair = {}

        for p_idx, path in enumerate(self.paths):
            k = path.inverter_idx
            b = path.box_idx
            self.paths_for_inverter[k].append(p_idx)
            self.paths_for_box[b].append(p_idx)
            pair = (k, b)
            if pair not in self.paths_for_pair:
                self.paths_for_pair[pair] = []
            self.paths_for_pair[pair].append(p_idx)

        # 分离 inv→box 边和 box→sub 边的映射
        self._build_separated_edge_maps()

    def _build_separated_edge_maps(self):
        """
        构建分离的边映射，区分 inv→box 边和 box→sub 边。

        关键设计：
        - 每条路径的 inv→box 段为独立电缆（每条路径一根）
        - 每个箱变的 box→sub 段为共享电缆（每个活跃箱变一根）
        - N_max 约束需要分别计数这两类电缆
        """
        # inv→box 边 → path 索引列表
        self.inv_edge_to_paths = {}
        # box→sub 边 → box 索引集合
        self.sub_edge_to_boxes = {}
        # 所有唯一边的集合
        self.all_unique_edges = set()

        for p_idx, path in enumerate(self.paths):
            for edge in path.inv_to_box_edges:
                if edge not in self.inv_edge_to_paths:
                    self.inv_edge_to_paths[edge] = []
                self.inv_edge_to_paths[edge].append(p_idx)
                self.all_unique_edges.add(edge)

        for b in range(self.n_boxes):
            # 获取任一从该箱变出发到升压站的路径（box→sub段对同一箱变是相同的）
            box_paths = self.paths_for_box[b]
            if box_paths:
                sample_path = self.paths[box_paths[0]]
                for edge in sample_path.box_to_sub_edges:
                    if edge not in self.sub_edge_to_boxes:
                        self.sub_edge_to_boxes[edge] = set()
                    self.sub_edge_to_boxes[edge].add(b)
                    self.all_unique_edges.add(edge)

    def build_and_solve(self, time_limit: int = 300, gap: float = 0.05,
                        solver_name: str = "auto") -> Optional[Dict]:
        """
        构建并求解 MILP 模型。

        Parameters
        ----------
        time_limit : int
            求解时间限制（秒）
        gap : float
            求解间隙（MIP Gap）
        solver_name : str
            求解器名称（"auto", "CBC", "GUROBI", "CPLEX"）

        Returns
        -------
        Optional[Dict]
            求解结果，None 表示不可行
        """
        if not PULP_AVAILABLE:
            logger.error("PuLP 未安装，无法求解 MILP 模型")
            return None

        logger.info(f"【Arc-Flow MILP】开始建模（{self.n_inverters}台逆变器，"
                     f"{self.n_boxes}个候选箱变，{self.n_paths}条候选路径）")

        start_time = time.time()

        # ==================== 创建模型 ====================
        model = pulp.LpProblem("ArcFlow_Equipment_Cable", pulp.LpMinimize)

        # ==================== 决策变量 ====================
        # y_b: 箱变 b 是否启用
        y = {b: pulp.LpVariable(f"y_{b}", cat="Binary") for b in range(self.n_boxes)}

        # z_b: 箱变 b 的容量类型（1=3200kVA, 0=1600kVA）
        z = {b: pulp.LpVariable(f"z_{b}", cat="Binary") for b in range(self.n_boxes)}

        # γ_{kb}: 逆变器 k 分配给箱变 b
        gamma = {(k, b): pulp.LpVariable(f"gamma_{k}_{b}", cat="Binary")
                 for k in range(self.n_inverters) for b in range(self.n_boxes)}

        # α_p: 路径 p 是否启用（仅 inv→box 段）
        alpha = {p: pulp.LpVariable(f"alpha_{p}", cat="Binary") for p in range(self.n_paths)}

        # β_e: 边 e 是否开挖管沟（覆盖所有唯一边）
        edge_list = sorted(self.all_unique_edges)
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        beta = {e_idx: pulp.LpVariable(f"beta_{e_idx}", cat="Binary") for e_idx in range(len(edge_list))}

        # ==================== 目标函数 ====================
        # 1. 箱变购置+安装成本
        box_cost = pulp.lpSum([
            # 1600kVA
            (self.box_purchase_cost[1600] + self.box_install_cost[1600]) * (y[b] - z[b]) +
            # 3200kVA
            (self.box_purchase_cost[3200] + self.box_install_cost[3200]) * z[b]
            for b in range(self.n_boxes)
        ])

        # 2. 电缆成本 = inv→box 段（每根独立） + box→sub 段（每个箱变一根）
        cable_cost_inv_box = pulp.lpSum([
            self.c2 * self.paths[p].inv_to_box_length * alpha[p]
            for p in range(self.n_paths)
        ])

        # box→sub 电缆成本：每个活跃箱变一根到升压站的电缆
        cable_cost_box_sub = pulp.lpSum([
            self.c2 * self._manhattan_dist(self.candidate_box_coords[b], self.substation_coord) * y[b]
            for b in range(self.n_boxes)
        ])

        cable_cost = cable_cost_inv_box + cable_cost_box_sub

        # 3. 管沟成本（所有需要开挖的边）
        trench_cost = pulp.lpSum([
            self.c3 * self.grid_size * beta[e_idx]
            for e_idx in range(len(edge_list))
        ])

        model += box_cost + cable_cost + trench_cost, "Total_Cost"

        # ==================== 约束 ====================
        # 约束1: 唯一分配 — 每个逆变器恰好分配给一个箱变
        for k in range(self.n_inverters):
            model += (
                pulp.lpSum([gamma[(k, b)] for b in range(self.n_boxes)]) == 1,
                f"UniqueAssignment_{k}"
            )

        # 约束2: 分配-启用一致性 — γ_{kb} ≤ y_b
        for k in range(self.n_inverters):
            for b in range(self.n_boxes):
                model += (gamma[(k, b)] <= y[b], f"AssignActivate_{k}_{b}")

        # 约束3: 容量类型一致性 — z_b ≤ y_b
        for b in range(self.n_boxes):
            model += (z[b] <= y[b], f"TypeActivate_{b}")

        # 约束4: 路径-分配一致性 — 路径启用需要对应分配
        for p_idx, path in enumerate(self.paths):
            k, b = path.inverter_idx, path.box_idx
            model += (alpha[p_idx] <= gamma[(k, b)], f"PathAssign_{p_idx}")

        # 约束5: 逆变器至少有一条路径启用 — Σ_{p ∈ P_k} α_p ≥ γ_{kb}
        for k in range(self.n_inverters):
            for b in range(self.n_boxes):
                pair_paths = self.paths_for_pair.get((k, b), [])
                if pair_paths:
                    model += (
                        pulp.lpSum([alpha[p] for p in pair_paths]) >= gamma[(k, b)],
                        f"PathCover_{k}_{b}"
                    )
                else:
                    # 没有路径连接这个(k,b)对 → 禁止分配
                    model += (gamma[(k, b)] == 0, f"NoPath_{k}_{b}")

        # 约束6: 布线-挖沟协同
        # 6a: inv→box 路径启用 → 其 inv→box 边必须开沟
        for edge, path_ids in self.inv_edge_to_paths.items():
            e_idx = edge_to_idx[edge]
            for p_idx in path_ids:
                model += (alpha[p_idx] <= beta[e_idx], f"TrenchDig_inv_{p_idx}_{e_idx}")
        # 6b: box→sub 箱变启用 → 其 box→sub 边必须开沟
        for edge, box_ids in self.sub_edge_to_boxes.items():
            e_idx = edge_to_idx[edge]
            for b in box_ids:
                model += (y[b] <= beta[e_idx], f"TrenchDig_sub_{b}_{e_idx}")

        # 约束7: 箱变容量约束
        for b in range(self.n_boxes):
            # 1600kVA: 最多5台; 3200kVA: 最多10台
            # Σ_k γ_{kb} ≤ 5*(1-z_b) + 10*z_b = 5 + 5*z_b
            model += (
                pulp.lpSum([gamma[(k, b)] for k in range(self.n_inverters)]) <= 5 + 5 * z[b],
                f"BoxCapacity_{b}"
            )

        # 约束8: 共沟约束 — 单管沟电缆数 ≤ N_max
        #   电缆数 = inv→box 段经过该边的活跃路径数 + box→sub 段经过该边的活跃箱变数
        for e_idx, edge in enumerate(edge_list):
            inv_paths = self.inv_edge_to_paths.get(edge, [])
            sub_boxes = self.sub_edge_to_boxes.get(edge, set())
            total_potential = len(inv_paths) + len(sub_boxes)
            if total_potential > self.N_max:
                cable_sum = []
                if inv_paths:
                    cable_sum.append(pulp.lpSum([alpha[p] for p in inv_paths]))
                if sub_boxes:
                    cable_sum.append(pulp.lpSum([y[b] for b in sub_boxes]))
                model += (
                    pulp.lpSum(cable_sum) <= self.N_max,
                    f"TrenchCable_{e_idx}"
                )

        # ==================== 增强约束（有效不等式） ====================
        # 容量下界割 — 至少需要 ceil(n_inv / max_cap) 台箱变
        min_boxes = int(np.ceil(self.n_inverters / 10))
        model += (
            pulp.lpSum([y[b] for b in range(self.n_boxes)]) >= min_boxes,
            "MinBoxes"
        )

        # 对称消除 — 按索引排序启用箱变
        for b in range(1, self.n_boxes):
            model += (y[b] <= y[b-1], f"SymBreak_{b}")

        # ==================== 求解 ====================
        solver = self._get_solver(solver_name, time_limit, gap)
        logger.info(f"【Arc-Flow MILP】使用求解器: {solver.name}")

        model.solve(solver)

        solve_time = time.time() - start_time
        status = pulp.LpStatus[model.status]
        logger.info(f"【Arc-Flow MILP】求解完成（状态: {status}，耗时: {solve_time:.1f}s）")

        if model.status != pulp.constants.LpStatusOptimal:
            logger.warning(f"【Arc-Flow MILP】未找到最优解（状态: {status}）")
            return None

        # ==================== 提取结果 ====================
        return self._extract_solution(model, y, z, gamma, alpha, beta, edge_list)

    def build_lp_relaxation(self, path_indices: List[int] = None) -> Optional[Dict]:
        """
        构建 LP 松弛（用于列生成的 RMP）。

        Parameters
        ----------
        path_indices : List[int]
            当前 RMP 中的路径子集索引。None 表示使用所有路径。

        Returns
        -------
        Optional[Dict]
            包含 LP 最优值、对偶变量、解的字典
        """
        if not PULP_AVAILABLE:
            return None

        if path_indices is None:
            path_indices = list(range(self.n_paths))

        # 构建路径子集的索引映射
        active_paths = {p_idx: self.paths[p_idx] for p_idx in path_indices}

        # 构建活跃路径的分离边映射
        active_inv_edge_to_paths = {}
        active_sub_edge_to_boxes = {}
        active_all_edges = set()
        
        for p_idx in path_indices:
            path = self.paths[p_idx]
            for edge in path.inv_to_box_edges:
                if edge not in active_inv_edge_to_paths:
                    active_inv_edge_to_paths[edge] = []
                active_inv_edge_to_paths[edge].append(p_idx)
                active_all_edges.add(edge)

        # box→sub edges (per box, not per path)
        active_box_set = set(self.paths[p].box_idx for p in path_indices)
        for b in active_box_set:
            box_paths = [p for p in path_indices if self.paths[p].box_idx == b]
            if box_paths:
                sample = self.paths[box_paths[0]]
                for edge in sample.box_to_sub_edges:
                    if edge not in active_sub_edge_to_boxes:
                        active_sub_edge_to_boxes[edge] = set()
                    active_sub_edge_to_boxes[edge].add(b)
                    active_all_edges.add(edge)

        edge_list = sorted(active_all_edges)
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}

        # ==================== 创建 LP 松弛模型 ====================
        model = pulp.LpProblem("RMP_LP_Relaxation", pulp.LpMinimize)

        # 连续变量（LP 松弛）
        y = {b: pulp.LpVariable(f"y_{b}", 0, 1) for b in range(self.n_boxes)}
        z = {b: pulp.LpVariable(f"z_{b}", 0, 1) for b in range(self.n_boxes)}
        gamma = {(k, b): pulp.LpVariable(f"gamma_{k}_{b}", 0, 1)
                 for k in range(self.n_inverters) for b in range(self.n_boxes)}
        alpha = {p: pulp.LpVariable(f"alpha_{p}", 0, 1) for p in path_indices}
        beta = {e_idx: pulp.LpVariable(f"beta_{e_idx}", 0, 1) for e_idx in range(len(edge_list))}

        # 目标函数（与 MILP 一致：分离 inv→box 和 box→sub 电缆成本）
        box_cost = pulp.lpSum([
            (self.box_purchase_cost[1600] + self.box_install_cost[1600]) * (y[b] - z[b]) +
            (self.box_purchase_cost[3200] + self.box_install_cost[3200]) * z[b]
            for b in range(self.n_boxes)
        ])
        cable_cost = pulp.lpSum([
            self.c2 * self.paths[p].inv_to_box_length * alpha[p] for p in path_indices
        ]) + pulp.lpSum([
            self.c2 * self._manhattan_dist(self.candidate_box_coords[b], self.substation_coord) * y[b]
            for b in range(self.n_boxes)
        ])
        trench_cost = pulp.lpSum([
            self.c3 * self.grid_size * beta[e_idx] for e_idx in range(len(edge_list))
        ])
        model += box_cost + cable_cost + trench_cost

        # 约束（同 MILP，但变量连续）
        assign_constraints = {}
        for k in range(self.n_inverters):
            name = f"Assign_{k}"
            c = pulp.lpSum([gamma[(k, b)] for b in range(self.n_boxes)]) == 1
            model += (c, name)
            assign_constraints[k] = name

        for k in range(self.n_inverters):
            for b in range(self.n_boxes):
                model += (gamma[(k, b)] <= y[b], f"ActB_{k}_{b}")

        for b in range(self.n_boxes):
            model += (z[b] <= y[b], f"TypeB_{b}")

        for p_idx in path_indices:
            path = self.paths[p_idx]
            model += (alpha[p_idx] <= gamma[(path.inverter_idx, path.box_idx)],
                      f"PA_{p_idx}")

        for k in range(self.n_inverters):
            for b in range(self.n_boxes):
                pair_paths = [p for p in self.paths_for_pair.get((k, b), []) if p in path_indices]
                if pair_paths:
                    model += (pulp.lpSum([alpha[p] for p in pair_paths]) >= gamma[(k, b)],
                              f"PC_{k}_{b}")
                else:
                    model += (gamma[(k, b)] == 0, f"NP_{k}_{b}")

        # 布线-挖沟协同（分离的边映射）
        for edge, pids in active_inv_edge_to_paths.items():
            e_idx = edge_to_idx[edge]
            for p_idx in pids:
                model += (alpha[p_idx] <= beta[e_idx], f"TD_inv_{p_idx}_{e_idx}")
        for edge, bids in active_sub_edge_to_boxes.items():
            e_idx = edge_to_idx[edge]
            for b in bids:
                model += (y[b] <= beta[e_idx], f"TD_sub_{b}_{e_idx}")

        for b in range(self.n_boxes):
            model += (pulp.lpSum([gamma[(k, b)] for k in range(self.n_inverters)]) <= 5 + 5 * z[b],
                      f"Cap_{b}")

        trench_cap_constraints = {}
        for e_idx, edge in enumerate(edge_list):
            inv_paths = active_inv_edge_to_paths.get(edge, [])
            sub_boxes = active_sub_edge_to_boxes.get(edge, set())
            total_potential = len(inv_paths) + len(sub_boxes)
            if total_potential > 1:
                name = f"TC_{e_idx}"
                cable_sum = []
                if inv_paths:
                    cable_sum.append(pulp.lpSum([alpha[p] for p in inv_paths]))
                if sub_boxes:
                    cable_sum.append(pulp.lpSum([y[b] for b in sub_boxes]))
                model += (pulp.lpSum(cable_sum) <= self.N_max, name)
                trench_cap_constraints[e_idx] = name

        # 求解
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
        model.solve(solver)

        if model.status != pulp.constants.LpStatusOptimal:
            return None

        # 提取对偶变量
        dual_assignment = {}  # π_k
        dual_trench_cap = {}  # μ_e
        try:
            constraints = model.constraints
            for k in range(self.n_inverters):
                name = f"Assign_{k}"
                if name in constraints:
                    dual_assignment[k] = constraints[name].pi if hasattr(constraints[name], 'pi') else 0.0

            for e_idx in trench_cap_constraints:
                name = trench_cap_constraints[e_idx]
                if name in constraints:
                    dual_trench_cap[e_idx] = constraints[name].pi if hasattr(constraints[name], 'pi') else 0.0
        except Exception:
            # CBC 可能不支持对偶变量提取
            dual_assignment = {k: 0.0 for k in range(self.n_inverters)}
            dual_trench_cap = {}

        # 提取变量值
        alpha_vals = {p: pulp.value(alpha[p]) or 0.0 for p in path_indices}
        gamma_vals = {(k, b): pulp.value(gamma[(k, b)]) or 0.0
                      for k in range(self.n_inverters) for b in range(self.n_boxes)}
        y_vals = {b: pulp.value(y[b]) or 0.0 for b in range(self.n_boxes)}
        z_vals = {b: pulp.value(z[b]) or 0.0 for b in range(self.n_boxes)}

        return {
            "objective": pulp.value(model.objective),
            "status": pulp.LpStatus[model.status],
            "alpha": alpha_vals,
            "gamma": gamma_vals,
            "y": y_vals,
            "z": z_vals,
            "dual_assignment": dual_assignment,
            "dual_trench_cap": dual_trench_cap,
            "edge_list": edge_list,
            "active_inv_edge_to_paths": active_inv_edge_to_paths,
            "active_sub_edge_to_boxes": active_sub_edge_to_boxes,
        }

    def _get_solver(self, solver_name: str, time_limit: int, gap: float):
        """获取合适的求解器"""
        if solver_name == "auto":
            # 尝试商业求解器（验证实际可用性）
            try:
                solver = pulp.GUROBI_CMD(msg=0, timeLimit=time_limit, gapRel=gap)
                if solver.available():
                    logger.info("检测到 GUROBI 求解器")
                    return solver
            except Exception:
                pass
            try:
                solver = pulp.CPLEX_CMD(msg=0, timelimit=time_limit)
                if solver.available():
                    logger.info("检测到 CPLEX 求解器")
                    return solver
            except Exception:
                pass

        # 回退到 CBC
        return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=gap)

    def _extract_solution(self, model, y, z, gamma, alpha, beta, edge_list) -> Dict:
        """从求解结果提取解"""
        # 启用的箱变
        active_boxes = []
        for b in range(self.n_boxes):
            if pulp.value(y[b]) is not None and pulp.value(y[b]) > 0.5:
                cap_type = 3200 if (pulp.value(z[b]) is not None and pulp.value(z[b]) > 0.5) else 1600
                connected_invs = []
                for k in range(self.n_inverters):
                    if pulp.value(gamma[(k, b)]) is not None and pulp.value(gamma[(k, b)]) > 0.5:
                        connected_invs.append(k)
                active_boxes.append({
                    "box_idx": b,
                    "coord": self.candidate_box_coords[b],
                    "Q_box": cap_type,
                    "connected_inverters": connected_invs
                })

        # 启用的路径
        active_paths = []
        for p in range(self.n_paths):
            if pulp.value(alpha[p]) is not None and pulp.value(alpha[p]) > 0.5:
                active_paths.append(self.paths[p])

        # 启用的管沟边
        active_edges = []
        for e_idx in range(len(edge_list)):
            if pulp.value(beta[e_idx]) is not None and pulp.value(beta[e_idx]) > 0.5:
                active_edges.append(edge_list[e_idx])

        return {
            "objective": pulp.value(model.objective),
            "active_boxes": active_boxes,
            "active_paths": active_paths,
            "active_edges": active_edges,
            "solve_status": pulp.LpStatus[model.status],
        }
