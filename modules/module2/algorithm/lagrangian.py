"""
拉格朗日松弛模块
================
松弛复杂约束，提供分支定界的下界估计。

松弛策略：
- 松弛共沟约束(式17)和容量约束(式16)
- 次梯度法更新拉格朗日乘子
- 提供 LP 下界用于分支定界剪枝
"""

import numpy as np
import pulp
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class LagrangianRelaxation:
    """
    拉格朗日松弛求解器

    松弛共沟约束(式17)和箱变容量约束(式16)到目标函数中，
    通过次梯度法迭代更新乘子，提供下界估计。

    L(λ,μ) = min cx + Σ_e μ_e(Σ_p α_p - N_max·β_e) + Σ_b λ_b(Σ_k γ_kb - Q_b)
    s.t. 唯一分配约束(式12)、路径一致性(式14)、布线挖沟协同(式15)
    """

    def __init__(self, n_inverters: int, n_sites: int,
                 paths: List[Dict], edges: Dict,
                 Q_box_inv: Dict, N_max: int,
                 c_box: Dict, c_install: Dict,
                 c2: float, c3: float):
        self.n_inv = n_inverters
        self.n_sites = n_sites
        self.paths = paths
        self.edges = edges
        self.Q_box_inv = Q_box_inv
        self.N_max = N_max
        self.c_box = c_box
        self.c_install = c_install
        self.c2 = c2
        self.c3 = c3

        # 拉格朗日乘子
        edge_keys = list(edges.keys())
        self.mu = {ek: 0.0 for ek in edge_keys}  # 共沟约束乘子
        self.lam = {b: 0.0 for b in range(n_sites)}  # 容量约束乘子

        # 收敛历史
        self.history: List[Dict] = []
        self.best_lb = -float('inf')

    def _build_solver(self):
        """Pick best available solver for relaxed subproblem."""
        candidates = [
            ("GUROBI", lambda: pulp.GUROBI(msg=0)),
            ("CPLEX_PY", lambda: pulp.CPLEX_PY(msg=0)),
            ("SCIP_PY", lambda: pulp.SCIP_PY(msg=0)),
            ("HiGHS", lambda: pulp.HiGHS(msg=0)),
            ("PULP_CBC_CMD", lambda: pulp.PULP_CBC_CMD(msg=0, timeLimit=30)),
        ]
        for name, factory in candidates:
            try:
                solver = factory()
                if solver is not None and solver.available():
                    logger.info(f"Lagrangian using solver: {name}")
                    return solver
            except Exception:
                continue

        return pulp.PULP_CBC_CMD(msg=0, timeLimit=30)

    def solve(self, max_iterations: int = 30, step_size: float = 1.0,
              step_decay: float = 0.95) -> Dict:
        """
        次梯度法求解拉格朗日对偶问题。

        Parameters
        ----------
        max_iterations : int
            最大迭代次数
        step_size : float
            初始步长
        step_decay : float
            步长衰减率

        Returns
        -------
        Dict
            {lower_bound, mu, lam, history}
        """
        logger.info(f"【拉格朗日松弛】开始（max_iter={max_iterations}）")

        current_step = step_size

        for iteration in range(max_iterations):
            # 1. 固定乘子，求解松弛问题
            result = self._solve_relaxed_problem()

            if result is None:
                logger.warning(f"第 {iteration + 1} 轮：松弛问题不可行")
                current_step *= step_decay
                continue

            lb = result["objective"]
            self.best_lb = max(self.best_lb, lb)

            # 2. 计算次梯度
            sg_mu = result["subgradient_mu"]
            sg_lam = result["subgradient_lam"]

            # 3. 更新乘子（次梯度法）
            sg_norm_sq = (sum(v ** 2 for v in sg_mu.values()) +
                          sum(v ** 2 for v in sg_lam.values()))

            if sg_norm_sq < 1e-12:
                logger.info(f"第 {iteration + 1} 轮：次梯度为零，收敛")
                break

            for ek in self.mu:
                self.mu[ek] = max(0.0, self.mu[ek] + current_step * sg_mu.get(ek, 0.0))
            for b in self.lam:
                self.lam[b] = max(0.0, self.lam[b] + current_step * sg_lam.get(b, 0.0))

            current_step *= step_decay

            self.history.append({
                "iteration": iteration + 1,
                "lower_bound": lb,
                "best_lb": self.best_lb,
                "step_size": current_step,
                "sg_norm": float(np.sqrt(sg_norm_sq)),
            })

            if (iteration + 1) % 10 == 0:
                logger.info(f"第 {iteration + 1} 轮：LB = {lb:.2f}，"
                            f"best = {self.best_lb:.2f}，step = {current_step:.4f}")

        logger.info(f"【拉格朗日松弛】完成，best LB = {self.best_lb:.2f}")

        return {
            "lower_bound": self.best_lb,
            "mu": dict(self.mu),
            "lam": dict(self.lam),
            "history": self.history,
        }

    def _solve_relaxed_problem(self) -> Optional[Dict]:
        """
        求解拉格朗日松弛问题（保留式12/14/15，松弛式16/17）

        目标 = 原始目标 + Σ μ_e·(违反量) + Σ λ_b·(违反量)
        """
        n_inv = self.n_inv
        n_sites = self.n_sites
        n_paths = len(self.paths)
        edge_keys = list(self.edges.keys())
        edge_to_idx = {ek: i for i, ek in enumerate(edge_keys)}
        n_edges = len(edge_keys)

        prob = pulp.LpProblem("LR_Relaxed", pulp.LpMinimize)

        path_ids = [p["id"] for p in self.paths]
        alpha = pulp.LpVariable.dicts("a", path_ids, cat="Binary")
        beta = pulp.LpVariable.dicts("b", range(n_edges), cat="Binary")
        gamma = pulp.LpVariable.dicts("g",
                                       [(k, b) for k in range(n_inv) for b in range(n_sites)],
                                       cat="Binary")
        y = pulp.LpVariable.dicts("y", range(n_sites), cat="Binary")
        z = pulp.LpVariable.dicts("z", range(n_sites), cat="Binary")

        c2w = self.c2 / 10000.0
        c3w = self.c3 / 10000.0
        c_box_1600 = self.c_box["1600"]
        c_box_3200 = self.c_box["3200"]
        c_inst_1600 = self.c_install["1600"]
        c_inst_3200 = self.c_install["3200"]

        # --- 原始目标 ---
        box_cost = pulp.lpSum([
            y[b] * (c_box_1600 + c_inst_1600) +
            z[b] * ((c_box_3200 - c_box_1600) + (c_inst_3200 - c_inst_1600))
            for b in range(n_sites)
        ])
        cable_cost = pulp.lpSum([alpha[p["id"]] * p["length"] * c2w for p in self.paths])
        trench_cost = pulp.lpSum([beta[edge_to_idx[ek]] * self.edges[ek]["length"] * c3w
                                   for ek in edge_keys])

        # --- 拉格朗日惩罚项 ---
        edge_to_paths_map = defaultdict(list)
        for p in self.paths:
            for ek in p["edges"]:
                edge_to_paths_map[ek].append(p["id"])

        # 共沟约束惩罚
        lr_trench_penalty = pulp.lpSum([
            self.mu[ek] * (
                pulp.lpSum([alpha[pid] for pid in edge_to_paths_map.get(ek, [])]) -
                self.N_max * beta[edge_to_idx[ek]]
            )
            for ek in edge_keys if ek in self.mu
        ])

        # 容量约束惩罚
        Q_max = self.Q_box_inv[3200]
        lr_cap_penalty = pulp.lpSum([
            self.lam[b] * (
                pulp.lpSum([gamma[(k, b)] for k in range(n_inv)]) - Q_max * y[b]
            )
            for b in range(n_sites)
        ])

        prob += box_cost + cable_cost + trench_cost + lr_trench_penalty + lr_cap_penalty

        # --- 保留的约束 ---
        # 式12：唯一分配
        for k in range(n_inv):
            prob += pulp.lpSum([gamma[(k, b)] for b in range(n_sites)]) == 1

        # 式14：路径一致性
        for k in range(n_inv):
            for b in range(n_sites):
                paths_kb = [p for p in self.paths
                            if p["type"] == "inv_to_box" and p["inv_idx"] == k and p["box_idx"] == b]
                if paths_kb:
                    prob += pulp.lpSum([alpha[p["id"]] for p in paths_kb]) == gamma[(k, b)]

        # 式15：布线挖沟协同
        for p in self.paths:
            for ek in p["edges"]:
                prob += alpha[p["id"]] <= beta[edge_to_idx[ek]]

        # γ ≤ y, z ≤ y
        for k in range(n_inv):
            for b in range(n_sites):
                prob += gamma[(k, b)] <= y[b]
        for b in range(n_sites):
            prob += z[b] <= y[b]

        # 箱变→升压站
        for b in range(n_sites):
            psub = [p for p in self.paths if p["type"] == "box_to_sub" and p["box_idx"] == b]
            if psub:
                prob += pulp.lpSum([alpha[p["id"]] for p in psub]) == y[b]

        # --- 求解 ---
        solver = self._build_solver()
        prob.solve(solver)

        if prob.status != pulp.constants.LpStatusOptimal:
            return None

        # --- 提取次梯度 ---
        sg_mu = {}
        for ek in edge_keys:
            pids = edge_to_paths_map.get(ek, [])
            flow = sum(pulp.value(alpha[pid]) or 0 for pid in pids)
            trench_opened = pulp.value(beta[edge_to_idx[ek]]) or 0
            sg_mu[ek] = flow - self.N_max * trench_opened

        sg_lam = {}
        for b in range(n_sites):
            conn = sum(pulp.value(gamma[(k, b)]) or 0 for k in range(n_inv))
            cap = Q_max * (pulp.value(y[b]) or 0)
            sg_lam[b] = conn - cap

        return {
            "objective": pulp.value(prob.objective),
            "subgradient_mu": sg_mu,
            "subgradient_lam": sg_lam,
        }

    def get_edge_duals(self) -> Dict[Tuple, float]:
        """获取当前共沟约束的拉格朗日乘子（用于定价子问题）"""
        return dict(self.mu)

    def get_lower_bound(self) -> float:
        return self.best_lb
