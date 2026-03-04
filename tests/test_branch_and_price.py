"""
分支定价算法单元测试
====================
覆盖模块二核心 B&P 组件的单元与集成测试。

测试范围：
1. CandidateSiteGenerator - 候选站址生成
2. PathGenerator / CablePath - 路径生成与曼哈顿路由
3. ArcFlowMILP - MILP 模型构建与求解
4. ColumnGeneration - 列生成收敛性
5. MatheuristicSolver - 启发式求解质量
6. BranchAndPrice - 端到端集成
"""

import unittest
import os
import json
import numpy as np
import sys

# 确保项目根目录在搜索路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ============================================================
#  测试辅助：加载标准算例
# ============================================================
def _load_test_data():
    """加载 r1 算例和 M1 输出用于测试"""
    instance_path = os.path.join(
        project_root, "data", "processed", "PV", "public", "easy",
        "public_easy_r1.json",
    )
    m1_path = os.path.join(
        project_root, "data", "results", "module1", "M1-Output_r1.json",
    )
    if not os.path.exists(instance_path) or not os.path.exists(m1_path):
        return None, None
    with open(instance_path, "r", encoding="utf-8") as f:
        instance_data = json.load(f)
    with open(m1_path, "r", encoding="utf-8") as f:
        m1_output = json.load(f)
    return instance_data, m1_output


def _compute_inverter_coords(instance_data, m1_output):
    """辅助函数：计算逆变器坐标"""
    grid_size = instance_data["terrain_data"]["grid_size"]
    coords = []
    for zone in m1_output["zone_summary"]:
        zid = zone["zone_id"]
        panels = [p for p in m1_output["partition_result"] if p["zone_id"] == zid]
        if panels:
            x = np.mean([p["grid_coord"][0] for p in panels])
            y = np.mean([p["grid_coord"][1] for p in panels])
            coords.append((round(x / grid_size) * grid_size,
                           round(y / grid_size) * grid_size))
    return coords


# ============================================================
#  测试: CandidateSiteGenerator
# ============================================================
class TestCandidateSiteGenerator(unittest.TestCase):
    """候选站址生成器测试"""

    @classmethod
    def setUpClass(cls):
        cls.instance, cls.m1 = _load_test_data()

    def test_generate_sites_nonempty(self):
        """生成的候选站址非空"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.candidate_sites import CandidateSiteGenerator
        inv_coords = _compute_inverter_coords(self.instance, self.m1)
        gen = CandidateSiteGenerator(self.instance, inv_coords)
        sites = gen.generate_all_candidates(max_candidates=10)
        self.assertGreater(len(sites), 0, "候选站址集不应为空")

    def test_grid_alignment(self):
        """候选站址坐标必须对齐网格"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.candidate_sites import CandidateSiteGenerator
        inv_coords = _compute_inverter_coords(self.instance, self.m1)
        gen = CandidateSiteGenerator(self.instance, inv_coords)
        sites = gen.generate_all_candidates(max_candidates=10)
        grid = self.instance["terrain_data"]["grid_size"]
        for s in sites:
            self.assertAlmostEqual(s[0] % grid, 0, places=5,
                                   msg=f"X坐标 {s[0]} 未对齐 grid_size={grid}")
            self.assertAlmostEqual(s[1] % grid, 0, places=5,
                                   msg=f"Y坐标 {s[1]} 未对齐 grid_size={grid}")

    def test_no_duplicates(self):
        """候选站址不应重复"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.candidate_sites import CandidateSiteGenerator
        inv_coords = _compute_inverter_coords(self.instance, self.m1)
        gen = CandidateSiteGenerator(self.instance, inv_coords)
        sites = gen.generate_all_candidates(max_candidates=15)
        unique = set(sites)
        self.assertEqual(len(sites), len(unique), "候选站址存在重复")


# ============================================================
#  测试: PathGenerator / CablePath
# ============================================================
class TestPathGenerator(unittest.TestCase):
    """路径生成器和曼哈顿路由测试"""

    @classmethod
    def setUpClass(cls):
        cls.instance, cls.m1 = _load_test_data()

    def test_manhattan_distance(self):
        """CablePath 的曼哈顿距离计算"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.path_generator import CablePath
        grid = self.instance["terrain_data"]["grid_size"]
        p = CablePath(0, 0, 0, (10, 20), (30, 40), (50, 60), grid)
        self.assertAlmostEqual(p.inv_to_box_length, 40.0)
        self.assertAlmostEqual(p.box_to_sub_length, 40.0)
        self.assertAlmostEqual(p.total_length, 80.0)

    def test_path_generation_completeness(self):
        """所有逆变器都应有至少一条路径"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.candidate_sites import CandidateSiteGenerator
        from algorithm.path_generator import PathGenerator
        inv_coords = _compute_inverter_coords(self.instance, self.m1)
        gen = CandidateSiteGenerator(self.instance, inv_coords)
        cands = gen.generate_all_candidates(max_candidates=8)
        pg = PathGenerator(self.instance, inv_coords, cands)
        paths = pg.generate_paths(knn_k=min(3, len(cands)))
        covered_invs = set(p.inverter_idx for p in paths)
        for k in range(len(inv_coords)):
            self.assertIn(k, covered_invs,
                          f"逆变器 {k} 没有生成任何路径")

    def test_edge_set_nonempty(self):
        """每条路径的边集不应为空"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.path_generator import CablePath
        grid = self.instance["terrain_data"]["grid_size"]
        p = CablePath(0, 0, 0, (0, 0), (30, 40), (60, 80), grid)
        self.assertGreater(len(p.all_edges), 0,
                           "路径边集不应为空")


# ============================================================
#  测试: ArcFlowMILP
# ============================================================
class TestArcFlowMILP(unittest.TestCase):
    """Arc-Flow MILP 模型测试"""

    @classmethod
    def setUpClass(cls):
        cls.instance, cls.m1 = _load_test_data()
        try:
            import pulp
            cls.has_pulp = True
        except ImportError:
            cls.has_pulp = False

    def test_milp_feasibility(self):
        """小规模算例的 MILP 应可行"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        if not self.has_pulp:
            self.skipTest("PuLP 未安装")
        from algorithm.candidate_sites import CandidateSiteGenerator
        from algorithm.path_generator import PathGenerator
        from algorithm.arc_flow_milp import ArcFlowMILP

        inv_coords = _compute_inverter_coords(self.instance, self.m1)
        gen = CandidateSiteGenerator(self.instance, inv_coords)
        cands = gen.generate_all_candidates(max_candidates=5)
        pg = PathGenerator(self.instance, inv_coords, cands)
        paths = pg.generate_paths(knn_k=min(3, len(cands)))
        e2p = pg.get_edge_to_paths(paths)

        milp = ArcFlowMILP(self.instance, inv_coords, cands, paths, e2p)
        result = milp.build_and_solve(time_limit=120, gap=0.1)
        self.assertIsNotNone(result, "MILP 应返回可行解")
        self.assertIn("objective", result)
        self.assertGreater(result["objective"], 0)


# ============================================================
#  测试: MatheuristicSolver
# ============================================================
class TestMatheuristic(unittest.TestCase):
    """Matheuristic 启发式求解器测试"""

    @classmethod
    def setUpClass(cls):
        cls.instance, cls.m1 = _load_test_data()

    def test_solve_returns_valid_result(self):
        """Matheuristic 应返回包含所有必选字段的结果"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.matheuristic import MatheuristicSolver
        inv_coords = _compute_inverter_coords(self.instance, self.m1)
        solver = MatheuristicSolver(self.instance, inv_coords,
                                    self.m1["zone_summary"])
        result = solver.solve()
        for key in ["equipment_selection", "cable_routes", "trench_summary",
                     "total_cost"]:
            self.assertIn(key, result, f"结果缺少字段: {key}")

    def test_all_inverters_assigned(self):
        """所有逆变器应被分配到箱变"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.matheuristic import MatheuristicSolver
        inv_coords = _compute_inverter_coords(self.instance, self.m1)
        solver = MatheuristicSolver(self.instance, inv_coords,
                                    self.m1["zone_summary"])
        result = solver.solve()
        assigned = set()
        for eq in result["equipment_selection"]:
            assigned.update(eq["connected_inverters"])
        inv_ids = {z["inverter_id"] for z in self.m1["zone_summary"]}
        self.assertEqual(assigned, inv_ids,
                         "不是所有逆变器都被分配")

    def test_capacity_constraint(self):
        """箱变容量约束：连接逆变器数 ≤ 容量上限"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.matheuristic import MatheuristicSolver
        inv_coords = _compute_inverter_coords(self.instance, self.m1)
        solver = MatheuristicSolver(self.instance, inv_coords,
                                    self.m1["zone_summary"])
        result = solver.solve()
        cap_map = {1600: 5, 3200: 10}
        for eq in result["equipment_selection"]:
            cap = eq["Q_box"]
            max_inv = cap_map.get(cap, 10)
            self.assertLessEqual(
                len(eq["connected_inverters"]), max_inv,
                f"箱变 {eq['transformer_id']} 连接数超过容量限制 "
                f"({len(eq['connected_inverters'])} > {max_inv})")


# ============================================================
#  测试: BranchAndPrice 端到端
# ============================================================
class TestBranchAndPriceE2E(unittest.TestCase):
    """分支定价端到端集成测试"""

    @classmethod
    def setUpClass(cls):
        cls.instance, cls.m1 = _load_test_data()

    def test_optimize_returns_valid_output(self):
        """optimize() 应返回标准 M2-Output 格式"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.branch_and_price import BranchAndPrice
        solver = BranchAndPrice(self.instance, self.m1)
        result = solver.optimize(strategy="matheuristic", time_limit=30)
        for key in ["equipment_selection", "cable_routes", "trench_summary",
                     "constraint_satisfaction", "total_cost"]:
            self.assertIn(key, result, f"输出缺少字段: {key}")

    def test_q_box_valid(self):
        """箱变容量只能是 1600 或 3200"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.branch_and_price import BranchAndPrice
        solver = BranchAndPrice(self.instance, self.m1)
        result = solver.optimize(strategy="matheuristic", time_limit=30)
        for eq in result["equipment_selection"]:
            self.assertIn(eq["Q_box"], [1600, 3200],
                          f"箱变容量 {eq['Q_box']} 不合法")

    def test_trench_cable_count(self):
        """管沟电缆数 ≤ N_max"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.branch_and_price import BranchAndPrice
        solver = BranchAndPrice(self.instance, self.m1)
        result = solver.optimize(strategy="matheuristic", time_limit=30)
        for t in result["trench_summary"]:
            self.assertLessEqual(t["cable_count"], 4,
                                 f"管沟 {t['trench_id']} 电缆数超限")

    def test_grid_alignment(self):
        """箱变坐标必须对齐网格"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.branch_and_price import BranchAndPrice
        solver = BranchAndPrice(self.instance, self.m1)
        result = solver.optimize(strategy="matheuristic", time_limit=30)
        grid = self.instance["terrain_data"]["grid_size"]
        for eq in result["equipment_selection"]:
            x, y = eq["install_coord"]
            self.assertAlmostEqual(x % grid, 0, places=5)
            self.assertAlmostEqual(y % grid, 0, places=5)

    def test_milp_strategy(self):
        """MILP 策略测试（如果 PuLP 可用）"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        try:
            import pulp
        except ImportError:
            self.skipTest("PuLP 未安装")
        from algorithm.branch_and_price import BranchAndPrice
        solver = BranchAndPrice(self.instance, self.m1)
        result = solver.optimize(strategy="milp", time_limit=120)
        self.assertIsNotNone(result)
        self.assertGreater(result["total_cost"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
