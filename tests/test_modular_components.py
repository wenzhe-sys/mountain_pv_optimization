"""
模块化分支定价组件单元测试
============================
覆盖重构后的模块二子组件：
1. BBTree - 分支定界树搜索
2. ColumnManager - 列管理与聚类
3. RMPSolver - 结构化对偶返回 & 变量固定
4. PricingSubproblem - 检验数计算
5. BranchAndPrice - B&P 策略集成
"""

import unittest
import os
import json
import sys
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _load_test_data():
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


# ============================================================
#  测试: BBTree 分支定界树
# ============================================================
class TestBBTree(unittest.TestCase):
    """分支定界树组件测试"""

    def test_create_root(self):
        from modules.module2.algorithm.bb_tree import BranchAndBoundTree
        tree = BranchAndBoundTree(n_sites=5, time_limit=10)
        root = tree.create_root()
        self.assertEqual(root.node_id, 0)
        self.assertEqual(root.depth, 0)
        self.assertEqual(tree.stats["nodes_created"], 1)

    def test_process_integer_solution(self):
        from modules.module2.algorithm.bb_tree import BranchAndBoundTree
        tree = BranchAndBoundTree(n_sites=3)
        root = tree.create_root()
        result = {
            "status": "Optimal",
            "objective": 100.0,
            "y_values": {0: 1.0, 1: 0.0, 2: 1.0},
        }
        action = tree.process_node_result(root, result)
        self.assertEqual(action, "integer")
        self.assertEqual(tree.global_ub, 100.0)

    def test_process_fractional_triggers_branch(self):
        from modules.module2.algorithm.bb_tree import BranchAndBoundTree
        tree = BranchAndBoundTree(n_sites=3)
        root = tree.create_root()
        result = {
            "status": "Optimal",
            "objective": 80.0,
            "y_values": {0: 0.7, 1: 0.3, 2: 1.0},
        }
        action = tree.process_node_result(root, result)
        self.assertEqual(action, "branch")

    def test_branch_creates_two_children(self):
        from modules.module2.algorithm.bb_tree import BranchAndBoundTree
        tree = BranchAndBoundTree(n_sites=3)
        root = tree.create_root()
        result = {
            "status": "Optimal",
            "objective": 80.0,
            "y_values": {0: 0.5, 1: 0.3, 2: 1.0},
        }
        tree.process_node_result(root, result)
        left, right = tree.branch(root)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertEqual(tree.stats["nodes_created"], 3)  # root + 2 children

    def test_pruning_by_upper_bound(self):
        from modules.module2.algorithm.bb_tree import BranchAndBoundTree
        tree = BranchAndBoundTree(n_sites=3)
        tree.global_ub = 50.0
        root = tree.create_root()
        result = {
            "status": "Optimal",
            "objective": 60.0,
            "y_values": {0: 0.5, 1: 0.5, 2: 0.0},
        }
        action = tree.process_node_result(root, result)
        self.assertEqual(action, "pruned")

    def test_infeasible_node(self):
        from modules.module2.algorithm.bb_tree import BranchAndBoundTree
        tree = BranchAndBoundTree(n_sites=3)
        root = tree.create_root()
        action = tree.process_node_result(root, None)
        self.assertEqual(action, "infeasible")


# ============================================================
#  测试: ColumnManager 列管理器
# ============================================================
class TestColumnManager(unittest.TestCase):

    def _make_path(self, pid, inv_idx, box_idx, length=10.0, ptype="inv_to_box"):
        return {
            "id": pid, "inv_idx": inv_idx, "box_idx": box_idx,
            "type": ptype, "length": length,
            "from": (0.0, 0.0), "to": (10.0, 10.0),
            "edges": [((0.0, 0.0), (10.0, 0.0)), ((10.0, 0.0), (10.0, 10.0))],
        }

    def test_add_and_get_active(self):
        from modules.module2.algorithm.column_manager import ColumnManager
        cm = ColumnManager(n_inverters=2, n_sites=2)
        cm.initialize([])
        p1 = self._make_path("p1", 0, 0)
        p2 = self._make_path("p2", 1, 1)
        cm.add_paths([p1, p2])
        active = cm.get_active_paths()
        self.assertEqual(len(active), 2)

    def test_no_duplicate_paths(self):
        from modules.module2.algorithm.column_manager import ColumnManager
        cm = ColumnManager(n_inverters=2, n_sites=2)
        cm.initialize([])
        p1 = self._make_path("p1", 0, 0)
        p1_dup = self._make_path("p1_dup", 0, 0)
        cm.add_paths([p1])
        cm.add_paths([p1_dup])
        # Should not add duplicate (same inv_idx, box_idx, type)
        self.assertEqual(len(cm.all_paths), 1)


# ============================================================
#  测试: RMPSolver 对偶 & 变量固定
# ============================================================
class TestRMPSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import pulp
            cls.has_pulp = True
        except ImportError:
            cls.has_pulp = False

    def test_structured_duals_returned(self):
        """LP松弛应返回 dual_assignment 和 dual_trench"""
        if not self.has_pulp:
            self.skipTest("PuLP 未安装")
        from modules.module2.algorithm.rmp_solver import RMPSolver
        solver = RMPSolver(
            c_box={0: 50000, 1: 80000}, c_install={0: 10000, 1: 15000},
            c1=100, c2=80, c3=50, Q_box_inv={0: 5, 1: 10},
            N_max=4, Q_substation=100, substation_coord=(0.0, 0.0)
        )
        inv = [{"id": 0, "centroid": (10.0, 10.0)}]
        boxes = [(20.0, 20.0)]
        path = {
            "id": "test_p", "inv_idx": 0, "box_idx": 0,
            "type": "inv_to_box", "length": 20.0,
            "edges": [(10.0, 10.0, 20.0, 20.0)],
        }
        sub_path = {
            "id": "test_sub", "inv_idx": None, "box_idx": 0,
            "type": "box_to_sub", "length": 30.0,
            "edges": [(20.0, 20.0, 0.0, 0.0)],
        }
        edges_info = {
            (10.0, 10.0, 20.0, 20.0): {"length": 20.0},
            (20.0, 20.0, 0.0, 0.0): {"length": 30.0},
        }
        result = solver.build_and_solve_rmp(inv, boxes, [path, sub_path], edges_info, is_relaxation=True)
        self.assertEqual(result["status"], "optimal")
        self.assertIn("dual_assignment", result)
        self.assertIn("dual_trench", result)
        self.assertIn("y_values", result)

    def test_fixed_to_zero_makes_infeasible(self):
        """将唯一可用箱变固定为0应导致不可行"""
        if not self.has_pulp:
            self.skipTest("PuLP 未安装")
        from modules.module2.algorithm.rmp_solver import RMPSolver
        solver = RMPSolver(
            c_box={0: 50000, 1: 80000}, c_install={0: 10000, 1: 15000},
            c1=100, c2=80, c3=50, Q_box_inv={0: 5, 1: 10},
            N_max=4, Q_substation=100, substation_coord=(0.0, 0.0)
        )
        inv = [{"id": 0}]
        boxes = [(20.0, 20.0)]
        path = {"id": "p0", "inv_idx": 0, "box_idx": 0, "type": "inv_to_box", "length": 10.0, "edges": []}
        sub_path = {"id": "s0", "inv_idx": None, "box_idx": 0, "type": "box_to_sub", "length": 10.0, "edges": []}
        result = solver.build_and_solve_rmp(inv, boxes, [path, sub_path], {},
                                            is_relaxation=False, fixed_to_zero={0})
        self.assertEqual(result["status"], "infeasible")


# ============================================================
#  测试: BranchAndPrice 策略集成
# ============================================================
class TestBranchAndPriceStrategies(unittest.TestCase):
    """B&P 各策略集成测试"""

    @classmethod
    def setUpClass(cls):
        cls.instance, cls.m1 = _load_test_data()

    def test_bp_strategy_returns_result(self):
        """branch_and_price 策略应返回有效结果"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.branch_and_price import BranchAndPrice
        solver = BranchAndPrice(self.instance, self.m1)
        result = solver.optimize(strategy="branch_and_price", time_limit=60)
        self.assertIsNotNone(result)
        self.assertIn("total_cost", result)
        self.assertGreater(result["total_cost"], 0)
        # Should include perf_stats and convergence_history
        self.assertIn("perf_stats", result)
        self.assertIn("convergence_history", result)
        self.assertIn("bb_summary", result)

    def test_milp_strategy_returns_result(self):
        """milp 策略应返回有效结果"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.branch_and_price import BranchAndPrice
        solver = BranchAndPrice(self.instance, self.m1)
        result = solver.optimize(strategy="milp", time_limit=60)
        self.assertIsNotNone(result)
        self.assertIn("total_cost", result)
        self.assertIn("perf_stats", result)

    def test_matheuristic_strategy(self):
        """matheuristic 策略应返回有效结果"""
        if self.instance is None:
            self.skipTest("算例文件不存在")
        from algorithm.branch_and_price import BranchAndPrice
        solver = BranchAndPrice(self.instance, self.m1)
        result = solver.optimize(strategy="matheuristic", time_limit=30)
        self.assertIsNotNone(result)
        for key in ["equipment_selection", "cable_routes", "total_cost"]:
            self.assertIn(key, result)


# ============================================================
#  测试: LagrangianRelaxation 拉格朗日松弛
# ============================================================
class TestLagrangianRelaxation(unittest.TestCase):

    def test_lagrangian_returns_lower_bound(self):
        """拉格朗日松弛应返回有效下界"""
        try:
            import pulp
        except ImportError:
            self.skipTest("PuLP 未安装")
        from modules.module2.algorithm.lagrangian import LagrangianRelaxation
        # Minimal problem setup
        paths = [
            {"id": 0, "inv_idx": 0, "box_idx": 0, "type": "inv_to_box",
             "length": 10.0, "edges": [(0, 0, 5, 5)]},
            {"id": 1, "inv_idx": None, "box_idx": 0, "type": "box_to_sub",
             "length": 15.0, "edges": [(5, 5, 10, 10)]},
        ]
        edges = {
            (0, 0, 5, 5): {"length": 10.0},
            (5, 5, 10, 10): {"length": 15.0},
        }
        lr = LagrangianRelaxation(
            n_inverters=1, n_sites=1, paths=paths, edges=edges,
            Q_box_inv={1600: 5, 3200: 10}, N_max=4,
            c_box={"1600": 50000, "3200": 80000},
            c_install={"1600": 10000, "3200": 15000},
            c2=80, c3=50,
        )
        result = lr.solve(max_iterations=5)
        self.assertIn("lower_bound", result)
        self.assertIn("history", result)

    def test_lagrangian_integration_in_bp(self):
        """B&P 编排器应成功调用拉格朗日松弛"""
        inst, m1 = _load_test_data()
        if inst is None:
            self.skipTest("算例文件不存在")
        from algorithm.branch_and_price import BranchAndPrice
        solver = BranchAndPrice(inst, m1)
        solver.column_manager.initialize([])
        heur_sol = solver.heuristic.solve_kmeans_heuristic()
        solver._initialize_columns(heur_sol)
        lb = solver._compute_lagrangian_bound()
        # Should not crash and return a float
        self.assertIsInstance(lb, float)


# ============================================================
#  测试: ValidInequalities 有效不等式
# ============================================================
class TestValidInequalities(unittest.TestCase):

    def test_symmetry_breaking(self):
        """对称消除约束应减少等价解"""
        try:
            import pulp
        except ImportError:
            self.skipTest("PuLP 未安装")
        from modules.module2.algorithm.rmp_solver import RMPSolver
        solver = RMPSolver(
            c_box={0: 50000, 1: 80000}, c_install={0: 10000, 1: 15000},
            c1=100, c2=80, c3=50, Q_box_inv={0: 5, 1: 10},
            N_max=4, Q_substation=100, substation_coord=(0.0, 0.0)
        )
        inv = [{"id": 0}, {"id": 1}]
        boxes = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
        paths = [
            {"id": "p0", "inv_idx": 0, "box_idx": 0, "type": "inv_to_box", "length": 10.0, "edges": [(0, 0, 10, 10)]},
            {"id": "p1", "inv_idx": 1, "box_idx": 1, "type": "inv_to_box", "length": 15.0, "edges": [(0, 0, 20, 20)]},
            {"id": "s0", "inv_idx": None, "box_idx": 0, "type": "box_to_sub", "length": 10.0, "edges": [(10, 10, 0, 0)]},
            {"id": "s1", "inv_idx": None, "box_idx": 1, "type": "box_to_sub", "length": 20.0, "edges": [(20, 20, 0, 0)]},
            {"id": "s2", "inv_idx": None, "box_idx": 2, "type": "box_to_sub", "length": 30.0, "edges": [(30, 30, 0, 0)]},
        ]
        edges_info = {
            (0, 0, 10, 10): {"length": 10.0},
            (0, 0, 20, 20): {"length": 20.0},
            (10, 10, 0, 0): {"length": 10.0},
            (20, 20, 0, 0): {"length": 20.0},
            (30, 30, 0, 0): {"length": 30.0},
        }
        result = solver.build_and_solve_rmp(inv, boxes, paths, edges_info, is_relaxation=False)
        # Should solve (symmetry constraints don't make it infeasible)
        self.assertEqual(result["status"], "optimal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
