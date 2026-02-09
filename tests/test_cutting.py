"""切割主问题 MIP 模型单元测试"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cutting_master import CuttingMasterProblem, estimate_demand


class TestCuttingMasterProblem(unittest.TestCase):
    """测试切割主问题 MIP 求解"""

    def test_basic_solve(self):
        """基础求解：需求 10 块 6.0m 规格面板"""
        solver = CuttingMasterProblem(D=12.0)
        demand = {6.0: 10}
        result = solver.solve(demand)

        self.assertEqual(result.status, "Optimal")
        self.assertEqual(result.materials_used, 5)  # 12/6=2块/原材料, 需10块→5个
        self.assertGreaterEqual(result.utilization_rate, 0.99)

    def test_mixed_demand(self):
        """混合规格需求"""
        solver = CuttingMasterProblem(D=12.0)
        demand = {2.0: 20, 4.0: 10}
        result = solver.solve(demand)

        self.assertEqual(result.status, "Optimal")
        # 验证需求满足
        for spec, info in result.demand_satisfaction.items():
            self.assertTrue(info["satisfied"], f"规格 {spec} 需求未满足")

    def test_integer_constraint(self):
        """验证切割结果为非负整数"""
        solver = CuttingMasterProblem(D=12.0)
        demand = {6.0: 7}
        result = solver.solve(demand)

        for material in result.cut_result:
            for cut in material["cuts"]:
                self.assertIsInstance(cut["quantity"], int)
                self.assertGreaterEqual(cut["quantity"], 0)

    def test_spec_validation(self):
        """验证切割规格合法性（必须是 2.0 的整数倍）"""
        with self.assertRaises(AssertionError):
            CuttingMasterProblem(D=12.0, t_l_options=[3.0, 5.0])

    def test_material_length_constraint(self):
        """验证原材料长度约束：切割总长 <= D"""
        solver = CuttingMasterProblem(D=12.0)
        demand = {6.0: 5}
        result = solver.solve(demand)

        for material in result.cut_result:
            if material["is_used"]:
                total_length = sum(c["spec_l"] * c["quantity"] for c in material["cuts"])
                self.assertLessEqual(total_length, 12.0 + 0.01)

    def test_output_format(self):
        """验证输出格式符合 M1-Output 规范"""
        solver = CuttingMasterProblem(D=12.0)
        demand = {6.0: 4}
        result = solver.solve(demand)

        for material in result.cut_result:
            self.assertIn("material_id", material)
            self.assertIn("is_used", material)
            self.assertIn("cuts", material)
            for cut in material["cuts"]:
                self.assertIn("spec_l", cut)
                self.assertIn("quantity", cut)


class TestEstimateDemand(unittest.TestCase):
    """测试需求估算"""

    def test_default_demand(self):
        demand = estimate_demand(108, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        total = sum(demand.values())
        self.assertEqual(total, 108)


if __name__ == "__main__":
    unittest.main()
