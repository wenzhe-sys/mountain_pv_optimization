"""Benders 分解框架单元测试"""

import unittest
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithm.benders_decomposition import BendersDecomposition


class TestBendersDecomposition(unittest.TestCase):
    """测试 Benders 分解求解流程"""

    @classmethod
    def setUpClass(cls):
        """加载算例 r1"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(project_root, "data", "processed", "PV",
                            "public", "easy", "public_easy_r1.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cls.instance = json.load(f)
            cls.has_data = True
        else:
            cls.has_data = False

    def test_basic_solve(self):
        """基本求解：应产生可行输出"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        solver = BendersDecomposition(
            self.instance, max_iter=3, verbose=False
        )
        output = solver.optimize()

        self.assertIn("instance_id", output)
        self.assertIn("cut_result", output)
        self.assertIn("partition_result", output)
        self.assertIn("zone_summary", output)
        self.assertIn("constraint_satisfaction", output)

    def test_output_instance_id(self):
        """输出的 instance_id 应与输入一致"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        solver = BendersDecomposition(
            self.instance, max_iter=2, verbose=False
        )
        output = solver.optimize()
        self.assertEqual(output["instance_id"], "r1")

    def test_zone_count(self):
        """分区数应等于逆变器数"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        solver = BendersDecomposition(
            self.instance, max_iter=3, verbose=False
        )
        output = solver.optimize()
        n_inverters = self.instance["equipment_params"]["inverter"]["p"]
        self.assertEqual(len(output["zone_summary"]), n_inverters)

    def test_all_panels_assigned(self):
        """所有面板应出现在 partition_result 中"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        solver = BendersDecomposition(
            self.instance, max_iter=3, verbose=False
        )
        output = solver.optimize()
        n_panels = self.instance["instance_info"]["n_nodes"]
        self.assertEqual(len(output["partition_result"]), n_panels)

    def test_history_recorded(self):
        """迭代历史应被记录"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        solver = BendersDecomposition(
            self.instance, max_iter=3, verbose=False
        )
        output = solver.optimize()
        self.assertIn("optimization_history", output)
        self.assertGreater(len(output["optimization_history"]), 0)

    def test_constraint_satisfaction_keys(self):
        """约束满足字段应包含所有必要项"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        solver = BendersDecomposition(
            self.instance, max_iter=2, verbose=False
        )
        output = solver.optimize()
        cs = output["constraint_satisfaction"]
        self.assertIn("整数切割", cs)
        self.assertIn("分区连通性", cs)
        self.assertIn("逆变器容量约束", cs)
        self.assertIn("分区周长约束", cs)


if __name__ == "__main__":
    unittest.main()
