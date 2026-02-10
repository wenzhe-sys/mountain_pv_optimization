"""
模块一端到端测试

验证 CuttingPartitionModel 的完整流程：
  数据加载 → Benders 求解 → M1-Output 校验

依据：《算例处理规范与模块接口协议》4.1 节校验规则
  E101: 必选字段完整性
  E102: 切割长度整数列约束 (spec_l 为 2.0 的整数倍)
  E103: 分区面板数范围 [18, 26]
  E104: ID 唯一性
"""

import unittest
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_cutting_partition import CuttingPartitionModel, validate_m1_output
from utils.load_instance import InstanceLoader


class TestModel1EndToEnd(unittest.TestCase):
    """模块一端到端测试（使用算例 r1）"""

    @classmethod
    def setUpClass(cls):
        """加载并运行模块一"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.instance_path = os.path.join(
            project_root, "data", "processed", "PV", "public", "easy", "public_easy_r1.json"
        )
        if os.path.exists(cls.instance_path):
            cls.model = CuttingPartitionModel(cls.instance_path)
            cls.output = cls.model.run(verbose=False, max_iter=3)
            cls.has_data = True

            # 加载原始算例
            loader = InstanceLoader()
            cls.instance = loader.load_instance("r1")
        else:
            cls.has_data = False

    def test_m1_output_fields(self):
        """E101: M1-Output 必选字段完整性"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        required = ["instance_id", "cut_result", "partition_result",
                     "zone_summary", "constraint_satisfaction"]
        for field in required:
            self.assertIn(field, self.output, f"缺失字段 {field}（E101）")

    def test_cutting_integer_constraint(self):
        """E102: 切割长度为 2.0 的整数倍"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        valid_specs = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0}
        for material in self.output["cut_result"]:
            if material["is_used"]:
                for cut in material["cuts"]:
                    self.assertIn(cut["spec_l"], valid_specs,
                                  f"非法切割长度 {cut['spec_l']}（E102）")

    def test_pva_count_range(self):
        """E103: 分区面板数在 [18, 26] 范围"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        for zone in self.output["zone_summary"]:
            pva_count = zone["pva_count"]
            self.assertTrue(
                18 <= pva_count <= 26,
                f"分区 {zone['zone_id']} 面板数 {pva_count} 超出范围（E103）"
            )

    def test_unique_panel_ids(self):
        """E104: 面板 ID 唯一"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        panel_ids = [p["panel_id"] for p in self.output["partition_result"]]
        self.assertEqual(len(panel_ids), len(set(panel_ids)), "面板 ID 重复（E104）")

    def test_unique_zone_ids(self):
        """E104: 分区 ID 唯一"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        zone_ids = [z["zone_id"] for z in self.output["zone_summary"]]
        self.assertEqual(len(zone_ids), len(set(zone_ids)), "分区 ID 重复（E104）")

    def test_unique_inverter_ids(self):
        """E104: 逆变器 ID 唯一"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        inv_ids = [z["inverter_id"] for z in self.output["zone_summary"]]
        self.assertEqual(len(inv_ids), len(set(inv_ids)), "逆变器 ID 重复（E104）")

    def test_all_panels_assigned(self):
        """所有面板应出现在分区结果中"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        n_panels = self.instance["instance_info"]["n_nodes"]
        self.assertEqual(len(self.output["partition_result"]), n_panels)

    def test_zone_count_matches_inverters(self):
        """分区数应等于逆变器数"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        n_inverters = self.instance["equipment_params"]["inverter"]["p"]
        self.assertEqual(len(self.output["zone_summary"]), n_inverters)

    def test_constraint_satisfaction_structure(self):
        """约束满足字段应包含所有必要项"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        cs = self.output["constraint_satisfaction"]
        self.assertIn("整数切割", cs)
        self.assertIn("分区连通性", cs)
        self.assertIn("逆变器容量约束", cs)
        self.assertIn("分区周长约束", cs)

    def test_validate_function(self):
        """独立校验函数测试"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        result = validate_m1_output(self.output)
        self.assertIn("is_valid", result)
        self.assertIn("errors", result)
        self.assertIn("warnings", result)

    def test_output_json_serializable(self):
        """输出应可 JSON 序列化"""
        if not self.has_data:
            self.skipTest("算例数据不可用")

        # 移除 history 后序列化
        output_clean = {k: v for k, v in self.output.items()
                        if k != "optimization_history"}
        serialized = json.dumps(output_clean, ensure_ascii=False)
        self.assertIsInstance(serialized, str)
        self.assertGreater(len(serialized), 100)


if __name__ == "__main__":
    unittest.main()
