import unittest
import os
import json
from model.model_cutting_partition import CuttingPartitionModel
from utils.load_instance import instance_loader

class TestModel1(unittest.TestCase):
    """测试模块一核心功能：切割约束、分区约束、输出接口合规性"""
    @classmethod
    def setUpClass(cls):
        """初始化测试环境：加载算例、运行模块一"""
        # 预处理后的算例路径（符合算例存储规范）
        cls.instance_path = r"C:\mountain_pv_optimization\data\processed\PV\public\easy\public_easy_r1.json"
        # 初始化并运行模块一
        cls.model1 = CuttingPartitionModel(cls.instance_path)
        cls.module1_output = cls.model1.run()
        # 加载原始算例用于对比
        cls.instance = instance_loader.load_instance("r1")

    def test_m1_output_fields_completeness(self):
        """测试M1-Output接口字段完整性（符合模块接口协议4.1）"""
        required_fields = ["instance_id", "cut_result", "partition_result", "zone_summary", "constraint_satisfaction"]
        for field in required_fields:
            with self.subTest(field=field):
                self.assertIn(field, self.module1_output, f"M1-Output缺失必选字段：{field}（错误码E101）")

    def test_cutting_integer_constraint(self):
        """测试整数切割约束（spec_l为2.0的整数倍，符合数据字典）"""
        t_l_options = self.instance["pva_params"]["t_l_options"]  # 合法切割长度：2.0/4.0...12.0
        for material in self.module1_output["cut_result"]:
            if material["is_used"]:
                for cut in material["cuts"]:
                    with self.subTest(spec_l=cut["spec_l"], material_id=material["material_id"]):
                        self.assertIn(cut["spec_l"], t_l_options, f"切割长度{cut['spec_l']}非法（错误码E102）")
                        self.assertTrue(cut["spec_l"] % 2.0 == 0, f"切割长度{cut['spec_l']}非2.0整数倍（错误码E102）")

    def test_partition_pva_count_constraint(self):
        """测试分区面板数约束（18-26块，符合模块一约束）"""
        for zone in self.module1_output["zone_summary"]:
            with self.subTest(zone_id=zone["zone_id"], pva_count=zone["pva_count"]):
                self.assertTrue(18 <= zone["pva_count"] <= 26, f"分区{zone['zone_id']}面板数{zone['pva_count']}超出范围（错误码E103）")

    def test_partition_perimeter_constraint(self):
        """测试分区周长约束（60-90m，符合数据字典）"""
        LB = self.instance["pva_params"]["LB"]  # 60.0m
        UB = self.instance["pva_params"]["UB"]  # 90.0m
        for zone in self.module1_output["zone_summary"]:
            with self.subTest(zone_id=zone["zone_id"], perimeter=zone["perimeter"]):
                self.assertTrue(LB <= zone["perimeter"] <= UB, f"分区{zone['zone_id']}周长{zone['perimeter']}超出[{LB},{UB}]（约束违规）")

    def test_constraint_satisfaction(self):
        """测试约束满足度（高优先级约束100%满足）"""
        constraint_satisfaction = self.module1_output["constraint_satisfaction"]
        self.assertEqual(constraint_satisfaction["整数切割"], "100%", "整数切割约束未满足")
        self.assertEqual(constraint_satisfaction["分区连通性"], "100%", "分区连通性约束未满足")
        self.assertEqual(constraint_satisfaction["逆变器容量约束"], "100%", "逆变器容量约束未满足")

    def test_unique_ids(self):
        """测试ID唯一性（panel_id、zone_id、inverter_id无重复）"""
        panel_ids = [p["panel_id"] for p in self.module1_output["partition_result"]]
        zone_ids = [z["zone_id"] for z in self.module1_output["zone_summary"]]
        inverter_ids = [z["inverter_id"] for z in self.module1_output["zone_summary"]]
        
        self.assertEqual(len(panel_ids), len(set(panel_ids)), "面板ID重复（错误码E104）")
        self.assertEqual(len(zone_ids), len(set(zone_ids)), "分区ID重复（错误码E104）")
        self.assertEqual(len(inverter_ids), len(set(inverter_ids)), "逆变器ID重复（错误码E104）")

if __name__ == "__main__":
    unittest.main()