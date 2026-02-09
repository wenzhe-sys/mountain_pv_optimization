import unittest
import os
import json
from model.model_cutting_partition import CuttingPartitionModel
from model.model_equipment_cable import EquipmentCableModel
from utils.load_instance import instance_loader

class TestModel2(unittest.TestCase):
    """测试模块二核心功能：设备选型约束、共沟约束、输出接口合规性"""
    @classmethod
    def setUpClass(cls):
        """初始化测试环境：运行模块一→获取M1-Output→运行模块二"""
        # 路径配置
        cls.instance_path = r"C:\mountain_pv_optimization\data\processed\PV\public\easy\public_easy_r1.json"
        cls.module1_output_path = r"C:\mountain_pv_optimization\data\results\module1\M1-Output_r1.json"
        
        # 确保模块一已运行（若未运行则自动执行）
        if not os.path.exists(cls.module1_output_path):
            model1 = CuttingPartitionModel(cls.instance_path)
            model1.run()
        
        # 初始化并运行模块二
        cls.model2 = EquipmentCableModel(cls.instance_path, cls.module1_output_path)
        cls.module2_output = cls.model2.run()
        # 加载原始算例和模块一输出
        cls.instance = instance_loader.load_instance("r1")
        with open(cls.module1_output_path, "r", encoding="utf-8") as f:
            cls.module1_output = json.load(f)
        cls.inverter_ids = [z["inverter_id"] for z in cls.module1_output["zone_summary"]]
        cls.grid_size = cls.instance["terrain_data"]["grid_size"]  # 10m

    def test_m2_output_fields_completeness(self):
        """测试M2-Output接口字段完整性（符合模块接口协议4.2）"""
        required_fields = ["instance_id", "module1_output", "equipment_selection", "cable_routes", "trench_summary", "constraint_satisfaction", "total_cost"]
        for field in required_fields:
            with self.subTest(field=field):
                self.assertIn(field, self.module2_output, f"M2-Output缺失必选字段：{field}")

    def test_transformer_capacity_constraint(self):
        """测试箱变容量约束（仅支持1600/3200kVA，错误码E201）"""
        allowed_Q_box = [1600, 3200]
        for eq in self.module2_output["equipment_selection"]:
            with self.subTest(transformer_id=eq["transformer_id"], Q_box=eq["Q_box"]):
                self.assertIn(eq["Q_box"], allowed_Q_box, f"箱变{eq['transformer_id']}容量{eq['Q_box']}不支持（错误码E201）")

    def test_trench_cable_count_constraint(self):
        """测试单沟电缆数约束（≤4根，错误码E202）"""
        N_max = self.instance["equipment_params"]["cable"]["I_max"]  # 4根
        for trench in self.module2_output["trench_summary"]:
            with self.subTest(trench_id=trench["trench_id"], cable_count=trench["cable_count"]):
                self.assertLessEqual(trench["cable_count"], N_max, f"管沟{trench['trench_id']}电缆数{trench['cable_count']}超限（错误码E202）")

    def test_install_coord_constraint(self):
        """测试箱变安装坐标约束（x/y为grid_size整数倍，错误码E203）"""
        for eq in self.module2_output["equipment_selection"]:
            x, y = eq["install_coord"]
            with self.subTest(transformer_id=eq["transformer_id"], install_coord=(x, y)):
                self.assertEqual(x % self.grid_size, 0, f"箱变{eq['transformer_id']}X坐标{x}未对齐网格（错误码E203）")
                self.assertEqual(y % self.grid_size, 0, f"箱变{eq['transformer_id']}Y坐标{y}未对齐网格（错误码E203）")

    def test_connected_inverter_validity(self):
        """测试连接的逆变器ID有效性（必须存在于M1-Output，错误码E204）"""
        for eq in self.module2_output["equipment_selection"]:
            for inv_id in eq["connected_inverters"]:
                with self.subTest(transformer_id=eq["transformer_id"], inverter_id=inv_id):
                    self.assertIn(inv_id, self.inverter_ids, f"箱变{eq['transformer_id']}连接无效逆变器ID：{inv_id}（错误码E204）")

    def test_constraint_satisfaction(self):
        """测试约束满足度（共沟、箱变容量等约束100%满足）"""
        constraint_satisfaction = self.module2_output["constraint_satisfaction"]
        self.assertEqual(constraint_satisfaction["共沟约束"], "100%", "共沟约束未满足")
        self.assertEqual(constraint_satisfaction["箱变容量"], "100%", "箱变容量约束未满足")
        self.assertEqual(constraint_satisfaction["路由连续性"], "100%", "路由连续性约束未满足")

    def test_cost_calculation_rationality(self):
        """测试成本计算合理性（总成本为正数，各分项成本非负）"""
        self.assertGreater(self.module2_output["total_cost"], 0, "总成本计算异常（非正数）")
        # 验证设备成本
        for eq in self.module2_output["equipment_selection"]:
            self.assertGreaterEqual(eq["cost"]["purchase"], 0, f"箱变{eq['transformer_id']}购置成本异常")
            self.assertGreaterEqual(eq["cost"]["installation"], 0, f"箱变{eq['transformer_id']}安装成本异常")
        # 验证电缆和管沟成本
        for route in self.module2_output["cable_routes"]:
            self.assertGreaterEqual(route["cost"]["cable"], 0, f"路由{route['route_id']}电缆成本异常")
            self.assertGreaterEqual(route["cost"]["trenching"], 0, f"路由{route['route_id']}管沟成本异常")

if __name__ == "__main__":
    unittest.main()