import json
import os
from typing import Dict
from algorithm.branch_and_price import BranchAndPrice

class EquipmentCableModel:
    def __init__(self, instance_path: str, module1_output_path: str):
        """加载算例和模块一输出，初始化模型（修正属性赋值顺序）"""
        self.instance_path = instance_path
        self.module1_output_path = module1_output_path
        
        # 步骤1：先加载算例和模块一输出（依赖优先）
        self.instance_data = self.load_instance()
        self.module1_output = self.load_module1_output()
        
        # 步骤2：提取逆变器ID（在加载module1_output之后）
        self.inverter_ids = [zone["inverter_id"] for zone in self.module1_output["zone_summary"]]
        
        # 步骤3：初始化分支定价求解器（依赖instance_data和module1_output）
        self.bap_solver = BranchAndPrice(self.instance_data, self.module1_output)
        
        # 步骤4：其他属性初始化
        self.results_path = r"C:\mountain_pv_optimization\data\results\module2"
        os.makedirs(self.results_path, exist_ok=True)
        self.grid_size = self.instance_data["terrain_data"]["grid_size"]

    def load_instance(self) -> Dict:
        """加载预处理后的算例（严格校验必选字段）"""
        with open(self.instance_path, "r", encoding="utf-8") as f:
            instance_data = json.load(f)
        
        # 校验算例字段完整性（遵循算例处理规范）
        required_fields = ["instance_info", "terrain_data", "equipment_params", "constraint_info"]
        for field in required_fields:
            if field not in instance_data:
                raise ValueError(f"算例缺失必选字段：{field}，错误码E205")
        
        print(f"【模块二】成功加载算例：{instance_data['instance_info']['instance_id']}（升压站容量：{instance_data['equipment_params']['substation']['Q_substation']}台逆变器）")
        return instance_data

    def load_module1_output(self) -> Dict:
        """加载模块一输出，验证接口合规性（遵循M1-Output协议）"""
        with open(self.module1_output_path, "r", encoding="utf-8") as f:
            module1_output = json.load(f)
        
        # 接口校验（对应错误码E101-E104）
        if "instance_id" not in module1_output:
            raise ValueError("模块一输出缺失instance_id，错误码E101")
        if not all(18 <= zone["pva_count"] <= 26 for zone in module1_output["zone_summary"]):
            raise ValueError("分区面板数超出18-26范围，错误码E103")
        if len(set([zone["inverter_id"] for zone in module1_output["zone_summary"]])) != len(module1_output["zone_summary"]):
            raise ValueError("逆变器ID重复，错误码E104")
        
        # 修正：直接从module1_output获取逆变器数（无需self.inverter_ids）
        inverter_count = len(module1_output["zone_summary"])
        print(f"【模块二】成功加载模块一结果：{module1_output['instance_id']}（分区数：{len(module1_output['zone_summary'])}，逆变器数：{inverter_count}）")
        return module1_output

    def validate_output(self, module2_output: Dict) -> None:
        """验证模块二输出是否符合M2-Output接口规范（对应错误码E201-E204）"""
        # 校验箱变容量（仅支持1600/3200kVA）
        for eq in module2_output["equipment_selection"]:
            if eq["Q_box"] not in [1600, 3200]:
                raise ValueError(f"箱变容量{eq['Q_box']}不支持，错误码E201")
            # 校验安装坐标是否为grid_size整数倍
            if eq["install_coord"][0] % self.grid_size != 0 or eq["install_coord"][1] % self.grid_size != 0:
                raise ValueError(f"箱变坐标{eq['install_coord']}未对齐网格，错误码E203")
        
        # 校验管沟电缆数（≤4根）
        for trench in module2_output["trench_summary"]:
            if trench["cable_count"] > 4:
                raise ValueError(f"管沟{trench['trench_id']}电缆数{trench['cable_count']}超限，错误码E202")
        
        # 校验逆变器ID存在性
        for eq in module2_output["equipment_selection"]:
            for inv_id in eq["connected_inverters"]:
                if inv_id not in self.inverter_ids:
                    raise ValueError(f"箱变连接无效逆变器ID：{inv_id}，错误码E204")

    def run(self) -> Dict:
        """运行模块二：设备选型+电缆共沟优化，输出符合M2-Output规范的结果"""
        print(f"【模块二】开始电气设备选型及电缆共沟优化（算例ID：{self.instance_data['instance_info']['instance_id']}）")
        
        # 1. 调用分支定价算法求解（融合K-means列管理）
        bap_result = self.bap_solver.optimize()
        
        # 2. 构建模块二输出结构（严格遵循接口协议）
        module2_output = {
            "instance_id": self.instance_data["instance_info"]["instance_id"],
            "module1_output": self.module1_output,  # 嵌入完整模块一输出
            "equipment_selection": bap_result["equipment_selection"],
            "cable_routes": bap_result["cable_routes"],
            "trench_summary": bap_result["trench_summary"],
            "constraint_satisfaction": bap_result["constraint_satisfaction"],
            "total_cost": bap_result["total_cost"]  # 总成本（万元）
        }
        
        # 3. 接口合规性校验
        self.validate_output(module2_output)
        
        # 4. 保存结果（文件名遵循规范：M2-Output_[instance_id].json）
        save_path = os.path.join(self.results_path, f"M2-Output_{module2_output['instance_id']}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(module2_output, f, ensure_ascii=False, indent=2)
        
        # 输出优化Summary
        total_box_cost = sum([eq["cost"]["purchase"] + eq["cost"]["installation"] for eq in module2_output["equipment_selection"]])
        total_cable_cost = sum([route["cost"]["cable"] for route in module2_output["cable_routes"]])
        total_trench_cost = sum([trench["cost"] for trench in module2_output["trench_summary"]])
        
        print(f"【模块二】优化完成！")
        print(f"  - 箱变配置：{len(module2_output['equipment_selection'])}台（3200kVA：{sum(1 for eq in module2_output['equipment_selection'] if eq['Q_box']==3200)}台，1600kVA：{sum(1 for eq in module2_output['equipment_selection'] if eq['Q_box']==1600)}台）")
        print(f"  - 成本构成：箱变总成本{total_box_cost:.1f}万元，电缆成本{total_cable_cost:.1f}万元，管沟成本{total_trench_cost:.1f}万元")
        print(f"  - 约束满足度：{module2_output['constraint_satisfaction']}")
        print(f"  - 结果文件：{save_path}")
        
        return module2_output

# 测试代码（单独运行时执行）
if __name__ == "__main__":
    # 示例：加载预处理后的算例和模块一输出
    test_instance_path = r"C:\mountain_pv_optimization\data\processed\PV\public\easy\public_easy_r1.json"
    test_module1_output_path = r"C:\mountain_pv_optimization\data\results\module1\M1-Output_r1.json"
    
    if os.path.exists(test_instance_path) and os.path.exists(test_module1_output_path):
        model = EquipmentCableModel(test_instance_path, test_module1_output_path)
        model.run()
    else:
        print("请先运行模块一生成输出文件，或检查算例路径是否正确！")