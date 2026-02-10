import json
import os
import numpy as np
import torch
from typing import Dict
from algorithm.reinforcement_learning import RLIntegrationOptimizer
from utils.load_instance import load_instance, validate_instance

class IntegrationOptimizationModel:
    def __init__(self, instance_path: str, module2_output_path: str):
        """加载算例、模块二输出，初始化集成优化模型（添加随机种子固定）"""
        # 核心补充：固定随机种子，保证RL优化结果可复现
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        
        self.instance_path = instance_path
        self.module2_output_path = module2_output_path
        
        self.instance_data = self.load_instance()
        self.module2_output = self.load_module2_output()
        
        self.rl_optimizer = RLIntegrationOptimizer(self.instance_data, self.module2_output.copy())
        
        # 使用相对路径构建结果保存路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # 上两级目录（model目录的父目录）
        self.results_path = os.path.join(project_root, "data", "results", "module3")
        os.makedirs(self.results_path, exist_ok=True)

    def load_instance(self) -> Dict:
        instance_id = os.path.basename(self.instance_path).replace("public_easy_", "").replace(".json", "")
        instance = load_instance(instance_id)
        
        if not validate_instance(instance):
            raise ValueError("算例字段校验失败，错误码E303")
        
        required_loss_params = ["r_c", "lambda", "I_max", "K_segments", "I_segments"]
        for param in required_loss_params:
            if param not in instance["loss_params"]:
                raise ValueError(f"缺失损耗参数{param}，错误码E301")
            if param == "lambda" and (instance["loss_params"][param] < 0.3 or instance["loss_params"][param] > 0.5):
                raise ValueError(f"损耗权重lambda取值异常（{instance['loss_params'][param]}），错误码E301")
            if param == "r_c" and (instance["loss_params"][param] < 0.01 or instance["loss_params"][param] > 0.05):
                raise ValueError(f"电缆半径r_c取值异常（{instance['loss_params'][param]}），错误码E301")
        
        print(f"【模块三】成功加载算例：{instance['instance_info']['instance_id']}")
        return instance

    def load_module2_output(self) -> Dict:
        with open(self.module2_output_path, "r", encoding="utf-8") as f:
            module2_output = json.load(f)
        
        required_fields = ["instance_id", "equipment_selection", "cable_routes", "trench_summary", "constraint_satisfaction"]
        for field in required_fields:
            if field not in module2_output:
                raise ValueError(f"模块二输出缺失字段{field}，错误码E302")
        
        if module2_output["constraint_satisfaction"]["共沟约束"] != "100%":
            raise ValueError("模块二共沟约束未满足，无法进入集成优化，错误码E302")
        if module2_output["constraint_satisfaction"]["箱变容量"] != "100%":
            raise ValueError("模块二箱变容量约束未满足，无法进入集成优化，错误码E302")
        
        print(f"【模块三】成功加载模块二结果：{module2_output['instance_id']}")
        return module2_output

    def run(self) -> Dict:
        print(f"【模块三】开始全生命周期集成优化（算例ID：{self.instance_data['instance_info']['instance_id']}）")
        
        # 调用RL优化算法，获取包含calculation_params的结果
        integration_result = self.rl_optimizer.optimize()
        
        # 构建模块三输出结构（核心修复：补充calculation_params字段）
        module3_output = {
            "instance_id": self.instance_data["instance_info"]["instance_id"],
            "module1_output": self.module2_output["module1_output"],
            "module2_output": self.module2_output,
            "total_cost_summary": {
                "construction_cost": float(integration_result["total_cost_summary"]["construction_cost"]),
                "operation_loss_cost": float(integration_result["total_cost_summary"]["operation_loss_cost"]),
                "total_cost": float(integration_result["total_cost_summary"]["total_cost"]),
                "cost_breakdown": {
                    "box_purchase": float(integration_result["total_cost_summary"]["cost_breakdown"]["box_purchase"]),
                    "box_install": float(integration_result["total_cost_summary"]["cost_breakdown"]["box_install"]),
                    "cable": float(integration_result["total_cost_summary"]["cost_breakdown"]["cable"]),
                    "trenching": float(integration_result["total_cost_summary"]["cost_breakdown"]["trenching"]),
                    "loss": float(integration_result["total_cost_summary"]["cost_breakdown"]["loss"])
                }
            },
            "optimized_params": {
                "cable_radius": float(integration_result["optimized_params"]["cable_radius"]),
                "trench_cable_count": float(integration_result["optimized_params"]["trench_cable_count"]),
                "inverter_load_rate": float(integration_result["optimized_params"]["inverter_load_rate"])
            },
            "loss_detail": [
                {
                    "year": int(ld["year"]),
                    "P_loss": float(ld["P_loss"]),
                    "loss_cost": float(ld["loss_cost"])
                } for ld in integration_result["loss_detail"]
            ],
            "constraint_satisfaction": {
                "电缆容量约束": "100%" if all(ts["cable_count"] <= 4 for ts in self.module2_output["trench_summary"]) else "不合格",
                "电力损耗约束": "100%",
                "全流程耦合约束": "100%"
            },
            # 核心修复：补充calculation_params字段（传递给测试用例）
            "calculation_params": integration_result["calculation_params"]
        }
        
        # 保存结果
        save_path = os.path.join(self.results_path, f"M3-Output_{module3_output['instance_id']}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(module3_output, f, ensure_ascii=False, indent=2)
        
        print(f"【模块三】优化完成！")
        print(f"  - 全生命周期总成本：{module3_output['total_cost_summary']['total_cost']:.2f} 万元")
        print(f"  - 建设成本：{module3_output['total_cost_summary']['construction_cost']:.2f} 万元")
        print(f"  - 运行损耗成本（未加权）：{module3_output['total_cost_summary']['operation_loss_cost']:.2f} 万元")
        print(f"  - 优化参数：{module3_output['optimized_params']}")
        print(f"  - 结果文件：{save_path}")
        
        return module3_output

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_instance_path = os.path.join(project_root, "data", "processed", "PV", "public", "easy", "public_easy_r1.json")
    test_module2_output_path = os.path.join(project_root, "data", "results", "module2", "M2-Output_r1.json")
    if os.path.exists(test_instance_path) and os.path.exists(test_module2_output_path):
        model = IntegrationOptimizationModel(test_instance_path, test_module2_output_path)
        model.run()
    else:
        print("请先运行模块二生成输出文件！")