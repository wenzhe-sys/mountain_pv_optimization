import json
import os
from typing import Dict
from algorithm.benders_decomposition import BendersDecomposition

class CuttingPartitionModel:
    def __init__(self, instance_path: str):
        """加载算例并初始化模型"""
        self.instance_path = instance_path
        self.instance_data = self.load_instance()
        self.benders_solver = BendersDecomposition(self.instance_data)
        self.results_path = r"C:\mountain_pv_optimization\data\results\module1"
        os.makedirs(self.results_path, exist_ok=True)

    def load_instance(self) -> Dict:
        """加载预处理后的算例"""
        with open(self.instance_path, "r", encoding="utf-8") as f:
            instance_data = json.load(f)
        print(f"【模块一】成功加载算例：{instance_data['instance_info']['instance_id']}（面板数：{instance_data['instance_info']['n_nodes']}）")
        return instance_data

    def run(self) -> Dict:
        """运行模块一：切割及分区优化"""
        print(f"【模块一】开始光伏面板切割及分区优化")
        
        # 调用Benders分解算法
        module1_output = self.benders_solver.optimize()
        
        # 补充实例ID
        module1_output["instance_id"] = self.instance_data["instance_info"]["instance_id"]
        
        # 保存结果
        save_path = os.path.join(self.results_path, f"M1-Output_{module1_output['instance_id']}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(module1_output, f, ensure_ascii=False, indent=2)
        
        print(f"【模块一】优化完成，结果保存至：{save_path}")
        return module1_output