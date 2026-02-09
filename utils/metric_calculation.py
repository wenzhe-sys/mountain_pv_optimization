import numpy as np
from typing import Dict, List

class MetricCalculator:
    @staticmethod
    def calculate_coverage_rate(module1_output: Dict, instance: Dict) -> float:
        """计算模块一：覆盖面积利用率（符合模块专用指标）"""
        covered_pva = sum([len([p for p in module1_output["partition_result"] if p["zone_id"] == z["zone_id"]]) 
                          for z in module1_output["zone_summary"]])
        total_buildable_pva = len(instance["pva_list"])
        return round((covered_pva / total_buildable_pva) * 100, 2)

    @staticmethod
    def calculate_trench_optimization_rate(module2_output: Dict) -> float:
        """计算模块二：共沟成本优化率（符合模块专用指标）"""
        # 核心修复1：从module2_output顶级字段获取管沟汇总（遵循M2-Output接口规范）
        trench_summary = module2_output["trench_summary"]
        if not trench_summary:
            return 0.0  # 边界处理：无管沟时优化率为0
        
        # 核心修复2：正确计算非共沟方案成本与共沟方案成本
        # 非共沟方案：每根电缆单独挖沟，成本=Σ(管沟成本×该管沟电缆数)
        non_trench_cost = sum([trench["cost"] * trench["cable_count"] 
                              for trench in trench_summary])
        # 共沟方案：实际挖沟总成本=Σ(管沟成本)
        trench_cost = sum([trench["cost"] for trench in trench_summary])
        
        # 核心修复3：避免除以零（非共沟成本为0时优化率为0）
        if non_trench_cost <= 1e-6:
            return 0.0
        
        # 计算优化率（保留2位小数）
        optimization_rate = ((non_trench_cost - trench_cost) / non_trench_cost) * 100
        return round(optimization_rate, 2)

    @staticmethod
    def calculate_lifecycle_cost_reduction(module3_output: Dict, traditional_cost: float) -> float:
        """计算模块三：全生命周期成本降低率（符合模块专用指标）"""
        optimized_total_cost = module3_output["total_cost_summary"]["total_cost"]
        # 避免除以零和负数（传统成本异常时返回0）
        if traditional_cost <= 1e-6 or optimized_total_cost >= traditional_cost:
            return 0.0
        reduction_rate = ((traditional_cost - optimized_total_cost) / traditional_cost) * 100
        return round(reduction_rate, 2)

    @staticmethod
    def calculate_constraint_satisfaction_rate(constraint_satisfaction: Dict) -> float:
        """计算通用指标：约束满足度"""
        satisfied = sum([1 for v in constraint_satisfaction.values() if v == "100%"])
        total = len(constraint_satisfaction)
        return round((satisfied / total) * 100, 2) if total > 0 else 100.0

    @staticmethod
    def calculate_average_perimeter(module1_output: Dict) -> float:
        """计算分区平均周长（模块一辅助指标）"""
        perimeters = [z["perimeter"] for z in module1_output["zone_summary"]]
        return round(np.mean(perimeters), 2) if perimeters else 0.0

# 单例调用
metric_calculator = MetricCalculator()