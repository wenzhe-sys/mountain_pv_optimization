import os
import json
from utils.data_preprocess import PVDataPreprocessor
from model.model_cutting_partition import CuttingPartitionModel
from model.model_equipment_cable import EquipmentCableModel
from model.model_integration import IntegrationOptimizationModel
from utils.load_instance import instance_loader
from utils.metric_calculation import metric_calculator
from utils.visualization import result_visualizer

def main(instance_id: str = "r1"):
    """
    主流程：数据预处理 → 模块一 → 模块二 → 模块三 → 指标计算 → 可视化
    """
    print("="*50)
    print(f"开始大型山地光伏电站设计优化（算例ID：{instance_id}）")
    print("="*50)

    # 步骤1：数据预处理（raw→processed）
    print("\n【步骤1/6】数据预处理...")
    preprocessor = PVDataPreprocessor()
    preprocessor.process_single_file(f"{instance_id}.txt")
    processed_instance_path = os.path.join(preprocessor.processed_pv_path, f"public_easy_{instance_id}.json")

    # 步骤2：加载标准化算例
    print("\n【步骤2/6】加载算例...")
    instance = instance_loader.load_instance(instance_id)

    # 步骤3：运行模块一（切割及分区）
    print("\n【步骤3/6】运行模块一：光伏面板切割及分区...")
    model1 = CuttingPartitionModel(processed_instance_path)
    module1_output = model1.run()

    # 步骤4：运行模块二（设备选型+电缆共沟）
    print("\n【步骤4/6】运行模块二：电气设备选型及电缆共沟...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    module1_output_path = os.path.join(project_root, "data", "results", "module1", f"M1-Output_{instance_id}.json")
    model2 = EquipmentCableModel(processed_instance_path, module1_output_path)
    module2_output = model2.run()

    # 步骤5：运行模块三（集成优化）
    print("\n【步骤5/6】运行模块三：全生命周期集成优化...")
    module2_output_path = os.path.join(project_root, "data", "results", "module2", f"M2-Output_{instance_id}.json")
    model3 = IntegrationOptimizationModel(processed_instance_path, module2_output_path)
    module3_output = model3.run()

    # 步骤6：指标计算与可视化
    print("\n【步骤6/6】指标计算与结果可视化...")
    # 计算核心指标
    coverage_rate = metric_calculator.calculate_coverage_rate(module1_output, instance)
    trench_optimization_rate = metric_calculator.calculate_trench_optimization_rate(module2_output)
    constraint_satisfaction = metric_calculator.calculate_constraint_satisfaction_rate(module3_output["constraint_satisfaction"])
    # 输出指标汇总
    print(f"\n===== 核心指标汇总 =====")
    print(f"1. 覆盖面积利用率：{coverage_rate}%")
    print(f"2. 共沟成本优化率：{trench_optimization_rate}%")
    print(f"3. 约束满足度：{constraint_satisfaction}%")
    print(f"4. 全生命周期总成本：{module3_output['total_cost_summary']['total_cost']:.2f}万元")
    # 可视化
    result_visualizer.plot_partition(module1_output, instance_id)
    result_visualizer.plot_cost_breakdown(module3_output, instance_id)
    result_visualizer.plot_loss_trend(module3_output, instance_id)

    print("\n" + "="*50)
    print(f"优化流程全部完成！所有结果已保存至 data/results 目录")
    print("="*50)

if __name__ == "__main__":
    # 支持指定算例ID运行（如r1、r2...r17）
    main(instance_id="r1")