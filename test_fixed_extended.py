#!/usr/bin/env python3
"""
测试修复后的扩展算例
"""

import os
import json
import argparse
import logging
import numpy as np
import torch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 尝试导入依赖库
try:
    from modules.module1.model.model_cutting_partition import CuttingPartitionModel
    from modules.module2.model.model_equipment_cable import EquipmentCableModel
    from modules.module3.model.model_integration import IntegrationOptimizationModel
    from utils.load_instance import load_instance
    from utils.metric_calculation import metric_calculator
    from utils.visualization import result_visualizer
    IMPORT_SUCCESS = True
except Exception as e:
    logger.warning(f"导入库时遇到错误: {e}")
    logger.warning("将尝试使用最小依赖运行...")
    IMPORT_SUCCESS = False

def set_random_seed(seed=42):
    """设置随机种子，保证结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"已设置随机种子: {seed}")

def test_extended_instance(instance_id: str = "18"):
    """
    测试修复后的扩展算例
    """
    print("="*60)
    print(f"开始测试修复后的扩展算例（算例ID：r{instance_id}）")
    print("="*60)

    if not IMPORT_SUCCESS:
        print("错误：无法导入必要的库，请检查Python环境和依赖安装。")
        return

    try:
        # 设置随机种子
        set_random_seed()
        
        # 查找扩展算例文件
        print("\n【步骤1】查找扩展算例...")
        project_root = os.path.dirname(os.path.abspath(__file__))
        extended_dir = os.path.join(project_root, "data", "processed", "PV", "public", "extended")
        
        # 尝试查找扩展算例（遍历不同难度级别）
        processed_instance_path = None
        for difficulty in ['easy', 'medium', 'hard']:
            ext_path = os.path.join(extended_dir, f"public_{difficulty}_r{instance_id}.json")
            if os.path.exists(ext_path):
                processed_instance_path = ext_path
                print(f"  - 找到扩展算例: {ext_path}")
                break
        
        if not processed_instance_path:
            print(f"错误：未找到扩展算例 r{instance_id}")
            # 打印可用的算例列表
            print("\n可用的扩展算例：")
            for file in os.listdir(extended_dir):
                if file.endswith('.json'):
                    print(f"  - {file}")
            return
        
        # 加载算例数据
        print("\n【步骤2】加载算例数据...")
        with open(processed_instance_path, 'r', encoding='utf-8') as f:
            instance_data = json.load(f)
        
        # 检查字段是否存在
        print("\n【步骤3】检查算例字段...")
        required_fields = ["pva_params", "equipment_params"]
        for field in required_fields:
            if field in instance_data:
                print(f"  - ✓ {field} 字段存在")
            else:
                print(f"  - ✗ {field} 字段缺失")
                return
        
        # 运行模块一
        print("\n【步骤4】运行模块一：光伏面板切割及分区...")
        model1 = CuttingPartitionModel(processed_instance_path)
        module1_output = model1.run(verbose=True, max_iter=20)
        
        # 运行模块二
        if module1_output:
            print("\n【步骤5】运行模块二：电气设备选型及电缆共沟...")
            module1_output_path = os.path.join(project_root, "data", "results", "module1", f"M1-Output_r{instance_id}.json")
            model2 = EquipmentCableModel(processed_instance_path, module1_output_path, module1_output)
            module2_output = model2.run()
            
            # 运行模块三
            if module2_output:
                print("\n【步骤6】运行模块三：全生命周期集成优化...")
                module2_output_path = os.path.join(project_root, "data", "results", "module2", f"M2-Output_r{instance_id}.json")
                model3 = IntegrationOptimizationModel(processed_instance_path, module2_output_path, module1_output)
                module3_output = model3.run()
        
        print("\n" + "="*60)
        print(f"测试完成！")
        print("="*60)
        
    except Exception as e:
        logger.error(f"运行过程中遇到错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description="测试修复后的扩展算例")
    parser.add_argument("--instance_id", type=str, default="18", help="算例ID（纯数字，如18、19...）")
    args = parser.parse_args()
    
    # 运行测试
    test_extended_instance(args.instance_id)