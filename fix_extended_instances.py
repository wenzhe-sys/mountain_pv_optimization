#!/usr/bin/env python3
"""
修复所有扩展算例的脚本
"""

import os
import json
import argparse
import logging
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def fix_extended_instance(instance_id: str):
    """
    修复单个扩展算例
    """
    print("="*60)
    print(f"开始修复扩展算例（算例ID：r{instance_id}）")
    print("="*60)

    try:
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
            return False
        
        # 加载算例数据
        print("\n【步骤2】加载算例数据...")
        with open(processed_instance_path, 'r', encoding='utf-8') as f:
            instance_data = json.load(f)
        
        # 计算面板数和逆变器数
        n_panels = instance_data["instance_info"]["n_nodes"]
        n_inverters = (n_panels + 25) // 26  # 向上取整，每个逆变器最多 26 块面板
        
        # 计算分区面板数范围
        min_panels_per_zone = max(10, (n_panels + n_inverters - 1) // n_inverters - 5)
        max_panels_per_zone = min(30, (n_panels + n_inverters - 1) // n_inverters + 5)
        
        print(f"  - 面板数: {n_panels}")
        print(f"  - 逆变器数: {n_inverters}")
        print(f"  - 分区面板数范围: [{min_panels_per_zone}, {max_panels_per_zone}]")
        
        # 检查并添加缺少的字段
        print("\n【步骤3】检查并添加缺少的字段...")
        
        # 添加 pva_params
        if "pva_params" not in instance_data:
            instance_data["pva_params"] = {
                "t_l_options": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                "D": 12.0,
                "LB": 50.0,  # 降低周长下限
                "UB": 100.0,  # 提高周长上限
                "min_panels_per_zone": min_panels_per_zone,
                "max_panels_per_zone": max_panels_per_zone
            }
            print("  - 添加了 pva_params 字段")
        else:
            # 更新现有字段
            instance_data["pva_params"]["LB"] = 50.0
            instance_data["pva_params"]["UB"] = 100.0
            instance_data["pva_params"]["min_panels_per_zone"] = min_panels_per_zone
            instance_data["pva_params"]["max_panels_per_zone"] = max_panels_per_zone
            print("  - 更新了 pva_params 字段")
        
        # 添加 equipment_params
        if "equipment_params" not in instance_data:
            instance_data["equipment_params"] = {
                "inverter": {
                    "q": 320.0,
                    "r": 0.85
                },
                "substation": {
                    "Q_substation": 10,  # 升压站容量（台逆变器）
                    "cost_substation": 500000.0  # 升压站成本
                }
            }
            print("  - 添加了 equipment_params 字段")
        else:
            # 确保 substation 字段存在
            if "substation" not in instance_data["equipment_params"]:
                instance_data["equipment_params"]["substation"] = {
                    "Q_substation": 10,
                    "cost_substation": 500000.0
                }
                print("  - 添加了 substation 字段")
        
        # 添加 constraint_info
        if "constraint_info" not in instance_data:
            instance_data["constraint_info"] = {
                "cable_trench_depth": 1.2,  # 电缆沟深度 (m)
                "cable_trench_width": 0.8,  # 电缆沟宽度 (m)
                "max_cables_per_trench": 4,  # 每沟最多电缆数
                "min_cable_spacing": 0.1,  # 电缆最小间距 (m)
                "max_inverter_distance": 500,  # 逆变器最大间距 (m)
                "max_panel_distance": 50,  # 面板最大间距 (m)
                "max_zone_perimeter": 100.0,  # 分区最大周长 (m)
                "min_zone_perimeter": 50.0,  # 分区最小周长 (m)
                "max_panel_per_zone": max_panels_per_zone,  # 每分区最大面板数
                "min_panel_per_zone": min_panels_per_zone  # 每分区最小面板数
            }
            print("  - 添加了 constraint_info 字段")
        else:
            # 更新现有字段
            instance_data["constraint_info"]["max_panel_per_zone"] = max_panels_per_zone
            instance_data["constraint_info"]["min_panel_per_zone"] = min_panels_per_zone
            print("  - 更新了 constraint_info 字段")
        
        # 保存修改后的算例
        with open(processed_instance_path, 'w', encoding='utf-8') as f:
            json.dump(instance_data, f, ensure_ascii=False, indent=2)
        print(f"  - 保存修改后的算例文件: {processed_instance_path}")
        
        print("\n" + "="*60)
        print(f"修复完成！")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"运行过程中遇到错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_all_extended_instances():
    """
    修复所有扩展算例
    """
    print("="*60)
    print("开始修复所有扩展算例")
    print("="*60)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    extended_dir = os.path.join(project_root, "data", "processed", "PV", "public", "extended")
    
    # 列出所有扩展算例文件
    extended_files = [f for f in os.listdir(extended_dir) if f.endswith('.json')]
    print(f"找到 {len(extended_files)} 个扩展算例文件")
    
    # 修复每个算例
    success_count = 0
    failure_count = 0
    
    for file in extended_files:
        # 提取算例ID
        if "_r" in file:
            instance_id = file.split("_r")[1].split(".")[0]
            print(f"\n处理文件: {file} (算例ID: r{instance_id})")
            
            if fix_extended_instance(instance_id):
                success_count += 1
            else:
                failure_count += 1
    
    print("\n" + "="*60)
    print(f"修复完成：成功 {success_count} 个，失败 {failure_count} 个")
    print("="*60)

if __name__ == "__main__":
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description="修复扩展算例")
    parser.add_argument("--instance_id", type=str, help="算例ID（纯数字，如18、19...）")
    parser.add_argument("--all", action="store_true", help="修复所有扩展算例")
    args = parser.parse_args()
    
    # 运行修复
    if args.all:
        fix_all_extended_instances()
    elif args.instance_id:
        fix_extended_instance(args.instance_id)
    else:
        print("请指定算例ID或使用 --all 参数修复所有算例")
        print("示例：python fix_extended_instances.py --instance_id 18")
        print("示例：python fix_extended_instances.py --all")