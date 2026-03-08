#!/usr/bin/env python3
"""
算法评估脚本

用于评估和比较不同算法的性能，包括：
1. 启发式算法
2. S2V-DQN算法
3. 混合策略算法

生成性能报告和可视化结果。
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Any

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from modules.module1.algorithm.benders_decomposition import BendersDecomposition
from modules.module1.algorithm.dqn_agent import DQNPartitionAgent
from modules.module1.algorithm.partition_heuristic import GreedyPartitioner
from modules.module1.visualization.layout_visualizer import LayoutVisualizer
from utils.load_instance import load_instance
from utils.graph_utils import build_adjacency_graph


def evaluate_algorithm(instance_data: Dict, algorithm_name: str, 
                      dqn_agent: DQNPartitionAgent = None) -> Dict:
    """
    评估单个算法的性能
    
    Args:
        instance_data: 算例数据
        algorithm_name: 算法名称 ('heuristic', 'dqn', 'hybrid')
        dqn_agent: DQN智能体实例（仅在使用DQN时需要）
    
    Returns:
        评估结果字典
    """
    start_time = time.time()
    
    if algorithm_name == 'heuristic':
        # 使用启发式算法
        solver = BendersDecomposition(instance_data, partition_solver='heuristic', verbose=False)
    elif algorithm_name == 'dqn':
        # 使用S2V-DQN算法
        solver = BendersDecomposition(instance_data, partition_solver='dqn', verbose=False)
        if dqn_agent:
            solver.set_dqn_solver(dqn_agent)
    elif algorithm_name == 'hybrid':
        # 使用混合策略
        solver = BendersDecomposition(instance_data, partition_solver='dqn', verbose=False)
        if dqn_agent:
            solver.set_dqn_solver(dqn_agent)
    else:
        raise ValueError(f"未知算法: {algorithm_name}")
    
    # 执行优化
    result = solver.optimize()
    
    # 计算评估指标
    total_time = time.time() - start_time
    
    # 提取指标
    # 处理 constraint_satisfaction 可能是布尔值的情况
    constraint_satisfaction = result.get('constraint_satisfaction', {})
    if isinstance(constraint_satisfaction, bool):
        is_feasible = constraint_satisfaction
    else:
        is_feasible = all(z.get('capacity_ok', True) for z in constraint_satisfaction.get('zone_details', []))
    
    metrics = {
        'algorithm': algorithm_name,
        'instance_id': instance_data['instance_info']['instance_id'],
        'n_nodes': instance_data['instance_info']['n_nodes'],
        'n_zones': len(result.get('zone_summary', [])),
        'total_perimeter': sum(z['perimeter'] for z in result.get('zone_summary', [])),
        'avg_perimeter': np.mean([z['perimeter'] for z in result.get('zone_summary', [])]) if result.get('zone_summary') else 0,
        'execution_time': total_time,
        'is_feasible': is_feasible,
        'optimization_history': solver.history
    }
    
    return metrics

def compare_algorithms(instances: List[Dict], dqn_agent: DQNPartitionAgent = None) -> Dict:
    """
    比较不同算法的性能
    
    Args:
        instances: 算例列表
        dqn_agent: DQN智能体实例
    
    Returns:
        比较结果字典
    """
    algorithms = ['heuristic', 'dqn', 'hybrid']
    results = {}
    
    for instance in instances:
        instance_id = instance['instance_info']['instance_id']
        results[instance_id] = {}
        
        for algorithm in algorithms:
            print(f"\n=== 评估算例 {instance_id} 使用 {algorithm} 算法 ===")
            metrics = evaluate_algorithm(instance, algorithm, dqn_agent)
            results[instance_id][algorithm] = metrics
            print(f"算法: {algorithm}")
            print(f"执行时间: {metrics['execution_time']:.2f} 秒")
            print(f"总周长: {metrics['total_perimeter']:.1f} 米")
            print(f"可行性: {'可行' if metrics['is_feasible'] else '不可行'}")
    
    return results

def generate_performance_report(results: Dict, output_dir: str = 'outputs/evaluation'):
    """
    生成性能报告
    
    Args:
        results: 评估结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成综合报告
    report = {
        'overall_metrics': {},
        'instance_details': results
    }
    
    # 确定实际运行的算法
    algorithms = []
    if results:
        first_instance = list(results.values())[0]
        algorithms = list(first_instance.keys())
    
    # 计算总体指标
    for algorithm in algorithms:
        all_times = []
        all_perimeters = []
        all_feasible = []
        
        for instance_id, instance_results in results.items():
            if algorithm in instance_results:
                metrics = instance_results[algorithm]
                all_times.append(metrics['execution_time'])
                all_perimeters.append(metrics['total_perimeter'])
                all_feasible.append(metrics['is_feasible'])
        
        report['overall_metrics'][algorithm] = {
            'average_time': np.mean(all_times) if all_times else 0,
            'average_perimeter': np.mean(all_perimeters) if all_perimeters else 0,
            'feasibility_rate': sum(all_feasible) / len(all_feasible) if all_feasible else 0
        }
    
    # 保存报告
    report_path = os.path.join(output_dir, 'performance_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"性能报告已保存到: {report_path}")
    
    # 生成可视化
    generate_visualizations(results, output_dir)

def generate_visualizations(results: Dict, output_dir: str):
    """
    生成可视化结果
    
    Args:
        results: 评估结果
        output_dir: 输出目录
    """
    visualizations_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # 确定实际运行的算法
    instance_ids = list(results.keys())
    if not instance_ids:
        return
    
    # 从第一个实例中获取实际运行的算法列表
    algorithms = list(results[instance_ids[0]].keys())
    
    # 1. 算法性能对比
    if len(algorithms) > 0:
        # 时间对比
        plt.figure(figsize=(12, 6))
        for algorithm in algorithms:
            times = [results[inst_id][algorithm]['execution_time'] for inst_id in instance_ids]
            plt.bar([f"{inst_id}_{algorithm[:3]}" for inst_id in instance_ids], times, label=algorithm)
        plt.xlabel('算例_算法')
        plt.ylabel('执行时间 (秒)')
        plt.title('不同算法执行时间对比')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_dir, 'execution_time_comparison.png'), dpi=300)
        plt.close()
        
        # 周长对比
        plt.figure(figsize=(12, 6))
        for algorithm in algorithms:
            perimeters = [results[inst_id][algorithm]['total_perimeter'] for inst_id in instance_ids]
            plt.bar([f"{inst_id}_{algorithm[:3]}" for inst_id in instance_ids], perimeters, label=algorithm)
        plt.xlabel('算例_算法')
        plt.ylabel('总周长 (m)')
        plt.title('不同算法总周长对比')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(visualizations_dir, 'perimeter_comparison.png'), dpi=300)
        plt.close()
    
    # 2. 性能雷达图
    for instance_id in instance_ids:
        metrics = {}
        for algorithm in algorithms:
            if algorithm in results[instance_id]:
                algo_metrics = results[instance_id][algorithm]
                # 归一化指标
                metrics[algorithm] = {
                    '时间效率': 1.0 / (algo_metrics['execution_time'] / 100 + 1),  # 时间越短越好
                    '周长优化': 1.0 / (algo_metrics['total_perimeter'] / 1000 + 1),  # 周长越短越好
                    '可行性': 1.0 if algo_metrics['is_feasible'] else 0.0,
                    '分区平衡性': 1.0 - np.std([z['pva_count'] for z in results[instance_id][algorithm].get('zone_summary', [])]) / 10 if results[instance_id][algorithm].get('zone_summary') else 0.0
                }
        
        # 生成雷达图
        if metrics:
            categories = list(list(metrics.values())[0].keys())
            N = len(categories)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]
            
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(111, polar=True)
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
            for i, (algorithm, values) in enumerate(metrics.items()):
                data = list(values.values())
                data += data[:1]
                ax.plot(angles, data, linewidth=2, linestyle='solid', label=algorithm, color=colors[i])
                ax.fill(angles, data, alpha=0.25, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            plt.title(f'算例 {instance_id} 算法性能雷达图')
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.tight_layout()
            plt.savefig(os.path.join(visualizations_dir, f'radar_{instance_id}.png'), dpi=300)
            plt.close()

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='算法评估脚本')
    parser.add_argument('--model-path', type=str, default='outputs/checkpoints/best_model.pt',
                      help='预训练模型路径')
    parser.add_argument('--compare-heuristic', action='store_true',
                      help='是否与启发式算法比较')
    args = parser.parse_args()
    
    # 加载算例
    instances_dir = 'data/processed/PV/public/easy'
    instances = []
    
    for filename in os.listdir(instances_dir):
        if filename.endswith('.json'):
            # 提取实例ID（如从public_easy_r1.json中提取r1）
            instance_id = filename.replace('public_easy_', '').replace('.json', '')
            instance_data = load_instance(instance_id)
            instances.append(instance_data)
    
    # 限制测试算例数量（可选）
    # instances = instances[:5]  # 只测试前5个算例
    # 评估所有算例
    print(f"共加载 {len(instances)} 个算例进行评估")
    
    # 初始化DQN智能体（如果需要）
    dqn_agent = None
    try:
        from modules.module1.algorithm.dqn_agent import DQNPartitionAgent
        dqn_agent = DQNPartitionAgent(dim_embed=128, T=6)
        # 加载预训练模型（如果有）
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, args.model_path)
        if os.path.exists(model_path):
            import torch
            dqn_agent.load_model(model_path)
            print(f"已加载预训练DQN模型: {model_path}")
        else:
            print(f"未找到预训练模型: {model_path}")
    except Exception as e:
        print(f"初始化DQN智能体失败: {e}")
    
    # 比较算法
    print("开始评估算法...")
    if args.compare_heuristic:
        # 只比较启发式算法
        algorithms = ['heuristic']
        results = {}
        for instance in instances:
            instance_id = instance['instance_info']['instance_id']
            results[instance_id] = {}
            for algorithm in algorithms:
                print(f"评估算例 {instance_id} 使用 {algorithm} 算法...")
                metrics = evaluate_algorithm(instance, algorithm, dqn_agent)
                results[instance_id][algorithm] = metrics
    else:
        # 比较所有算法
        results = compare_algorithms(instances, dqn_agent)
    
    # 生成报告
    print("生成性能报告...")
    generate_performance_report(results)
    
    print("评估完成！")


if __name__ == '__main__':
    main()