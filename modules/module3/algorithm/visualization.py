import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import os
import logging
import numpy as np
from typing import List, Dict

class VisualizationManager:
    """
    可视化管理器
    实现更丰富的可视化功能
    """
    def __init__(self, save_dir: str = "visualizations"):
        """
        初始化可视化管理器
        Args:
            save_dir: 可视化结果保存目录
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置可视化风格
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def visualize_pareto_front(self, pareto_front: List[Dict], filename: str = "pareto_front.png"):
        """
        可视化帕累托最优前沿
        Args:
            pareto_front: 帕累托最优解列表
            filename: 保存文件名
        """
        if not pareto_front:
            return
        
        try:
            # 提取数据
            costs = [item["cost"] for item in pareto_front]
            efficiencies = [item["efficiency"] for item in pareto_front]
            reliabilities = [item["reliability"] for item in pareto_front]
            
            # 创建3D散点图
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制散点
            scatter = ax.scatter(costs, efficiencies, reliabilities, 
                               c=costs, cmap='viridis', s=50, alpha=0.7)
            
            # 添加颜色条
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('成本 (万元)')
            
            # 设置标签
            ax.set_xlabel('成本 (万元)')
            ax.set_ylabel('效率')
            ax.set_zlabel('可靠性')
            ax.set_title('帕累托最优前沿')
            
            # 保存图片
            save_path = os.path.join(self.save_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            logging.info(f"成功保存帕累托前沿可视化: {save_path}")
        except Exception as e:
            logging.error(f"可视化帕累托前沿失败: {e}")
    
    def visualize_optimization_history(self, history: List[Dict], filename: str = "optimization_history.png"):
        """
        可视化优化历史
        Args:
            history: 优化历史记录
            filename: 保存文件名
        """
        if not history:
            return
        
        try:
            # 提取数据
            iterations = [item["iteration"] for item in history]
            costs = [item["cost"] for item in history]
            efficiencies = [item["efficiency"] for item in history]
            reliabilities = [item["reliability"] for item in history]
            
            # 创建DataFrame
            df = pd.DataFrame({
                'Iteration': iterations,
                'Cost': costs,
                'Efficiency': efficiencies,
                'Reliability': reliabilities
            })
            
            # 创建子图
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # 绘制成本曲线
            sns.lineplot(x='Iteration', y='Cost', data=df, ax=axes[0], linewidth=2)
            axes[0].set_ylabel('成本 (万元)')
            axes[0].set_title('优化过程')
            axes[0].grid(True, alpha=0.3)
            
            # 绘制效率曲线
            sns.lineplot(x='Iteration', y='Efficiency', data=df, ax=axes[1], linewidth=2, color='green')
            axes[1].set_ylabel('效率')
            axes[1].grid(True, alpha=0.3)
            
            # 绘制可靠性曲线
            sns.lineplot(x='Iteration', y='Reliability', data=df, ax=axes[2], linewidth=2, color='orange')
            axes[2].set_ylabel('可靠性')
            axes[2].set_xlabel('迭代次数')
            axes[2].grid(True, alpha=0.3)
            
            # 保存图片
            save_path = os.path.join(self.save_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            logging.info(f"成功保存优化历史可视化: {save_path}")
        except Exception as e:
            logging.error(f"可视化优化历史失败: {e}")
    
    def visualize_performance_metrics(self, metrics: Dict, filename: str = "performance_metrics.png"):
        """
        可视化性能指标
        Args:
            metrics: 性能指标字典
            filename: 保存文件名
        """
        if not metrics:
            return
        
        try:
            # 提取数据
            improvement_keys = ['cost_improvement', 'efficiency_improvement', 'reliability_improvement']
            improvement_values = [metrics.get(key, 0) for key in improvement_keys]
            
            # 创建柱状图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制柱状图
            bars = ax.bar(improvement_keys, improvement_values, color=['blue', 'green', 'orange'])
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.2f}%', ha='center', va='bottom')
            
            # 设置标签
            ax.set_ylabel('改进百分比 (%)')
            ax.set_title('性能改进指标')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 保存图片
            save_path = os.path.join(self.save_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            logging.info(f"成功保存性能指标可视化: {save_path}")
        except Exception as e:
            logging.error(f"可视化性能指标失败: {e}")
    
    def create_dashboard(self, optimization_result: Dict, filename: str = "dashboard.png"):
        """
        创建综合仪表盘
        Args:
            optimization_result: 优化结果字典
            filename: 保存文件名
        """
        try:
            # 创建子图
            fig = plt.figure(figsize=(15, 12))
            
            # 1. 成本摘要
            ax1 = fig.add_subplot(2, 2, 1)
            cost_summary = optimization_result.get('total_cost_summary', {})
            cost_breakdown = cost_summary.get('cost_breakdown', {})
            
            if cost_breakdown:
                labels = list(cost_breakdown.keys())
                values = list(cost_breakdown.values())
                ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax1.set_title('成本 breakdown')
            
            # 2. 性能指标
            ax2 = fig.add_subplot(2, 2, 2)
            performance_metrics = optimization_result.get('performance_metrics', {})
            performance_evaluation = performance_metrics.get('performance_evaluation', {})
            
            if performance_evaluation:
                improvement_keys = ['cost_improvement', 'efficiency_improvement', 'reliability_improvement']
                improvement_values = [performance_evaluation.get(key, 0) for key in improvement_keys]
                ax2.bar(improvement_keys, improvement_values, color=['blue', 'green', 'orange'])
                ax2.set_ylabel('改进百分比 (%)')
                ax2.set_title('性能改进指标')
                ax2.grid(True, alpha=0.3, axis='y')
            
            # 3. 优化参数
            ax3 = fig.add_subplot(2, 2, 3)
            optimized_params = optimization_result.get('optimized_params', {})
            
            if optimized_params:
                params = list(optimized_params.keys())[:6]  # 只显示前6个参数
                values = [optimized_params.get(key, 0) for key in params]
                ax3.barh(params, values, color='skyblue')
                ax3.set_xlabel('值')
                ax3.set_title('优化参数')
            
            # 4. 帕累托前沿
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            pareto_front = optimization_result.get('pareto_front', [])
            
            if pareto_front:
                costs = [item["cost"] for item in pareto_front]
                efficiencies = [item["efficiency"] for item in pareto_front]
                reliabilities = [item["reliability"] for item in pareto_front]
                
                scatter = ax4.scatter(costs, efficiencies, reliabilities, 
                                   c=costs, cmap='viridis', s=30, alpha=0.7)
                ax4.set_xlabel('成本')
                ax4.set_ylabel('效率')
                ax4.set_zlabel('可靠性')
                ax4.set_title('帕累托最优前沿')
            
            # 保存图片
            save_path = os.path.join(self.save_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            logging.info(f"成功保存综合仪表盘: {save_path}")
        except Exception as e:
            logging.error(f"创建综合仪表盘失败: {e}")