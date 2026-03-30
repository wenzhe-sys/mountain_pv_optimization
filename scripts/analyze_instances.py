#!/usr/bin/env python3
"""
分析17个开源算例的基本信息和差异
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (12, 8)  # 默认图表大小
plt.rcParams['figure.dpi'] = 100  # 默认分辨率

class InstanceAnalyzer:
    def __init__(self, instances_dir):
        self.instances_dir = instances_dir
        self.instances = []
        self.load_instances()
    
    def load_instances(self):
        """加载所有算例数据"""
        for i in range(1, 18):
            instance_id = f"r{i}"
            file_path = os.path.join(self.instances_dir, f"public_easy_{instance_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.instances.append({
                        'id': instance_id,
                        'data': data
                    })
                print(f"加载算例 {instance_id} 成功")
            else:
                print(f"算例 {instance_id} 文件不存在")
    
    def analyze_basic_info(self):
        """分析基本信息"""
        print("\n=== 算例基本信息 ===")
        for instance in self.instances:
            info = instance['data']['instance_info']
            print(f"算例 {instance['id']}:")
            print(f"  节点数量: {info['n_nodes']}")
            print(f"  逆变器坐标: {info['inverter_coord']}")
            print(f"  面板数量: {len(instance['data']['pva_list'])}")
    
    def analyze_panel_distribution(self):
        """分析面板分布"""
        print("\n=== 面板分布分析 ===")
        
        # 统计每个算例的面板数量
        panel_counts = []
        instance_ids = []
        
        for instance in self.instances:
            instance_ids.append(instance['id'])
            panel_counts.append(len(instance['data']['pva_list']))
        
        # 绘制面板数量柱状图
        plt.figure(figsize=(14, 7))
        bars = plt.bar(instance_ids, panel_counts, color='skyblue')
        plt.xlabel('算例ID', fontsize=14)
        plt.ylabel('面板数量', fontsize=14)
        plt.title('各算例面板数量分布', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/instance_panel_counts.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("面板数量分布图已保存")
    
    def analyze_inverter_positions(self):
        """分析逆变器位置"""
        print("\n=== 逆变器位置分析 ===")
        
        x_coords = []
        y_coords = []
        instance_ids = []
        
        for instance in self.instances:
            instance_ids.append(instance['id'])
            x, y = instance['data']['instance_info']['inverter_coord']
            x_coords.append(x)
            y_coords.append(y)
        
        # 绘制逆变器位置散点图
        plt.figure(figsize=(15, 10))
        scatter = plt.scatter(x_coords, y_coords, s=150, c=range(len(self.instances)), cmap='viridis', alpha=0.8)
        cbar = plt.colorbar(scatter, label='算例序号')
        cbar.ax.tick_params(labelsize=12)
        
        # 添加算例标签（优化布局，避免重叠）
        for i, (x, y, instance_id) in enumerate(zip(x_coords, y_coords, instance_ids)):
            plt.annotate(instance_id, (x, y), xytext=(10, 10), textcoords='offset points',
                        fontsize=11, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
        
        plt.xlabel('X坐标 (m)', fontsize=14)
        plt.ylabel('Y坐标 (m)', fontsize=14)
        plt.title('各算例逆变器位置分布', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig('outputs/visualizations/inverter_positions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("逆变器位置分布图已保存")
    
    def analyze_terrain(self):
        """分析地形数据"""
        print("\n=== 地形数据分析 ===")
        
        # 存储地形数据用于可视化
        grid_sizes = []
        avg_slopes = []
        buildable_ratios = []
        instance_ids = []
        
        for instance in self.instances:
            instance_ids.append(instance['id'])
            terrain = instance['data']['terrain_data']
            grid_size = terrain['grid_size']
            slope_matrix = np.array(terrain['slope_matrix'])
            buildable_matrix = np.array(terrain['buildable_matrix'])
            
            grid_sizes.append(grid_size)
            avg_slopes.append(np.mean(slope_matrix))
            buildable_ratios.append(np.mean(buildable_matrix))
            
            print(f"算例 {instance['id']}:")
            print(f"  网格大小: {grid_size}")
            print(f"  平均坡度: {np.mean(slope_matrix):.4f}")
            print(f"  可建区域比例: {np.mean(buildable_matrix):.4f}")
        
        # 绘制地形参数对比图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 平均坡度图
        axes[0].bar(instance_ids, avg_slopes, color='lightgreen')
        axes[0].set_xlabel('算例ID', fontsize=12)
        axes[0].set_ylabel('平均坡度', fontsize=12)
        axes[0].set_title('各算例平均坡度', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 可建区域比例图
        axes[1].bar(instance_ids, buildable_ratios, color='lightblue')
        axes[1].set_xlabel('算例ID', fontsize=12)
        axes[1].set_ylabel('可建区域比例', fontsize=12)
        axes[1].set_title('各算例可建区域比例', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/terrain_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("地形分析图已保存")
    
    def generate_summary(self):
        """生成总结报告"""
        print("\n=== 算例分析总结 ===")
        print(f"总共有 {len(self.instances)} 个算例")
        
        # 统计面板数量
        panel_counts = [len(instance['data']['pva_list']) for instance in self.instances]
        print(f"面板数量范围: {min(panel_counts)} - {max(panel_counts)}")
        print(f"平均面板数量: {np.mean(panel_counts):.1f}")
        
        # 统计逆变器位置
        x_coords = [instance['data']['instance_info']['inverter_coord'][0] for instance in self.instances]
        y_coords = [instance['data']['instance_info']['inverter_coord'][1] for instance in self.instances]
        print(f"逆变器X坐标范围: {min(x_coords):.1f} - {max(x_coords):.1f}")
        print(f"逆变器Y坐标范围: {min(y_coords):.1f} - {max(y_coords):.1f}")
        
        # 生成综合对比图
        plt.figure(figsize=(18, 10))
        
        # 面板数量分布
        plt.subplot(2, 2, 1)
        panel_counts = [len(instance['data']['pva_list']) for instance in self.instances]
        plt.bar([instance['id'] for instance in self.instances], panel_counts, color='skyblue')
        plt.title('面板数量分布', fontsize=14, fontweight='bold')
        plt.xlabel('算例ID')
        plt.ylabel('面板数量')
        plt.xticks(rotation=45)
        
        # 逆变器X坐标分布
        plt.subplot(2, 2, 2)
        x_coords = [instance['data']['instance_info']['inverter_coord'][0] for instance in self.instances]
        plt.bar([instance['id'] for instance in self.instances], x_coords, color='lightgreen')
        plt.title('逆变器X坐标分布', fontsize=14, fontweight='bold')
        plt.xlabel('算例ID')
        plt.ylabel('X坐标 (m)')
        plt.xticks(rotation=45)
        
        # 逆变器Y坐标分布
        plt.subplot(2, 2, 3)
        y_coords = [instance['data']['instance_info']['inverter_coord'][1] for instance in self.instances]
        plt.bar([instance['id'] for instance in self.instances], y_coords, color='lightcoral')
        plt.title('逆变器Y坐标分布', fontsize=14, fontweight='bold')
        plt.xlabel('算例ID')
        plt.ylabel('Y坐标 (m)')
        plt.xticks(rotation=45)
        
        # 面板分布密度
        plt.subplot(2, 2, 4)
        x_positions = []
        y_positions = []
        for instance in self.instances[:3]:  # 只显示前3个算例的面板分布
            for pva in instance['data']['pva_list'][:20]:  # 每个算例只显示前20个面板
                x_positions.append(pva['x'])
                y_positions.append(pva['y'])
        plt.scatter(x_positions, y_positions, alpha=0.6, s=50)
        plt.title('面板分布密度示例', fontsize=14, fontweight='bold')
        plt.xlabel('X坐标 (m)')
        plt.ylabel('Y坐标 (m)')
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n所有分析图表已保存到 outputs/visualizations/ 目录")
        print("综合分析图已保存")

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs('outputs/visualizations', exist_ok=True)
    
    # 分析算例
    analyzer = InstanceAnalyzer('data/processed/PV/public/easy')
    analyzer.analyze_basic_info()
    analyzer.analyze_panel_distribution()
    analyzer.analyze_inverter_positions()
    analyzer.analyze_terrain()
    analyzer.generate_summary()
    
    print("\n分析完成！")