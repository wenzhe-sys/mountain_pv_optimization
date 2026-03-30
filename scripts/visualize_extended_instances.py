#!/usr/bin/env python3
"""
可视化扩展算例的地形和面板布局
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

class ExtendedInstanceVisualizer:
    def __init__(self, instances_dir):
        self.instances_dir = instances_dir
        self.instances = []
        self.load_instances()
        
        # 确保输出目录存在
        os.makedirs('outputs/visualizations/extended', exist_ok=True)
    
    def load_instances(self):
        """加载扩展算例"""
        for file in os.listdir(self.instances_dir):
            if file.endswith('.json'):
                file_path = os.path.join(self.instances_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.instances.append({
                        'file': file,
                        'data': data
                    })
                print(f"加载扩展算例 {file} 成功")
    
    def visualize_terrain_heatmap(self, instance):
        """可视化地形热力图"""
        terrain = instance['data']['terrain_data']
        slope_matrix = np.array(terrain['slope_matrix'])
        instance_id = instance['data']['instance_info']['instance_id']
        terrain_type = instance['data']['instance_info']['terrain_type']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(slope_matrix, annot=True, cmap='viridis', fmt='.2f', cbar_kws={'label': '坡度'})
        plt.title(f'算例 {instance_id} 地形坡度热力图 ({terrain_type})', fontsize=14, fontweight='bold')
        plt.xlabel('网格X坐标')
        plt.ylabel('网格Y坐标')
        plt.tight_layout()
        
        output_file = f'outputs/visualizations/extended/{instance_id}_terrain.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"地形热力图已保存: {output_file}")
    
    def visualize_panel_layout(self, instance):
        """可视化面板布局"""
        pva_list = instance['data']['pva_list']
        instance_id = instance['data']['instance_info']['instance_id']
        panel_count = len(pva_list)
        layout_type = instance['data']['instance_info']['panel_layout']
        
        x_coords = [pva['x'] for pva in pva_list]
        y_coords = [pva['y'] for pva in pva_list]
        inverter_x, inverter_y = instance['data']['instance_info']['inverter_coord']
        
        plt.figure(figsize=(12, 10))
        plt.scatter(x_coords, y_coords, s=50, alpha=0.7, label=f'光伏面板 ({panel_count}块)')
        plt.scatter(inverter_x, inverter_y, s=200, c='red', marker='^', label='逆变器')
        
        # 连接面板到逆变器
        for x, y in zip(x_coords, y_coords):
            plt.plot([x, inverter_x], [y, inverter_y], 'k-', alpha=0.1)
        
        plt.xlabel('X坐标 (m)', fontsize=12)
        plt.ylabel('Y坐标 (m)', fontsize=12)
        plt.title(f'算例 {instance_id} 面板布局 ({layout_type}布局)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = f'outputs/visualizations/extended/{instance_id}_layout.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"面板布局图已保存: {output_file}")
    
    def visualize_terrain_comparison(self):
        """对比不同地形类型"""
        terrain_types = {}
        for instance in self.instances:
            terrain_type = instance['data']['instance_info']['terrain_type']
            if terrain_type not in terrain_types:
                terrain_types[terrain_type] = []
            terrain_types[terrain_type].append(instance)
        
        # 绘制不同地形类型的坡度分布
        plt.figure(figsize=(15, 10))
        for i, (terrain_type, instances) in enumerate(terrain_types.items()):
            slopes = []
            for instance in instances:
                slope_matrix = np.array(instance['data']['terrain_data']['slope_matrix'])
                slopes.extend(slope_matrix.flatten())
            
            plt.subplot(len(terrain_types), 1, i+1)
            plt.hist(slopes, bins=20, alpha=0.7, label=terrain_type)
            plt.title(f'{terrain_type} 地形坡度分布', fontsize=14, fontweight='bold')
            plt.xlabel('坡度')
            plt.ylabel('频率')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = 'outputs/visualizations/extended/terrain_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"地形对比图已保存: {output_file}")
    
    def visualize_layout_comparison(self):
        """对比不同布局类型"""
        layout_types = {}
        for instance in self.instances:
            layout_type = instance['data']['instance_info']['panel_layout']
            if layout_type not in layout_types:
                layout_types[layout_type] = []
            layout_types[layout_type].append(instance)
        
        # 绘制不同布局类型的面板分布
        plt.figure(figsize=(18, 6))
        for i, (layout_type, instances) in enumerate(layout_types.items()):
            if instances:
                instance = instances[0]  # 取每个类型的第一个算例
                pva_list = instance['data']['pva_list']
                x_coords = [pva['x'] for pva in pva_list]
                y_coords = [pva['y'] for pva in pva_list]
                inverter_x, inverter_y = instance['data']['instance_info']['inverter_coord']
                
                plt.subplot(1, len(layout_types), i+1)
                plt.scatter(x_coords, y_coords, s=30, alpha=0.7, label='光伏面板')
                plt.scatter(inverter_x, inverter_y, s=150, c='red', marker='^', label='逆变器')
                plt.title(f'{layout_type} 布局', fontsize=14, fontweight='bold')
                plt.xlabel('X坐标 (m)')
                plt.ylabel('Y坐标 (m)')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = 'outputs/visualizations/extended/layout_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"布局对比图已保存: {output_file}")
    
    def visualize_panel_count_comparison(self):
        """对比不同面板数量"""
        panel_counts = {}
        for instance in self.instances:
            panel_count = instance['data']['instance_info']['panel_count']
            if panel_count not in panel_counts:
                panel_counts[panel_count] = []
            panel_counts[panel_count].append(instance)
        
        # 绘制不同面板数量的分布
        plt.figure(figsize=(12, 8))
        counts = [len(instances) for instances in panel_counts.values()]
        labels = list(panel_counts.keys())
        
        plt.bar(labels, counts, color='skyblue')
        plt.title('不同面板数量的算例分布', fontsize=16, fontweight='bold')
        plt.xlabel('面板数量', fontsize=14)
        plt.ylabel('算例数量', fontsize=14)
        
        # 添加数据标签
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, f'{count}', ha='center', fontsize=12)
        
        plt.tight_layout()
        output_file = 'outputs/visualizations/extended/panel_count_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"面板数量对比图已保存: {output_file}")
    
    def generate_all_visualizations(self):
        """生成所有可视化"""
        print("\n=== 生成扩展算例可视化 ===")
        
        # 为每个算例生成地形和布局图
        for instance in self.instances[:5]:  # 只处理前5个算例以节省时间
            self.visualize_terrain_heatmap(instance)
            self.visualize_panel_layout(instance)
        
        # 生成对比图
        self.visualize_terrain_comparison()
        self.visualize_layout_comparison()
        self.visualize_panel_count_comparison()
        
        print("\n所有可视化已完成！")

if __name__ == "__main__":
    # 扩展算例目录
    instances_dir = 'data/processed/PV/public/extended'
    
    visualizer = ExtendedInstanceVisualizer(instances_dir)
    visualizer.generate_all_visualizations()
    
    print("\n扩展算例可视化完成！")