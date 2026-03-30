#!/usr/bin/env python3
"""
生成扩展算例，包括不同地形类型、面板数量和布局
"""

import json
import os
import numpy as np
import random

class ExtendedInstanceGenerator:
    def __init__(self, base_instances_dir, output_dir):
        self.base_instances_dir = base_instances_dir
        self.output_dir = output_dir
        self.base_instances = []
        self.load_base_instances()
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_base_instances(self):
        """加载基础算例"""
        for i in range(1, 18):
            instance_id = f"r{i}"
            file_path = os.path.join(self.base_instances_dir, f"public_easy_{instance_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.base_instances.append({
                        'id': instance_id,
                        'data': data
                    })
                print(f"加载基础算例 {instance_id} 成功")
    
    def generate_mountain_terrain(self, grid_size, difficulty='medium'):
        """生成山地地形"""
        if difficulty == 'easy':
            slope_range = (0, 0.1)
        elif difficulty == 'medium':
            slope_range = (0, 0.3)
        else:  # hard
            slope_range = (0, 0.5)
        
        slope_matrix = np.random.uniform(*slope_range, (grid_size, grid_size)).tolist()
        
        # 生成可建区域（部分区域不可建）
        buildable_matrix = np.ones((grid_size, grid_size), dtype=bool)
        if difficulty in ['medium', 'hard']:
            # 随机设置一些区域不可建
            for i in range(grid_size):
                for j in range(grid_size):
                    if random.random() < 0.1:  # 10%的概率不可建
                        buildable_matrix[i, j] = False
        
        return {
            'grid_size': grid_size,
            'slope_matrix': slope_matrix,
            'buildable_matrix': buildable_matrix.tolist()
        }
    
    def generate_panel_layout(self, base_instance, panel_count, layout_type='random'):
        """生成不同的面板布局"""
        base_pva_list = base_instance['data']['pva_list']
        base_x = [pva['x'] for pva in base_pva_list]
        base_y = [pva['y'] for pva in base_pva_list]
        
        x_min, x_max = min(base_x), max(base_x)
        y_min, y_max = min(base_y), max(base_y)
        
        pva_list = []
        for i in range(panel_count):
            if layout_type == 'random':
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
            elif layout_type == 'grid':
                grid_cols = int(np.sqrt(panel_count))
                grid_rows = (panel_count + grid_cols - 1) // grid_cols
                col = i % grid_cols
                row = i // grid_cols
                x = x_min + (x_max - x_min) * col / (grid_cols - 1) if grid_cols > 1 else x_min
                y = y_min + (y_max - y_min) * row / (grid_rows - 1) if grid_rows > 1 else y_min
            elif layout_type == 'clustered':
                # 生成几个聚类中心
                num_clusters = 3
                cluster_centers = [(random.uniform(x_min, x_max), random.uniform(y_min, y_max)) 
                                 for _ in range(num_clusters)]
                center_idx = i % num_clusters
                center_x, center_y = cluster_centers[center_idx]
                x = center_x + random.uniform(-20, 20)
                y = center_y + random.uniform(-20, 20)
            
            # 计算网格坐标
            grid_x = int(x // 10)
            grid_y = int(y // 10)
            
            pva_list.append({
                'panel_id': f"pva_{i}",
                'x': round(x, 2),
                'y': round(y, 2),
                'grid_coord': [grid_y, grid_x],  # 注意顺序
                'cut_spec': [2.0, 3.0]  # 保持面板尺寸一致
            })
        
        return pva_list
    
    def generate_extended_instances(self):
        """生成扩展算例"""
        extended_instances = []
        
        # 1. 不同地形类型
        terrain_types = [
            {'name': 'flat', 'difficulty': 'easy', 'description': '平坦地形'},
            {'name': 'gentle_hill', 'difficulty': 'medium', 'description': '平缓山地'},
            {'name': 'steep_hill', 'difficulty': 'hard', 'description': '陡峭山地'}
        ]
        
        # 2. 不同面板数量
        panel_counts = [50, 150, 200, 300]
        
        # 3. 不同布局类型
        layout_types = ['random', 'grid', 'clustered']
        
        instance_id = 18  # 从18开始编号
        
        for base_instance in self.base_instances[:3]:  # 使用前3个基础算例作为模板
            for terrain in terrain_types:
                for panel_count in panel_counts:
                    for layout in layout_types:
                        # 生成新算例
                        new_instance = {
                            'instance_info': {
                                'instance_id': f"r{instance_id}",
                                'type': 'extended',
                                'difficulty': terrain['difficulty'],
                                'n_nodes': panel_count,
                                'inverter_coord': base_instance['data']['instance_info']['inverter_coord'],
                                'unit': 'm',
                                'source': 'Luo开源PV算例（扩展）',
                                'version': 'v1.1',
                                'desensitization_info': {
                                    'is_desensitized': False,
                                    'note': '扩展算例，无敏感信息'
                                },
                                'terrain_type': terrain['name'],
                                'terrain_description': terrain['description'],
                                'panel_layout': layout,
                                'panel_count': panel_count
                            },
                            'pva_list': self.generate_panel_layout(base_instance, panel_count, layout),
                            'terrain_data': self.generate_mountain_terrain(10, terrain['difficulty'])
                        }
                        
                        # 保存算例
                        output_file = os.path.join(self.output_dir, f"public_{terrain['difficulty']}_r{instance_id}.json")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(new_instance, f, ensure_ascii=False, indent=2)
                        
                        print(f"生成扩展算例 r{instance_id} - 地形: {terrain['name']}, 面板数: {panel_count}, 布局: {layout}")
                        extended_instances.append(new_instance)
                        instance_id += 1
        
        return extended_instances
    
    def generate_summary(self, extended_instances):
        """生成扩展算例总结"""
        print("\n=== 扩展算例生成总结 ===")
        print(f"共生成 {len(extended_instances)} 个扩展算例")
        
        # 统计不同类型的算例
        terrain_counts = {}
        panel_counts = {}
        layout_counts = {}
        
        for instance in extended_instances:
            terrain = instance['instance_info']['terrain_type']
            panel_count = instance['instance_info']['panel_count']
            layout = instance['instance_info']['panel_layout']
            
            terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
            panel_counts[panel_count] = panel_counts.get(panel_count, 0) + 1
            layout_counts[layout] = layout_counts.get(layout, 0) + 1
        
        print("\n地形类型分布:")
        for terrain, count in terrain_counts.items():
            print(f"  {terrain}: {count} 个")
        
        print("\n面板数量分布:")
        for panel_count, count in panel_counts.items():
            print(f"  {panel_count}: {count} 个")
        
        print("\n布局类型分布:")
        for layout, count in layout_counts.items():
            print(f"  {layout}: {count} 个")
        
        print("\n扩展算例已保存到:", self.output_dir)

if __name__ == "__main__":
    # 基础算例目录
    base_dir = 'data/processed/PV/public/easy'
    # 扩展算例输出目录
    output_dir = 'data/processed/PV/public/extended'
    
    generator = ExtendedInstanceGenerator(base_dir, output_dir)
    extended_instances = generator.generate_extended_instances()
    generator.generate_summary(extended_instances)
    
    print("\n扩展算例生成完成！")