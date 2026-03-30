# modules/module2/visualization/cable_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple


class CableVisualizer:
    """电缆路由和共沟可视化工具"""
    
    def __init__(self, grid_size: float):
        """
        初始化可视化工具
        
        Args:
            grid_size: 网格大小
        """
        self.grid_size = grid_size
    
    def visualize_cable_routes(self, cable_routes: List[Dict],
                              substation_coord: Tuple[float, float],
                              title: str = "电缆路由图",
                              save_path: str = None) -> None:
        """
        可视化电缆路由
        
        Args:
            cable_routes: 电缆路由列表
            substation_coord: 升压站坐标
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 创建图形
        plt.figure(figsize=(15, 12))
        
        # 绘制升压站
        plt.plot(substation_coord[0], substation_coord[1], 'rs', markersize=15, label='升压站')
        
        # 绘制电缆路由
        colors = plt.cm.get_cmap('tab20', len(cable_routes))
        for i, route in enumerate(cable_routes):
            path = route.get('path', [])
            if path:
                x = [p[0] for p in path]
                y = [p[1] for p in path]
                plt.plot(x, y, '-', color=colors(i), linewidth=2, 
                        label=f'电缆 {route.get("cable_id", i+1)}')
                # 绘制起点和终点
                plt.plot(x[0], y[0], 'bo', markersize=8)
                plt.plot(x[-1], y[-1], 'go', markersize=8)
        
        # 设置标题和布局
        plt.title(title)
        plt.xlabel('X 坐标 (m)')
        plt.ylabel('Y 坐标 (m)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"电缆路由图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_trench_summary(self, trench_summary: List[Dict],
                               title: str = "管沟汇总图",
                               save_path: str = None) -> None:
        """
        可视化管沟汇总
        
        Args:
            trench_summary: 管沟汇总列表
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 提取数据
        trench_ids = [trench.get('trench_id', f'trench_{i}') for i, trench in enumerate(trench_summary)]
        cable_counts = [trench.get('cable_count', 0) for trench in trench_summary]
        lengths = [trench.get('length', 0) for trench in trench_summary]
        costs = [trench.get('cost', 0) for trench in trench_summary]
        
        # 创建子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 绘制电缆数
        ax1.bar(trench_ids, cable_counts, color='skyblue')
        ax1.set_xlabel('管沟ID')
        ax1.set_ylabel('电缆数')
        ax1.set_title('各管沟电缆数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 绘制长度
        ax2.bar(trench_ids, lengths, color='lightgreen')
        ax2.set_xlabel('管沟ID')
        ax2.set_ylabel('长度 (m)')
        ax2.set_title('各管沟长度')
        ax2.tick_params(axis='x', rotation=45)
        
        # 绘制成本
        ax3.bar(trench_ids, costs, color='salmon')
        ax3.set_xlabel('管沟ID')
        ax3.set_ylabel('成本 (万元)')
        ax3.set_title('各管沟成本')
        ax3.tick_params(axis='x', rotation=45)
        
        # 设置标题和布局
        plt.suptitle(title)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"管沟汇总图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_equipment_selection(self, equipment_selection: List[Dict],
                                    substation_coord: Tuple[float, float],
                                    title: str = "箱变选址图",
                                    save_path: str = None) -> None:
        """
        可视化箱变选址
        
        Args:
            equipment_selection: 设备选型列表
            substation_coord: 升压站坐标
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 创建图形
        plt.figure(figsize=(15, 12))
        
        # 绘制升压站
        plt.plot(substation_coord[0], substation_coord[1], 'rs', markersize=15, label='升压站')
        
        # 绘制箱变
        colors = {1600: 'blue', 3200: 'green'}
        for equip in equipment_selection:
            coord = equip.get('install_coord', (0, 0))
            capacity = equip.get('Q_box', 1600)
            plt.plot(coord[0], coord[1], 'o', markersize=12, 
                    color=colors.get(capacity, 'black'), 
                    label=f'箱变 {equip.get("box_id", "")} ({capacity}kVA)')
            
            # 绘制箱变到升压站的连线
            plt.plot([coord[0], substation_coord[0]], 
                    [coord[1], substation_coord[1]], 
                    '--', color=colors.get(capacity, 'black'), alpha=0.5)
        
        # 设置标题和布局
        plt.title(title)
        plt.xlabel('X 坐标 (m)')
        plt.ylabel('Y 坐标 (m)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"箱变选址图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_cost_breakdown(self, total_cost: float,
                               equipment_cost: float,
                               cable_cost: float,
                               trench_cost: float,
                               title: str = "成本构成饼图",
                               save_path: str = None) -> None:
        """
        可视化成本构成
        
        Args:
            total_cost: 总成本
            equipment_cost: 设备成本
            cable_cost: 电缆成本
            trench_cost: 管沟成本
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 准备数据
        labels = ['设备成本', '电缆成本', '管沟成本']
        sizes = [equipment_cost, cable_cost, trench_cost]
        colors = ['gold', 'lightcoral', 'lightskyblue']
        explode = (0.1, 0, 0)  # 突出显示设备成本
        
        # 绘制饼图
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')  # 确保饼图是圆形
        
        # 设置标题
        plt.title(f"{title}\n总成本: {total_cost:.2f}万元")
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"成本构成饼图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_column_generation(self, convergence_history: List[Dict],
                                  title: str = "列生成收敛曲线",
                                  save_path: str = None) -> None:
        """
        可视化列生成收敛曲线
        
        Args:
            convergence_history: 收敛历史记录
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 提取数据
        iterations = [h['iteration'] for h in convergence_history]
        objectives = [h['objective'] for h in convergence_history]
        active_paths = [h['n_active_paths'] for h in convergence_history]
        total_paths = [h['n_total_paths'] for h in convergence_history]
        
        # 创建图形
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 绘制目标函数值
        ax1.plot(iterations, objectives, 'b-', label='目标函数值')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('目标函数值')
        ax1.legend(loc='upper left')
        
        # 绘制路径数量
        ax2 = ax1.twinx()
        ax2.plot(iterations, active_paths, 'g--', label='活跃路径数')
        ax2.plot(iterations, total_paths, 'r--', label='总路径数')
        ax2.set_ylabel('路径数量')
        ax2.legend(loc='upper right')
        
        # 设置标题和布局
        plt.title(title)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"列生成收敛曲线已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_branch_and_bound(self, branch_and_bound_data: List[Dict],
                                 title: str = "分支定界树探索过程",
                                 save_path: str = None) -> None:
        """
        可视化分支定界树探索过程
        
        Args:
            branch_and_bound_data: 分支定界数据
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 创建图形
        plt.figure(figsize=(15, 10))
        
        # 提取数据
        node_ids = [node['node_id'] for node in branch_and_bound_data]
        depths = [node['depth'] for node in branch_and_bound_data]
        objectives = [node.get('objective', 0) for node in branch_and_bound_data]
        is_integer = [node.get('is_integer', False) for node in branch_and_bound_data]
        
        # 绘制节点
        colors = ['red' if is_int else 'blue' for is_int in is_integer]
        sizes = [100 if is_int else 50 for is_int in is_integer]
        
        plt.scatter(depths, node_ids, c=colors, s=sizes, alpha=0.6)
        
        # 绘制连接线
        for node in branch_and_bound_data:
            if 'children' in node:
                for child_id in node['children']:
                    child_node = next((n for n in branch_and_bound_data if n['node_id'] == child_id), None)
                    if child_node:
                        plt.plot([node['depth'], child_node['depth']], 
                                [node['node_id'], child_node['node_id']], 
                                'k-', alpha=0.3)
        
        # 设置标题和布局
        plt.title(title)
        plt.xlabel('深度')
        plt.ylabel('节点ID')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分支定界树探索过程图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()