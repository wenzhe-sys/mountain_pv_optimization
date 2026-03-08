# modules/module1/visualization/layout_visualizer.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple


class LayoutVisualizer:
    """布局结果可视化工具"""
    
    def __init__(self, graph: nx.Graph):
        """
        初始化可视化工具
        
        Args:
            graph: 面板邻接图
        """
        self.graph = graph
        # 预计算节点位置
        self.pos = self._compute_positions()
    
    def _compute_positions(self) -> Dict[str, Tuple[float, float]]:
        """计算节点位置"""
        pos = {}
        for node in self.graph.nodes():
            # 假设节点属性中包含坐标信息
            if 'x' in self.graph.nodes[node] and 'y' in self.graph.nodes[node]:
                pos[node] = (self.graph.nodes[node]['x'], self.graph.nodes[node]['y'])
            else:
                # 如果没有坐标信息，使用Spring布局
                pos = nx.spring_layout(self.graph)
                break
        return pos
    
    def visualize_partition(self, zones: List[Set[str]], 
                           title: str = "光伏面板分区布局",
                           save_path: str = None) -> None:
        """
        可视化分区布局
        
        Args:
            zones: 分区列表，每个分区是节点ID的集合
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 生成颜色列表
        colors = plt.cm.get_cmap('tab20', len(zones))
        
        # 创建图形
        plt.figure(figsize=(12, 10))
        
        # 绘制节点
        for i, zone in enumerate(zones):
            node_list = list(zone)
            nx.draw_networkx_nodes(
                self.graph, self.pos,
                nodelist=node_list,
                node_size=300,
                node_color=[colors(i)],
                label=f'分区 {i+1}'
            )
        
        # 绘制边
        nx.draw_networkx_edges(self.graph, self.pos, alpha=0.3)
        
        # 绘制标签
        nx.draw_networkx_labels(self.graph, self.pos, font_size=8)
        
        # 添加图例
        plt.legend(scatterpoints=1, loc='best')
        
        # 设置标题和布局
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"布局图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_cutting_plan(self, cutting_result: Dict,
                              title: str = "光伏面板切割计划",
                              save_path: str = None) -> None:
        """
        可视化切割计划
        
        Args:
            cutting_result: 切割结果
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 提取切割信息
        cut_patterns = []
        for cut in cutting_result:
            length = cut.get('length', 0)
            count = cut.get('count', 0)
            if count > 0:
                cut_patterns.append((length, count))
        
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        # 绘制切割长度分布
        if cut_patterns:
            lengths, counts = zip(*cut_patterns)
            plt.bar([str(l) for l in lengths], counts)
            plt.xlabel('切割长度')
            plt.ylabel('数量')
            plt.title(title)
            plt.tight_layout()
            
            # 保存或显示
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"切割计划图已保存到: {save_path}")
            else:
                plt.show()
        else:
            plt.text(0.5, 0.5, '无切割计划', ha='center', va='center')
            plt.axis('off')
        
        plt.close()
    
    def visualize_performance(self, history: List[Dict],
                             title: str = "优化性能曲线",
                             save_path: str = None) -> None:
        """
        可视化优化性能曲线
        
        Args:
            history: 优化历史记录
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 提取数据
        iterations = [h['iteration'] for h in history]
        upper_bounds = [h['ub'] for h in history]
        lower_bounds = [h['lb'] for h in history]
        gaps = [h['ub'] - h['lb'] if h['ub'] is not None and h['lb'] is not None else 0 for h in history]
        
        # 创建图形
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 绘制上下界
        ax1.plot(iterations, upper_bounds, 'b-', label='上界 (UB)')
        ax1.plot(iterations, lower_bounds, 'r-', label='下界 (LB)')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('目标函数值')
        ax1.legend(loc='upper left')
        
        # 绘制差距
        ax2 = ax1.twinx()
        ax2.plot(iterations, gaps, 'g--', label='差距 (UB-LB)')
        ax2.set_ylabel('差距')
        ax2.legend(loc='upper right')
        
        # 设置标题和布局
        plt.title(title)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能曲线已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_algorithms(self, results: Dict[str, List[Dict]], 
                          title: str = "算法性能对比",
                          save_path: str = None) -> None:
        """
        对比不同算法的性能
        
        Args:
            results: 算法结果字典，键为算法名称，值为优化历史记录
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 为每个算法绘制曲线
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for i, (algorithm, history) in enumerate(results.items()):
            iterations = [h['iteration'] for h in history]
            upper_bounds = [h['ub'] for h in history]
            plt.plot(iterations, upper_bounds, label=algorithm, 
                    marker='o', color=colors[i], linewidth=2)
        
        # 设置图表属性
        plt.xlabel('迭代次数')
        plt.ylabel('上界 (UB)')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"算法对比图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_performance_radar(self, metrics: Dict[str, Dict], 
                                   title: str = "算法性能雷达图",
                                   save_path: str = None) -> None:
        """
        可视化算法性能雷达图
        
        Args:
            metrics: 性能指标字典，键为算法名称，值为指标字典
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 获取指标类别
        categories = list(list(metrics.values())[0].keys())
        N = len(categories)
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 创建图形
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        # 为每个算法绘制雷达图
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
        for i, (algorithm, values) in enumerate(metrics.items()):
            data = list(values.values())
            data += data[:1]  # 闭合
            ax.plot(angles, data, linewidth=2, linestyle='solid', 
                   label=algorithm, color=colors[i])
            ax.fill(angles, data, alpha=0.25, color=colors[i])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # 设置标题
        plt.title(title, size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能雷达图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_zone_details(self, zones: List[Set[str]], 
                             zone_details: List[Dict],
                             title: str = "分区详细信息",
                             save_path: str = None) -> None:
        """
        可视化分区详细信息
        
        Args:
            zones: 分区列表
            zone_details: 分区详细信息列表
            title: 图表标题
            save_path: 保存路径，None表示不保存
        """
        # 提取数据
        zone_ids = [detail.get('zone_id', f'zone_{i}') for i, detail in enumerate(zone_details)]
        panel_counts = [detail.get('n_panels', len(zone)) for zone, detail in zip(zones, zone_details)]
        perimeters = [detail.get('perimeter', 0) for detail in zone_details]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制面板数量
        ax1.bar(zone_ids, panel_counts, color='skyblue')
        ax1.set_xlabel('分区')
        ax1.set_ylabel('面板数量')
        ax1.set_title('各分区面板数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 绘制周长
        ax2.bar(zone_ids, perimeters, color='lightgreen')
        ax2.set_xlabel('分区')
        ax2.set_ylabel('周长 (m)')
        ax2.set_title('各分区周长')
        ax2.tick_params(axis='x', rotation=45)
        
        # 设置标题和布局
        plt.suptitle(title)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分区详细信息图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()