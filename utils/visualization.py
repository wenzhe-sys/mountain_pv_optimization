import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import os

class ResultVisualizer:
    def __init__(self, save_dir: str = None):
        if save_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(project_root, "data", "results", "visualization")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # 支持中文（跨平台：macOS用PingFang/Heiti，Windows用SimHei，Linux用WenQuanYi）
        import platform
        system = platform.system()
        if system == "Darwin":
            plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"]
        elif system == "Linux":
            plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei"]
        else:
            plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    def plot_partition(self, module1_output: Dict, instance_id: str):
        """可视化模块一：面板分区结果"""
        fig, ax = plt.subplots(figsize=(10, 8))
        zones = module1_output["zone_summary"]
        colors = plt.cm.Set3(np.linspace(0, 1, len(zones)))
        
        for idx, zone in enumerate(zones):
            zone_pva = [p for p in module1_output["partition_result"] if p["zone_id"] == zone["zone_id"]]
            x = [p["grid_coord"][1] for p in zone_pva]
            y = [p["grid_coord"][0] for p in zone_pva]
            ax.scatter(x, y, c=[colors[idx]], label=f"分区{idx+1}（{zone['pva_count']}块）", s=50)
        
        ax.set_xlabel("网格列坐标")
        ax.set_ylabel("网格行坐标")
        ax.set_title(f"光伏面板分区规划结果（算例：{instance_id}）")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_path = os.path.join(self.save_dir, f"partition_{instance_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"分区可视化图保存至：{save_path}")

    def plot_cost_breakdown(self, module3_output: Dict, instance_id: str):
        """可视化模块三：成本构成（优化版，解决文字重叠问题）"""
        cost = module3_output["total_cost_summary"]["cost_breakdown"]
        labels = ["箱变购置", "箱变安装", "电缆成本", "管沟开挖", "电力损耗"]
        values = [cost["box_purchase"], cost["box_install"], cost["cable"], cost["trenching"], cost["loss"]]
        
        # 1. 合并占比小于5%的小项，避免过多扇区导致重叠
        total_value = sum(values)
        threshold = 0.05  # 5%阈值，可根据需要调整
        small_items_sum = 0
        new_labels = []
        new_values = []
        
        for label, value in zip(labels, values):
            if value / total_value < threshold:
                small_items_sum += value
            else:
                new_labels.append(label)
                new_values.append(value)
        
        # 如果有合并的小项，添加"其他"类别
        if small_items_sum > 0:
            new_labels.append("其他")
            new_values.append(small_items_sum)
        
        # 2. 为小扇区设置爆炸效果，占比小于10%的扇区分离出来
        explode = [0.1 if v / sum(new_values) < 0.1 else 0 for v in new_values]
        
        # 3. 绘制优化后的饼图
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(
            new_values,
            labels=new_labels,
            autopct="%1.1f%%",
            startangle=90,
            shadow=True,
            explode=explode,  # 扇区分离
            labeldistance=1.1,  # 标签位置外移
            pctdistance=0.85,   # 百分比位置调整
            textprops={"fontsize": 11}  # 字体大小优化
        )
        
        # 4. 优化百分比文字样式（白色加粗，提高可读性）
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_weight("bold")
            autotext.set_fontsize(10)
        
        # 5. 设置标题，显示总成本
        total_cost = module3_output['total_cost_summary']['total_cost']
        ax.set_title(f"全生命周期成本构成（算例：{instance_id}）\n总成本：{total_cost:.1f}万元", fontsize=14)
        
        # 6. 保存图片
        save_path = os.path.join(self.save_dir, f"cost_breakdown_{instance_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"成本构成图保存至：{save_path}")

    def plot_loss_trend(self, module3_output: Dict, instance_id: str):
        """可视化模块三：年度电力损耗趋势"""
        years = [ld["year"] for ld in module3_output["loss_detail"]]
        loss_costs = [ld["loss_cost"] for ld in module3_output["loss_detail"]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years, loss_costs, marker="o", linewidth=2, color="#2E86AB")
        ax.set_xlabel("年份")
        ax.set_ylabel("年度损耗成本（万元）")
        ax.set_title(f"年度电力损耗成本趋势（算例：{instance_id}）")
        ax.grid(True, alpha=0.3)
        save_path = os.path.join(self.save_dir, f"loss_trend_{instance_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"损耗趋势图保存至：{save_path}")

# 单例调用
result_visualizer = ResultVisualizer()