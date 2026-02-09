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

    # ─── 模块一新增可视化 ───

    def plot_partition_detailed(self, module1_output: Dict, instance_id: str):
        """
        可视化模块一分区结果（增强版）。

        每个分区用不同颜色着色，标注面板数、逆变器 ID 和周长。
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        zones = module1_output["zone_summary"]
        n_zones = len(zones)
        colors = plt.cm.Set3(np.linspace(0, 1, max(n_zones, 1)))

        for idx, zone in enumerate(zones):
            zone_pva = [p for p in module1_output["partition_result"]
                        if p["zone_id"] == zone["zone_id"]]
            x = [p["grid_coord"][1] for p in zone_pva]
            y = [p["grid_coord"][0] for p in zone_pva]

            label = (f"{zone['zone_id']} ({zone['pva_count']}块, "
                     f"周长{zone['perimeter']:.0f}m)")
            ax.scatter(x, y, c=[colors[idx]], label=label,
                       s=80, edgecolors="black", linewidths=0.5, zorder=2)

            # 在分区重心处标注逆变器 ID
            if x and y:
                cx, cy = np.mean(x), np.mean(y)
                ax.annotate(zone["inverter_id"], (cx, cy),
                           fontsize=7, ha="center", va="center",
                           fontweight="bold", color="darkred", zorder=3)

        ax.set_xlabel("网格列坐标")
        ax.set_ylabel("网格行坐标")
        ax.set_title(f"光伏面板分区规划结果（算例: {instance_id}, "
                     f"{n_zones}个分区）")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal")

        save_path = os.path.join(self.save_dir, f"partition_detail_{instance_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  分区详情图保存至: {save_path}")

    def plot_cutting_utilization(self, module1_output: Dict, instance_id: str):
        """
        可视化切割方案：每种原材料的利用率柱状图。
        """
        cut_result = module1_output.get("cut_result", [])
        used_materials = [m for m in cut_result if m.get("is_used")]

        if not used_materials:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        names = [m["material_id"] for m in used_materials]
        utils = [m.get("utilization", 0) * 100 for m in used_materials]

        bars = ax.bar(range(len(names)), utils, color="#4ECDC4", edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, fontsize=8)
        ax.set_ylabel("利用率 (%)")
        ax.set_title(f"原材料切割利用率（算例: {instance_id}）")
        ax.set_ylim(0, 110)
        ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="理论上限")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        save_path = os.path.join(self.save_dir, f"cutting_util_{instance_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  切割利用率图保存至: {save_path}")

    def plot_benders_convergence(self, history: list, instance_id: str):
        """
        可视化 Benders 分解收敛过程：UB/LB 随迭代变化。
        """
        if not history:
            return

        iters = [h["iteration"] for h in history]
        lbs = [h["lb"] for h in history]
        ubs = [h.get("ub") for h in history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iters, lbs, "b-o", label="下界 (LB)", markersize=6)

        ub_iters = [i for i, u in zip(iters, ubs) if u is not None]
        ub_vals = [u for u in ubs if u is not None]
        if ub_vals:
            ax.plot(ub_iters, ub_vals, "r-s", label="上界 (UB)", markersize=6)

        ax.set_xlabel("迭代次数")
        ax.set_ylabel("目标值")
        ax.set_title(f"Benders 分解收敛过程（算例: {instance_id}）")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(self.save_dir, f"benders_convergence_{instance_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Benders 收敛曲线保存至: {save_path}")

    def plot_dqn_training_curves(self, training_history: list):
        """
        可视化 DQN 训练曲线：奖励、损失、epsilon 三幅子图。
        """
        if not training_history:
            return

        epochs = [h["epoch"] for h in training_history]
        rewards = [h["avg_reward"] for h in training_history]
        losses = [h["avg_loss"] for h in training_history]
        epsilons = [h["epsilon"] for h in training_history]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # 奖励曲线
        axes[0].plot(epochs, rewards, color="#2E86AB", linewidth=1.5)
        axes[0].set_ylabel("平均奖励")
        axes[0].set_title("S2V-DQN 训练曲线")
        axes[0].grid(True, alpha=0.3)

        # 标注最优点
        best_idx = np.argmax(rewards)
        axes[0].annotate(f"最优: {rewards[best_idx]:.2f}",
                         xy=(epochs[best_idx], rewards[best_idx]),
                         fontsize=9, color="red", fontweight="bold")

        # 损失曲线
        axes[1].plot(epochs, losses, color="#FF6B6B", linewidth=1.5)
        axes[1].set_ylabel("平均损失")
        axes[1].grid(True, alpha=0.3)

        # Epsilon 曲线
        axes[2].plot(epochs, epsilons, color="#4ECDC4", linewidth=1.5)
        axes[2].set_ylabel("探索率 (ε)")
        axes[2].set_xlabel("训练轮次")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "dqn_training_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  DQN 训练曲线保存至: {save_path}")

# 单例调用
result_visualizer = ResultVisualizer()