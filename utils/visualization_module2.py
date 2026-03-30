"""
模块二可视化模块
================
提供设备选型与电缆共沟优化结果的可视化。

包含：
1. 设备布局图（逆变器 + 箱变 + 升压站）
2. 电缆路由图（曼哈顿路径 + 管沟共享）
3. 管沟共享热力图
4. 列生成收敛曲线
5. 成本构成对比柱状图
6. 箱变容量利用率
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import platform

# 中文字体配置
system = platform.system()
if system == "Darwin":
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "STHeiti"]
elif system == "Linux":
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC"]
else:
    plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class Module2Visualizer:
    """模块二可视化器"""

    def __init__(self, save_dir: str = None):
        if save_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(project_root, "data", "results", "visualization")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # ================================================================
    #  1. 设备布局图
    # ================================================================
    def plot_equipment_layout(
        self, module2_output: Dict, module1_output: Dict,
        instance_data: Dict, instance_id: str = "",
    ):
        """
        绘制设备布局图：逆变器、箱变、升压站位置及连接关系。

        Parameters
        ----------
        module2_output : Dict
            M2 输出
        module1_output : Dict
            M1 输出（用于获取面板、逆变器坐标）
        instance_data : Dict
            算例数据（用于获取升压站坐标、地形信息）
        instance_id : str
            算例 ID
        """
        fig, ax = plt.subplots(figsize=(14, 11))

        # ------ 面板散点 ------
        partitions = module1_output["partition_result"]
        zones = module1_output["zone_summary"]
        n_zones = len(zones)
        colors = plt.cm.Set3(np.linspace(0, 1, max(n_zones, 3)))

        for idx, zone in enumerate(zones):
            zone_pva = [p for p in partitions if p["zone_id"] == zone["zone_id"]]
            xs = [p["grid_coord"][0] for p in zone_pva]
            ys = [p["grid_coord"][1] for p in zone_pva]
            ax.scatter(xs, ys, c=[colors[idx]], s=18, alpha=0.5,
                       label=f"分区{idx+1}（{zone['pva_count']}块）")

        # ------ 逆变器位置 ------
        for idx, zone in enumerate(zones):
            zone_pva = [p for p in partitions if p["zone_id"] == zone["zone_id"]]
            if zone_pva:
                inv_x = np.mean([p["grid_coord"][0] for p in zone_pva])
                inv_y = np.mean([p["grid_coord"][1] for p in zone_pva])
                ax.scatter(inv_x, inv_y, marker="^", c="blue", s=150,
                           zorder=5, edgecolors="navy", linewidths=0.8)
                ax.annotate(zone.get("inverter_id", f"inv_{idx}"),
                            (inv_x, inv_y), textcoords="offset points",
                            xytext=(5, 5), fontsize=7, color="blue")

        # ------ 箱变位置 ------
        eq_selection = module2_output["equipment_selection"]
        box_colors = {"1600": "orange", "3200": "red"}
        for eq in eq_selection:
            bx, by = eq["install_coord"]
            cap = str(eq["Q_box"])
            c = box_colors.get(cap, "red")
            ax.scatter(bx, by, marker="s", c=c, s=250, zorder=6,
                       edgecolors="black", linewidths=1.2)
            ax.annotate(f"{eq['transformer_id']}\n{cap}kVA",
                        (bx, by), textcoords="offset points",
                        xytext=(8, -12), fontsize=7, fontweight="bold", color=c)

        # ------ 升压站 ------
        sub_coord = instance_data["equipment_params"]["substation"]["coord"]
        ax.scatter(sub_coord[0], sub_coord[1], marker="*", c="gold",
                   s=400, zorder=7, edgecolors="black", linewidths=1.5)
        ax.annotate("升压站", (sub_coord[0], sub_coord[1]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=9, fontweight="bold", color="darkgoldenrod")

        # ------ 连接线 ------
        for eq in eq_selection:
            bx, by = eq["install_coord"]
            # 箱变 → 升压站
            ax.plot([bx, sub_coord[0]], [by, sub_coord[1]],
                    'k--', linewidth=1.5, alpha=0.4)
            # 逆变器 → 箱变
            for inv_id in eq["connected_inverters"]:
                for idx, zone in enumerate(zones):
                    if zone.get("inverter_id") == inv_id:
                        zone_pva = [p for p in partitions
                                    if p["zone_id"] == zone["zone_id"]]
                        if zone_pva:
                            ix = np.mean([p["grid_coord"][0] for p in zone_pva])
                            iy = np.mean([p["grid_coord"][1] for p in zone_pva])
                            ax.plot([ix, bx], [iy, by], 'b-',
                                    linewidth=0.8, alpha=0.6)

        ax.set_xlabel("X 坐标 (m)")
        ax.set_ylabel("Y 坐标 (m)")
        ax.set_title(f"设备布局图（算例: {instance_id}）\n"
                     f"箱变: {len(eq_selection)} 台  逆变器: {n_zones} 台",
                     fontsize=13)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal")

        path = os.path.join(self.save_dir, f"m2_layout_{instance_id}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"设备布局图保存至: {path}")

    # ================================================================
    #  2. 电缆路由图
    # ================================================================
    def plot_cable_routing(
        self, module2_output: Dict, module1_output: Dict,
        instance_data: Dict, instance_id: str = "",
    ):
        """绘制电缆路由图（曼哈顿 L 形路径）"""
        fig, ax = plt.subplots(figsize=(14, 11))

        sub_coord = instance_data["equipment_params"]["substation"]["coord"]
        zones = module1_output["zone_summary"]
        partitions = module1_output["partition_result"]

        # 计算逆变器坐标映射
        inv_coord_map = {}
        for idx, zone in enumerate(zones):
            zone_pva = [p for p in partitions if p["zone_id"] == zone["zone_id"]]
            if zone_pva:
                ix = np.mean([p["grid_coord"][0] for p in zone_pva])
                iy = np.mean([p["grid_coord"][1] for p in zone_pva])
                inv_coord_map[zone.get("inverter_id", f"inv_{idx}")] = (ix, iy)

        # 箱变坐标映射
        box_coord_map = {}
        for eq in module2_output["equipment_selection"]:
            box_coord_map[eq["transformer_id"]] = tuple(eq["install_coord"])

        # 绘制路由
        route_colors = plt.cm.tab10(np.linspace(0, 1, max(len(module2_output["cable_routes"]), 1)))
        for r_idx, route in enumerate(module2_output["cable_routes"]):
            inv_id = route["inverter_id"]
            box_id = route["transformer_id"]
            color = route_colors[r_idx % len(route_colors)]

            if inv_id in inv_coord_map and box_id in box_coord_map:
                ix, iy = inv_coord_map[inv_id]
                bx, by = box_coord_map[box_id]

                # 曼哈顿 L 形：先水平再垂直
                ax.plot([ix, bx, bx], [iy, iy, by], '-',
                        color=color, linewidth=1.8, alpha=0.7,
                        label=f"{route['route_id']}")

                # 箱变 → 升压站
                ax.plot([bx, sub_coord[0], sub_coord[0]],
                        [by, by, sub_coord[1]], '--',
                        color=color, linewidth=1.2, alpha=0.5)

        # 标记节点
        for inv_id, (x, y) in inv_coord_map.items():
            ax.scatter(x, y, marker="^", c="blue", s=120, zorder=5,
                       edgecolors="navy")
        for box_id, (x, y) in box_coord_map.items():
            ax.scatter(x, y, marker="s", c="red", s=200, zorder=6,
                       edgecolors="black")
        ax.scatter(sub_coord[0], sub_coord[1], marker="*", c="gold",
                   s=350, zorder=7, edgecolors="black")

        ax.set_xlabel("X 坐标 (m)")
        ax.set_ylabel("Y 坐标 (m)")
        ax.set_title(f"电缆路由图（算例: {instance_id}）\n"
                     f"路由: {len(module2_output['cable_routes'])} 条",
                     fontsize=13)
        if len(module2_output["cable_routes"]) <= 10:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal")

        path = os.path.join(self.save_dir, f"m2_routing_{instance_id}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"电缆路由图保存至: {path}")

    # ================================================================
    #  3. 管沟共享可视化
    # ================================================================
    def plot_trench_sharing(self, module2_output: Dict, instance_id: str = ""):
        """绘制管沟共享柱状图：每条管沟的电缆数"""
        trench = module2_output["trench_summary"]
        if not trench:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ids = [t["trench_id"] for t in trench]
        counts = [t["cable_count"] for t in trench]
        lengths = [t.get("length", 0) for t in trench]

        x = np.arange(len(ids))
        width = 0.35

        bars1 = ax.bar(x - width / 2, counts, width, label="电缆数",
                        color="steelblue", edgecolor="black")
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width / 2, lengths, width, label="管沟长度 (m)",
                         color="coral", edgecolor="black", alpha=0.7)

        # N_max 参考线
        ax.axhline(y=4, color="red", linestyle="--", linewidth=1, alpha=0.7,
                    label="N_max=4")

        ax.set_xlabel("管沟")
        ax.set_ylabel("电缆数")
        ax2.set_ylabel("长度 (m)")
        ax.set_title(f"管沟共享分析（算例: {instance_id}）", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=30)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

        path = os.path.join(self.save_dir, f"m2_trench_{instance_id}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"管沟共享图保存至: {path}")

    # ================================================================
    #  4. 列生成收敛曲线
    # ================================================================
    def plot_convergence(self, convergence_history: List[Dict],
                         instance_id: str = ""):
        """绘制列生成收敛曲线"""
        if not convergence_history:
            print("无列生成收敛历史，跳过绘图")
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))

        iters = list(range(1, len(convergence_history) + 1))
        objectives = [h.get("objective", 0) for h in convergence_history]
        n_columns = [h.get("n_active_paths", 0) for h in convergence_history]

        color1 = "steelblue"
        ax1.plot(iters, objectives, 'o-', color=color1, linewidth=2,
                 markersize=5, label="RMP 目标值")
        ax1.set_xlabel("迭代")
        ax1.set_ylabel("目标值", color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "coral"
        ax2.plot(iters, n_columns, 's--', color=color2, linewidth=1.5,
                 markersize=4, alpha=0.7, label="活跃路径数")
        ax2.set_ylabel("活跃路径数", color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.set_title(f"列生成收敛曲线（算例: {instance_id}）", fontsize=13)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.grid(True, alpha=0.3)

        path = os.path.join(self.save_dir, f"m2_convergence_{instance_id}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"收敛曲线保存至: {path}")

    # ================================================================
    #  5. 成本分解柱状图
    # ================================================================
    def plot_cost_breakdown(self, module2_output: Dict, instance_id: str = ""):
        """绘制模块二成本分解柱状图"""
        eq_selection = module2_output["equipment_selection"]
        cable_routes = module2_output["cable_routes"]
        trench_summary = module2_output["trench_summary"]

        box_purchase = sum(e["cost"]["purchase"] for e in eq_selection)
        box_install = sum(e["cost"]["installation"] for e in eq_selection)
        cable_cost = sum(r["cost"]["cable"] for r in cable_routes)
        trench_cost = sum(t["cost"] for t in trench_summary)

        categories = ["箱变购置", "箱变安装", "电缆", "管沟开挖"]
        values = [box_purchase, box_install, cable_cost, trench_cost]
        colors_bar = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.bar(categories, values, color=colors_bar,
                       edgecolor="black", linewidth=0.8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10)

        total = module2_output["total_cost"]
        ax.set_ylabel("成本 (万元)")
        ax.set_title(f"模块二成本分解（算例: {instance_id}）\n"
                     f"总成本: {total:.1f} 万元", fontsize=13)
        ax.grid(axis="y", alpha=0.3)

        path = os.path.join(self.save_dir, f"m2_cost_{instance_id}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"成本分解图保存至: {path}")

    # ================================================================
    #  6. 箱变容量利用率
    # ================================================================
    def plot_box_utilization(self, module2_output: Dict, instance_id: str = ""):
        """绘制箱变容量利用率图"""
        eq_selection = module2_output["equipment_selection"]
        if not eq_selection:
            return

        capacity_map = {1600: 5, 3200: 10}
        ids = [e["transformer_id"] for e in eq_selection]
        connected = [len(e["connected_inverters"]) for e in eq_selection]
        capacity = [capacity_map.get(e["Q_box"], 10) for e in eq_selection]
        utilization = [c / cap * 100 for c, cap in zip(connected, capacity)]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(ids))

        bars = ax.bar(x, utilization, color="steelblue",
                       edgecolor="black", linewidth=0.8)

        for bar, (c, cap) in zip(bars, zip(connected, capacity)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{c}/{cap}", ha="center", va="bottom", fontsize=9)

        ax.axhline(y=100, color="red", linestyle="--", linewidth=1,
                    alpha=0.7, label="容量上限")
        ax.set_xlabel("箱变")
        ax.set_ylabel("利用率 (%)")
        ax.set_title(f"箱变容量利用率（算例: {instance_id}）", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=30)
        ax.legend()
        ax.set_ylim(0, 120)
        ax.grid(axis="y", alpha=0.3)

        path = os.path.join(self.save_dir, f"m2_utilization_{instance_id}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"利用率图保存至: {path}")

    # ================================================================
    #  综合可视化
    # ================================================================
    def generate_all_plots(
        self, module2_output: Dict, module1_output: Dict,
        instance_data: Dict, instance_id: str = "",
    ):
        """生成所有模块二可视化图表"""
        print(f"==== 生成模块二可视化（算例: {instance_id}） ====")
        self.plot_equipment_layout(
            module2_output, module1_output, instance_data, instance_id)
        self.plot_cable_routing(
            module2_output, module1_output, instance_data, instance_id)
        self.plot_trench_sharing(module2_output, instance_id)
        self.plot_cost_breakdown(module2_output, instance_id)
        self.plot_box_utilization(module2_output, instance_id)

        # 列生成收敛（如果有）
        conv = module2_output.get("convergence_history", [])
        if conv:
            self.plot_convergence(conv, instance_id)

        # B&B 树摘要（如果有）
        bb = module2_output.get("bb_summary", {})
        perf = module2_output.get("perf_stats", {})
        if perf:
            self.plot_performance_breakdown(perf, instance_id)

        print(f"==== 模块二可视化完成 ====")

    # ================================================================
    #  7. 性能分解图
    # ================================================================
    def plot_performance_breakdown(self, perf_stats: Dict, instance_id: str = ""):
        """绘制求解过程各阶段耗时分解图"""
        labels = ["RMP求解", "定价子问题", "拉格朗日松弛", "B&B搜索", "其它"]
        rmp_t = perf_stats.get("rmp_time", 0)
        price_t = perf_stats.get("pricing_time", 0)
        lr_t = perf_stats.get("lagrangian_time", 0)
        bb_t = perf_stats.get("bb_time", 0)
        total_t = max(perf_stats.get("total_time", 1), 0.001)
        other_t = max(0, total_t - rmp_t - price_t - lr_t - bb_t)
        values = [rmp_t, price_t, lr_t, bb_t, other_t]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#cccccc"]
        nonzero_idx = [i for i, v in enumerate(values) if v > 0.001]
        if nonzero_idx:
            ax1.pie([values[i] for i in nonzero_idx],
                    labels=[labels[i] for i in nonzero_idx],
                    colors=[colors[i] for i in nonzero_idx],
                    autopct="%1.1f%%", startangle=90)
        ax1.set_title(f"求解耗时分解（算例: {instance_id}）\n总时间: {total_t:.1f}s")

        # Stats bar chart
        stat_labels = ["CG迭代数", "B&B节点数", "活跃路径数"]
        stat_values = [
            perf_stats.get("cg_iterations", 0),
            perf_stats.get("bb_nodes", 0),
            0,  # placeholder
        ]
        ax2.barh(stat_labels, stat_values, color=["#4e79a7", "#76b7b2", "#f28e2b"])
        ax2.set_xlabel("数量")
        ax2.set_title(f"求解统计指标（算例: {instance_id}）")
        for i, v in enumerate(stat_values):
            ax2.text(v + 0.5, i, str(v), va="center", fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"m2_performance_{instance_id}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"性能分解图保存至: {path}")

    # ================================================================
    #  8. 策略对比雷达图
    # ================================================================
    def plot_strategy_radar(self, benchmark_results: List[Dict], instance_id: str = ""):
        """
        绘制多策略对比雷达图。

        Parameters
        ----------
        benchmark_results : List[Dict]
            benchmark 输出的行列表，每行含 strategy, total_cost, time_sec 等
        """
        strategies = list(set(r["strategy"] for r in benchmark_results
                              if r.get("total_cost", -1) > 0))
        if len(strategies) < 2:
            return

        # Normalize metrics to [0, 1] (lower is better for cost/time)
        metrics = ["total_cost", "time_sec", "n_boxes"]
        labels = ["成本(万元)", "求解时间(s)", "箱变数量"]

        data = {}
        for s in strategies:
            row = next(r for r in benchmark_results if r["strategy"] == s)
            data[s] = [row.get(m, 0) for m in metrics]

        # Normalize
        maxvals = [max(data[s][i] for s in strategies) or 1 for i in range(len(metrics))]
        for s in strategies:
            data[s] = [1 - v / mx if mx > 0 else 0 for v, mx in zip(data[s], maxvals)]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = ["#4e79a7", "#f28e2b", "#e15759"]
        for idx, s in enumerate(strategies):
            vals = data[s] + data[s][:1]
            ax.plot(angles, vals, 'o-', linewidth=2, label=s,
                    color=colors[idx % len(colors)])
            ax.fill(angles, vals, alpha=0.15, color=colors[idx % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(f"策略对比（算例: {instance_id}）\n(越大越好)", fontsize=13)
        ax.legend(loc="upper right", fontsize=9)

        path = os.path.join(self.save_dir, f"m2_radar_{instance_id}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"策略对比雷达图保存至: {path}")
