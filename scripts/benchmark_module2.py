"""
模块二基线对比测试
==================
对比三种求解策略在所有算例上的性能：

1. **Matheuristic**：KMeans 聚类 + 邻域搜索（启发式回退）
2. **Direct MILP**：直接求解 Arc-Flow MILP（精确方法）
3. **Branch & Price**：分支定价（列生成 + 分支定界）

输出：
- 各策略的成本、求解时间和约束满足率
- 对比汇总表（CSV）
- 柱状对比图

用法：
    python scripts/benchmark_module2.py
    python scripts/benchmark_module2.py --instances r1 r2 r3
    python scripts/benchmark_module2.py --strategies matheuristic milp
"""

import argparse
import json
import os
import sys
import time
import csv
from typing import List, Dict

# 项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # Support the new directory structure
    sys.path.insert(0, os.path.join(project_root, "modules", "module2"))

from modules.module2.algorithm.branch_and_price import BranchAndPrice


def load_data(instance_id: str):
    """加载算例与 M1 输出"""
    inst_path = os.path.join(
        project_root, "data", "processed", "PV", "public", "easy",
        f"public_easy_{instance_id}.json",
    )
    m1_path = os.path.join(
        project_root, "data", "results", "module1",
        f"M1-Output_{instance_id}.json",
    )
    if not os.path.exists(inst_path):
        raise FileNotFoundError(f"算例不存在: {inst_path}")
    if not os.path.exists(m1_path):
        raise FileNotFoundError(f"M1 输出不存在: {m1_path}")

    with open(inst_path, "r", encoding="utf-8") as f:
        instance_data = json.load(f)
    with open(m1_path, "r", encoding="utf-8") as f:
        m1_output = json.load(f)
    return instance_data, m1_output


def run_benchmark(instance_id: str, strategies: List[str],
                  time_limit: int = 300) -> List[Dict]:
    """对单个算例运行指定策略并收集结果"""
    instance_data, m1_output = load_data(instance_id)
    results = []

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"算例: {instance_id} | 策略: {strategy}")
        print(f"{'='*60}")

        try:
            solver = BranchAndPrice(instance_data, m1_output)
            t0 = time.time()
            result = solver.optimize(strategy=strategy, time_limit=time_limit)
            elapsed = time.time() - t0

            # 检查约束
            cs = result.get("constraint_satisfaction", {})
            def _is_constraint_ok(v):
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    return v == "100%"
                return bool(v)
            all_satisfied = all(_is_constraint_ok(v) for v in cs.values())

            row = {
                "instance_id": instance_id,
                "strategy": strategy,
                "total_cost": round(result["total_cost"], 2),
                "n_boxes": len(result["equipment_selection"]),
                "n_routes": len(result["cable_routes"]),
                "time_sec": round(elapsed, 2),
                "constraints_ok": all_satisfied,
                "constraint_detail": str(cs),
            }

            # Capture performance stats if available
            perf = result.get("perf_stats", {})
            row["cg_iterations"] = perf.get("cg_iterations", 0)
            row["cg_time"] = round(perf.get("cg_time", 0), 2)
            row["rmp_time"] = round(perf.get("rmp_time", 0), 2)
            row["pricing_time"] = round(perf.get("pricing_time", 0), 2)
            row["bb_nodes"] = perf.get("bb_nodes", 0)
            row["bb_time"] = round(perf.get("bb_time", 0), 2)
            row["lagrangian_time"] = round(perf.get("lagrangian_time", 0), 2)

            # Capture B&B summary if available
            bb_sum = result.get("bb_summary", {})
            row["bb_gap"] = bb_sum.get("gap", None)
            row["bb_lb"] = round(bb_sum.get("global_lb", 0), 2) if bb_sum.get("global_lb") else None
            row["bb_ub"] = round(bb_sum.get("global_ub", 0), 2) if bb_sum.get("global_ub") else None

            results.append(row)

            print(f"  总成本: {row['total_cost']:.2f} 万元")
            print(f"  箱变数: {row['n_boxes']}")
            print(f"  求解时间: {row['time_sec']:.2f}s")
            print(f"  约束满足: {'通过' if all_satisfied else '未通过'}")

        except Exception as e:
            print(f"  [错误] {strategy}: {e}")
            results.append({
                "instance_id": instance_id,
                "strategy": strategy,
                "total_cost": -1,
                "n_boxes": -1,
                "n_routes": -1,
                "time_sec": -1,
                "constraints_ok": False,
                "constraint_detail": str(e),
            })

    return results


def save_results(all_results: List[Dict], output_dir: str):
    """保存结果到 CSV"""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_module2.csv")

    if not all_results:
        print("无结果可保存")
        return

    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n对比结果保存至: {csv_path}")


def print_summary(all_results: List[Dict]):
    """打印汇总表"""
    if not all_results:
        return

    print(f"\n{'='*80}")
    print(f"{'算例':<12} {'策略':<20} {'成本':>10} {'箱变':>5} {'时间(s)':>10} {'约束':>6}")
    print(f"{'-'*80}")
    for r in all_results:
        ok_str = "通过" if r["constraints_ok"] else "失败"
        cost_str = f"{r['total_cost']:.1f}" if r['total_cost'] > 0 else "ERROR"
        print(f"{r['instance_id']:<12} {r['strategy']:<20} {cost_str:>10} "
              f"{r['n_boxes']:>5} {r['time_sec']:>10.1f} {ok_str:>6}")
    print(f"{'='*80}")


def generate_comparison_chart(all_results: List[Dict], output_dir: str):
    """生成策略对比柱状图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import platform
        import numpy as np
        system = platform.system()
        if system == "Windows":
            matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("matplotlib 未安装，跳过图表生成")
        return

    # 按算例分组
    instances = sorted(set(r["instance_id"] for r in all_results))
    strategies = sorted(set(r["strategy"] for r in all_results))
    n_strategies = len(strategies)

    if not instances or not strategies:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    x = np.arange(len(instances))
    width = 0.8 / max(n_strategies, 1)

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    for s_idx, strat in enumerate(strategies):
        costs = []
        times = []
        for inst in instances:
            match = [r for r in all_results
                     if r["instance_id"] == inst and r["strategy"] == strat]
            if match:
                costs.append(max(match[0]["total_cost"], 0))
                times.append(max(match[0]["time_sec"], 0))
            else:
                costs.append(0)
                times.append(0)

        offset = (s_idx - n_strategies / 2 + 0.5) * width
        ax1.bar(x + offset, costs, width, label=strat,
                color=colors[s_idx % len(colors)])
        ax2.bar(x + offset, times, width, label=strat,
                color=colors[s_idx % len(colors)])

    ax1.set_ylabel("总成本 (万元)")
    ax1.set_title("各策略成本对比")
    ax1.set_xticks(x)
    ax1.set_xticklabels(instances, rotation=30)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_ylabel("求解时间 (s)")
    ax2.set_title("各策略求解时间对比")
    ax2.set_xticks(x)
    ax2.set_xticklabels(instances, rotation=30)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"对比图表保存至: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="模块二基线对比测试")
    parser.add_argument("--instances", nargs="+", default=None,
                        help="算例 ID 列表 (默认: r1 r2)")
    parser.add_argument("--strategies", nargs="+",
                        default=["matheuristic", "milp", "branch_and_price"],
                        help="策略列表")
    parser.add_argument("--time-limit", type=int, default=300,
                        help="单次求解时间限制 (秒)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="结果输出目录")
    args = parser.parse_args()

    # 默认算例
    if args.instances is None:
        args.instances = ["r1", "r2"]
        # 检测可用算例
        available = []
        for i in range(1, 18):
            inst_id = f"r{i}"
            inst_path = os.path.join(
                project_root, "data", "processed", "PV", "public", "easy",
                f"public_easy_{inst_id}.json",
            )
            m1_path = os.path.join(
                project_root, "data", "results", "module1",
                f"M1-Output_{inst_id}.json",
            )
            if os.path.exists(inst_path) and os.path.exists(m1_path):
                available.append(inst_id)
        if available:
            args.instances = available[:5]  # 默认最多 5 个
            print(f"检测到可用算例: {available}")
            print(f"将运行前 {len(args.instances)} 个: {args.instances}")

    if args.output_dir is None:
        args.output_dir = os.path.join(
            project_root, "data", "results", "visualization")

    print(f"模块二基线对比测试")
    print(f"  算例: {args.instances}")
    print(f"  策略: {args.strategies}")
    print(f"  时限: {args.time_limit}s")
    print(f"  输出: {args.output_dir}")

    # 运行基线测试
    all_results = []
    for inst_id in args.instances:
        try:
            results = run_benchmark(inst_id, args.strategies, args.time_limit)
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"跳过 {inst_id}: {e}")

    # 输出与保存
    print_summary(all_results)
    save_results(all_results, args.output_dir)

    try:
        import numpy as np
        generate_comparison_chart(all_results, args.output_dir)
    except Exception as e:
        print(f"生成对比图表失败: {e}")


if __name__ == "__main__":
    main()
