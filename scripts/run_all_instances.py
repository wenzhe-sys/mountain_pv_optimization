#!/usr/bin/env python3
"""
批量运行 17 个算例 + 生成对比报告

对每个算例运行 Benders 分解（启发式分区），输出 M1-Output 并汇总对比。

使用方式:
  python scripts/run_all_instances.py
  python scripts/run_all_instances.py --method dqn --model checkpoints/best_model.pt
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from algorithm.benders_decomposition import BendersDecomposition
from utils.load_instance import InstanceLoader


def run_all(args):
    """运行所有算例。"""
    print()
    print("══════════════════════════════════════════════════════════")
    print("  模块一批量求解 | 光伏面板切割及分区规划")
    print(f"  求解方法: {args.method} | "
          f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("══════════════════════════════════════════════════════════")

    loader = InstanceLoader()
    results_dir = os.path.join(project_root, "data", "results", "module1")
    os.makedirs(results_dir, exist_ok=True)

    # DQN 模型加载
    dqn_solver = None
    if args.method == "dqn" and args.model:
        from algorithm.dqn_agent import DQNPartitionAgent
        dqn_solver = DQNPartitionAgent(device="cpu")
        dqn_solver.load_checkpoint(args.model)
        print(f"  DQN 模型: {args.model}")

    summary = []
    total_start = time.time()

    for i in range(1, 18):
        inst_id = f"r{i}"
        try:
            instance = loader.load_instance(inst_id)
        except Exception as e:
            print(f"\n  ⚠ 算例 {inst_id} 加载失败: {e}")
            continue

        print(f"\n{'─' * 56}")
        solver = BendersDecomposition(
            instance, partition_solver=args.method,
            max_iter=args.max_iter, verbose=True
        )
        if dqn_solver:
            solver.set_dqn_solver(dqn_solver)

        start = time.time()
        output = solver.optimize()
        elapsed = time.time() - start

        # 保存 M1-Output
        save_path = os.path.join(results_dir, f"M1-Output_{inst_id}.json")
        # 移除 optimization_history 以减小文件大小
        output_save = {k: v for k, v in output.items() if k != "optimization_history"}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_save, f, ensure_ascii=False, indent=2)

        # 汇总
        n_zones = len(output["zone_summary"])
        panel_counts = [z["pva_count"] for z in output["zone_summary"]]
        avg_perimeter = sum(z["perimeter"] for z in output["zone_summary"]) / n_zones if n_zones > 0 else 0
        constraints = output["constraint_satisfaction"]

        summary.append({
            "算例": inst_id,
            "面板数": instance["instance_info"]["n_nodes"],
            "分区数": n_zones,
            "面板分布": f"{min(panel_counts)}-{max(panel_counts)}" if panel_counts else "N/A",
            "平均周长": f"{avg_perimeter:.1f}m",
            "容量": constraints.get("逆变器容量约束", "N/A"),
            "连通": constraints.get("分区连通性", "N/A"),
            "周长": constraints.get("分区周长约束", "N/A"),
            "耗时": f"{elapsed:.1f}s",
        })

    total_time = time.time() - total_start

    # 打印汇总表
    print()
    print("══════════════════════════════════════════════════════════")
    print("  全部算例求解完成 | 汇总报告")
    print("══════════════════════════════════════════════════════════")
    print()

    # 表头
    header = f"{'算例':>4} | {'面板':>4} | {'分区':>4} | {'面板分布':>8} | " \
             f"{'平均周长':>8} | {'容量':>5} | {'连通':>5} | {'周长':>5} | {'耗时':>6}"
    print(header)
    print("─" * len(header))

    for row in summary:
        print(f"{row['算例']:>4} | {row['面板数']:>4} | {row['分区数']:>4} | "
              f"{row['面板分布']:>8} | {row['平均周长']:>8} | "
              f"{row['容量']:>5} | {row['连通']:>5} | {row['周长']:>5} | "
              f"{row['耗时']:>6}")

    print()
    print(f"  总耗时: {total_time:.1f}秒 | 结果目录: {results_dir}/")
    print("══════════════════════════════════════════════════════════")


def main():
    parser = argparse.ArgumentParser(description="模块一批量求解")
    parser.add_argument("--method", type=str, default="heuristic",
                        choices=["heuristic", "dqn"], help="分区求解方法")
    parser.add_argument("--model", type=str, default=None, help="DQN 模型路径")
    parser.add_argument("--max-iter", type=int, default=10, help="Benders 最大迭代次数")
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
