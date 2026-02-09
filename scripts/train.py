#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2V-DQN 训练管线 CLI

支持：
  - 从零开始训练:  python scripts/train.py --epochs 300
  - 恢复训练:      python scripts/train.py --resume checkpoints/epoch_150.pt --epochs 300
  - 查看进度:      python scripts/train.py --status checkpoints/
  - 评估模型:      python scripts/train.py --eval checkpoints/best_model.pt

所有 checkpoint 保存到 checkpoints/ 目录，支持团队成员间交接。
"""

import os
import sys

# 强制 stdout 无缓冲，确保日志实时输出
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import json
import time
import glob
import argparse
from datetime import datetime, timedelta

# 将项目根目录加入 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np

from algorithm.dqn_agent import DQNPartitionAgent
from utils.load_instance import InstanceLoader


def load_all_instances() -> list:
    """加载全部 17 个 PV 算例。"""
    loader = InstanceLoader()
    instances = []
    for i in range(1, 18):
        try:
            inst = loader.load_instance(f"r{i}")
            instances.append(inst)
        except Exception as e:
            print(f"  ⚠ 加载算例 r{i} 失败: {e}")
    return instances


def detect_device() -> str:
    """自动检测最佳计算设备。"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_time(seconds: float) -> str:
    """将秒数格式化为可读字符串。"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}分{seconds % 60:.0f}秒"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}小时{m}分"


def train(args):
    """执行训练。"""
    device = args.device or detect_device()

    print()
    print("══════════════════════════════════════════════════════════")
    print("  S2V-DQN 训练管线 | 模块一：光伏面板切割及分区规划")
    print(f"  设备: {device} | 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("══════════════════════════════════════════════════════════")

    # 加载算例
    print("\n【准备数据】加载 PV 算例...")
    instances = load_all_instances()
    print(f"  成功加载 {len(instances)} 个算例")

    if not instances:
        print("  ✗ 未找到可用算例，请先运行数据预处理")
        return

    # 初始化智能体
    # CPU 模式用较小 batch 加速；GPU 模式用标准 batch
    batch_size = 8 if device == "cpu" else 32

    agent = DQNPartitionAgent(
        node_feature_dim=5, hidden_dim=64,
        n_iterations=4, context_dim=4,
        lr=args.lr, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=args.epsilon_decay,
        buffer_size=10000, batch_size=batch_size,
        target_update=100, device=device
    )

    start_epoch = 1

    # 恢复 checkpoint
    if args.resume:
        print(f"\n【恢复训练】从 {args.resume} 加载 checkpoint...")
        meta = agent.load_checkpoint(args.resume)
        start_epoch = meta["epoch"] + 1
        print(f"  ├─ 已完成轮次: {meta['epoch']}")
        print(f"  ├─ 最优奖励: {meta['best_reward']:.4f} (轮次 {meta['best_epoch']})")
        print(f"  └─ 存档时间: {meta['save_time']}")

    # 创建 checkpoint 目录
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n【开始训练】轮次 {start_epoch} → {args.epochs} | "
          f"学习率 {args.lr} | ε衰减 {args.epsilon_decay}")
    print(f"  存档目录: {ckpt_dir}/")
    print(f"  存档间隔: 每 {args.save_every} 轮")

    # 早停计数器
    patience_counter = 0
    patience = args.patience
    global_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # 训练一个 epoch（每 4 步学习一次，首 epoch 显示算例详情）
        verbose_inst = (epoch == start_epoch)  # 第一轮显示每个算例进度
        stats = agent.train_epoch(instances, epoch, learn_every=4,
                                    verbose_instances=verbose_inst)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - global_start
        remaining = elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch) if epoch > start_epoch else 0

        # 早停检查
        if stats["is_best"]:
            patience_counter = 0
            # 保存最优模型
            best_path = os.path.join(ckpt_dir, "best_model.pt")
            agent.save_checkpoint(best_path)
        else:
            patience_counter += 1

        # 打印日志
        if epoch % args.log_every == 0 or stats["is_best"] or epoch == args.epochs:
            mark = " ★ 新最优" if stats["is_best"] else ""
            print(f"\n【轮次 {epoch}/{args.epochs}】{mark}")
            print(f"  ├─ 平均奖励: {stats['avg_reward']:.4f} "
                  f"(最优: {agent.best_reward:.4f}, 轮次 {agent.best_epoch})")
            print(f"  ├─ 平均损失: {stats['avg_loss']:.6f}")
            print(f"  ├─ 探索率(ε): {stats['epsilon']:.4f}")
            print(f"  ├─ 约束满足率: "
                  f"容量 {stats['capacity_rate']:.1%} | "
                  f"连通 {stats['connected_rate']:.1%} | "
                  f"周长 {stats['perimeter_rate']:.1%}")
            print(f"  ├─ 本轮耗时: {format_time(epoch_time)} | "
                  f"已用总时: {format_time(elapsed)}")
            print(f"  ├─ 预计剩余: {format_time(remaining)}")

            # 定期存档
            if epoch % args.save_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
                agent.save_checkpoint(ckpt_path)
                print(f"  └─ 存档: {ckpt_path} ✓")
            else:
                print(f"  └─ 经验池: {stats['buffer_size']} 条")

        # 早停
        if patience_counter >= patience:
            print(f"\n  ⚠ 连续 {patience} 轮无改善，触发早停")
            break

    # 训练完成
    total_time = time.time() - global_start
    final_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
    agent.save_checkpoint(final_path)

    print()
    print("══════════════════════════════════════════════════════════")
    print(f"  训练完成 | 总耗时: {format_time(total_time)} | "
          f"最优轮次: 第 {agent.best_epoch} 轮")
    print(f"  最终模型: {ckpt_dir}/best_model.pt")
    print(f"  最终存档: {final_path}")
    print("══════════════════════════════════════════════════════════")


def show_status(args):
    """显示训练进度。"""
    ckpt_dir = args.status
    print(f"\n【训练进度】目录: {ckpt_dir}/")

    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not ckpt_files:
        print("  未找到任何 checkpoint 文件")
        return

    print(f"  找到 {len(ckpt_files)} 个 checkpoint:")
    for f in ckpt_files:
        try:
            ckpt = torch.load(f, map_location="cpu")
            epoch = ckpt.get("current_epoch", "?")
            reward = ckpt.get("best_reward", "?")
            save_time = ckpt.get("save_time", "?")
            name = os.path.basename(f)
            print(f"  ├─ {name}: 轮次 {epoch}, 最优奖励 {reward:.4f}, "
                  f"保存于 {save_time}")
        except Exception as e:
            print(f"  ├─ {os.path.basename(f)}: 读取失败 ({e})")


def evaluate(args):
    """评估模型。"""
    print(f"\n【模型评估】加载 {args.eval}...")
    agent = DQNPartitionAgent(device="cpu")
    meta = agent.load_checkpoint(args.eval)
    print(f"  轮次: {meta['epoch']}, 最优奖励: {meta['best_reward']:.4f}")

    instances = load_all_instances()
    print(f"  评估 {len(instances)} 个算例...\n")

    from utils.graph_utils import build_adjacency_graph

    for inst in instances:
        inst_id = inst["instance_info"]["instance_id"]
        graph = build_adjacency_graph(inst["pva_list"], inst["terrain_data"]["grid_size"])
        n_zones = inst["equipment_params"]["inverter"]["p"]

        result = agent.solve(graph, n_zones, inst["pva_params"])
        panel_counts = [len(z) for z in result.zones]

        status = "✓" if result.is_feasible else "✗"
        print(f"  {status} 算例 {inst_id}: {len(result.zones)} 个分区, "
              f"面板数 {panel_counts}, 总周长 {result.total_perimeter:.1f}m, "
              f"违规 {len(result.violations)} 条")


def main():
    parser = argparse.ArgumentParser(
        description="S2V-DQN 训练管线 | 模块一：光伏面板切割及分区规划"
    )
    parser.add_argument("--epochs", type=int, default=300, help="训练总轮次")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--epsilon-decay", type=int, default=5000, help="ε衰减步数")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cpu/cuda/mps)")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="存档目录")
    parser.add_argument("--save-every", type=int, default=50, help="存档间隔（轮次）")
    parser.add_argument("--log-every", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--patience", type=int, default=30, help="早停耐心值")
    parser.add_argument("--status", type=str, default=None, help="查看指定目录的训练进度")
    parser.add_argument("--eval", type=str, default=None, help="评估指定 checkpoint")

    args = parser.parse_args()

    if args.status:
        show_status(args)
    elif args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
