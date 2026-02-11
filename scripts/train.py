#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2V-DQN 训练管线 CLI

严格对齐 Khalil et al. 2017 论文训练流程。

使用方式：
  python scripts/train.py --epochs 300
  python scripts/train.py --resume outputs/checkpoints/epoch_0150.pt --epochs 300
  python scripts/train.py --status outputs/checkpoints/
  python scripts/train.py --eval outputs/checkpoints/best_model.pt
"""

import os
import sys

# 强制 stdout 无缓冲
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import json
import time
import glob
import argparse
from datetime import datetime

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
            print(f"  ⚠ 加载算例 r{i} 失败: {e}", flush=True)
    return instances


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}分{seconds % 60:.0f}秒"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}小时{m}分"


def train(args):
    device = args.device or detect_device()

    print(flush=True)
    print("══════════════════════════════════════════════════════════", flush=True)
    print("  S2V-DQN 训练管线 | 模块一：光伏面板切割及分区规划", flush=True)
    print(f"  设备: {device} | 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("══════════════════════════════════════════════════════════", flush=True)

    print("\n【准备数据】加载 PV 算例...", flush=True)
    instances = load_all_instances()
    print(f"  成功加载 {len(instances)} 个算例", flush=True)

    if not instances:
        print("  ✗ 未找到可用算例", flush=True)
        return

    # train_every=5: 每 5 步训练一次（减少 80% 计算量，对齐原作的 n-step 策略）
    agent = DQNPartitionAgent(
        dim_in=1, dim_embed=64, T=4,
        lr=args.lr, gamma=0.95, tau=0.005,
        buffer_size=10000, batch_size=64,
        train_every=5,
        device=device
    )

    # ─── 阶段一：专家经验生成 + 行为克隆预训练 ───
    if not args.resume and not args.skip_pretrain:
        print("\n" + "─" * 56, flush=True)
        print("【阶段一】专家经验生成 + 行为克隆预训练", flush=True)
        print("─" * 56, flush=True)

        print("\n  生成专家经验（启发式多种子运行）...", flush=True)
        expert_data = agent.generate_expert_data(
            instances, n_runs_per_instance=args.expert_runs
        )

        if expert_data:
            # 行为克隆用 1e-3（监督学习可用较高学习率），RL 微调再用 args.lr（1e-4）
            bc_lr = 1e-3
            print(f"\n  开始行为克隆预训练（{args.pretrain_epochs} 轮, lr={bc_lr}）...", flush=True)
            bc_losses = agent.pretrain_from_expert(
                expert_data, n_epochs=args.pretrain_epochs, lr=bc_lr
            )
            print(f"  行为克隆完成，最终损失: {bc_losses[-1]:.6f}", flush=True)

            # Save BC model as both pretrained.pt and best_model.pt
            pretrain_path = os.path.join(args.checkpoint_dir, "pretrained.pt")
            agent.save_checkpoint(pretrain_path)
            print(f"  预训练模型保存: {pretrain_path}", flush=True)

            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            agent.save_checkpoint(best_path)
            print(f"  最优模型保存: {best_path}", flush=True)

    if args.skip_rl:
        print("\n" + "─" * 56, flush=True)
        print("【跳过 RL 微调】BC 模型已保存为 best_model.pt", flush=True)
        print("─" * 56, flush=True)
        print("\n══════════════════════════════════════════════════════════", flush=True)
        print(f"  训练完成 | 模型: {best_path}", flush=True)
        print("══════════════════════════════════════════════════════════", flush=True)
        return

    # RL fine-tuning: freeze S2V, reset Q-heads, only train Q-function
    import torch.nn as nn
    nn.init.xavier_uniform_(agent.q_policy.theta5)
    nn.init.xavier_uniform_(agent.q_policy.theta6)
    nn.init.xavier_uniform_(agent.q_policy.theta7)
    agent.q_target.load_state_dict(agent.q_policy.state_dict())
    for p in agent.s2v_policy.parameters():
        p.requires_grad = False
    agent.optimizer = torch.optim.Adam(
        agent.q_policy.parameters(), lr=args.lr
    )

    print("\n" + "─" * 56, flush=True)
    print("【阶段二】RL 微调（在预训练基础上继续优化）", flush=True)
    print("─" * 56, flush=True)

    start_epoch = 1

    if args.resume:
        print(f"\n【恢复训练】从 {args.resume} 加载...", flush=True)
        meta = agent.load_checkpoint(args.resume)
        start_epoch = meta["epoch"] + 1
        # Re-freeze S2V after loading checkpoint (requires_grad not saved)
        for p in agent.s2v_policy.parameters():
            p.requires_grad = False
        agent.optimizer = torch.optim.Adam(
            agent.q_policy.parameters(), lr=args.lr
        )
        print(f"  ├─ 已完成轮次: {meta['epoch']}", flush=True)
        print(f"  ├─ 最优奖励: {meta['best_reward']:.4f} (轮次 {meta['best_epoch']})", flush=True)
        print(f"  ├─ S2V 已冻结，仅训练 Q 头", flush=True)
        print(f"  └─ 存档时间: {meta['save_time']}", flush=True)

    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # 线性 epsilon 衰减：前 eps_decay_epochs 轮从 1.0 衰减到 0.05，之后保持 0.05
    epsilon_start = 1.0
    epsilon_end = 0.05
    eps_decay_epochs = args.eps_decay_epochs

    print(f"\n【开始训练】轮次 {start_epoch} → {args.epochs} | "
          f"学习率 {args.lr} | gamma={agent.gamma} | tau={agent.tau}", flush=True)
    print(f"  存档目录: {ckpt_dir}/", flush=True)

    patience_counter = 0
    global_start = time.time()

    # Policy degradation detector: auto-stop if reward drops while epsilon drops
    # Reward variance is very high (single epoch: -44 to +42), so use wide
    # window and generous threshold to avoid false positives.
    DEGRADE_WINDOW = 20   # compare moving averages every N epochs
    DEGRADE_STRIKES = 5   # N consecutive degradations -> stop
    reward_history = []
    degrade_strikes = 0

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # 线性 epsilon 衰减（在 eps_decay_epochs 轮内完成）
        progress = min(1.0, (epoch - 1) / max(eps_decay_epochs - 1, 1))
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * progress

        verbose_inst = (epoch == start_epoch)
        stats = agent.train_epoch(instances, epoch, epsilon,
                                    verbose_instances=verbose_inst,
                                    n_workers=args.n_workers)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - global_start
        remaining = elapsed / max(epoch - start_epoch + 1, 1) * (args.epochs - epoch)

        reward_history.append(stats["avg_reward"])

        if stats["is_best"]:
            patience_counter = 0
            agent.save_checkpoint(os.path.join(ckpt_dir, "best_model.pt"))
        else:
            patience_counter += 1

        # Policy degradation check every DEGRADE_WINDOW epochs
        # (only after epsilon has started decaying meaningfully)
        if len(reward_history) >= 2 * DEGRADE_WINDOW and epsilon < 0.85:
            recent_avg = np.mean(reward_history[-DEGRADE_WINDOW:])
            prev_avg = np.mean(reward_history[-2 * DEGRADE_WINDOW:-DEGRADE_WINDOW])
            if recent_avg < prev_avg - 3.0:
                degrade_strikes += 1
                print(f"\n  ⚠ 策略退化信号 [{degrade_strikes}/{DEGRADE_STRIKES}]: "
                      f"近{DEGRADE_WINDOW}轮均值 {recent_avg:.1f} < "
                      f"前{DEGRADE_WINDOW}轮均值 {prev_avg:.1f}", flush=True)
            else:
                degrade_strikes = 0

        if epoch % args.log_every == 0 or stats["is_best"] or epoch == args.epochs:
            mark = " ★ 新最优" if stats["is_best"] else ""
            print(f"\n【轮次 {epoch}/{args.epochs}】{mark}", flush=True)
            print(f"  ├─ 平均奖励: {stats['avg_reward']:.4f} "
                  f"(最优: {agent.best_reward:.4f}, 轮次 {agent.best_epoch})", flush=True)
            print(f"  ├─ 平均损失: {stats['avg_loss']:.6f}", flush=True)
            print(f"  ├─ 探索率(ε): {epsilon:.4f}", flush=True)
            print(f"  ├─ 约束满足率: "
                  f"容量 {stats['capacity_rate']:.1%} | "
                  f"连通 {stats['connected_rate']:.1%} | "
                  f"周长 {stats['perimeter_rate']:.1%}", flush=True)
            print(f"  ├─ 本轮耗时: {format_time(epoch_time)} | "
                  f"已用总时: {format_time(elapsed)}", flush=True)
            print(f"  ├─ 预计剩余: {format_time(remaining)}", flush=True)

            if epoch % args.save_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
                agent.save_checkpoint(ckpt_path)
                print(f"  └─ 存档: {ckpt_path} ✓", flush=True)
            else:
                print(f"  └─ 经验池: {stats['buffer_size']} 条", flush=True)

        if degrade_strikes >= DEGRADE_STRIKES:
            print(f"\n  ✗ 策略持续退化（连续 {DEGRADE_STRIKES} 次检测到奖励下降），自动停止训练", flush=True)
            print(f"    最优模型保留在 best_model.pt（轮次 {agent.best_epoch}，奖励 {agent.best_reward:.2f}）", flush=True)
            break

        if patience_counter >= args.patience:
            print(f"\n  ⚠ 连续 {args.patience} 轮无改善，触发早停", flush=True)
            break

    total_time = time.time() - global_start
    final_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
    agent.save_checkpoint(final_path)

    print(flush=True)
    print("══════════════════════════════════════════════════════════", flush=True)
    print(f"  训练完成 | 总耗时: {format_time(total_time)} | "
          f"最优轮次: 第 {agent.best_epoch} 轮", flush=True)
    print(f"  最终模型: {ckpt_dir}/best_model.pt", flush=True)
    print(f"  最终存档: {final_path}", flush=True)
    print("══════════════════════════════════════════════════════════", flush=True)


def show_status(args):
    ckpt_dir = args.status
    print(f"\n【训练进度】目录: {ckpt_dir}/", flush=True)
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not ckpt_files:
        print("  未找到任何 checkpoint", flush=True)
        return
    for f in ckpt_files:
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            print(f"  ├─ {os.path.basename(f)}: 轮次 {ckpt.get('current_epoch', '?')}, "
                  f"最优奖励 {ckpt.get('best_reward', 0):.4f}", flush=True)
        except Exception as e:
            print(f"  ├─ {os.path.basename(f)}: 读取失败 ({e})", flush=True)


def evaluate(args):
    print(f"\n【模型评估】加载 {args.eval}...", flush=True)
    agent = DQNPartitionAgent(device="cpu")
    meta = agent.load_checkpoint(args.eval)
    print(f"  轮次: {meta['epoch']}, 最优奖励: {meta['best_reward']:.4f}", flush=True)

    instances = load_all_instances()
    from utils.graph_utils import build_adjacency_graph

    for inst in instances:
        inst_id = inst["instance_info"]["instance_id"]
        graph = build_adjacency_graph(inst["pva_list"], inst["terrain_data"]["grid_size"])
        n_zones = inst["equipment_params"]["inverter"]["p"]
        result = agent.solve(graph, n_zones, inst["pva_params"])
        sizes = [len(z) for z in result.zones]
        mark = "✓" if result.is_feasible else "✗"
        print(f"  {mark} {inst_id}: 分区={sizes}, 周长={result.total_perimeter:.0f}m, "
              f"违规={len(result.violations)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="S2V-DQN 训练管线")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--eps-decay-epochs", type=int, default=100)
    parser.add_argument("--status", type=str, default=None)
    parser.add_argument("--eval", type=str, default=None)
    # 预训练参数
    parser.add_argument("--skip-pretrain", action="store_true", help="跳过行为克隆预训练")
    parser.add_argument("--skip-rl", action="store_true", help="跳过 RL 微调，仅使用 BC 模型")
    parser.add_argument("--expert-runs", type=int, default=20, help="每算例启发式运行次数")
    parser.add_argument("--pretrain-epochs", type=int, default=50, help="行为克隆训练轮数")
    parser.add_argument("--n-workers", type=int, default=8, help="并行 worker 数量（利用多核CPU）")
    args = parser.parse_args()

    if args.status:
        show_status(args)
    elif args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
