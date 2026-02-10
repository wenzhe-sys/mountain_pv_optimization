#!/usr/bin/env python3
"""最小端到端 dry run 测试：验证训练 pipeline 全链路可用 + 速度。"""
import sys, os, time
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.dqn_agent import DQNPartitionAgent
from utils.load_instance import InstanceLoader
from utils.graph_utils import build_adjacency_graph

# 只加载 3 个算例
loader = InstanceLoader()
instances = []
for rid in ["r1", "r2", "r3"]:
    try:
        inst = loader.load_instance(rid)
        n = inst["instance_info"]["n_nodes"]
        k = inst["equipment_params"]["inverter"]["p"]
        print(f"  加载 {rid}: {n} 节点, {k} 分区")
        instances.append(inst)
    except Exception as e:
        print(f"  加载 {rid} 失败: {e}")

agent = DQNPartitionAgent(
    dim_in=1, dim_embed=64, T=4,
    lr=1e-4, gamma=1.0, tau=0.01,
    buffer_size=50000, batch_size=64,
    train_every=5, device="cpu"
)

# ─── 阶段一：专家生成 ───
print("\n=== 阶段一: 专家生成 (1 run/instance) ===")
t0 = time.time()
expert_data = agent.generate_expert_data(instances, n_runs_per_instance=1)
print(f"  耗时: {time.time()-t0:.1f}秒, 轨迹数: {len(expert_data)}")

# ─── 阶段二：行为克隆 ───
if expert_data:
    print("\n=== 阶段二: 行为克隆 (2 epochs) ===")
    t0 = time.time()
    losses = agent.pretrain_from_expert(expert_data, n_epochs=2, lr=1e-3)
    print(f"  损失: {[f'{l:.4f}' for l in losses]}, 耗时: {time.time()-t0:.1f}秒")

# ─── 阶段三：RL 微调 ───
print("\n=== 阶段三: RL 微调 (2 epochs) ===")
for epoch in range(1, 3):
    t0 = time.time()
    epsilon = 1.0 - 0.475 * (epoch - 1)
    stats = agent.train_epoch(instances, epoch, epsilon, verbose_instances=(epoch == 1))
    elapsed = time.time() - t0
    print(f"  轮次 {epoch}: 奖励={stats['avg_reward']:.2f}, 损失={stats['avg_loss']:.6f}, "
          f"容量={stats['capacity_rate']:.1%}, 连通={stats['connected_rate']:.1%}, "
          f"周长={stats['perimeter_rate']:.1%}, 经验池={stats['buffer_size']}, "
          f"耗时={elapsed:.1f}秒")

# ─── 阶段四：推理 ───
print("\n=== 阶段四: 推理测试 ===")
for inst in instances:
    graph = build_adjacency_graph(inst["pva_list"], inst["terrain_data"]["grid_size"])
    n_zones = inst["equipment_params"]["inverter"]["p"]
    result = agent.solve(graph, n_zones, inst["pva_params"])
    sizes = [len(z) for z in result.zones]
    mark = "OK" if result.is_feasible else "NG"
    print(f"  [{mark}] {inst['instance_info']['instance_id']}: "
          f"分区={sizes}, 周长={result.total_perimeter:.0f}m, 违规={len(result.violations)}")

# ─── 阶段五：Checkpoint ───
print("\n=== 阶段五: Checkpoint 测试 ===")
ckpt_path = "/tmp/dry_run_test.pt"
agent.save_checkpoint(ckpt_path)
agent2 = DQNPartitionAgent(device="cpu")
meta = agent2.load_checkpoint(ckpt_path)
print(f"  保存/加载成功: epoch={meta['epoch']}, best_reward={meta['best_reward']:.4f}")
os.remove(ckpt_path)

print("\n" + "=" * 50)
print("  全链路 dry run 测试通过！")
print("=" * 50)
