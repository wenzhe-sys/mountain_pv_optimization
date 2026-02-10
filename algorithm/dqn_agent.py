"""
S2V-DQN 分区智能体

严格对齐 Khalil et al. 2017 论文和参考实现 jiayuanzhang0/S2V-DQN。

核心改动（相比之前的错误实现）：
  1. 奖励：每步立即奖励 = 周长变化量（密集信号）
  2. 训练：batch 训练（多个 transition 一起处理）
  3. 状态：二进制向量（0/1），dim_in=1，与原始 MaxCut 一致
  4. 分区策略：分 K 轮构建，每轮构建一个分区

参考实现：jiayuanzhang0/S2V-DQN
  - ReplayBuffer: 存 (graph_id, state, action, reward, next_state, done)
  - GSet: 存储图结构，训练时按 graph_id 查找
  - 训练: batch 采样 -> 编码 -> Q值 -> TD target -> 更新
  - 软更新: target_net = tau * policy_net + (1-tau) * target_net
"""

import os
import time
import random
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from algorithm.s2v_network import Structure2Vec, QFunction, encode_graph, get_graph_embedding
from model.partition_sub import PartitionValidator, PartitionResult
from utils.graph_utils import (
    build_adjacency_graph, build_coord_index, check_connectivity,
    calculate_perimeter_fast
)

logger = logging.getLogger(__name__)


# ─── 图数据存储 ───

class GraphData:
    """单个图的数据（邻接矩阵 + 边权重 + 坐标索引）。"""

    def __init__(self, graph: nx.Graph, device: str = "cpu"):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.coord_index = build_coord_index(graph)
        self.device = torch.device(device)

        # 构建密集邻接矩阵和边权重
        self.adj = torch.zeros(self.n_nodes, self.n_nodes, device=self.device)
        self.edge_weight = torch.zeros(self.n_nodes, self.n_nodes, device=self.device)

        for u, v, data in graph.edges(data=True):
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            self.adj[i, j] = 1.0
            self.adj[j, i] = 1.0
            w = 1.0  # 统一边权重
            self.edge_weight[i, j] = w
            self.edge_weight[j, i] = w


class GSet:
    """图结构存储，训练时按 graph_id 查找（与参考实现一致）。"""

    def __init__(self):
        self.graphs: List[GraphData] = []

    def push(self, graph_data: GraphData) -> int:
        gid = len(self.graphs)
        self.graphs.append(graph_data)
        return gid

    def __getitem__(self, gid: int) -> GraphData:
        return self.graphs[gid]


# ─── 经验回放 ───

class ReplayBuffer:
    """
    经验回放缓冲区。

    存储 (graph_id, state, action, reward, next_state, done)。
    graph_id 指向 GSet 中的图结构，避免重复存储邻接矩阵。
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, gid: int, state: torch.Tensor, action: int,
             reward: float, next_state: torch.Tensor, done: bool):
        self.buffer.append((
            torch.tensor([gid], dtype=torch.long),
            state.clone(),
            torch.tensor([[action]], dtype=torch.long),
            torch.tensor([[reward]], dtype=torch.float32),
            next_state.clone(),
            torch.tensor([[int(done)]], dtype=torch.long),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        gid, state, action, reward, next_state, done = zip(*batch)
        return (torch.cat(gid), list(state), torch.cat(action),
                torch.cat(reward), list(next_state), torch.cat(done))

    def __len__(self):
        return len(self.buffer)


# ─── 分区环境 ───

class PartitionEnv:
    """
    分区环境（单轮：构建一个分区）。

    与 MaxCut 环境的对应关系：
      MaxCut: 选节点 -> cut value 变化 -> reward
      分区:   选节点加入当前分区 -> 周长变化 -> reward

    状态：长度 N 的二进制向量（0=可选, 1=已选入当前分区）
    动作：选一个相邻的未选节点
    奖励：old_perimeter - new_perimeter（周长减少=正奖励）
    终止：达到目标大小 或 无相邻可扩展节点
    """

    def __init__(self, graph_data: GraphData,
                 target_size: int = 22,
                 min_size: int = 18, max_size: int = 26,
                 excluded: Set[int] = None):
        """
        Args:
            graph_data: 图数据
            target_size: 目标分区大小
            min_size: 最小分区大小
            max_size: 最大分区大小
            excluded: 已被其他分区占用的节点索引集合
        """
        self.gd = graph_data
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
        self.excluded = excluded or set()

    def reset(self) -> torch.Tensor:
        """重置环境，返回初始状态。"""
        self.state = torch.zeros(self.gd.n_nodes, dtype=torch.long,
                                  device=self.gd.device)
        # 标记已排除的节点（不可选）
        for idx in self.excluded:
            self.state[idx] = -1  # -1 = 已被其他分区占用

        self.current_zone_indices = set()  # 当前分区的节点索引集合
        self.current_perimeter = 0.0
        self.step_count = 0
        self.done = False

        return self.state.clone()

    def get_valid_actions(self) -> List[int]:
        """获取合法动作（未选且与当前分区相邻的节点）。"""
        if not self.current_zone_indices:
            # 分区为空，所有非排除节点都可选
            return [i for i in range(self.gd.n_nodes)
                    if self.state[i].item() == 0]

        valid = []
        for i in range(self.gd.n_nodes):
            if self.state[i].item() != 0:
                continue  # 已选或已排除
            # 检查是否与当前分区相邻
            for j in self.current_zone_indices:
                if self.gd.adj[i, j].item() > 0:
                    valid.append(i)
                    break
        return valid

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """
        执行动作。

        Args:
            action: 节点索引

        Returns:
            (next_state, reward, done)
        """
        if self.state[action].item() != 0:
            # 非法动作
            self.done = True
            return self.state.clone(), -5.0, True

        # 计算旧周长
        old_perimeter = self._compute_perimeter()

        # 加入节点
        self.state[action] = 1
        self.current_zone_indices.add(action)
        self.step_count += 1

        # 计算新周长
        new_perimeter = self._compute_perimeter()
        self.current_perimeter = new_perimeter

        # 奖励 = 周长减少量（论文风格：目标函数变化量）
        reward = old_perimeter - new_perimeter

        # 第一步没有有意义的周长变化（从 0 到单节点周长），给 0 奖励
        if self.step_count == 1:
            reward = 0.0

        # 终止条件
        if self.step_count >= self.max_size:
            self.done = True
        elif self.step_count >= self.min_size:
            valid = self.get_valid_actions()
            if not valid:
                self.done = True
            # 达到目标大小时也可以选择停止
            elif self.step_count >= self.target_size:
                self.done = True

        return self.state.clone(), reward, self.done

    def _compute_perimeter(self) -> float:
        """计算当前分区的周长。"""
        if not self.current_zone_indices:
            return 0.0

        zone_nodes = {self.gd.nodes[i] for i in self.current_zone_indices}
        return calculate_perimeter_fast(zone_nodes, self.gd.graph, self.gd.coord_index)

    def get_zone_nodes(self) -> Set[str]:
        """获取当前分区的面板 ID 集合。"""
        return {self.gd.nodes[i] for i in self.current_zone_indices}


# ─── DQN 智能体 ───

class DQNPartitionAgent:
    """
    S2V-DQN 分区智能体（对齐论文）。

    训练流程：
      for each episode:
          graph_data = load_instance()
          for k in range(n_zones):
              env = PartitionEnv(graph_data, excluded=已分配节点)
              state = env.reset()
              while not done:
                  embed = s2v.encode(state)
                  action = epsilon_greedy(Q, embed)
                  next_state, reward, done = env.step(action)
                  buffer.push(gid, state, action, reward, next_state, done)
                  train_step()
    """

    def __init__(self, dim_in: int = 1, dim_embed: int = 64,
                 T: int = 4, lr: float = 1e-3,
                 gamma: float = 1.0, tau: float = 0.05,
                 buffer_size: int = 10000, batch_size: int = 16,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.dim_in = dim_in
        self.dim_embed = dim_embed
        self.T = T
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # 策略网络
        self.s2v_policy = Structure2Vec(dim_in, dim_embed).to(self.device)
        self.q_policy = QFunction(dim_embed).to(self.device)

        # 目标网络（冻结梯度）
        self.s2v_target = Structure2Vec(dim_in, dim_embed).to(self.device)
        self.q_target = QFunction(dim_embed).to(self.device)
        self.s2v_target.load_state_dict(self.s2v_policy.state_dict())
        self.q_target.load_state_dict(self.q_policy.state_dict())
        for p in self.s2v_target.parameters():
            p.requires_grad = False
        for p in self.q_target.parameters():
            p.requires_grad = False

        # 优化器（S2V + Q 联合优化）
        self.optimizer = optim.Adam(
            list(self.s2v_policy.parameters()) + list(self.q_policy.parameters()),
            lr=lr
        )

        # 存储
        self.gset = GSet()
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 训练统计
        self.training_history = []
        self.best_reward = float("-inf")
        self.best_epoch = 0
        self.current_epoch = 0

    def _encode(self, gd: GraphData, state: torch.Tensor,
                 net_type: str = "policy") -> torch.Tensor:
        """用 S2V 编码状态，返回节点嵌入。"""
        # 特征 = 状态向量（0/1/-1 -> float），dim_in=1
        feat = state.float().unsqueeze(1).to(self.device)  # [N, 1]
        adj = gd.adj.to(self.device)
        ew = gd.edge_weight.to(self.device)

        s2v = self.s2v_policy if net_type == "policy" else self.s2v_target
        return encode_graph(s2v, feat, adj, ew, self.T)

    def select_action(self, gd: GraphData, state: torch.Tensor,
                       epsilon: float, valid_actions: List[int]) -> int:
        """
        Epsilon-greedy 动作选择。

        Args:
            gd: 图数据
            state: 当前状态
            epsilon: 探索率
            valid_actions: 合法动作列表

        Returns:
            选择的节点索引
        """
        if not valid_actions:
            return 0  # fallback

        if random.random() < epsilon:
            # 探索：随机选
            return random.choice(valid_actions)

        # 利用：选 Q 值最大的
        with torch.no_grad():
            embed = self._encode(gd, state, "policy")
            state_embed = get_graph_embedding(embed)  # [1, dim_embed]
            q_values = self.q_policy(state_embed, embed)  # [N, 1]

            # 只在合法动作中选最大
            valid_q = q_values[valid_actions]
            best_idx = torch.argmax(valid_q).item()
            return valid_actions[best_idx]

    def train_step(self) -> Optional[float]:
        """
        单步训练（从 replay buffer 采样并更新）。

        对齐参考实现的 train() 方法。

        Returns:
            loss 值或 None
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        gid, states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        B = len(states)
        total_loss = torch.tensor(0.0, device=self.device)

        # 逐样本计算（图结构不同，无法真正 batch）
        # 但每个样本只做 1 次前向传播，比之前快很多
        q_values_list = []
        q_targets_list = []

        for i in range(B):
            gd = self.gset[gid[i].item()]
            s = states[i].to(self.device)
            a = actions[i, 0].item()
            r = rewards[i, 0].item()
            s_next = next_states[i].to(self.device)
            d = dones[i, 0].item()

            # 当前 Q(s, a)
            embed = self._encode(gd, s, "policy")
            state_embed = get_graph_embedding(embed)
            q = self.q_policy(state_embed, embed[a:a+1])  # [1, 1]
            q_values_list.append(q.squeeze())

            # TD target
            if d:
                target = torch.tensor(r, device=self.device, dtype=torch.float32)
            else:
                with torch.no_grad():
                    # 找 next_state 的合法动作
                    remaining = (s_next == 0).nonzero(as_tuple=True)[0].tolist()

                    if remaining:
                        # Double DQN: 用 policy 网络选动作，用 target 网络评估
                        embed_next_p = self._encode(gd, s_next, "policy")
                        state_embed_next_p = get_graph_embedding(embed_next_p)
                        q_next_p = self.q_policy(state_embed_next_p, embed_next_p[remaining])
                        best_action_idx = remaining[torch.argmax(q_next_p).item()]

                        embed_next_t = self._encode(gd, s_next, "target")
                        state_embed_next_t = get_graph_embedding(embed_next_t)
                        q_next_t = self.q_target(state_embed_next_t,
                                                   embed_next_t[best_action_idx:best_action_idx+1])
                        target = r + self.gamma * q_next_t.squeeze()
                    else:
                        target = torch.tensor(r, device=self.device, dtype=torch.float32)

            q_targets_list.append(target)

        # 合并计算 loss
        q_values = torch.stack(q_values_list)
        q_targets = torch.stack(q_targets_list)
        loss = nn.MSELoss()(q_values, q_targets)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络
        for p, tp in zip(self.s2v_policy.parameters(), self.s2v_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.q_policy.parameters(), self.q_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return loss.item()

    def train_epoch(self, instances: List[Dict], epoch: int,
                     epsilon: float, verbose_instances: bool = False) -> Dict:
        """
        训练一个 epoch（遍历所有算例）。

        每个算例分 K 轮构建 K 个分区，每轮是一个独立的 episode。

        Args:
            instances: 算例列表
            epoch: 当前 epoch
            epsilon: 当前探索率
            verbose_instances: 是否打印每个算例进度

        Returns:
            训练统计
        """
        epoch_rewards = []
        epoch_losses = []
        constraint_stats = {"capacity": 0, "connected": 0, "perimeter": 0, "total_zones": 0}

        for inst in instances:
            inst_id = inst["instance_info"]["instance_id"]
            graph = build_adjacency_graph(inst["pva_list"],
                                           inst["terrain_data"]["grid_size"])
            n_zones = inst["equipment_params"]["inverter"]["p"]
            n_nodes = inst["instance_info"]["n_nodes"]
            target_size = n_nodes // n_zones
            pva_params = inst["pva_params"]

            gd = GraphData(graph, str(self.device))
            gid = self.gset.push(gd)

            # 分 K 轮构建 K 个分区（所有分区都由 DQN 决策）
            excluded_indices = set()
            zones = []
            instance_reward = 0.0
            instance_losses = []

            for k in range(n_zones):
                remaining_count = n_nodes - len(excluded_indices)
                zones_left = n_zones - k
                current_target = remaining_count // zones_left
                current_max = min(26, remaining_count - (zones_left - 1) * 18) if zones_left > 1 else remaining_count

                env = PartitionEnv(
                    gd, target_size=current_target,
                    min_size=min(18, remaining_count),
                    max_size=current_max,
                    excluded=excluded_indices
                )
                state = env.reset()

                episode_reward = 0.0
                while not env.done:
                    valid = env.get_valid_actions()
                    if not valid:
                        break

                    action = self.select_action(gd, state, epsilon, valid)
                    next_state, reward, done = env.step(action)

                    self.replay_buffer.push(gid, state, action, reward, next_state, done)

                    loss = self.train_step()
                    if loss is not None:
                        instance_losses.append(loss)

                    state = next_state
                    episode_reward += reward

                instance_reward += episode_reward
                zone_nodes = env.get_zone_nodes()
                # 最后一个分区补充剩余节点
                if k == n_zones - 1:
                    for i in range(gd.n_nodes):
                        if i not in excluded_indices and i not in env.current_zone_indices:
                            zone_nodes.add(gd.nodes[i])
                zones.append(zone_nodes)
                excluded_indices |= env.current_zone_indices

            epoch_rewards.append(instance_reward)
            if instance_losses:
                epoch_losses.append(np.mean(instance_losses))

            # 验证约束
            coord_index = build_coord_index(graph)
            validator = PartitionValidator(
                graph, n_zones, min_panels=18, max_panels=26,
                perimeter_lb=pva_params["LB"], perimeter_ub=pva_params["UB"]
            )
            result = validator.validate(zones)
            constraint_stats["total_zones"] += len(result.zone_details)
            for detail in result.zone_details:
                if detail["capacity_ok"]:
                    constraint_stats["capacity"] += 1
                if detail["is_connected"]:
                    constraint_stats["connected"] += 1
                if detail["perimeter_ok"]:
                    constraint_stats["perimeter"] += 1

            if verbose_instances:
                sizes = [len(z) for z in zones]
                print(f"    算例 {inst_id}: 奖励={instance_reward:.2f}, "
                      f"分区={sizes}, 可行={result.is_feasible}", flush=True)

        # 汇总
        n_total = max(constraint_stats["total_zones"], 1)
        stats = {
            "epoch": epoch,
            "avg_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
            "avg_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "epsilon": epsilon,
            "capacity_rate": constraint_stats["capacity"] / n_total,
            "connected_rate": constraint_stats["connected"] / n_total,
            "perimeter_rate": constraint_stats["perimeter"] / n_total,
            "buffer_size": len(self.replay_buffer),
            "is_best": False,
        }

        if stats["avg_reward"] > self.best_reward:
            self.best_reward = stats["avg_reward"]
            self.best_epoch = epoch
            stats["is_best"] = True

        self.current_epoch = epoch
        self.training_history.append(stats)
        return stats

    def solve(self, graph: nx.Graph, n_zones: int,
              pva_params: Dict = None) -> PartitionResult:
        """推理：用训练好的策略构建分区方案。所有分区都由 DQN 决策。"""
        if pva_params is None:
            pva_params = {"LB": 60.0, "UB": 90.0}

        gd = GraphData(graph, str(self.device))
        n_nodes = gd.n_nodes
        target_size = n_nodes // n_zones

        excluded_indices = set()
        zones = []

        for k in range(n_zones):
            remaining_count = n_nodes - len(excluded_indices)
            zones_left = n_zones - k

            # 动态调整目标大小，确保剩余面板能均匀分配
            current_target = remaining_count // zones_left
            current_max = min(26, remaining_count - (zones_left - 1) * 18) if zones_left > 1 else remaining_count

            env = PartitionEnv(gd, target_size=current_target,
                                min_size=min(18, remaining_count),
                                max_size=current_max,
                                excluded=excluded_indices)
            state = env.reset()

            while not env.done:
                valid = env.get_valid_actions()
                if not valid:
                    break
                action = self.select_action(gd, state, epsilon=0.0, valid_actions=valid)
                state, _, _ = env.step(action)

            zone_nodes = env.get_zone_nodes()
            # 如果是最后一个分区且还有剩余，补进来
            if k == n_zones - 1:
                for i in range(gd.n_nodes):
                    if i not in excluded_indices and i not in env.current_zone_indices:
                        zone_nodes.add(gd.nodes[i])

            zones.append(zone_nodes)
            excluded_indices |= env.current_zone_indices
            # 补充最后分区新加的节点到 excluded
            if k == n_zones - 1:
                excluded_indices |= {gd.node_to_idx[n] for n in zone_nodes if n in gd.node_to_idx}

        validator = PartitionValidator(
            graph, n_zones, min_panels=18, max_panels=26,
            perimeter_lb=pva_params.get("LB", 60.0),
            perimeter_ub=pva_params.get("UB", 90.0)
        )
        result = validator.validate(zones)
        result.solver_method = "dqn"
        return result

    # ─── 专家经验生成与行为克隆 ───

    def generate_expert_data(self, instances: List[Dict],
                               n_runs_per_instance: int = 60,
                               only_feasible: bool = True) -> List[Dict]:
        """
        用启发式生成专家经验，拆解为 (state, action) 动作序列。

        启发式（GreedyPartitioner）产出的可行分区方案就是"专家方案"。
        将专家方案倒推为一步步的动作选择序列，供行为克隆使用。

        Args:
            instances: 算例列表
            n_runs_per_instance: 每个算例跑多少次（不同随机种子）
            only_feasible: 是否只保留可行方案

        Returns:
            专家轨迹列表，每条轨迹包含 {gid, transitions: [(state, action), ...]}
        """
        from algorithm.partition_heuristic import GreedyPartitioner

        expert_trajectories = []
        total_feasible = 0
        total_runs = 0

        for inst in instances:
            inst_id = inst["instance_info"]["instance_id"]
            graph = build_adjacency_graph(inst["pva_list"],
                                           inst["terrain_data"]["grid_size"])
            n_zones = inst["equipment_params"]["inverter"]["p"]
            n_nodes = inst["instance_info"]["n_nodes"]
            target_size = n_nodes // n_zones
            pva_params = inst["pva_params"]

            gd = GraphData(graph, str(self.device))
            gid = self.gset.push(gd)

            for seed in range(n_runs_per_instance):
                total_runs += 1
                partitioner = GreedyPartitioner(
                    graph, n_zones, random_seed=seed,
                    min_panels=18, max_panels=26,
                    perimeter_lb=pva_params["LB"],
                    perimeter_ub=pva_params["UB"],
                    local_search_iters=200
                )
                result = partitioner.solve()

                if only_feasible and not result.is_feasible:
                    continue

                total_feasible += 1

                # 将分区方案拆解为动作序列
                for zone_nodes in result.zones:
                    transitions = self._zone_to_trajectory(gd, zone_nodes, n_zones)
                    if transitions:
                        expert_trajectories.append({
                            "gid": gid,
                            "transitions": transitions,
                        })

        print(f"  专家经验生成完成: {total_runs} 次运行, "
              f"{total_feasible} 个可行方案, "
              f"{len(expert_trajectories)} 条轨迹", flush=True)

        return expert_trajectories

    def _zone_to_trajectory(self, gd: GraphData, zone_nodes: Set[str],
                              n_zones: int) -> List[Tuple[torch.Tensor, int]]:
        """
        将一个分区方案（一组节点）拆解为动作序列。

        策略：从分区的某个"边缘"节点开始，用 BFS 顺序重构添加过程。
        每一步记录 (当前state, 选择的action)。

        Args:
            gd: 图数据
            zone_nodes: 分区中的面板 ID 集合
            n_zones: 分区总数

        Returns:
            动作序列 [(state_tensor, action_index), ...]
        """
        # 将面板 ID 转为索引
        zone_indices = set()
        for node_name in zone_nodes:
            if node_name in gd.node_to_idx:
                zone_indices.add(gd.node_to_idx[node_name])

        if len(zone_indices) < 2:
            return []

        # 用 BFS 从重心最近的节点开始，确定添加顺序
        zone_list = list(zone_indices)
        rows = [gd.graph.nodes[gd.nodes[i]]["row"] for i in zone_list]
        cols = [gd.graph.nodes[gd.nodes[i]]["col"] for i in zone_list]
        center_r = np.mean(rows)
        center_c = np.mean(cols)

        # 找最靠近重心的节点作为起点
        start_idx = min(zone_list, key=lambda i: (
            abs(gd.graph.nodes[gd.nodes[i]]["row"] - center_r) +
            abs(gd.graph.nodes[gd.nodes[i]]["col"] - center_c)
        ))

        # BFS 确定添加顺序
        ordered = []
        visited = {start_idx}
        queue = deque([start_idx])
        while queue:
            current = queue.popleft()
            ordered.append(current)
            # 找 current 在图中的邻居中属于 zone 但未访问的
            for j in zone_indices:
                if j not in visited and gd.adj[current, j].item() > 0:
                    visited.add(j)
                    queue.append(j)

        # 如果 BFS 没覆盖所有节点（不连通），追加剩余
        for idx in zone_indices:
            if idx not in visited:
                ordered.append(idx)

        # 构建 (state, action) 序列
        transitions = []
        state = torch.zeros(gd.n_nodes, dtype=torch.long, device=gd.device)

        for action_idx in ordered:
            transitions.append((state.clone(), action_idx))
            state[action_idx] = 1

        return transitions

    def pretrain_from_expert(self, expert_trajectories: List[Dict],
                               n_epochs: int = 50,
                               lr: float = 1e-3) -> List[float]:
        """
        行为克隆预训练：用交叉熵损失让 DQN 模仿专家的动作选择。

        对每个 (state, expert_action) 对：
          1. 用 S2V 编码 state → 得到所有节点嵌入
          2. 用 Q 函数计算所有节点的 Q 值
          3. 让 expert_action 对应的 Q 值最高（交叉熵损失）

        Args:
            expert_trajectories: generate_expert_data() 的输出
            n_epochs: 训练轮数
            lr: 学习率

        Returns:
            每轮的平均损失
        """
        # 收集所有 (gid, state, action) 样本
        all_samples = []
        for traj in expert_trajectories:
            gid = traj["gid"]
            for state, action in traj["transitions"]:
                all_samples.append((gid, state, action))

        if not all_samples:
            print("  ⚠ 无专家样本可用", flush=True)
            return []

        print(f"  行为克隆样本数: {len(all_samples)}", flush=True)

        # 用独立优化器（不干扰 RL 的优化器状态）
        bc_optimizer = optim.Adam(
            list(self.s2v_policy.parameters()) + list(self.q_policy.parameters()),
            lr=lr
        )

        epoch_losses = []

        for epoch in range(1, n_epochs + 1):
            random.shuffle(all_samples)
            total_loss = 0.0
            n_batches = 0

            # mini-batch 训练
            batch_size = 32
            for i in range(0, len(all_samples), batch_size):
                batch = all_samples[i:i + batch_size]
                batch_loss = torch.tensor(0.0, device=self.device)

                for gid, state, action in batch:
                    gd = self.gset[gid]
                    embed = self._encode(gd, state.to(self.device), "policy")
                    state_embed = get_graph_embedding(embed)

                    # 计算所有节点的 Q 值
                    q_values = self.q_policy(state_embed, embed).squeeze(-1)  # [N]

                    # 找合法动作（state == 0 的节点）
                    valid_mask = (state == 0)
                    if valid_mask.sum() == 0:
                        continue

                    # 交叉熵损失：让 expert_action 的 Q 值最高
                    # 只在合法动作中计算 softmax
                    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                    valid_q = q_values[valid_indices]

                    # expert_action 在 valid_indices 中的位置
                    expert_pos = (valid_indices == action).nonzero(as_tuple=True)[0]
                    if len(expert_pos) == 0:
                        continue  # 专家动作不在合法范围内，跳过

                    target = expert_pos[0]
                    loss = nn.CrossEntropyLoss()(valid_q.unsqueeze(0), target.unsqueeze(0))
                    batch_loss = batch_loss + loss

                if batch_loss.requires_grad:
                    avg_batch_loss = batch_loss / len(batch)
                    bc_optimizer.zero_grad()
                    avg_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.s2v_policy.parameters()) + list(self.q_policy.parameters()),
                        10.0
                    )
                    bc_optimizer.step()
                    total_loss += avg_batch_loss.item()
                    n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  【行为克隆 {epoch}/{n_epochs}】损失: {avg_loss:.6f}", flush=True)

        # 同步到目标网络
        self.s2v_target.load_state_dict(self.s2v_policy.state_dict())
        self.q_target.load_state_dict(self.q_policy.state_dict())

        # 更新 RL 优化器（用预训练后的参数）
        self.optimizer = optim.Adam(
            list(self.s2v_policy.parameters()) + list(self.q_policy.parameters()),
            lr=self.optimizer.defaults.get("lr", 1e-3)
        )

        return epoch_losses

    # ─── Checkpoint ───

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "s2v_policy": self.s2v_policy.state_dict(),
            "s2v_target": self.s2v_target.state_dict(),
            "q_policy": self.q_policy.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "best_reward": self.best_reward,
            "best_epoch": self.best_epoch,
            "training_history": self.training_history,
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.random.get_rng_state(),
            "save_time": datetime.now().isoformat(),
        }, path)

    def load_checkpoint(self, path: str) -> Dict:
        ckpt = torch.load(path, map_location=self.device)
        self.s2v_policy.load_state_dict(ckpt["s2v_policy"])
        self.s2v_target.load_state_dict(ckpt["s2v_target"])
        self.q_policy.load_state_dict(ckpt["q_policy"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.current_epoch = ckpt["current_epoch"]
        self.best_reward = ckpt["best_reward"]
        self.best_epoch = ckpt["best_epoch"]
        self.training_history = ckpt["training_history"]
        if "random_state" in ckpt:
            random.setstate(ckpt["random_state"])
        if "numpy_state" in ckpt:
            np.random.set_state(ckpt["numpy_state"])
        if "torch_state" in ckpt:
            torch.random.set_rng_state(ckpt["torch_state"])
        return {"epoch": self.current_epoch, "best_reward": self.best_reward,
                "best_epoch": self.best_epoch, "save_time": ckpt.get("save_time", "?")}
