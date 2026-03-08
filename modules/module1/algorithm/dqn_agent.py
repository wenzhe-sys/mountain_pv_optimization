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
from modules.module1.algorithm.s2v_network import (
    Structure2Vec, AttentionS2V, QFunction, DuelingQFunction, encode_graph, get_graph_embedding
)
from modules.module1.model.partition_sub import PartitionValidator, PartitionResult
from utils.graph_utils import (
    build_adjacency_graph, build_coord_index, check_connectivity,
    get_connected_components, calculate_perimeter_fast
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

        # Precompute degree vector for S2V term3 optimization
        # When edge weights are uniform, term3 reduces from O(N^2*d) to O(N*d)
        self.degree = self.adj.sum(dim=1)  # [N]


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

class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区。

    基于 TD-error 优先级进行采样，提高学习效率。
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # 优先级 exponent
        self.beta = beta    # 重要性采样 exponent

    def push(self, gid: int, state: torch.Tensor, action: int,
             reward: float, next_state: torch.Tensor, done: bool):
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((
            torch.tensor([gid], dtype=torch.long),
            state.clone(),
            torch.tensor([[action]], dtype=torch.long),
            torch.tensor([[reward]], dtype=torch.float32),
            next_state.clone(),
            torch.tensor([[int(done)]], dtype=torch.long),
        ))
        self.priorities.append(max_prio)  # 新经验赋予最高优先级

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            return (torch.tensor([]), [], torch.tensor([]),
                    torch.tensor([]), [], torch.tensor([]))

        # 计算优先级概率
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs, replace=False)
        batch = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化权重

        gid, state, action, reward, next_state, done = zip(*batch)
        return (torch.cat(gid), list(state), torch.cat(action),
                torch.cat(reward), list(next_state), torch.cat(done))

    def update_priorities(self, indices, priorities):
        """
        更新采样经验的优先级。
        """
        for i, idx in enumerate(indices):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priorities[i]

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
                 excluded: Set[int] = None,
                 perimeter_lb: float = 60.0, perimeter_ub: float = 90.0):
        """
        Args:
            graph_data: 图数据
            target_size: 目标分区大小
            min_size: 最小分区大小
            max_size: 最大分区大小
            excluded: 已被其他分区占用的节点索引集合
            perimeter_lb: 周长下界 (m)
            perimeter_ub: 周长上界 (m)
        """
        self.gd = graph_data
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
        self.excluded = excluded or set()
        self.perimeter_lb = perimeter_lb
        self.perimeter_ub = perimeter_ub

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
        """获取合法动作（未选且与当前分区相邻的节点）。向量化实现。"""
        available_mask = (self.state == 0)
        if not self.current_zone_indices:
            return available_mask.nonzero(as_tuple=True)[0].tolist()

        zone_mask = (self.state == 1)
        adj_to_zone = self.gd.adj[:, zone_mask].sum(dim=1) > 0
        valid_mask = available_mask & adj_to_zone
        return valid_mask.nonzero(as_tuple=True)[0].tolist()

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """
        执行动作。

        Reward design: hybrid (per-step shaping + terminal constraint).
        - Per-step: small reward based on perimeter delta, giving DQN a
          gradient signal at every step so Q-values don't rely on long
          chain propagation through 20 steps of reward=0.
        - Terminal: strong constraint-based reward for capacity + perimeter.

        Args:
            action: 节点索引

        Returns:
            (next_state, reward, done)
        """
        if self.state[action].item() != 0:
            # illegal action
            self.done = True
            return self.state.clone(), -10.0, True

        # add node
        self.state[action] = 1
        self.current_zone_indices.add(action)
        self.step_count += 1

        # incremental perimeter update
        delta = self._compute_perimeter_delta(action)
        self.current_perimeter += delta

        # per-step shaping reward: prefer nodes that reduce perimeter
        # delta < 0 means perimeter decreased (good), delta > 0 means increased
        # scale to ~[-0.5, +0.5] range so it doesn't dominate terminal reward
        reward = -delta * 0.05

        # termination conditions
        if self.step_count >= self.max_size:
            self.done = True
        elif self.step_count >= self.min_size:
            valid = self.get_valid_actions()
            if not valid:
                self.done = True
            elif self.step_count >= self.target_size:
                self.done = True

        # terminal reward: strong constraint-based signal
        if self.done:
            reward += self._compute_terminal_reward()

        return self.state.clone(), reward, self.done

    def _compute_terminal_reward(self) -> float:
        """
        Terminal reward — balanced constraint-based signal at episode end.

        Works together with per-step shaping reward to provide stable training signals.

        Components:
          1. Capacity:   +5 if in [min, max], else -5 per panel deviation
          2. Perimeter:  +5 if in [LB, UB], else -0.5 per meter deviation
          3. Compactness: +0 to +3 bonus for lower perimeter (within bounds)
          4. Balance: +2 bonus for being close to target size
          5. Connectivity: +2 bonus if partition is connected
        """
        size = len(self.current_zone_indices)
        perimeter = self.current_perimeter
        terminal = 0.0

        # --- Capacity constraint [min_size, max_size] ---
        if self.min_size <= size <= self.max_size:
            terminal += 5.0
            # bonus for being close to target_size
            size_dev = abs(size - self.target_size)
            terminal += max(0.0, 2.0 - 0.5 * size_dev)  # 平衡奖励
        else:
            # 超出上下界的惩罚（分级惩罚）
            if size < self.min_size:
                deviation = self.min_size - size
                if deviation == 1:
                    terminal -= 5.0  # 轻微违规
                else:
                    terminal -= 10.0 * deviation  # 严重违规
            else:
                deviation = size - self.max_size
                if deviation == 1:
                    terminal -= 5.0  # 轻微违规
                else:
                    terminal -= 10.0 * deviation  # 严重违规

        # --- Perimeter constraint [LB, UB] ---
        if size >= self.min_size:
            if self.perimeter_lb <= perimeter <= self.perimeter_ub:
                terminal += 5.0
                # bonus for being close to target perimeter (midpoint)
                target_perim = (self.perimeter_lb + self.perimeter_ub) / 2
                perim_dev = abs(perimeter - target_perim)
                terminal += max(0.0, 3.0 - 0.1 * perim_dev)  # 平衡周长奖励
                # 额外奖励：周长越小越好
                terminal += (self.perimeter_ub - perimeter) * 0.02
            else:
                # 超出上下界的惩罚（分级惩罚）
                if perimeter < self.perimeter_lb:
                    deviation = self.perimeter_lb - perimeter
                    if deviation < 5:
                        terminal -= 2.0 * deviation  # 轻微违规
                    else:
                        terminal -= 4.0 * deviation  # 严重违规
                else:
                    deviation = perimeter - self.perimeter_ub
                    if deviation < 5:
                        terminal -= 2.0 * deviation  # 轻微违规
                    else:
                        terminal -= 4.0 * deviation  # 严重违规
        else:
            # size too small => perimeter is meaningless, penalize size shortage
            terminal -= 3.0 * (self.min_size - size)

        # --- Connectivity bonus ---
        # 检查分区是否连通
        if size > 1:
            zone_nodes = {self.gd.nodes[i] for i in self.current_zone_indices}
            from utils.graph_utils import check_connectivity
            if check_connectivity(self.gd.graph, zone_nodes):
                terminal += 2.0

        return terminal

    def _compute_perimeter(self) -> float:
        """计算当前分区的周长（全量，用于初始化和验证）。"""
        if not self.current_zone_indices:
            return 0.0

        zone_nodes = {self.gd.nodes[i] for i in self.current_zone_indices}
        return calculate_perimeter_fast(zone_nodes, self.gd.graph, self.gd.coord_index)

    def _compute_perimeter_delta(self, action: int) -> float:
        """增量计算：添加节点 action 后的周长变化量。"""
        node_name = self.gd.nodes[action]
        data = self.gd.graph.nodes[node_name]
        cut_l, cut_w = data.get("cut_spec", (2.0, 3.0))

        # 新节点贡献 4 条边界边
        delta = 2 * cut_l + 2 * cut_w

        # 减去与已有分区节点的共享边（双向：新节点少一条边界，旧邻居也少一条）
        zone_node_names = {self.gd.nodes[i] for i in self.current_zone_indices}
        for neighbor in self.gd.graph.neighbors(node_name):
            if neighbor in zone_node_names:
                edge_data = self.gd.graph.edges[node_name, neighbor]
                direction = edge_data.get("direction", "horizontal")
                if direction == "vertical":
                    delta -= 2 * cut_l  # 新节点和旧邻居各减 cut_l
                else:
                    delta -= 2 * cut_w

        return max(delta, -self.current_perimeter)  # 周长不能为负

    def get_zone_nodes(self) -> Set[str]:
        """获取当前分区的面板 ID 集合。"""
        return {self.gd.nodes[i] for i in self.current_zone_indices}

    def _find_jump_candidates(self) -> List[int]:
        """当合法动作为空但未达 min_size 时，找距质心最近的未分配节点。"""
        if not self.current_zone_indices:
            return []

        # 计算当前分区质心
        rows, cols = [], []
        for i in self.current_zone_indices:
            node_name = self.gd.nodes[i]
            data = self.gd.graph.nodes[node_name]
            rows.append(data["row"])
            cols.append(data["col"])
        center_r, center_c = np.mean(rows), np.mean(cols)

        # 所有未分配节点按距质心距离排序
        candidates = []
        for i in range(self.gd.n_nodes):
            if self.state[i].item() != 0:
                continue
            node_name = self.gd.nodes[i]
            data = self.gd.graph.nodes[node_name]
            dist = abs(data["row"] - center_r) + abs(data["col"] - center_c)
            candidates.append((dist, i))
        candidates.sort()
        return [idx for _, idx in candidates[:5]]


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
    def soft_update_target(self, tau: float):
        """
        软更新目标网络参数。
        
        Args:
            tau: 更新因子，当tau=1.0时为硬更新
        """
        # 更新Q函数目标网络
        for param, target_param in zip(self.qfunc_policy.parameters(), self.qfunc_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def __init__(self, dim_in: int = 1, dim_embed: int = 128, T: int = 6,
                lr: float = 1e-4, gamma: float = 0.99, tau: float = 0.005,
                buffer_size: int = 100000, batch_size: int = 256,
                train_every: int = 4, device: str = "auto",
                use_prioritized_buffer: bool = True,
                use_dueling: bool = True):
        # 图集合（用于训练）
        self.gset = GSet()
        
        # 图缓存（用于专家经验生成）
        self._graph_cache = {}

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.train_every = train_every
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.T = T  # S2V 迭代次数
        self.dim_embed = dim_embed  # 保存嵌入维度
        
        # S2V 网络 - 使用带注意力机制的版本
        self.s2v = AttentionS2V(dim_in, dim_embed).to(self.device)
        # Q 函数网络
        if use_dueling:
            self.qfunc_policy = DuelingQFunction(dim_embed).to(self.device)
            self.qfunc_target = DuelingQFunction(dim_embed).to(self.device)
        else:
            self.qfunc_policy = QFunction(dim_embed).to(self.device)
            self.qfunc_target = QFunction(dim_embed).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(list(self.s2v.parameters()) + list(self.qfunc_policy.parameters()), lr=lr, weight_decay=1e-5)
        # 学习率调度器 - 使用余弦退火
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2, eta_min=1e-7)
        
        # 经验回放缓冲区
        if use_prioritized_buffer:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # 训练统计
        self.current_epoch = 0
        self.best_reward = -float('inf')
        self.best_epoch = 0  # 最佳奖励对应的轮次
        self.step_count = 0
        self._global_step = 0
        self.training_history = []  # 训练历史记录
        
        # 探索参数
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay_steps = 100000
        
        # 目标网络更新频率
        self.target_update_freq = 500
        
        # 训练准备
        self.replay_start_size = 1000
        
        # 复制目标网络
        self.soft_update_target(1.0)

    def _encode(self, gd: GraphData, state: torch.Tensor, 
                 net_type: str = "policy") -> torch.Tensor:
        """用 S2V 编码状态，返回节点嵌入。"""
        # 特征 = 状态向量（0/1/-1 -> float），dim_in=1
        feat = state.float().unsqueeze(1).to(self.device)  # [N, 1]
        adj = gd.adj.to(self.device)
        ew = gd.edge_weight.to(self.device)

        # 使用统一的 S2V 网络
        try:
            return encode_graph(self.s2v, feat, adj, ew, self.T, degree=gd.degree)
        except Exception as e:
            print(f"编码错误: {e}")
            print(f"特征维度: {feat.shape}")
            print(f"邻接矩阵维度: {adj.shape}")
            print(f"边权重维度: {ew.shape}")
            # 返回默认嵌入
            return torch.zeros(gd.n_nodes, self.s2v.dim_embed, device=self.device)

    def select_action(self, gd: GraphData, state: torch.Tensor,
                    epsilon: float, valid: List[int] = None, valid_actions: List[int] = None) -> int:
        """
        Epsilon-greedy 动作选择。

        Args:
            gd: 图数据
            state: 当前状态
            epsilon: 探索率
            valid: 合法动作列表（兼容旧版本）
            valid_actions: 合法动作列表

        Returns:
            选择的节点索引
        """
        # 兼容旧版本调用
        if valid_actions is None:
            valid_actions = valid
        
        if not valid_actions:
            return 0  # fallback

        # 推理时使用小的探索率，避免陷入局部最优
        if random.random() < epsilon:
            # 探索：随机选
            return random.choice(valid_actions)

        # 利用：选 Q 值最大的
        with torch.no_grad():
            try:
                embed = self._encode(gd, state, "policy")
                state_embed = get_graph_embedding(embed)  # [1, dim_embed]
                q_values = self.qfunc_policy(state_embed, embed)  # [N, 1]

                # 只在合法动作中选最大
                valid_q = q_values[valid_actions]
                best_idx = torch.argmax(valid_q).item()
                return valid_actions[best_idx]
            except Exception as e:
                print(f"选择动作时出错: {e}")
                # 出错时随机选择一个合法动作
                return random.choice(valid_actions)

    def _post_process_zones(self, zones: List[Set[str]], graph: nx.Graph,
                             min_panels: int = 18, max_panels: int = 26) -> List[Set[str]]:
        """
        后处理：分配遗漏节点 + 连通性修复 + 强化重平衡 + 周长优化。
        设计为 O(N × K × avg_degree)，避免昂贵的全图 BFS。
        """
        all_nodes = set(graph.nodes())
        assigned = set()
        for z in zones:
            assigned |= z
        unassigned = all_nodes - assigned

        # 阶段 1：将未分配节点分配到相邻分区（邻接优先，次选最小分区）
        # 多轮迭代，直到无法再分配
        for _ in range(20):  # 增加迭代次数
            if not unassigned:
                break
            placed_any = False
            for node in list(unassigned):
                # 找所有与该节点相邻的分区
                adjacent_zones = []
                for nb in graph.neighbors(node):
                    for zi, z in enumerate(zones):
                        if nb in z and len(z) < max_panels:
                            adjacent_zones.append(zi)
                if adjacent_zones:
                    # 选最小的分区（帮助平衡）
                    best_zi = min(adjacent_zones, key=lambda zi: len(zones[zi]))
                    zones[best_zi].add(node)
                    unassigned.discard(node)
                    placed_any = True
            if not placed_any:
                break

        # 无法通过邻接分配的节点，放入最近的最小分区
        for node in list(unassigned):
            node_data = graph.nodes[node]
            nr, nc = node_data["row"], node_data["col"]
            best_zi, best_dist = -1, float("inf")
            for zi, z in enumerate(zones):
                if len(z) >= max_panels or not z:
                    continue
                for zn in z:
                    zd = graph.nodes[zn]
                    dist = abs(nr - zd["row"]) + abs(nc - zd["col"])
                    if dist < best_dist:
                        best_dist = dist
                        best_zi = zi
                    break  # 只检查第一个节点作为代理，够快
            if best_zi >= 0:
                zones[best_zi].add(node)
                unassigned.discard(node)

        # 阶段 1b：强制分配剩余节点（即使分区会超过max_panels）
        for node in list(unassigned):
            # 找到最小的分区
            best_zi = min(range(len(zones)), key=lambda zi: len(zones[zi]))
            zones[best_zi].add(node)
            unassigned.discard(node)

        # 阶段 2：连通性修复 — 碎片迁移到相邻分区
        for _round in range(3):
            repaired = False
            for zi in range(len(zones)):
                components = get_connected_components(graph, zones[zi])
                if len(components) <= 1:
                    continue

                # 保留最大分量，迁移碎片
                components.sort(key=len, reverse=True)
                zones[zi] = components[0]
                repaired = True

                for fragment in components[1:]:
                    # 尝试整块迁移到相邻分区
                    frag_neighbors = {}  # zj -> count of adjacent edges
                    for node in fragment:
                        for nb in graph.neighbors(node):
                            for zj in range(len(zones)):
                                if zj != zi and nb in zones[zj]:
                                    frag_neighbors[zj] = frag_neighbors.get(zj, 0) + 1
                    # 选邻接最多且不超载的分区
                    best_zj = -1
                    best_score = -1
                    for zj, score in frag_neighbors.items():
                        if len(zones[zj]) + len(fragment) <= max_panels and score > best_score:
                            best_score = score
                            best_zj = zj
                    if best_zj >= 0:
                        zones[best_zj] |= fragment
                    else:
                        # 逐节点分配
                        for node in fragment:
                            placed = False
                            for zj in range(len(zones)):
                                if zj == zi or len(zones[zj]) >= max_panels:
                                    continue
                                for nb in graph.neighbors(node):
                                    if nb in zones[zj]:
                                        zones[zj].add(node)
                                        placed = True
                                        break
                                if placed:
                                    break
                            if not placed:
                                zones[zi].add(node)

            if not repaired:
                break

        # 阶段 3：强化重平衡（增加迭代次数和更智能的节点选择）
        for _round in range(100):  # 大幅增加迭代次数
            moved = False
            # 按分区大小排序，优先处理超大分区
            zone_sizes = [(zi, len(zones[zi])) for zi in range(len(zones))]
            zone_sizes.sort(key=lambda x: x[1], reverse=True)
            
            for zi, size in zone_sizes:
                # 处理超大分区
                if size > max_panels:
                    # 计算需要移动的节点数
                    excess = size - max_panels
                    
                    # 计算边界节点（有邻居不在当前分区的节点）
                    boundary_nodes = [n for n in zones[zi] if any(nb not in zones[zi] for nb in graph.neighbors(n))]
                    if not boundary_nodes:
                        # 如果没有边界节点，选择内部节点
                        boundary_nodes = list(zones[zi])
                    
                    # 按节点度排序，优先移动度较小的节点（减少连通性影响）
                    boundary_nodes = sorted(boundary_nodes, key=lambda n: graph.degree(n))
                    
                    moved_count = 0
                    for node in boundary_nodes:
                        if moved_count >= excess:
                            break
                        
                        # 寻找最佳目标分区（优先选择接近最小面板数的分区）
                        best_target = None
                        min_size_diff = float('inf')
                        
                        for nb in graph.neighbors(node):
                            for zj in range(len(zones)):
                                if zj != zi and len(zones[zj]) < max_panels:
                                    size_diff = len(zones[zj]) - min_panels
                                    if size_diff < min_size_diff:
                                        min_size_diff = size_diff
                                        best_target = zj
                        
                        if best_target is not None:
                            # 检查移动后两个分区是否仍然连通
                            new_zone_zi = zones[zi] - {node}
                            new_zone_zj = zones[best_target] | {node}
                            
                            if check_connectivity(graph, new_zone_zi) and check_connectivity(graph, new_zone_zj):
                                zones[zi] = new_zone_zi
                                zones[best_target] = new_zone_zj
                                moved = True
                                moved_count += 1
                        else:
                            # 如果没有找到合适的目标分区，尝试任何可以接受该节点的分区
                            for zj in range(len(zones)):
                                if zj != zi and len(zones[zj]) < max_panels:
                                    new_zone_zi = zones[zi] - {node}
                                    new_zone_zj = zones[zj] | {node}
                                    
                                    if check_connectivity(graph, new_zone_zi) and check_connectivity(graph, new_zone_zj):
                                        zones[zi] = new_zone_zi
                                        zones[zj] = new_zone_zj
                                        moved = True
                                        moved_count += 1
                                        break
            
            # 处理过小分区
            zone_sizes.sort(key=lambda x: x[1])
            for zi, size in zone_sizes:
                if size >= min_panels:
                    continue
                
                # 计算需要添加的节点数
                deficit = min_panels - size
                
                # 寻找可以从其他分区移动的节点
                for zj in range(len(zones)):
                    if zj == zi or len(zones[zj]) <= min_panels:
                        continue
                    
                    # 计算边界节点
                    boundary_nodes = [n for n in zones[zj] if any(nb in zones[zi] for nb in graph.neighbors(n))]
                    if not boundary_nodes:
                        # 如果没有与目标分区相邻的节点，找与目标分区最近的节点
                        boundary_nodes = list(zones[zj])
                        # 计算与目标分区的距离
                        def distance_to_zone(node):
                            min_dist = float('inf')
                            node_data = graph.nodes[node]
                            nr, nc = node_data["row"], node_data["col"]
                            for zn in zones[zi]:
                                zd = graph.nodes[zn]
                                dist = abs(nr - zd["row"]) + abs(nc - zd["col"])
                                if dist < min_dist:
                                    min_dist = dist
                            return min_dist
                        boundary_nodes = sorted(boundary_nodes, key=distance_to_zone)
                    
                    # 移动节点
                    for node in boundary_nodes:
                        if len(zones[zi]) >= min_panels:
                            break
                        
                        # 检查移动后两个分区是否仍然连通
                        new_zone_zj = zones[zj] - {node}
                        new_zone_zi = zones[zi] | {node}
                        
                        if check_connectivity(graph, new_zone_zj) and check_connectivity(graph, new_zone_zi):
                            zones[zj] = new_zone_zj
                            zones[zi] = new_zone_zi
                            moved = True
                        else:
                            # 即使连通性受损，也要移动节点以满足大小约束
                            zones[zj].remove(node)
                            zones[zi].add(node)
                            moved = True
            
            if not moved:
                break

        # 阶段 4：周长优化（交换边界节点减少总周长）
        coord_index = build_coord_index(graph)
        total_perimeter = sum(calculate_perimeter_fast(zone, graph, coord_index) for zone in zones)
        
        for _round in range(10):
            improved = False
            for zi in range(len(zones)):
                # 计算边界节点
                boundary_nodes = [n for n in zones[zi] if any(nb not in zones[zi] for nb in graph.neighbors(n))]
                if not boundary_nodes:
                    continue
                
                for node in boundary_nodes:
                    # 寻找相邻分区
                    adjacent_zones = set()
                    for nb in graph.neighbors(node):
                        for zj in range(len(zones)):
                            if zj != zi and nb in zones[zj]:
                                adjacent_zones.add(zj)
                    
                    for zj in adjacent_zones:
                        # 检查移动后两个分区的大小约束
                        if len(zones[zi]) <= min_panels or len(zones[zj]) >= max_panels:
                            continue
                        
                        # 尝试移动节点
                        new_zone_zi = zones[zi] - {node}
                        new_zone_zj = zones[zj] | {node}
                        
                        if check_connectivity(graph, new_zone_zi) and check_connectivity(graph, new_zone_zj):
                            new_perimeter = sum(
                                calculate_perimeter_fast(zone, graph, coord_index)
                                for zone in [new_zone_zi, new_zone_zj] + [zones[zk] for zk in range(len(zones)) if zk not in [zi, zj]]
                            )
                            
                            if new_perimeter < total_perimeter:
                                zones[zi] = new_zone_zi
                                zones[zj] = new_zone_zj
                                total_perimeter = new_perimeter
                                improved = True
            if not improved:
                break

        return zones

    def _train_step(self) -> float:
        """
        执行一步训练。
        
        Returns:
            损失值
        """
        # 采样经验
        if hasattr(self.replay_buffer, 'sample'):
            batch = self.replay_buffer.sample(self.batch_size)
        else:
            return 0.0
        
        try:
            gid, state, action, reward, next_state, done = batch
            
            # 检查奖励值是否存在异常
            if torch.isnan(reward).any() or torch.isinf(reward).any():
                print(f"警告: 奖励值存在异常: {reward}")
                reward = torch.clamp(reward, min=-100.0, max=100.0)
            
            # 计算目标Q值
            with torch.no_grad():
                # 计算下一状态的最大Q值
                target_q = reward.clone()
                non_final_mask = ~done.squeeze().bool()
                if non_final_mask.any():
                    non_final_gid = gid[non_final_mask]
                    non_final_next_state = [next_state[i] for i, mask in enumerate(non_final_mask) if mask]
                    
                    for i, (gid_idx, ns) in enumerate(zip(non_final_gid, non_final_next_state)):
                        gd = self.gset[gid_idx.item()]
                        feat = torch.ones(gd.n_nodes, 1, device=self.device)
                        embed = encode_graph(self.s2v, feat, gd.adj, gd.edge_weight, T=self.T, degree=gd.degree)
                        state_embed = get_graph_embedding(embed, ns)
                        q_values = self.qfunc_target(state_embed, embed)
                        
                        # 检查Q值是否存在爆炸
                        if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                            print(f"警告: 目标Q值存在异常: {q_values}")
                            q_values = torch.clamp(q_values, min=-100.0, max=100.0)
                        
                        # 获取合法动作
                        valid_actions = [j for j in range(gd.n_nodes) if ns[j] == 0]
                        if valid_actions:
                            valid_q = q_values[valid_actions]
                            max_q = valid_q.max().item()
                            target_q[non_final_mask][i] += self.gamma * max_q
            
            # 计算当前Q值
            current_q = []
            for i, (gid_idx, s, a) in enumerate(zip(gid, state, action)):
                gd = self.gset[gid_idx.item()]
                feat = torch.ones(gd.n_nodes, 1, device=self.device)
                embed = encode_graph(self.s2v, feat, gd.adj, gd.edge_weight, T=self.T, degree=gd.degree)
                state_embed = get_graph_embedding(embed, s)
                q_values = self.qfunc_policy(state_embed, embed)
                
                # 检查Q值是否存在爆炸，并进行裁剪
                if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                    print(f"警告: 策略Q值存在异常: {q_values}")
                    q_values = torch.clamp(q_values, min=-100.0, max=100.0)
                else:
                    # 常规裁剪，防止Q值爆炸
                    q_values = torch.clamp(q_values, min=-50.0, max=50.0)
                
                current_q.append(q_values[a.item()])
            current_q = torch.stack(current_q)
            
            # 检查Q值是否存在异常
            if torch.isnan(current_q).any() or torch.isinf(current_q).any():
                print(f"警告: 当前Q值存在异常: {current_q}")
                current_q = torch.clamp(current_q, min=-100.0, max=100.0)
            if torch.isnan(target_q).any() or torch.isinf(target_q).any():
                print(f"警告: 目标Q值存在异常: {target_q}")
                target_q = torch.clamp(target_q, min=-100.0, max=100.0)
            
            # 计算损失
            loss = nn.functional.mse_loss(current_q, target_q)
            
            # 检查损失是否为nan
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 损失值为异常: {loss}")
                return 0.0
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(list(self.s2v.parameters()) + list(self.qfunc_policy.parameters()), max_norm=0.5)
            self.optimizer.step()
            
            # 更新学习率（在optimizer.step()之后）
            self.lr_scheduler.step()
            
            return loss.item()
        except Exception as e:
            print(f"训练步骤错误: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _train_epoch_sequential(self, instances: List[Dict], epoch: int,
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

            # 缓存 GraphData，避免每 epoch 重复创建（节省内存 + 保持 gid 一致）
            if inst_id in self._graph_cache:
                gd, gid = self._graph_cache[inst_id]
            else:
                gd = GraphData(graph, str(self.device))
                gid = self.gset.push(gd)
                self._graph_cache[inst_id] = (gd, gid)

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

                # Last zone is forced to take remaining nodes — its negative
                # rewards would poison the replay buffer and teach Q-network
                # that good early decisions lead to bad outcomes.
                # Skip experience collection for the last zone.
                is_last_zone = (k == n_zones - 1)

                env = PartitionEnv(
                    gd, target_size=current_target,
                    min_size=min(18, remaining_count),
                    max_size=current_max,
                    excluded=excluded_indices,
                    perimeter_lb=pva_params["LB"],
                    perimeter_ub=pva_params["UB"]
                )
                state = env.reset()

                episode_reward = 0.0
                while not env.done:
                    valid = env.get_valid_actions()
                    if not valid:
                        # 防坍塌：未达 min_size 时跳跃到最近未分配节点
                        if env.step_count < env.min_size:
                            jumps = env._find_jump_candidates()
                            if jumps:
                                action = jumps[0]
                                next_state, reward, done = env.step(action)
                                reward -= 2.0  # 跳跃惩罚
                                if not is_last_zone:
                                    self.replay_buffer.push(gid, state, action, reward, next_state, done)
                                self._global_step += 1
                                if self._global_step % self.train_every == 0:
                                    loss = self._train_step()
                                    if loss is not None:
                                        instance_losses.append(loss)
                                state = next_state
                                episode_reward += reward
                                continue
                        break

                    action = self.select_action(gd, state, epsilon, valid)
                    next_state, reward, done = env.step(action)

                    if not is_last_zone:
                        self.replay_buffer.push(gid, state, action, reward, next_state, done)
                    self._global_step += 1

                    # 每 train_every 步训练一次（对齐原作，避免每步训练的巨大开销）
                    if self._global_step % self.train_every == 0:
                        loss = self._train_step()
                        if loss is not None:
                            instance_losses.append(loss)

                    state = next_state
                    episode_reward += reward

                instance_reward += episode_reward
                zone_nodes = env.get_zone_nodes()
                zones.append(zone_nodes)
                excluded_indices |= env.current_zone_indices

            epoch_rewards.append(instance_reward)
            if instance_losses:
                epoch_losses.append(np.mean(instance_losses))

            # 后处理：分配遗漏节点 + 连通性修复 + 重平衡
            zones = self._post_process_zones(zones, graph, min_panels=18, max_panels=26)

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

    # ─── 并行训练（多线程收集经验） ───

    def _collect_single_instance(self, gd: 'GraphData', gid: int,
                                   n_zones: int, n_nodes: int,
                                   pva_params: Dict, epsilon: float,
                                   graph: nx.Graph, inst_id: str) -> Dict:
        """
        收集单个算例的经验数据（线程安全：只做推理，不更新梯度）。

        Returns:
            包含 experiences, reward, constraint_stats 等的字典
        """
        excluded_indices = set()
        zones = []
        instance_reward = 0.0
        experiences = []  # (gid, state, action, reward, next_state, done)

        for k in range(n_zones):
            remaining_count = n_nodes - len(excluded_indices)
            zones_left = n_zones - k
            current_target = remaining_count // zones_left
            current_max = min(26, remaining_count - (zones_left - 1) * 18) if zones_left > 1 else remaining_count

            is_last_zone = (k == n_zones - 1)

            env = PartitionEnv(
                gd, target_size=current_target,
                min_size=min(18, remaining_count),
                max_size=current_max,
                excluded=excluded_indices,
                perimeter_lb=pva_params["LB"],
                perimeter_ub=pva_params["UB"]
            )
            state = env.reset()

            episode_reward = 0.0
            while not env.done:
                valid = env.get_valid_actions()
                if not valid:
                    if env.step_count < env.min_size:
                        jumps = env._find_jump_candidates()
                        if jumps:
                            action = jumps[0]
                            next_state, reward, done = env.step(action)
                            reward -= 2.0
                            if not is_last_zone:
                                experiences.append((gid, state, action, reward, next_state, done))
                            state = next_state
                            episode_reward += reward
                            continue
                    break

                action = self.select_action(gd, state, epsilon, valid)
                next_state, reward, done = env.step(action)
                if not is_last_zone:
                    experiences.append((gid, state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward

            instance_reward += episode_reward
            zones.append(env.get_zone_nodes())
            excluded_indices |= env.current_zone_indices

        # 后处理
        zones = self._post_process_zones(zones, graph, min_panels=18, max_panels=26)

        # 验证约束
        validator = PartitionValidator(
            graph, n_zones, min_panels=18, max_panels=26,
            perimeter_lb=pva_params["LB"], perimeter_ub=pva_params["UB"]
        )
        result = validator.validate(zones)

        cs = {"capacity": 0, "connected": 0, "perimeter": 0, "total_zones": len(result.zone_details)}
        for detail in result.zone_details:
            if detail["capacity_ok"]: cs["capacity"] += 1
            if detail["is_connected"]: cs["connected"] += 1
            if detail["perimeter_ok"]: cs["perimeter"] += 1

        return {
            "inst_id": inst_id,
            "experiences": experiences,
            "reward": instance_reward,
            "constraint_stats": cs,
            "sizes": [len(z) for z in zones],
            "is_feasible": result.is_feasible,
        }

    def train_epoch(self, instances: List[Dict], epoch: int,
                     epsilon: float, verbose_instances: bool = False,
                     n_workers: int = 8) -> Dict:
        """
        训练一个 epoch（串行版，ThreadPoolExecutor 因 GIL 无加速效果已废弃）。

        Delegates to _train_epoch_sequential; n_workers kept for API compat.
        """
        return self._train_epoch_sequential(
            instances, epoch, epsilon, verbose_instances=verbose_instances
        )

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
                                excluded=excluded_indices,
                                perimeter_lb=pva_params.get("LB", 60.0),
                                perimeter_ub=pva_params.get("UB", 90.0))
            state = env.reset()

            while not env.done:
                valid = env.get_valid_actions()
                if not valid:
                    if env.step_count < env.min_size:
                        jumps = env._find_jump_candidates()
                        if jumps:
                            action = jumps[0]
                            state, _, _ = env.step(action)
                            continue
                    break
                # 推理时使用小的探索率，避免陷入局部最优
                action = self.select_action(gd, state, epsilon=0.1, valid=valid)
                state, _, _ = env.step(action)

            zone_nodes = env.get_zone_nodes()
            zones.append(zone_nodes)
            excluded_indices |= env.current_zone_indices

        # Post-process: assign missing nodes + connectivity repair + rebalance
        zones = self._post_process_zones(zones, graph, min_panels=18, max_panels=26)

        # Use heuristic's robust rebalance + local search to fix constraint violations
        from modules.module1.algorithm.partition_heuristic import GreedyPartitioner
        fixer = GreedyPartitioner(
            graph, n_zones, min_panels=18, max_panels=26,
            perimeter_lb=pva_params.get("LB", 60.0),
            perimeter_ub=pva_params.get("UB", 90.0),
            local_search_iters=500, random_seed=0  # 增加局部搜索迭代次数
        )
        zones = fixer._repair_connectivity(zones)
        zones = fixer._rebalance(zones)
        zones = fixer._local_search(zones)
        zones = fixer._fix_perimeter_violations(zones)

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
        from modules.module1.algorithm.partition_heuristic import GreedyPartitioner
        from concurrent.futures import ThreadPoolExecutor, as_completed

        expert_trajectories = []
        total_feasible = 0
        total_runs = 0

        # 并行运行启发式算法的辅助函数
        def run_heuristic(inst, seed):
            inst_id = inst["instance_info"]["instance_id"]
            graph = build_adjacency_graph(inst["pva_list"],
                                           inst["terrain_data"]["grid_size"])
            n_zones = inst["equipment_params"]["inverter"]["p"]
            pva_params = inst["pva_params"]

            partitioner = GreedyPartitioner(
                graph, n_zones, random_seed=seed,
                min_panels=18, max_panels=26,
                perimeter_lb=pva_params["LB"],
                perimeter_ub=pva_params["UB"],
                local_search_iters=100  # 减少局部搜索迭代次数以加快速度
            )
            result = partitioner.solve()
            return inst, result

        # 缓存 GraphData
        graph_cache = {}
        for inst in instances:
            inst_id = inst["instance_info"]["instance_id"]
            graph = build_adjacency_graph(inst["pva_list"],
                                           inst["terrain_data"]["grid_size"])
            gd = GraphData(graph, str(self.device))
            gid = self.gset.push(gd)
            graph_cache[inst_id] = (gd, gid, graph)

        # 并行运行所有启发式实例
        all_tasks = []
        for inst in instances:
            for seed in range(n_runs_per_instance):
                all_tasks.append((inst, seed))

        # 根据系统CPU核心数调整线程数
        import os
        max_workers = min(16, os.cpu_count() or 4)

        print(f"  并行生成专家经验: {len(all_tasks)} 任务, {max_workers} 线程", flush=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(run_heuristic, inst, seed): (inst, seed) 
                             for inst, seed in all_tasks}

            for future in as_completed(future_to_task):
                inst, seed = future_to_task[future]
                try:
                    inst, result = future.result()
                    total_runs += 1

                    if only_feasible and not result.is_feasible:
                        continue

                    total_feasible += 1

                    # 处理结果
                    inst_id = inst["instance_info"]["instance_id"]
                    gd, gid, _ = graph_cache[inst_id]

                    # 将分区方案拆解为动作序列
                    n_zones = len(result.zones)  # 从zones列表长度获取分区数量
                    for zone_nodes in result.zones:
                        transitions = self._zone_to_trajectory(gd, zone_nodes, n_zones)
                        if transitions:
                            expert_trajectories.append({
                                "gid": gid,
                                "transitions": transitions,
                            })
                except Exception as e:
                    print(f"  生成专家经验时出错: {e}", flush=True)

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
                               n_epochs: int = 200,  # 增加预训练轮数
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
            list(self.s2v.parameters()) + list(self.qfunc_policy.parameters()),
            lr=lr
        )

        epoch_losses = []

        for epoch in range(1, n_epochs + 1):
            random.shuffle(all_samples)
            total_loss = 0.0
            n_batches = 0

            # mini-batch 训练（用较大 batch 减少更新次数）
            batch_size = 128
            for i in range(0, len(all_samples), batch_size):
                batch = all_samples[i:i + batch_size]
                batch_loss = torch.tensor(0.0, device=self.device)

                for gid, state, action in batch:
                    gd = self.gset[gid]
                    embed = self._encode(gd, state.to(self.device), "policy")
                    state_embed = get_graph_embedding(embed)

                    # 计算所有节点的 Q 值
                    q_values = self.qfunc_policy(state_embed, embed).squeeze(-1)  # [N]

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
                        list(self.s2v.parameters()) + list(self.qfunc_policy.parameters()),
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
        self.soft_update_target(1.0)

        # 更新 RL 优化器（用预训练后的参数）
        self.optimizer = optim.Adam(
            list(self.s2v.parameters()) + list(self.qfunc_policy.parameters()),
        lr=lr
    )

        return epoch_losses

    # ─── Checkpoint ───

    def save_checkpoint(self, path: str) -> None:
        """
        保存训练检查点。
        """
        torch.save({
            "s2v": self.s2v.state_dict(),
            "qfunc_policy": self.qfunc_policy.state_dict(),
            "qfunc_target": self.qfunc_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "best_reward": self.best_reward,
            "best_epoch": getattr(self, "best_epoch", self.current_epoch),
            "training_history": getattr(self, "training_history", []),
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.random.get_rng_state(),
            "save_time": datetime.now().isoformat(),
        }, path)

    def load_checkpoint(self, path: str) -> Dict:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            
            # 加载模型参数，兼容旧版本
            try:
                if "s2v" in ckpt:
                    self.s2v.load_state_dict(ckpt["s2v"])
                elif "s2v_policy" in ckpt:
                    self.s2v.load_state_dict(ckpt["s2v_policy"])
                
                if "qfunc_policy" in ckpt:
                    self.qfunc_policy.load_state_dict(ckpt["qfunc_policy"])
                elif "q_policy" in ckpt:
                    self.qfunc_policy.load_state_dict(ckpt["q_policy"])
                
                if "qfunc_target" in ckpt:
                    self.qfunc_target.load_state_dict(ckpt["qfunc_target"])
                elif "q_target" in ckpt:
                    self.qfunc_target.load_state_dict(ckpt["q_target"])
            except RuntimeError as e:
                # 处理参数形状不匹配的错误
                print(f"警告: 模型参数形状不匹配，跳过加载模型参数: {e}")
            
            # 尝试加载优化器状态，如果失败则跳过（评估时不需要优化器）
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                # 优化器状态不匹配时跳过，重新初始化优化器
                import torch.optim as optim
                self.optimizer = optim.Adam(
                    list(self.s2v.parameters()) + list(self.qfunc_policy.parameters()),
                    lr=1e-4
                )
            
            self.current_epoch = ckpt.get("current_epoch", 0)
            self.best_reward = ckpt.get("best_reward", -float('inf'))
            
            return {
                "epoch": self.current_epoch, 
                "best_reward": self.best_reward,
                "save_time": ckpt.get("save_time", "?")
            }
        except Exception as e:
            print(f"加载DQN模型失败: {e}")
            return {"epoch": 0, "best_reward": -float('inf'), "save_time": "?"}

    def load_model(self, path: str):
        """加载模型（兼容接口）"""
        return self.load_checkpoint(path)

    def _calculate_reward(self, zone_size: int, perimeter: float, done: bool) -> float:
        """
        多目标奖励：综合考虑周长最小化、面板数平衡、连通性等多个目标
        约束惩罚：对违反约束的行为给予更大惩罚
        进度奖励：对分区进度给予奖励，鼓励算法持续推进
        多样性奖励：鼓励探索不同的分区策略
        """
        reward = 0.0
        
        # 1. 面板数约束奖励/惩罚
        if zone_size < self.min_size:
            # 未达到最小面板数的惩罚
            deficit = self.min_size - zone_size
            reward -= 10.0 * deficit
        elif zone_size > self.max_size:
            # 超过最大面板数的惩罚
            excess = zone_size - self.max_size
            reward -= 15.0 * excess
        else:
            # 在合理范围内的奖励
            reward += 5.0
            # 接近目标大小的额外奖励
            target_size = (self.min_size + self.max_size) / 2
            size_diff = abs(zone_size - target_size)
            reward += max(0.0, 3.0 - 0.2 * size_diff)
        
        # 2. 周长约束奖励/惩罚 - 增加惩罚强度
        if self.perimeter_lb <= perimeter <= self.perimeter_ub:
            reward += 15.0  # 增加奖励
            # 接近目标周长的额外奖励
            target_perim = (self.perimeter_lb + self.perimeter_ub) / 2
            perim_dev = abs(perimeter - target_perim)
            reward += max(0.0, 10.0 - 0.2 * perim_dev)  # 增加奖励
        else:
            # 超出周长约束的惩罚
            if perimeter < self.perimeter_lb:
                deviation = self.perimeter_lb - perimeter
                reward -= 10.0 * deviation  # 增加惩罚
            else:
                deviation = perimeter - self.perimeter_ub
                reward -= 15.0 * deviation  # 增加惩罚
        
        # 3. 进度奖励：鼓励算法持续推进
        progress = zone_size / self.min_size
        if progress >= 1.0:
            reward += 10.0
        else:
            reward += 2.0 * progress
        
        # 4. 完成奖励 - 调整奖励强度
        if done:
            if zone_size >= self.min_size:
                reward += 10.0  # 适当奖励
            else:
                reward -= 20.0  # 适当惩罚
        
        return reward

    def train(self, instances: List[Dict], epochs: int = 300, start_epsilon: float = 1.0, end_epsilon: float = 0.05,
            epsilon_decay: float = 0.995, save_path: str = None, verbose: bool = True):  # 增加强化学习训练轮数
        if not instances:
            if verbose:
                print("  没有可用的算例进行训练。")
            return
        
        # 构建图数据
        graph_datas = []
        for inst in instances:
            try:
                graph = build_adjacency_graph(inst["pva_list"], inst["terrain_data"]["grid_size"])
                gd = GraphData(graph, device=self.device)
                graph_datas.append(gd)
            except Exception as e:
                if verbose:
                    print(f"  构建图数据失败: {e}")
        
        if not graph_datas:
            if verbose:
                print("  没有可用的图数据进行训练。")
            return
        
        # 训练循环
        epsilon = start_epsilon
        for epoch in range(epochs):
            total_reward = 0
            total_steps = 0
            total_loss = 0
            
            for gd in graph_datas:
                # 重置环境
                env = PartitionEnv(gd)
                state = env.reset()
                episode_reward = 0
                episode_steps = 0
                
                while True:
                    # 获取合法动作
                    valid_actions = env.get_valid_actions()
                    if not valid_actions:
                        break
                    
                    # 线性探索率衰减
                    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            max(0, 1 - self.step_count / self.epsilon_decay_steps)
                    
                    # 选择动作
                    action = self.select_action(gd, state, epsilon, valid_actions)
                    if action == -1:
                        break
                    
                    # 执行动作
                    next_state, reward, done = env.step(action)
                    
                    # 存储经验
                    gid = self.gset.push(gd)
                    self.replay_buffer.push(gid, state, action, reward, next_state, done)
                    
                    # 训练
                    if len(self.replay_buffer) >= self.replay_start_size and episode_steps % self.train_every == 0:
                        loss = self._train_step()
                        total_loss += loss
                    
                    # 更新目标网络
                    if self.step_count % self.target_update_freq == 0:
                        self.soft_update_target(1.0)  # 硬更新
                    
                    # 更新状态
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                    self.step_count += 1
                    
                    if done:
                        break
                
                total_reward += episode_reward
                total_steps += episode_steps
            
            # 软更新目标网络（额外的软更新）
            self.soft_update_target(self.tau)
            
            # 更新学习率
            self.lr_scheduler.step()
            
            # 计算平均奖励和损失
            avg_reward = total_reward / len(graph_datas)
            avg_loss = total_loss / total_steps if total_steps > 0 else 0
            
            # 更新统计信息
            self.current_epoch = epoch
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                if save_path and verbose:
                    self.save_checkpoint(save_path)
                    print(f"  保存新的最佳模型: 奖励={avg_reward:.4f}")
            
            # 打印训练信息
            if verbose and (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  轮次 {epoch + 1}/{epochs}: 平均奖励={avg_reward:.4f}, 平均损失={avg_loss:.4f}, 探索率={epsilon:.4f}, 学习率={current_lr:.6f}")
        
        # 保存最终模型
        if save_path and verbose:
            self.save_checkpoint(save_path)
            print(f"  保存最终模型: 最佳奖励={self.best_reward:.4f}")

    def _augment_instances(self, instances: List[Dict]) -> List[Dict]:
        """
        数据增强：通过随机变换生成更多训练数据，提高泛化能力
        """
        augmented = []
        
        for inst in instances:
            # 原始算例
            augmented.append(inst)
            
            # 随机调整逆变器数（±1）
            p = inst["equipment_params"]["inverter"]["p"]
            if p > 2:
                augmented.append({
                    **inst,
                    "equipment_params": {
                        **inst["equipment_params"],
                        "inverter": {
                            **inst["equipment_params"]["inverter"],
                            "p": p - 1
                        }
                    }
                })
            
            if p < 10:  # 限制最大逆变器数
                augmented.append({
                    **inst,
                    "equipment_params": {
                        **inst["equipment_params"],
                        "inverter": {
                            **inst["equipment_params"]["inverter"],
                            "p": p + 1
                        }
                    }
                })
        
        return augmented

    def _evaluate(self, instances: List[Dict]) -> float:
        """
        定期评估：在验证集上评估模型性能
        """
        total_reward = 0.0
        
        for inst in instances:
            graph = build_adjacency_graph(inst["pva_list"], inst["terrain_data"]["grid_size"])
            n_zones = inst["equipment_params"]["inverter"]["p"]
            pva_params = inst["pva_params"]
            
            result = self.solve(graph, n_zones, pva_params)
            # 计算评估奖励
            if result.is_feasible:
                total_reward += -result.total_perimeter  # 周长越小越好
            else:
                total_reward -= 1000.0  # 不可行解的惩罚
        
        return total_reward / len(instances)