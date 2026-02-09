"""
DQN 分区智能体

基于 S2V-DQN 框架的分区求解器，通过强化学习自动生成高质量分区方案。

核心组件：
- ReplayBuffer: 经验回放缓冲区
- PartitionEnv: 分区环境（状态/动作/奖励）
- DQNPartitionAgent: 完整的 DQN 智能体（训练+推理+checkpoint）

训练流程：
  for epoch in epochs:
      for instance in instances:
          env.reset(instance)
          while not done:
              action = agent.select_action(state)  # ε-greedy
              next_state, reward, done = env.step(action)
              agent.store(state, action, reward, next_state, done)
              agent.learn()  # 从 replay buffer 采样更新

参考：Khalil et al. 2017, 项目书"研究方法一"
"""

import os
import time
import random
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import deque, namedtuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from algorithm.s2v_network import (
    S2VDQNModel, build_node_features, build_adjacency_matrix,
    build_context_features
)
from model.partition_sub import PartitionValidator, PartitionResult
from utils.graph_utils import (
    build_adjacency_graph, build_coord_index, check_connectivity,
    calculate_perimeter_fast, get_adjacent_external_nodes
)

logger = logging.getLogger(__name__)

# 经验元组
Transition = namedtuple("Transition",
                         ["state", "action", "reward", "next_state", "done", "mask"])


class ReplayBuffer:
    """经验回放缓冲区。"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class PartitionEnv:
    """
    分区环境。

    状态：当前图的嵌入表示（通过 S2V 编码）
    动作：选择一个未分配的面板加入当前分区
    奖励：
      - 每步成功加入: +0.1
      - 违反约束: -1.0
      - 分区完成: -周长/UB（归一化，负值=周长越小越好）
      - 全部完成: +覆盖率 bonus
    """

    def __init__(self, graph: nx.Graph, n_zones: int,
                 min_panels: int = 18, max_panels: int = 26,
                 perimeter_lb: float = 60.0, perimeter_ub: float = 90.0):
        self.graph = graph
        self.n_zones = n_zones
        self.min_panels = min_panels
        self.max_panels = max_panels
        self.perimeter_lb = perimeter_lb
        self.perimeter_ub = perimeter_ub

        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.coord_index = build_coord_index(graph)

        # 预计算邻接矩阵
        self.adjacency = build_adjacency_matrix(graph)

        self.reset()

    def reset(self):
        """重置环境。"""
        self.zones = []           # 已完成的分区
        self.current_zone = set() # 当前正在构建的分区
        self.assigned = set()     # 已分配的面板
        self.done = False
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self) -> Dict:
        """获取当前状态。"""
        all_zones = self.zones + ([self.current_zone] if self.current_zone else [])
        node_features = build_node_features(
            self.graph, all_zones, len(self.zones)
        )
        context = build_context_features(
            self.zones, self.current_zone,
            self.n_nodes, self.max_panels, self.n_zones
        )
        action_mask = self._get_action_mask()

        return {
            "node_features": node_features,
            "adjacency": self.adjacency,
            "context": context,
            "action_mask": action_mask,
        }

    def _get_action_mask(self) -> torch.Tensor:
        """获取合法动作掩码。"""
        mask = torch.zeros(self.n_nodes, dtype=torch.bool)

        if self.done:
            return mask

        for i, node in enumerate(self.nodes):
            if node in self.assigned:
                continue

            # 如果当前分区为空，所有未分配节点都可选
            if not self.current_zone:
                mask[i] = True
                continue

            # 必须与当前分区相邻且加入后保持连通
            is_adjacent = any(
                nb in self.current_zone
                for nb in self.graph.neighbors(node)
            )
            if is_adjacent and len(self.current_zone) < self.max_panels:
                mask[i] = True

        return mask

    def step(self, action_idx: int) -> Tuple[Dict, float, bool]:
        """
        执行动作。

        Args:
            action_idx: 节点索引

        Returns:
            (next_state, reward, done)
        """
        node = self.nodes[action_idx]
        reward = 0.0

        if node in self.assigned:
            reward = -2.0  # 重复分配，严重惩罚
            self.done = True
            return self._get_state(), reward, self.done

        # 加入当前分区
        self.current_zone.add(node)
        self.assigned.add(node)

        # 检查连通性
        if not check_connectivity(self.graph, self.current_zone):
            reward = -1.0  # 连通性违规
        else:
            reward = 0.1   # 成功加入

        # 检查是否应该完成当前分区
        zone_complete = False
        if len(self.current_zone) >= self.max_panels:
            zone_complete = True
        elif len(self.current_zone) >= self.min_panels:
            # 如果没有可扩展的邻居了，强制完成
            expandable = get_adjacent_external_nodes(self.graph, self.current_zone)
            expandable = expandable - self.assigned
            if not expandable:
                zone_complete = True
            # 或者剩余面板不够再建一个分区
            remaining = self.n_nodes - len(self.assigned)
            zones_left = self.n_zones - len(self.zones) - 1
            if zones_left > 0 and remaining <= zones_left * self.max_panels:
                if len(self.current_zone) >= self.min_panels:
                    zone_complete = True

        if zone_complete:
            # 计算该分区的周长奖励
            perimeter = calculate_perimeter_fast(
                self.current_zone, self.graph, self.coord_index
            )
            # 归一化周长作为负奖励（周长越小奖励越高）
            peri_reward = -(perimeter / self.perimeter_ub)
            reward += peri_reward

            self.zones.append(self.current_zone.copy())
            self.current_zone = set()

        # 检查是否全部完成
        if len(self.zones) >= self.n_zones or len(self.assigned) >= self.n_nodes:
            # 将剩余面板分配到最后一个分区
            if self.current_zone:
                self.zones.append(self.current_zone.copy())
                self.current_zone = set()

            # 覆盖率 bonus
            coverage = len(self.assigned) / self.n_nodes
            reward += coverage * 2.0

            self.done = True

        self.total_reward += reward
        return self._get_state(), reward, self.done


class DQNPartitionAgent:
    """
    DQN 分区智能体。

    支持：
    - 训练（epsilon-greedy 探索）
    - 推理（贪心策略）
    - Checkpoint 保存/恢复/交接
    """

    def __init__(self, node_feature_dim: int = 5, hidden_dim: int = 64,
                 n_iterations: int = 4, context_dim: int = 4,
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: int = 5000,
                 buffer_size: int = 10000, batch_size: int = 32,
                 target_update: int = 100,
                 device: str = "cpu"):
        """
        Args:
            node_feature_dim: 节点特征维度
            hidden_dim: S2V 隐藏维度
            n_iterations: S2V 消息传递轮数
            context_dim: 上下文维度
            lr: 学习率
            gamma: 折扣因子
            epsilon_start/end/decay: ε-greedy 参数
            buffer_size: 经验回放缓冲区大小
            batch_size: 训练批次大小
            target_update: 目标网络更新频率
            device: 设备 ("cpu" / "cuda" / "mps")
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # epsilon 衰减
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # 网络
        self.policy_net = S2VDQNModel(
            node_feature_dim, hidden_dim, n_iterations, context_dim
        ).to(self.device)

        self.target_net = S2VDQNModel(
            node_feature_dim, hidden_dim, n_iterations, context_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 训练统计
        self.training_history = []
        self.best_reward = float("-inf")
        self.best_epoch = 0
        self.current_epoch = 0

    @property
    def epsilon(self) -> float:
        """当前 epsilon 值。"""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self.steps_done / self.epsilon_decay)

    def select_action(self, state: Dict, training: bool = True) -> int:
        """
        选择动作（ε-greedy 策略）。

        Args:
            state: 环境状态
            training: 是否为训练模式

        Returns:
            动作索引
        """
        mask = state["action_mask"]
        valid_actions = torch.where(mask)[0]

        if len(valid_actions) == 0:
            return 0  # 无合法动作，返回默认

        if training and random.random() < self.epsilon:
            # 随机探索
            return valid_actions[random.randint(0, len(valid_actions) - 1)].item()

        # 贪心选择
        with torch.no_grad():
            node_features = state["node_features"].to(self.device)
            adjacency = state["adjacency"].to(self.device)
            context = state["context"].to(self.device)

            q_values = self.policy_net(node_features, adjacency, context, mask.to(self.device))
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验。"""
        self.replay_buffer.push(
            state["node_features"], action, reward,
            next_state["node_features"], done,
            state["action_mask"]
        )

    def learn(self, state: Dict) -> Optional[float]:
        """
        从经验回放中学习。

        Returns:
            损失值（如果执行了学习）或 None
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)

        # 逐样本计算损失（因为图结构不同，无法 batch）
        total_loss = 0.0
        for trans in transitions:
            node_feat = trans.state.to(self.device)
            adjacency = state["adjacency"].to(self.device)
            context = state["context"].to(self.device)
            action = trans.action
            reward = trans.reward
            done = trans.done

            # Q(s, a)
            q_values = self.policy_net(node_feat, adjacency, context)
            q_value = q_values[action]

            # Target: r + γ max Q'(s', a')
            if done:
                target = torch.tensor(reward, device=self.device, dtype=torch.float32)
            else:
                next_feat = trans.next_state.to(self.device)
                with torch.no_grad():
                    next_q = self.target_net(next_feat, adjacency, context)
                    next_mask = trans.mask.to(self.device) if trans.mask is not None else None
                    if next_mask is not None:
                        next_q = next_q.masked_fill(~next_mask, float("-inf"))
                    target = reward + self.gamma * next_q.max()

            loss = nn.functional.smooth_l1_loss(q_value, target)
            total_loss += loss

        # 反向传播
        avg_loss = total_loss / len(transitions)
        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # 更新目标网络
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return avg_loss.item()

    def train_epoch(self, instances: List[Dict], epoch: int) -> Dict:
        """
        训练一个 epoch（遍历所有算例）。

        Args:
            instances: 算例列表
            epoch: 当前 epoch 编号

        Returns:
            训练统计信息
        """
        epoch_rewards = []
        epoch_losses = []
        epoch_constraints = {"capacity": 0, "connected": 0, "perimeter": 0, "total": 0}

        for inst in instances:
            # 构建图和环境
            graph = build_adjacency_graph(inst["pva_list"],
                                           inst["terrain_data"]["grid_size"])
            n_zones = inst["equipment_params"]["inverter"]["p"]
            pva_params = inst["pva_params"]

            env = PartitionEnv(
                graph, n_zones,
                min_panels=18, max_panels=26,
                perimeter_lb=pva_params["LB"],
                perimeter_ub=pva_params["UB"]
            )

            state = env.reset()
            episode_reward = 0.0
            episode_losses = []

            while not env.done:
                action = self.select_action(state, training=True)
                next_state, reward, done = env.step(action)
                self.store_transition(state, action, reward, next_state, done)

                loss = self.learn(state)
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                episode_reward += reward

            epoch_rewards.append(episode_reward)
            if episode_losses:
                epoch_losses.append(np.mean(episode_losses))

            # 验证约束
            validator = PartitionValidator(
                graph, n_zones, min_panels=18, max_panels=26,
                perimeter_lb=pva_params["LB"], perimeter_ub=pva_params["UB"]
            )
            result = validator.validate(env.zones)
            epoch_constraints["total"] += 1
            for detail in result.zone_details:
                if detail["capacity_ok"]:
                    epoch_constraints["capacity"] += 1
                if detail["is_connected"]:
                    epoch_constraints["connected"] += 1
                if detail["perimeter_ok"]:
                    epoch_constraints["perimeter"] += 1

        # 汇总统计
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        n_total_zones = epoch_constraints["total"] * n_zones if epoch_constraints["total"] > 0 else 1

        stats = {
            "epoch": epoch,
            "avg_reward": float(avg_reward),
            "avg_loss": float(avg_loss),
            "epsilon": float(self.epsilon),
            "capacity_rate": epoch_constraints["capacity"] / n_total_zones,
            "connected_rate": epoch_constraints["connected"] / n_total_zones,
            "perimeter_rate": epoch_constraints["perimeter"] / n_total_zones,
            "buffer_size": len(self.replay_buffer),
        }

        # 更新最优
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_epoch = epoch
            stats["is_best"] = True
        else:
            stats["is_best"] = False

        self.current_epoch = epoch
        self.training_history.append(stats)

        return stats

    def solve(self, graph: nx.Graph, n_zones: int,
              pva_params: Dict = None) -> PartitionResult:
        """
        使用训练好的 DQN 进行推理求解。

        Args:
            graph: 面板邻接图
            n_zones: 目标分区数

        Returns:
            PartitionResult 分区结果
        """
        if pva_params is None:
            pva_params = {"LB": 60.0, "UB": 90.0}

        env = PartitionEnv(
            graph, n_zones,
            min_panels=18, max_panels=26,
            perimeter_lb=pva_params.get("LB", 60.0),
            perimeter_ub=pva_params.get("UB", 90.0)
        )

        state = env.reset()
        while not env.done:
            action = self.select_action(state, training=False)
            state, _, _ = env.step(action)

        validator = PartitionValidator(
            graph, n_zones, min_panels=18, max_panels=26,
            perimeter_lb=pva_params.get("LB", 60.0),
            perimeter_ub=pva_params.get("UB", 90.0)
        )
        result = validator.validate(env.zones)
        result.solver_method = "dqn"
        return result

    # ─── Checkpoint 管理 ───

    def save_checkpoint(self, path: str, extra: Dict = None):
        """
        保存完整 checkpoint（支持交接恢复）。

        保存内容：模型权重、优化器、训练进度、随机种子、历史数据
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "current_epoch": self.current_epoch,
            "best_reward": self.best_reward,
            "best_epoch": self.best_epoch,
            "training_history": self.training_history,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "save_time": datetime.now().isoformat(),
        }

        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """
        加载 checkpoint（支持交接恢复）。

        Returns:
            checkpoint 元数据
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_reward = checkpoint["best_reward"]
        self.best_epoch = checkpoint["best_epoch"]
        self.training_history = checkpoint["training_history"]

        # 恢复随机状态
        if "random_state" in checkpoint:
            random.setstate(checkpoint["random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_random_state" in checkpoint:
            torch.random.set_rng_state(checkpoint["torch_random_state"])

        return {
            "epoch": self.current_epoch,
            "best_reward": self.best_reward,
            "best_epoch": self.best_epoch,
            "save_time": checkpoint.get("save_time", "未知"),
        }
