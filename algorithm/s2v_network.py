"""
Structure2Vec (S2V) 图嵌入网络

基于 Khalil et al. 2017 "Learning Combinatorial Optimization Algorithms over Graphs"。

将光伏面板网格图编码为固定维度的向量表示，捕捉：
- 节点自身属性（坐标、功率、分配状态）
- 邻居聚合信息（通过 T 轮消息传递）
- 全局图结构特征

输出用于 DQN 的 Q 值估计。

参考实现：
- Hanjun-Dai/graph_comb_opt (515 star, 原作者)
- jiayuanzhang0/S2V-DQN (纯 PyTorch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Tuple, Optional


class Structure2Vec(nn.Module):
    """
    S2V 图嵌入网络。

    每轮消息传递：
      μ_v^(t+1) = ReLU( θ1 × x_v + θ2 × Σ_{u∈N(v)} μ_u^(t) + θ3 × Σ_{u∈N(v)} w(v,u) )

    其中 x_v 是节点特征，w(v,u) 是边权重，μ_v 是节点嵌入。
    """

    def __init__(self, node_feature_dim: int = 5, hidden_dim: int = 64,
                 n_iterations: int = 4):
        """
        Args:
            node_feature_dim: 节点特征维度
            hidden_dim: 隐藏层维度（嵌入维度）
            n_iterations: 消息传递轮数 T
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_iterations = n_iterations

        # θ1: 节点特征变换
        self.theta1 = nn.Linear(node_feature_dim, hidden_dim)
        # θ2: 邻居嵌入聚合
        self.theta2 = nn.Linear(hidden_dim, hidden_dim)
        # θ3: 边权重变换
        self.theta3 = nn.Linear(1, hidden_dim)

    def forward(self, node_features: torch.Tensor,
                adjacency: torch.Tensor,
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播。

        Args:
            node_features: 节点特征矩阵 [N, node_feature_dim]
            adjacency: 邻接矩阵 [N, N]（0/1）
            edge_weights: 边权重矩阵 [N, N]（可选，默认全 1）

        Returns:
            节点嵌入矩阵 [N, hidden_dim]
        """
        N = node_features.size(0)

        if edge_weights is None:
            edge_weights = adjacency.clone()

        # 初始化节点嵌入为零
        mu = torch.zeros(N, self.hidden_dim, device=node_features.device)

        for t in range(self.n_iterations):
            # θ1 × x_v: 节点特征变换 [N, hidden_dim]
            feat_term = self.theta1(node_features)

            # θ2 × Σ μ_u: 邻居嵌入聚合 [N, hidden_dim]
            neighbor_sum = torch.matmul(adjacency, mu)  # [N, hidden_dim]
            neighbor_term = self.theta2(neighbor_sum)

            # θ3 × Σ w(v,u): 边权重聚合 [N, hidden_dim]
            edge_sum = torch.matmul(edge_weights, torch.ones(N, 1, device=node_features.device))  # [N, 1]
            edge_term = self.theta3(edge_sum)

            # 更新嵌入
            mu = F.relu(feat_term + neighbor_term + edge_term)

        return mu


class QNetwork(nn.Module):
    """
    Q 值网络。

    Q(s, a) = MLP(concat(μ_a, μ_global, context))

    其中：
    - μ_a: 候选动作节点的 S2V 嵌入
    - μ_global: 全局图嵌入（所有节点嵌入的 sum pooling）
    - context: 当前分区状态的上下文特征
    """

    def __init__(self, hidden_dim: int = 64, context_dim: int = 4):
        """
        Args:
            hidden_dim: S2V 嵌入维度
            context_dim: 上下文特征维度（当前分区大小、已分配比例等）
        """
        super().__init__()
        # 输入 = 节点嵌入 + 全局嵌入 + 上下文
        input_dim = hidden_dim * 2 + context_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, node_embedding: torch.Tensor,
                global_embedding: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        计算 Q 值。

        Args:
            node_embedding: 候选节点的嵌入 [batch, hidden_dim] 或 [N, hidden_dim]
            global_embedding: 全局图嵌入 [hidden_dim] 或 [batch, hidden_dim]
            context: 上下文特征 [context_dim] 或 [batch, context_dim]

        Returns:
            Q 值 [batch, 1] 或 [N, 1]
        """
        # 广播 global_embedding 和 context
        if global_embedding.dim() == 1:
            global_embedding = global_embedding.unsqueeze(0).expand(node_embedding.size(0), -1)
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(node_embedding.size(0), -1)

        x = torch.cat([node_embedding, global_embedding, context], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class S2VDQNModel(nn.Module):
    """
    完整的 S2V-DQN 模型。

    组合 S2V 图嵌入 + Q 值网络。
    """

    def __init__(self, node_feature_dim: int = 5, hidden_dim: int = 64,
                 n_iterations: int = 4, context_dim: int = 4):
        super().__init__()
        self.s2v = Structure2Vec(node_feature_dim, hidden_dim, n_iterations)
        self.q_net = QNetwork(hidden_dim, context_dim)
        self.hidden_dim = hidden_dim

    def forward(self, node_features: torch.Tensor,
                adjacency: torch.Tensor,
                context: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算所有候选动作的 Q 值。

        Args:
            node_features: [N, feat_dim]
            adjacency: [N, N]
            context: [context_dim]
            action_mask: [N] 布尔掩码，True=可选动作

        Returns:
            Q 值 [N]（被掩码的位置为 -inf）
        """
        # S2V 嵌入
        node_embeddings = self.s2v(node_features, adjacency)  # [N, hidden]

        # 全局嵌入（sum pooling）
        global_embedding = node_embeddings.sum(dim=0)  # [hidden]

        # 计算每个节点的 Q 值
        q_values = self.q_net(node_embeddings, global_embedding, context)  # [N, 1]
        q_values = q_values.squeeze(-1)  # [N]

        # 应用动作掩码
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, float("-inf"))

        return q_values

    def get_embeddings(self, node_features: torch.Tensor,
                        adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取节点嵌入和全局嵌入（用于外部调用）。"""
        node_embeddings = self.s2v(node_features, adjacency)
        global_embedding = node_embeddings.sum(dim=0)
        return node_embeddings, global_embedding


def build_node_features(graph: nx.Graph, zones: list,
                         current_zone_idx: int) -> torch.Tensor:
    """
    从图和当前分区状态构建节点特征矩阵。

    特征（dim=5）：
      0: 归一化 row 坐标
      1: 归一化 col 坐标
      2: 是否已分配 (0/1)
      3: 当前所属区域编号（归一化）
      4: 邻居中已分配节点的比例

    Args:
        graph: 面板邻接图
        zones: 当前已分配的分区列表 [Set[str], ...]
        current_zone_idx: 当前正在构建的分区索引

    Returns:
        节点特征矩阵 [N, 5]
    """
    nodes = list(graph.nodes())
    N = len(nodes)

    # 构建节点到索引的映射
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # 获取坐标范围用于归一化
    rows = [graph.nodes[n]["row"] for n in nodes]
    cols = [graph.nodes[n]["col"] for n in nodes]
    row_min, row_max = min(rows), max(rows)
    col_min, col_max = min(cols), max(cols)
    row_range = max(row_max - row_min, 1)
    col_range = max(col_max - col_min, 1)

    # 构建已分配集合
    assigned = set()
    node_zone = {}  # panel_id -> zone_idx
    for z_idx, zone in enumerate(zones):
        for panel_id in zone:
            assigned.add(panel_id)
            node_zone[panel_id] = z_idx

    n_zones = max(len(zones), 1)

    features = torch.zeros(N, 5)
    for i, node in enumerate(nodes):
        data = graph.nodes[node]
        # 归一化坐标
        features[i, 0] = (data["row"] - row_min) / row_range
        features[i, 1] = (data["col"] - col_min) / col_range
        # 是否已分配
        features[i, 2] = 1.0 if node in assigned else 0.0
        # 所属区域（归一化）
        features[i, 3] = node_zone.get(node, -1) / n_zones
        # 邻居已分配比例
        neighbors = list(graph.neighbors(node))
        if neighbors:
            assigned_neighbors = sum(1 for nb in neighbors if nb in assigned)
            features[i, 4] = assigned_neighbors / len(neighbors)

    return features


def build_adjacency_matrix(graph: nx.Graph) -> torch.Tensor:
    """从 NetworkX 图构建邻接矩阵（0/1）。"""
    nodes = list(graph.nodes())
    N = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    adj = torch.zeros(N, N)
    for u, v in graph.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return adj


def build_context_features(zones: list, current_zone: set,
                            n_total: int, max_panels: int = 26,
                            n_zones_target: int = 5) -> torch.Tensor:
    """
    构建上下文特征向量。

    特征（dim=4）：
      0: 当前分区大小 / max_panels
      1: 已完成的分区数 / 总目标分区数
      2: 已分配面板比例
      3: 剩余未分配面板数 / n_total

    Args:
        zones: 已完成的分区列表
        current_zone: 当前正在构建的分区
        n_total: 面板总数
        max_panels: 每分区最大面板数
        n_zones_target: 目标分区数

    Returns:
        上下文特征 [4]
    """
    n_assigned = sum(len(z) for z in zones) + len(current_zone)

    return torch.tensor([
        len(current_zone) / max_panels,
        len(zones) / max(n_zones_target, 1),
        n_assigned / max(n_total, 1),
        (n_total - n_assigned) / max(n_total, 1),
    ], dtype=torch.float32)
