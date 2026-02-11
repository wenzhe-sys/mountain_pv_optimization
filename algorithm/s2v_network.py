"""
Structure2Vec (S2V) 图嵌入网络 + Q 函数

严格对齐 Khalil et al. 2017 "Learning Combinatorial Optimization Algorithms over Graphs"。

S2V 消息传递公式（论文 Eq.1）：
  mu_v^(t+1) = ReLU( theta1 * x_v + theta2 * SUM_{u in N(v)} mu_u^(t)
                    + theta3 * SUM_{u in N(v)} ReLU(theta4 * w(v,u)) )

Q 函数公式（论文 Eq.3）：
  Q(h(S), v) = theta5^T * ReLU( theta6 * mu_graph(S) + theta7 * mu_v )
  其中 mu_graph(S) = SUM_{v in V} mu_v

参考实现：jiayuanzhang0/S2V-DQN (GitHub)
  - S2V 类对应 mod_agent.py 的 S2V 类
  - QFunction 类对应 mod_agent.py 的 QFunction 类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Structure2Vec(nn.Module):
    """
    S2V 图嵌入网络（密集矩阵版，不依赖 torch_geometric）。

    参数：
      theta1: [dim_in, dim_embed]  -- 节点特征变换
      theta2: [dim_embed, dim_embed] -- 邻居嵌入聚合
      theta3: [dim_embed, dim_embed] -- 边权重聚合（经 theta4 变换后）
      theta4: [1, dim_embed]        -- 边权重变换
    """

    def __init__(self, dim_in: int = 1, dim_embed: int = 64):
        super().__init__()
        self.dim_embed = dim_embed

        self.theta1 = nn.Parameter(torch.empty(dim_in, dim_embed))
        self.theta2 = nn.Parameter(torch.empty(dim_embed, dim_embed))
        self.theta3 = nn.Parameter(torch.empty(dim_embed, dim_embed))
        self.theta4 = nn.Parameter(torch.empty(1, dim_embed))

        # Xavier 初始化（与参考实现一致）
        nn.init.xavier_uniform_(self.theta1)
        nn.init.xavier_uniform_(self.theta2)
        nn.init.xavier_uniform_(self.theta3)
        nn.init.xavier_uniform_(self.theta4)

    def forward(self, feat: torch.Tensor, adj: torch.Tensor,
                edge_weight: torch.Tensor, embed: torch.Tensor,
                degree: torch.Tensor = None) -> torch.Tensor:
        """
        单轮消息传递。

        Args:
            feat: 节点特征 [N, dim_in]
            adj: 邻接矩阵 [N, N]（0/1，可以是稀疏或密集）
            edge_weight: 边权重矩阵 [N, N]
            embed: 当前节点嵌入 [N, dim_embed]
            degree: 预计算的度数向量 [N]（可选，用于 uniform edge weight 优化）

        Returns:
            更新后的节点嵌入 [N, dim_embed]
        """
        # theta1 * x_v: 节点特征变换 [N, dim_embed]
        term1 = torch.matmul(feat, self.theta1)

        # theta2 * SUM(mu_u): 邻居嵌入聚合 [N, dim_embed]
        neighbor_sum = torch.matmul(adj, embed)  # [N, dim_embed]
        term2 = torch.matmul(neighbor_sum, self.theta2)

        # theta3 * SUM(ReLU(theta4 * w_uv)):
        if degree is not None:
            # Optimized path: uniform edge weights (all 1.0)
            # w_sum[i,k] = degree(i) * ReLU(theta4[0,k])  =>  O(N*d) instead of O(N^2*d)
            relu_theta4 = F.relu(self.theta4)  # [1, dim_embed]
            w_sum = degree.unsqueeze(1) * relu_theta4  # [N, dim_embed]
        else:
            # General path: non-uniform edge weights
            w_transformed = F.relu(edge_weight.unsqueeze(-1) * self.theta4)  # [N, N, dim_embed]
            w_sum = (adj.unsqueeze(-1) * w_transformed).sum(dim=1)  # [N, dim_embed]
        term3 = torch.matmul(w_sum, self.theta3)

        return F.relu(term1 + term2 + term3)


class QFunction(nn.Module):
    """
    Q 值函数（论文 Eq.3）。

    Q(h(S), v) = theta5^T * ReLU(concat(theta6 * mu_graph, theta7 * mu_v))

    参数：
      theta5: [2 * dim_embed, 1]
      theta6: [dim_embed, dim_embed]
      theta7: [dim_embed, dim_embed]
    """

    def __init__(self, dim_embed: int = 64):
        super().__init__()
        self.theta5 = nn.Parameter(torch.empty(2 * dim_embed, 1))
        self.theta6 = nn.Parameter(torch.empty(dim_embed, dim_embed))
        self.theta7 = nn.Parameter(torch.empty(dim_embed, dim_embed))

        nn.init.xavier_uniform_(self.theta5)
        nn.init.xavier_uniform_(self.theta6)
        nn.init.xavier_uniform_(self.theta7)

    def forward(self, state_embed: torch.Tensor,
                node_embed: torch.Tensor) -> torch.Tensor:
        """
        计算 Q 值。

        Args:
            state_embed: 全局状态嵌入 [1, dim_embed] 或 [B, dim_embed]
            node_embed: 候选节点嵌入 [N, dim_embed] 或 [B, dim_embed]

        Returns:
            Q 值 [N, 1] 或 [B, 1]
        """
        # theta6 * mu_graph
        term6 = torch.matmul(state_embed, self.theta6)  # [1, dim_embed] or [B, dim_embed]

        # 广播 state_embed 到与 node_embed 相同的行数
        if term6.size(0) == 1 and node_embed.size(0) > 1:
            term6 = term6.expand(node_embed.size(0), -1)

        # theta7 * mu_v
        term7 = torch.matmul(node_embed, self.theta7)  # [N, dim_embed]

        # concat + ReLU + theta5
        concat = torch.cat([term6, term7], dim=1)  # [N, 2*dim_embed]
        q = torch.matmul(F.relu(concat), self.theta5)  # [N, 1]

        return q


def encode_graph(s2v: Structure2Vec, feat: torch.Tensor,
                  adj: torch.Tensor, edge_weight: torch.Tensor,
                  T: int = 4, degree: torch.Tensor = None) -> torch.Tensor:
    """
    多轮 S2V 编码。

    Args:
        s2v: S2V 网络
        feat: 节点特征 [N, dim_in]
        adj: 邻接矩阵 [N, N]
        edge_weight: 边权重矩阵 [N, N]
        T: 消息传递轮数
        degree: 预计算的度数向量 [N]（可选，用于 uniform edge weight 优化）

    Returns:
        节点嵌入 [N, dim_embed]
    """
    N = feat.size(0)
    embed = torch.zeros(N, s2v.dim_embed, device=feat.device)

    for _ in range(T):
        embed = s2v(feat, adj, edge_weight, embed, degree=degree)

    return embed


def get_graph_embedding(node_embed: torch.Tensor) -> torch.Tensor:
    """
    全局图嵌入 = 所有节点嵌入之和（论文做法）。

    Args:
        node_embed: [N, dim_embed]

    Returns:
        [1, dim_embed]
    """
    return node_embed.sum(dim=0, keepdim=True)
