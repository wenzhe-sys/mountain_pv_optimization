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

    def __init__(self, dim_in: int = 1, dim_embed: int = 128):
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


class AttentionS2V(nn.Module):
    """
    带注意力机制的 S2V 图嵌入网络。

    在标准 S2V 的基础上添加注意力机制，允许模型学习节点之间的重要性权重。
    """

    def __init__(self, dim_in: int = 1, dim_embed: int = 128):
        super().__init__()
        self.dim_embed = dim_embed

        # 标准 S2V 参数
        self.theta1 = nn.Parameter(torch.empty(dim_in, dim_embed))
        self.theta2 = nn.Parameter(torch.empty(dim_embed, dim_embed))
        self.theta3 = nn.Parameter(torch.empty(dim_embed, dim_embed))
        self.theta4 = nn.Parameter(torch.empty(1, dim_embed))

        # 注意力机制参数 - 多头注意力
        self.num_heads = 4
        self.head_dim = dim_embed // self.num_heads
        
        self.attention = nn.Sequential(
            nn.Linear(dim_embed * 2, dim_embed),
            nn.ReLU(),
            nn.Linear(dim_embed, self.num_heads),
            nn.Softmax(dim=1)
        )

        # 多头注意力融合
        self.fusion = nn.Linear(dim_embed * self.num_heads, dim_embed)

        # Xavier 初始化
        nn.init.xavier_uniform_(self.theta1)
        nn.init.xavier_uniform_(self.theta2)
        nn.init.xavier_uniform_(self.theta3)
        nn.init.xavier_uniform_(self.theta4)

    def forward(self, feat: torch.Tensor, adj: torch.Tensor, 
                edge_weight: torch.Tensor, embed: torch.Tensor, 
                degree: torch.Tensor = None) -> torch.Tensor:
        """
        单轮消息传递，带有注意力机制。

        Args:
            feat: 节点特征 [N, dim_in]
            adj: 邻接矩阵 [N, N]（0/1，可以是稀疏或密集）
            edge_weight: 边权重矩阵 [N, N]
            embed: 当前节点嵌入 [N, dim_embed]
            degree: 预计算的度数向量 [N]（可选，用于 uniform edge weight 优化）

        Returns:
            更新后的节点嵌入 [N, dim_embed]
        """
        N = feat.size(0)

        # theta1 * x_v: 节点特征变换 [N, dim_embed]
        term1 = torch.matmul(feat, self.theta1)

        # 注意力机制：批量计算邻居节点的注意力权重
        # 构建邻居索引
        adj_indices = adj.nonzero(as_tuple=True)
        if len(adj_indices[0]) == 0:
            # 无邻居情况
            weighted_neighbor_sum = torch.zeros(N, self.dim_embed, device=feat.device)
        else:
            # 批量计算注意力权重
            src_nodes = adj_indices[0]
            tgt_nodes = adj_indices[1]
            
            # 提取源节点和目标节点的嵌入
            src_embeds = embed[src_nodes]
            tgt_embeds = embed[tgt_nodes]
            
            # 拼接嵌入
            combined = torch.cat([src_embeds, tgt_embeds], dim=1)
            
            # 计算注意力权重
            weights = self.attention(combined)  # [E, num_heads]
            
            # 多头注意力处理
            all_head_outputs = []
            for head in range(self.num_heads):
                # 为每个头计算加权和
                head_weights = weights[:, head].unsqueeze(1)
                weighted_embeds = tgt_embeds * head_weights
                
                # 聚合到源节点
                head_output = torch.zeros(N, self.dim_embed, device=feat.device)
                head_output.index_add_(0, src_nodes, weighted_embeds)
                all_head_outputs.append(head_output)
            
            # 融合多头结果
            concatenated = torch.cat(all_head_outputs, dim=1)
            weighted_neighbor_sum = self.fusion(concatenated)

        # 带注意力的邻居嵌入聚合 [N, dim_embed]
        term2 = torch.matmul(weighted_neighbor_sum, self.theta2)

        # theta3 * SUM(ReLU(theta4 * w_uv)):
        if degree is not None:
            # Optimized path: uniform edge weights (all 1.0)
            relu_theta4 = F.relu(self.theta4)  # [1, dim_embed]
            w_sum = degree.unsqueeze(1) * relu_theta4  # [N, dim_embed]
        else:
            # General path: non-uniform edge weights
            # 修复维度不匹配问题
            w_transformed = F.relu(edge_weight.unsqueeze(-1) * self.theta4.unsqueeze(0))  # [N, N, dim_embed]
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

    def __init__(self, dim_embed: int = 128, dropout: float = 0.1):
        super().__init__()
        self.theta5 = nn.Parameter(torch.empty(2 * dim_embed, 1))
        self.theta6 = nn.Parameter(torch.empty(dim_embed, dim_embed))
        self.theta7 = nn.Parameter(torch.empty(dim_embed, dim_embed))
        self.dropout = nn.Dropout(dropout)

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

        # concat + ReLU + theta5 + dropout
        concat = torch.cat([term6, term7], dim=1)  # [N, 2*dim_embed]
        concat = self.dropout(concat)
        q = torch.matmul(F.relu(concat), self.theta5)  # [N, 1]

        return q

class DuelingQFunction(nn.Module):
    """
    Dueling Q 值函数。

    将 Q 值分解为状态价值 V 和优势函数 A：
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))

    这样可以更有效地学习状态价值，提高算法性能。
    """

    def __init__(self, dim_embed: int = 128, dropout: float = 0.1):
        super().__init__()
        # 价值流 (Value stream)
        self.value_fc = nn.Linear(dim_embed, dim_embed)
        self.value_out = nn.Linear(dim_embed, 1)
        
        # 优势流 (Advantage stream)
        self.adv_fc = nn.Linear(2 * dim_embed, dim_embed)
        self.adv_out = nn.Linear(dim_embed, 1)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)

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
        # 计算状态价值 V(s)
        v = F.relu(self.value_fc(state_embed))
        v = self.dropout(v)
        v = self.value_out(v)  # [1, 1] 或 [B, 1]

        # 广播 state_embed 到与 node_embed 相同的行数
        if state_embed.size(0) == 1 and node_embed.size(0) > 1:
            expanded_state = state_embed.expand(node_embed.size(0), -1)
        else:
            expanded_state = state_embed

        # 计算优势函数 A(s, a)
        adv_input = torch.cat([expanded_state, node_embed], dim=1)  # [N, 2*dim_embed]
        adv = F.relu(self.adv_fc(adv_input))
        adv = self.dropout(adv)
        adv = self.adv_out(adv)  # [N, 1]

        # 计算 Q 值：Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        adv_mean = adv.mean(dim=0, keepdim=True)  # [1, 1]
        q = v + (adv - adv_mean)  # [N, 1]

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
    try:
        N = feat.size(0)
        embed = torch.zeros(N, s2v.dim_embed, device=feat.device)

        for _ in range(T):
            embed = s2v(feat, adj, edge_weight, embed, degree=degree)

        return embed
    except Exception as e:
        print(f"编码图时出错: {e}")
        print(f"特征维度: {feat.shape}")
        print(f"S2V维度: {s2v.dim_embed}")
        # 返回默认嵌入
        return torch.zeros(feat.size(0), s2v.dim_embed, device=feat.device)


def get_graph_embedding(node_embed: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
    """
    全局图嵌入 = 所有节点嵌入之和（论文做法）。

    Args:
        node_embed: [N, dim_embed]
        state: 可选的状态张量，用于计算掩码

    Returns:
        [1, dim_embed]
    """
    if state is not None:
        # 只计算活跃节点的嵌入
        mask = (state == 1).float().unsqueeze(1)
        return (node_embed * mask).sum(dim=0, keepdim=True)
    return node_embed.sum(dim=0, keepdim=True)