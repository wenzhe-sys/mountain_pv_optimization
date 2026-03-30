import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import logging
from collections import deque

class S2VGraphEmbedder(nn.Module):
    """
    结构化图嵌入网络
    用于将图结构数据转换为低维向量表示
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            node_features: 节点特征矩阵
            adj_matrix: 邻接矩阵
        Returns:
            嵌入后的节点特征
        """
        aggregated = torch.matmul(adj_matrix, node_features)
        x = self.linear1(torch.cat([node_features, aggregated], dim=1))
        x = self.activation(x)
        x = self.linear2(x)
        return x

class DuelingDQN(nn.Module):
    """
    Dueling DQN网络架构
    将Q值分解为状态价值和优势函数
    """
    def __init__(self, state_dim: int, action_dim: int, num_objectives: int = 3):
        super().__init__()
        # 共享特征提取网络
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.activation = nn.ReLU()
        
        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_objectives)
        )
        
        # 优势函数流
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim * num_objectives)
        )
        
        self.num_objectives = num_objectives
        self.action_dim = action_dim
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            state: 状态张量
        Returns:
            Q值张量
        """
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        
        # 计算状态价值
        values = self.value_stream(x)  # (batch_size, num_objectives)
        
        # 计算优势函数
        advantages = self.advantage_stream(x)  # (batch_size, num_objectives * action_dim)
        advantages = advantages.view(-1, self.num_objectives, self.action_dim) if state.dim() > 1 else advantages.view(self.num_objectives, self.action_dim)
        
        # 计算Q值：Q = V + (A - mean(A))
        if state.dim() > 1:
            advantage_mean = advantages.mean(dim=2, keepdim=True)
            q_values = values.unsqueeze(2) + (advantages - advantage_mean)
        else:
            advantage_mean = advantages.mean(dim=1, keepdim=True)
            q_values = values.unsqueeze(1) + (advantages - advantage_mean)
        
        return q_values

class MultiObjectiveDQNAgent(nn.Module):
    """
    多目标DQN智能体
    支持多种网络架构和探索策略
    """
    def __init__(self, state_dim: int, action_dim: int, num_objectives: int = 3, use_dueling: bool = True):
        super().__init__()
        self.num_objectives = num_objectives
        self.action_dim = action_dim
        self.use_dueling = use_dueling
        
        # 根据配置选择网络架构
        if use_dueling:
            self.network = DuelingDQN(state_dim, action_dim, num_objectives)
        else:
            # 传统DQN架构
            self.network = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim * num_objectives)
            )
            self.loss_fn = nn.MSELoss()
            self.optimizer = optim.Adam(self.parameters(), lr=5e-4)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            state: 状态张量
        Returns:
            Q值张量
        """
        if self.use_dueling:
            return self.network(state)
        else:
            x = self.network(state)
            return x.view(-1, self.num_objectives, self.action_dim) if state.dim() > 1 else x.view(self.num_objectives, self.action_dim)
    
    def loss_fn(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        Args:
            predicted: 预测Q值
            target: 目标Q值
        Returns:
            损失值
        """
        if self.use_dueling:
            return self.network.loss_fn(predicted, target)
        else:
            return nn.MSELoss()(predicted, target)
    
    def optimizer(self) -> optim.Optimizer:
        """
        获取优化器
        Returns:
            优化器实例
        """
        if self.use_dueling:
            return self.network.optimizer
        else:
            return self.optimizer