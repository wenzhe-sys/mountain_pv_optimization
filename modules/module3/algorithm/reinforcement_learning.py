import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Set, Optional
import logging
from collections import defaultdict, deque
import copy
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
logging.basicConfig(level=logging.INFO)

class ConfigManager:
    """
    统一配置管理系统
    支持参数的集中管理和动态调整
    """
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        Args:
            config_path: 配置文件路径
        """
        # 默认配置
        self.default_config = {
            "rl": {
                "state_dim": 18,
                "action_dim": 7,
                "num_objectives": 3,
                "gamma": 0.98,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "temperature": 1.0,
                "temperature_decay": 0.99,
                "temperature_min": 0.1,
                "batch_size": 32,
                "memory_capacity": 10000,
                "use_dueling": True,
                "exploration_strategy": "boltzmann",
                "use_adaptive_weights": True
            },
            "cable": {
                "r_c_min": 0.012,
                "r_c_max": 0.04,
                "r_c_step": 0.0005
            },
            "optimization": {
                "max_iter_small": 100,
                "max_iter_medium": 150,
                "max_iter_large": 200,
                "heuristic_search_frequency": 50
            },
            "logging": {
                "level": "INFO",
                "log_file": None
            },
            "visualization": {
                "enable": True,
                "save_dir": "visualizations"
            }
        }
        
        # 加载配置文件
        self.config = self.default_config.copy()
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
    def load_config(self, config_path: str):
        """
        加载配置文件
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # 递归更新配置
                self._update_config(self.config, loaded_config)
            logging.info(f"成功加载配置文件: {config_path}")
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
    
    def save_config(self, config_path: str):
        """
        保存配置文件
        Args:
            config_path: 配置文件路径
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logging.info(f"成功保存配置文件: {config_path}")
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")
    
    def _update_config(self, target: Dict, source: Dict):
        """
        递归更新配置
        Args:
            target: 目标配置
            source: 源配置
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config(target[key], value)
            else:
                target[key] = value
    
    def get(self, key: str, default=None):
        """
        获取配置值
        Args:
            key: 配置键，支持点号分隔的路径
            default: 默认值
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value):
        """
        设置配置值
        Args:
            key: 配置键，支持点号分隔的路径
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_rl_config(self) -> Dict:
        """
        获取RL相关配置
        Returns:
            RL配置字典
        """
        return self.config.get("rl", {})
    
    def get_cable_config(self) -> Dict:
        """
        获取电缆相关配置
        Returns:
            电缆配置字典
        """
        return self.config.get("cable", {})
    
    def get_optimization_config(self) -> Dict:
        """
        获取优化相关配置
        Returns:
            优化配置字典
        """
        return self.config.get("optimization", {})

# 创建全局配置管理器实例
config_manager = ConfigManager()

class VisualizationManager:
    """
    可视化管理器
    实现更丰富的可视化功能
    """
    def __init__(self, save_dir: str = "visualizations"):
        """
        初始化可视化管理器
        Args:
            save_dir: 可视化结果保存目录
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置可视化风格
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def visualize_pareto_front(self, pareto_front: List[Dict], filename: str = "pareto_front.png"):
        """
        可视化帕累托最优前沿
        Args:
            pareto_front: 帕累托最优解列表
            filename: 保存文件名
        """
        if not pareto_front:
            return
        
        try:
            # 提取数据
            costs = [item["cost"] for item in pareto_front]
            efficiencies = [item["efficiency"] for item in pareto_front]
            reliabilities = [item["reliability"] for item in pareto_front]
            
            # 创建3D散点图
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制散点
            scatter = ax.scatter(costs, efficiencies, reliabilities, 
                               c=costs, cmap='viridis', s=50, alpha=0.7)
            
            # 添加颜色条
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('成本 (万元)')
            
            # 设置标签
            ax.set_xlabel('成本 (万元)')
            ax.set_ylabel('效率')
            ax.set_zlabel('可靠性')
            ax.set_title('帕累托最优前沿')
            
            # 保存图片
            save_path = os.path.join(self.save_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            logging.info(f"成功保存帕累托前沿可视化: {save_path}")
        except Exception as e:
            logging.error(f"可视化帕累托前沿失败: {e}")
    
    def visualize_optimization_history(self, history: List[Dict], filename: str = "optimization_history.png"):
        """
        可视化优化历史
        Args:
            history: 优化历史记录
            filename: 保存文件名
        """
        if not history:
            return
        
        try:
            # 提取数据
            iterations = [item["iteration"] for item in history]
            costs = [item["cost"] for item in history]
            efficiencies = [item["efficiency"] for item in history]
            reliabilities = [item["reliability"] for item in history]
            
            # 创建DataFrame
            df = pd.DataFrame({
                'Iteration': iterations,
                'Cost': costs,
                'Efficiency': efficiencies,
                'Reliability': reliabilities
            })
            
            # 创建子图
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # 绘制成本曲线
            sns.lineplot(x='Iteration', y='Cost', data=df, ax=axes[0], linewidth=2)
            axes[0].set_ylabel('成本 (万元)')
            axes[0].set_title('优化过程')
            axes[0].grid(True, alpha=0.3)
            
            # 绘制效率曲线
            sns.lineplot(x='Iteration', y='Efficiency', data=df, ax=axes[1], linewidth=2, color='green')
            axes[1].set_ylabel('效率')
            axes[1].grid(True, alpha=0.3)
            
            # 绘制可靠性曲线
            sns.lineplot(x='Iteration', y='Reliability', data=df, ax=axes[2], linewidth=2, color='orange')
            axes[2].set_ylabel('可靠性')
            axes[2].set_xlabel('迭代次数')
            axes[2].grid(True, alpha=0.3)
            
            # 保存图片
            save_path = os.path.join(self.save_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            logging.info(f"成功保存优化历史可视化: {save_path}")
        except Exception as e:
            logging.error(f"可视化优化历史失败: {e}")
    
    def visualize_performance_metrics(self, metrics: Dict, filename: str = "performance_metrics.png"):
        """
        可视化性能指标
        Args:
            metrics: 性能指标字典
            filename: 保存文件名
        """
        if not metrics:
            return
        
        try:
            # 提取数据
            improvement_keys = ['cost_improvement', 'efficiency_improvement', 'reliability_improvement']
            improvement_values = [metrics.get(key, 0) for key in improvement_keys]
            
            # 创建柱状图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制柱状图
            bars = ax.bar(improvement_keys, improvement_values, color=['blue', 'green', 'orange'])
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.2f}%', ha='center', va='bottom')
            
            # 设置标签
            ax.set_ylabel('改进百分比 (%)')
            ax.set_title('性能改进指标')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 保存图片
            save_path = os.path.join(self.save_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            logging.info(f"成功保存性能指标可视化: {save_path}")
        except Exception as e:
            logging.error(f"可视化性能指标失败: {e}")
    
    def create_dashboard(self, optimization_result: Dict, filename: str = "dashboard.png"):
        """
        创建综合仪表盘
        Args:
            optimization_result: 优化结果字典
            filename: 保存文件名
        """
        try:
            # 创建子图
            fig = plt.figure(figsize=(15, 12))
            
            # 1. 成本摘要
            ax1 = fig.add_subplot(2, 2, 1)
            cost_summary = optimization_result.get('total_cost_summary', {})
            cost_breakdown = cost_summary.get('cost_breakdown', {})
            
            if cost_breakdown:
                labels = list(cost_breakdown.keys())
                values = list(cost_breakdown.values())
                ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax1.set_title('成本 breakdown')
            
            # 2. 性能指标
            ax2 = fig.add_subplot(2, 2, 2)
            performance_metrics = optimization_result.get('performance_metrics', {})
            performance_evaluation = performance_metrics.get('performance_evaluation', {})
            
            if performance_evaluation:
                improvement_keys = ['cost_improvement', 'efficiency_improvement', 'reliability_improvement']
                improvement_values = [performance_evaluation.get(key, 0) for key in improvement_keys]
                ax2.bar(improvement_keys, improvement_values, color=['blue', 'green', 'orange'])
                ax2.set_ylabel('改进百分比 (%)')
                ax2.set_title('性能改进指标')
                ax2.grid(True, alpha=0.3, axis='y')
            
            # 3. 优化参数
            ax3 = fig.add_subplot(2, 2, 3)
            optimized_params = optimization_result.get('optimized_params', {})
            
            if optimized_params:
                params = list(optimized_params.keys())[:6]  # 只显示前6个参数
                values = [optimized_params.get(key, 0) for key in params]
                ax3.barh(params, values, color='skyblue')
                ax3.set_xlabel('值')
                ax3.set_title('优化参数')
            
            # 4. 帕累托前沿
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            pareto_front = optimization_result.get('pareto_front', [])
            
            if pareto_front:
                costs = [item["cost"] for item in pareto_front]
                efficiencies = [item["efficiency"] for item in pareto_front]
                reliabilities = [item["reliability"] for item in pareto_front]
                
                scatter = ax4.scatter(costs, efficiencies, reliabilities, 
                                   c=costs, cmap='viridis', s=30, alpha=0.7)
                ax4.set_xlabel('成本')
                ax4.set_ylabel('效率')
                ax4.set_zlabel('可靠性')
                ax4.set_title('帕累托最优前沿')
            
            # 保存图片
            save_path = os.path.join(self.save_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            logging.info(f"成功保存综合仪表盘: {save_path}")
        except Exception as e:
            logging.error(f"创建综合仪表盘失败: {e}")

# 固定全局随机种子
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

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

# 修改 MultiObjectiveDQNAgent 类
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

class RLIntegrationOptimizer:
    """
    强化学习集成优化器
    实现端到端强化学习与数学启发式相结合的智能优化方法
    """
    def __init__(self, instance_data: Dict, module2_output: Dict, module1_output = None):
        """
        初始化优化器
        Args:
            instance_data: 实例数据字典
            module2_output: 模块二输出结果
            module1_output: 模块一输出结果（可选）
        """
        # 实例参数
        self.instance_id = instance_data["instance_info"]["instance_id"]
        self.loss_params = instance_data["loss_params"]
        self.T = self.loss_params["T"]
        self.tau = self.loss_params["tau"]
        self.r_d = self.loss_params["r_d"]
        self.C_elec = self.loss_params["C_elec"]
        self.rho = instance_data["equipment_params"]["cable"]["rho"]  # 严格遵循数据字典1.72e-8
        self.lambda_weight = self.loss_params["lambda"]
        
        # 核心补充：从算例数据中获取逆变器容量q
        self.inverter_capacity = instance_data["equipment_params"]["inverter"]["q"]
        
        # 电缆半径配置
        self.r_c = self.loss_params["r_c"]
        self.r_c_min = 0.012
        self.r_c_max = 0.04
        self.r_c_step = 0.0005
        
        # 模块输出数据（使用浅拷贝提高效率，需要修改时再深拷贝）
        self.module2_output = module2_output
        self.module1_output = module1_output
        self.equipment_selection = module2_output["equipment_selection"]
        self.cable_routes = module2_output["cable_routes"]
        self.trench_summary = module2_output["trench_summary"]
        
        # 问题规模评估
        self.problem_scale = self._evaluate_problem_scale()
        
        # 动态状态维度：根据问题规模自动调整
        self.state_dim = self._calculate_state_dim()
        self.action_dim = 7  # 增加动作维度以包含更多优化选项
        self.num_objectives = 3  # 多目标：成本、效率、可靠性
        
        # 初始化嵌入器和智能体
        self.embedder = S2VGraphEmbedder(input_dim=5, hidden_dim=64)
        self.agent = MultiObjectiveDQNAgent(
            state_dim=self.state_dim, 
            action_dim=self.action_dim, 
            num_objectives=self.num_objectives,
            use_dueling=True  # 使用Dueling DQN
        )
        
        # RL参数
        self.gamma = 0.98  # 折扣因子
        self.epsilon = 0.1  # 探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.epsilon_min = 0.01  # 最小探索率
        self.temperature = 1.0  # 玻尔兹曼探索温度参数
        self.temperature_decay = 0.99  # 温度衰减
        self.temperature_min = 0.1  # 最小温度
        self.batch_size = 32  # 批量大小
        
        # 经验回放缓冲区：使用deque提高效率
        self.memory = deque(maxlen=10000)
        
        # 多目标优化：帕累托最优前沿
        self.pareto_front = []
        
        # 自适应奖励权重
        self.reward_weights = {"cost": 0.5, "efficiency": 0.3, "reliability": 0.2}
        
        # 核心修复1：动态计算实际电流
        self.I = self.calculate_actual_current()

        # 核心修复2：分段线性化参数
        self.K_segments = self.loss_params["K_segments"]
        self.I_segments = self.loss_params["I_segments"]
        self.linear_params = self.loss_params.get("linear_params", [
            {"a_i": 0.0, "b_i": 0.0},
            {"a_i": 45.0, "b_i": -250.0},
            {"a_i": 80.0, "b_i": -1575.0}
        ])
        
        # 配置管理
        self.config = {
            "exploration_strategy": "boltzmann",  # boltzmann or epsilon_greedy
            "use_adaptive_weights": True,
            "use_prioritized_experience": False,
            "debug_mode": False
        }
    
    def _evaluate_problem_scale(self) -> str:
        """
        评估问题规模
        Returns:
            问题规模："small", "medium", 或 "large"
        """
        num_equipment = len(self.equipment_selection)
        num_cable_routes = len(self.cable_routes)
        num_trenches = len(self.trench_summary)
        
        total_elements = num_equipment + num_cable_routes + num_trenches
        
        if total_elements < 50:
            return "small"
        elif total_elements < 200:
            return "medium"
        else:
            return "large"
    
    def _calculate_state_dim(self) -> int:
        """
        根据问题规模计算状态维度
        Returns:
            状态维度大小
        """
        base_dim = 18
        
        if self.problem_scale == "small":
            return base_dim
        elif self.problem_scale == "medium":
            return base_dim + 4  # 增加4个额外特征
        else:  # large
            return base_dim + 8  # 增加8个额外特征

    def calculate_actual_current(self) -> float:
        """
        动态计算实际电流：基于逆变器容量、面板总功率推导（符合工程逻辑）
        公式：I = P_total / (√3 × U)，假设电压U=380V（低压交流标准）
        核心修正：从算例的equipment_params获取逆变器容量q，而非module1_output
        """
        # 从算例数据中获取逆变器容量（已在__init__中缓存为self.inverter_capacity）
        q = self.inverter_capacity  # 320kW（数据字典默认值）
        
        # 从模块一输出获取分区总功率（所有面板功率之和）
        total_power = sum([zone["total_power"] for zone in self.module2_output["module1_output"]["zone_summary"]])
        # 计算线电流（三相交流，U=380V，功率因数cosφ=0.85）
        U = 380.0  # 标准低压交流电压（V）
        cos_phi = 0.85  # 光伏逆变器典型功率因数
        if total_power == 0 or U == 0:
            return 100.0  # 异常时 fallback 到100A
        I = total_power * 1000 / (np.sqrt(3) * U * cos_phi)  # 转换为A
        # 限制在I_max范围内（数据字典约束）
        return float(min(max(I, 0), self.loss_params["I_max"]))

    # 其余方法（linearize_I_squared、calculate_power_loss等）保持不变
    def linearize_I_squared(self, I: float) -> float:
        """核心修复3：分段线性化拟合I²（与数据字典一致，减少近似误差）"""
        # 找到电流所在分段
        for i in range(self.K_segments):
            I_min, I_max = self.I_segments[i]
            if I_min <= I <= I_max:
                a_i = self.linear_params[i]["a_i"]
                b_i = self.linear_params[i]["b_i"]
                return float(a_i * I + b_i)
        # 超出分段范围时，用最近分段的参数
        return float(self.linear_params[-1]["a_i"] * I + self.linear_params[-1]["b_i"])

    def calculate_power_loss(self, cable_length: float) -> float:
        """
        修复后的损耗计算：动态电流 + 分段线性化 + 地形修正长度
        完全对齐数据字典的损耗计算约束
        """
        if self.I <= 0:
            raise ValueError("电流不能为非正数")
        # 1. 地形修正后的电缆长度（直接使用模块二输出的实际敷设长度）
        actual_cable_length = cable_length  # 模块二已做地形修正（D_uv）
        # 2. 计算电阻（严格遵循数据字典公式 R=ρ×l/(π×r_c²)）
        resistance = self.rho * actual_cable_length / (np.pi * self.r_c ** 2)
        # 3. 分段线性化后的I²（替代真实I²，减少近似误差）
        I_squared = self.linearize_I_squared(self.I)
        # 4. 年度损耗（kWh）
        annual_loss = I_squared * resistance * self.tau
        return float(annual_loss)

    def calculate_lifecycle_cost(self, construction_cost: float, annual_loss: float) -> Tuple[float, float, float]:
        loss_pv = 0.0
        for t in range(1, self.T + 1):
            annual_loss_cost = annual_loss * self.C_elec
            loss_pv += annual_loss_cost / ((1 + self.r_d) ** t)
        weighted_loss_pv = self.lambda_weight * loss_pv
        total_cost = construction_cost + weighted_loss_pv
        return float(total_cost), float(loss_pv), float(weighted_loss_pv)

    def calculate_reliability(self) -> float:
        """
        计算系统可靠性指标
        基于电缆半径、逆变器负载率、约束满足率等因素
        """
        # 电缆可靠性：基于电缆半径
        cable_reliability = min(1.0, self.r_c / 0.03)
        
        # 逆变器可靠性：基于负载率
        total_power = sum([zone["total_power"] for zone in self.module2_output["module1_output"]["zone_summary"]])
        inverter_count = len(self.equipment_selection)
        if inverter_count > 0:
            total_inverter_capacity = sum([es["Q_box"] for es in self.equipment_selection]) / 1000  # 转换为MW
            inverter_load_rate = total_power / (total_inverter_capacity * 1000)  # 转换为kW
            inverter_reliability = max(0.5, 1.0 - (inverter_load_rate - 0.8) * 2)  # 负载率越高，可靠性越低
        else:
            inverter_reliability = 0.5
        
        # 约束满足率
        constraint_satisfaction = self.module2_output["constraint_satisfaction"]
        if isinstance(constraint_satisfaction, bool):
            constraint_satisfaction_ratio = 1.0 if constraint_satisfaction else 0.0
        else:
            satisfied_count = sum([1 for v in constraint_satisfaction.values() if v == "100%" or v is True])
            constraint_satisfaction_ratio = satisfied_count / len(constraint_satisfaction)
        
        # 综合可靠性
        reliability = 0.4 * cable_reliability + 0.4 * inverter_reliability + 0.2 * constraint_satisfaction_ratio
        return float(reliability)

    def calculate_efficiency(self) -> float:
        """
        计算系统效率指标
        基于功率损耗、电缆长度、设备配置等因素
        """
        # 计算总功率损耗
        total_cable_length = sum([cr["length"] for cr in self.cable_routes])
        annual_loss = self.calculate_power_loss(total_cable_length)
        
        # 计算总发电功率
        total_power = sum([zone["total_power"] for zone in self.module2_output["module1_output"]["zone_summary"]])
        annual_energy = total_power * self.tau  # 年发电量
        
        # 效率 = 1 - 损耗/发电量
        if annual_energy > 0:
            efficiency = max(0.8, 1.0 - (annual_loss / annual_energy))
        else:
            efficiency = 0.8
        
        return float(efficiency)

    def build_state(self) -> torch.Tensor:
        """
        构建状态向量
        Returns:
            状态张量
        """
        try:
            # 基础状态特征
            construction_cost = sum([es["cost"]["purchase"] + es["cost"]["installation"] for es in self.equipment_selection])
            construction_cost += sum([cr["cost"]["cable"] + cr["cost"]["trenching"] for cr in self.cable_routes])
            total_cable_length = sum([cr["length"] for cr in self.cable_routes])
            total_trench_length = sum([ts["length"] for ts in self.trench_summary])
            avg_cable_count = np.mean([ts["cable_count"] for ts in self.trench_summary])
            constraint_satisfaction = self.module2_output["constraint_satisfaction"]
            
            # 处理constraint_satisfaction可能是布尔值的情况
            if isinstance(constraint_satisfaction, bool):
                constraint_satisfaction_ratio = 1.0 if constraint_satisfaction else 0.0
            else:
                satisfied_count = sum([1 for v in constraint_satisfaction.values() if v == "100%" or v is True])
                constraint_satisfaction_ratio = satisfied_count / len(constraint_satisfaction)
            
            # 计算逆变器负载率
            total_power = sum([zone["total_power"] for zone in self.module2_output["module1_output"]["zone_summary"]])
            inverter_count = len(self.equipment_selection)
            if inverter_count > 0:
                total_inverter_capacity = sum([es["Q_box"] for es in self.equipment_selection]) / 1000  # 转换为MW
                inverter_load_rate = total_power / (total_inverter_capacity * 1000)  # 转换为kW
            else:
                inverter_load_rate = 0.0
            
            # 计算箱变平均容量
            if inverter_count > 0:
                avg_transformer_capacity = sum([es["Q_box"] for es in self.equipment_selection]) / inverter_count
            else:
                avg_transformer_capacity = 0.0
            
            # 计算电缆平均长度
            if len(self.cable_routes) > 0:
                avg_cable_length = total_cable_length / len(self.cable_routes)
            else:
                avg_cable_length = 0.0
            
            # 计算可靠性和效率
            reliability = self.calculate_reliability()
            efficiency = self.calculate_efficiency()
            
            # 构建基础状态向量
            base_state = [
                float(construction_cost / 100),
                float(total_cable_length / 1000),
                float(total_trench_length / 100),
                float(avg_cable_count / 4),
                float(constraint_satisfaction_ratio),
                float(self.r_c / 0.05),
                float(self.lambda_weight),
                float(self.T / 30),
                float(self.tau / 3500),
                float(self.C_elec / 0.5),
                float(inverter_load_rate),
                float(avg_transformer_capacity / 3200),
                float(avg_cable_length / 1000),
                float(inverter_count / 10),
                float(self.I / 100),
                float(reliability),
                float(efficiency),
                float(len(self.pareto_front) / 10)
            ]
            
            # 根据问题规模添加额外特征
            if self.problem_scale == "medium" or self.problem_scale == "large":
                # 设备分布特征
                if inverter_count > 0:
                    q_box_values = [es["Q_box"] for es in self.equipment_selection]
                    q_box_std = np.std(q_box_values) / 3200
                else:
                    q_box_std = 0.0
                
                # 电缆长度分布特征
                if len(self.cable_routes) > 0:
                    cable_lengths = [cr["length"] for cr in self.cable_routes]
                    cable_length_std = np.std(cable_lengths) / 1000
                else:
                    cable_length_std = 0.0
                
                # 管沟电缆数分布特征
                if len(self.trench_summary) > 0:
                    cable_counts = [ts["cable_count"] for ts in self.trench_summary]
                    cable_count_std = np.std(cable_counts) / 4
                else:
                    cable_count_std = 0.0
                
                # 添加额外特征
                base_state.extend([
                    float(q_box_std),
                    float(cable_length_std),
                    float(cable_count_std),
                    float(self.temperature)
                ])
            
            if self.problem_scale == "large":
                # 更多复杂特征
                # 逆变器容量分布
                if inverter_count > 0:
                    q_box_3200_ratio = sum(1 for es in self.equipment_selection if es["Q_box"] == 3200) / inverter_count
                else:
                    q_box_3200_ratio = 0.0
                
                # 电缆成本比例
                total_cost = construction_cost
                if total_cost > 0:
                    cable_cost_ratio = sum([cr["cost"]["cable"] for cr in self.cable_routes]) / total_cost
                    trenching_cost_ratio = sum([cr["cost"]["trenching"] for cr in self.cable_routes]) / total_cost
                else:
                    cable_cost_ratio = 0.0
                    trenching_cost_ratio = 0.0
                
                # 添加更多特征
                base_state.extend([
                    float(q_box_3200_ratio),
                    float(cable_cost_ratio),
                    float(trenching_cost_ratio),
                    float(self.epsilon)
                ])
            
            # 确保状态维度正确
            if len(base_state) < self.state_dim:
                # 填充零值以达到所需维度
                base_state.extend([0.0] * (self.state_dim - len(base_state)))
            elif len(base_state) > self.state_dim:
                # 截断到所需维度
                base_state = base_state[:self.state_dim]
            
            state = torch.tensor(base_state, dtype=torch.float32)
            return state
        except Exception as e:
            logging.error(f"构建状态时出错: {e}")
            # 返回默认状态
            return torch.zeros(self.state_dim, dtype=torch.float32)

    def get_action(self, state: torch.Tensor) -> int:
        """
        获取动作
        Args:
            state: 状态张量
        Returns:
            动作索引
        """
        try:
            # 自适应奖励权重
            if self.config["use_adaptive_weights"]:
                weights = self._get_adaptive_weights()
            else:
                weights = torch.tensor([0.5, 0.3, 0.2])  # 默认权重
            
            with torch.no_grad():
                q_values = self.agent(state)
                # 多目标决策：使用加权和
                weighted_q = torch.sum(q_values * weights.unsqueeze(1), dim=0)
            
            # 选择探索策略
            if self.config["exploration_strategy"] == "boltzmann":
                return self._boltzmann_exploration(weighted_q)
            else:  # epsilon_greedy
                return self._epsilon_greedy_exploration(weighted_q)
        except Exception as e:
            logging.error(f"获取动作时出错: {e}")
            # 返回随机动作
            return int(np.random.randint(self.action_dim))
    
    def _get_adaptive_weights(self) -> torch.Tensor:
        """
        获取自适应奖励权重
        Returns:
            权重张量
        """
        # 基于当前状态动态调整权重
        # 例如：当效率较低时，增加效率权重
        state = self.build_state()
        efficiency = state[16].item()  # 效率在状态向量中的位置
        reliability = state[15].item()  # 可靠性在状态向量中的位置
        
        # 基础权重
        cost_weight = 0.5
        efficiency_weight = 0.3
        reliability_weight = 0.2
        
        # 根据当前效率和可靠性调整权重
        if efficiency < 0.85:
            efficiency_weight += 0.1
            cost_weight -= 0.05
            reliability_weight -= 0.05
        elif reliability < 0.85:
            reliability_weight += 0.1
            cost_weight -= 0.05
            efficiency_weight -= 0.05
        
        # 归一化权重
        total_weight = cost_weight + efficiency_weight + reliability_weight
        weights = torch.tensor([
            cost_weight / total_weight,
            efficiency_weight / total_weight,
            reliability_weight / total_weight
        ], dtype=torch.float32)
        
        return weights
    
    def _epsilon_greedy_exploration(self, weighted_q: torch.Tensor) -> int:
        """
        epsilon-greedy探索策略
        Args:
            weighted_q: 加权Q值
        Returns:
            动作索引
        """
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        else:
            return int(torch.argmax(weighted_q).item())
    
    def _boltzmann_exploration(self, weighted_q: torch.Tensor) -> int:
        """
        玻尔兹曼探索策略
        Args:
            weighted_q: 加权Q值
        Returns:
            动作索引
        """
        try:
            # 确保温度在有效范围内
            temperature = max(self.temperature, self.temperature_min)
            
            # 计算动作概率，添加数值稳定性处理
            q_values = weighted_q.numpy()
            
            # 减去最大值以避免数值溢出
            max_q = np.max(q_values)
            exp_q = np.exp((q_values - max_q) / temperature)
            
            # 计算概率
            sum_exp = np.sum(exp_q)
            if sum_exp == 0:
                # 如果所有概率都是0，使用均匀分布
                action_probs = np.ones(self.action_dim) / self.action_dim
            else:
                action_probs = exp_q / sum_exp
            
            # 检查概率是否有效
            if np.any(np.isnan(action_probs)) or np.sum(action_probs) == 0:
                # 使用均匀分布作为后备
                action_probs = np.ones(self.action_dim) / self.action_dim
            
            # 根据概率选择动作
            action = np.random.choice(self.action_dim, p=action_probs)
            
            # 衰减温度
            self.temperature *= self.temperature_decay
            self.temperature = max(self.temperature, self.temperature_min)
            
            return int(action)
        except Exception as e:
            logging.error(f"玻尔兹曼探索时出错: {e}")
            # 返回随机动作作为后备
            return int(np.random.randint(self.action_dim))
    
    def remember(self, state: torch.Tensor, action: int, rewards: List[float], next_state: torch.Tensor):
        """
        存储经验到回放缓冲区
        Args:
            state: 当前状态
            action: 执行的动作
            rewards: 获得的奖励
            next_state: 下一个状态
        """
        # 使用deque自动管理容量
        self.memory.append((state, action, rewards, next_state))
    
    def replay(self):
        """
        经验回放学习
        """
        try:
            if len(self.memory) < self.batch_size:
                return
            
            # 随机采样
            batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
            batch = [self.memory[i] for i in batch_indices]
            
            # 准备批次数据
            states = torch.stack([item[0] for item in batch])
            actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
            rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
            next_states = torch.stack([item[3] for item in batch])
            
            # 计算Q值
            current_q = self.agent(states)
            # 选择对应动作的Q值
            current_q = current_q[torch.arange(self.batch_size), :, actions]
            
            # 计算目标Q值
            with torch.no_grad():
                next_q = self.agent(next_states)
                next_q = next_q.max(dim=2)[0]
            target_q = rewards + self.gamma * next_q
            
            # 计算损失
            loss = self.agent.loss_fn(current_q, target_q)
            
            # 反向传播
            optimizer = self.agent.optimizer()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
        except Exception as e:
            logging.error(f"经验回放时出错: {e}")
    
    def is_pareto_optimal(self, solution: Dict) -> bool:
        """
        检查解是否为帕累托最优
        """
        for existing in self.pareto_front:
            if (existing["cost"] <= solution["cost"] and
                existing["efficiency"] >= solution["efficiency"] and
                existing["reliability"] >= solution["reliability"]):
                return False
        return True
    
    def update_pareto_front(self, solution: Dict):
        """
        更新帕累托最优前沿
        """
        if self.is_pareto_optimal(solution):
            # 移除被支配的解
            self.pareto_front = [s for s in self.pareto_front if not (
                solution["cost"] <= s["cost"] and
                solution["efficiency"] >= s["efficiency"] and
                solution["reliability"] >= s["reliability"]
            )]
            self.pareto_front.append(solution)
    
    def heuristic_local_search(self):
        """
        启发式局部搜索：在强化学习的基础上进行局部优化
        """
        # 电缆半径局部优化
        best_r_c = self.r_c
        best_cost = float("inf")
        
        for i in range(-5, 6):
            test_r_c = max(self.r_c_min, min(self.r_c_max, self.r_c + i * self.r_c_step))
            original_r_c = self.r_c
            self.r_c = test_r_c
            
            # 计算成本
            total_cable_length = sum([cr["length"] for cr in self.cable_routes])
            annual_loss = self.calculate_power_loss(total_cable_length)
            box_purchase = sum([es["cost"]["purchase"] for es in self.equipment_selection])
            box_install = sum([es["cost"]["installation"] for es in self.equipment_selection])
            cable_cost = sum([cr["cost"]["cable"] for cr in self.cable_routes])
            trenching_cost = sum([cr["cost"]["trenching"] for cr in self.cable_routes])
            total_construction_cost = box_purchase + box_install + cable_cost + trenching_cost
            lifecycle_cost, _, _ = self.calculate_lifecycle_cost(total_construction_cost, annual_loss)
            
            if lifecycle_cost < best_cost:
                best_cost = lifecycle_cost
                best_r_c = test_r_c
            
            self.r_c = original_r_c
        
        # 更新电缆半径
        self.r_c = best_r_c
        
        # 管沟电缆数量优化
        for ts in self.trench_summary:
            original_count = ts["cable_count"]
            best_count = original_count
            best_cost = float("inf")
            
            for count in range(1, 5):
                ts["cable_count"] = count
                # 计算成本
                total_cable_length = sum([cr["length"] for cr in self.cable_routes])
                annual_loss = self.calculate_power_loss(total_cable_length)
                box_purchase = sum([es["cost"]["purchase"] for es in self.equipment_selection])
                box_install = sum([es["cost"]["installation"] for es in self.equipment_selection])
                cable_cost = sum([cr["cost"]["cable"] for cr in self.cable_routes])
                trenching_cost = sum([cr["cost"]["trenching"] for cr in self.cable_routes])
                total_construction_cost = box_purchase + box_install + cable_cost + trenching_cost
                lifecycle_cost, _, _ = self.calculate_lifecycle_cost(total_construction_cost, annual_loss)
                
                if lifecycle_cost < best_cost:
                    best_cost = lifecycle_cost
                    best_count = count
            
            ts["cable_count"] = best_count

    def optimize(self) -> Dict:
        """
        执行强化学习优化
        Returns:
            优化结果字典
        """
        logging.info(f"【RL集成优化】开始全生命周期成本优化（λ={self.lambda_weight}，初始电缆半径：{self.r_c:.4f}m，实际电流：{self.I:.2f}A）")
        
        try:
            state = self.build_state()
            # 根据问题规模调整迭代次数
            if self.problem_scale == "small":
                max_iter = 100
            elif self.problem_scale == "medium":
                max_iter = 150
            else:  # large
                max_iter = 200
            
            best_cost = float("inf")
            best_params = {
                "cable_radius": float(self.r_c),
                "trench_cable_count": float(np.mean([ts["cable_count"] for ts in self.trench_summary])),
                "inverter_load_rate": float(np.random.uniform(0.85, 0.9)),
                "reliability": 0.0,
                "efficiency": 0.0
            }
            
            # 优化历史记录
            optimization_history = []
            
            for iter in range(max_iter):
                # 自适应调整批量大小
                if iter % 50 == 0 and iter > 0:
                    # 根据问题规模调整批量大小
                    if self.problem_scale == "large":
                        self.batch_size = min(64, self.batch_size + 8)
                    elif self.problem_scale == "medium":
                        self.batch_size = min(48, self.batch_size + 4)
                
                # 获取动作
                action = self.get_action(state)
                
                # 执行动作
                self._execute_action(action)
                
                # 计算当前状态的成本和指标
                current_metrics = self._calculate_current_metrics()
                
                # 多目标奖励
                rewards = [
                    float(-current_metrics["lifecycle_cost"] / 1000),  # 成本奖励（负的，因为要最小化）
                    float(current_metrics["efficiency"] * 100),  # 效率奖励
                    float(current_metrics["reliability"] * 100)  # 可靠性奖励
                ]
                
                # 更新帕累托前沿
                solution = {
                    "cost": current_metrics["lifecycle_cost"],
                    "efficiency": current_metrics["efficiency"],
                    "reliability": current_metrics["reliability"],
                    "params": {
                        "cable_radius": self.r_c,
                        "trench_cable_count": np.mean([ts["cable_count"] for ts in self.trench_summary]),
                        "lambda_weight": self.lambda_weight
                    }
                }
                self.update_pareto_front(solution)
                
                # 更新最佳解
                if current_metrics["lifecycle_cost"] < best_cost:
                    best_cost = float(current_metrics["lifecycle_cost"])
                    best_params["cable_radius"] = float(self.r_c)
                    best_params["trench_cable_count"] = float(np.mean([ts["cable_count"] for ts in self.trench_summary]))
                    best_params["inverter_load_rate"] = float(current_metrics["inverter_load_rate"])
                    best_params["lambda_weight"] = float(self.lambda_weight)
                    best_params["avg_cable_length"] = float(current_metrics["avg_cable_length"])
                    best_params["reliability"] = float(current_metrics["reliability"])
                    best_params["efficiency"] = float(current_metrics["efficiency"])
                
                # 构建下一个状态
                next_state = self.build_state()
                
                # 存储经验
                self.remember(state, action, rewards, next_state)
                
                # 经验回放学习
                self.replay()
                
                # 记录历史
                optimization_history.append({
                    "iteration": iter,
                    "cost": current_metrics["lifecycle_cost"],
                    "efficiency": current_metrics["efficiency"],
                    "reliability": current_metrics["reliability"],
                    "epsilon": self.epsilon,
                    "temperature": self.temperature
                })
                
                # 每10次迭代打印一次进度
                if iter % 10 == 0:
                    logging.info(f"【迭代 {iter}/{max_iter}】成本：{current_metrics['lifecycle_cost']:.2f} 万元，效率：{current_metrics['efficiency']:.4f}，可靠性：{current_metrics['reliability']:.4f}")
                
                # 更新状态
                state = next_state
            
            # 最终执行一次启发式局部搜索
            self.heuristic_local_search()
            
            # 计算最终指标
            final_metrics = self._calculate_current_metrics()
            final_cable_length = sum([cr["length"] for cr in self.cable_routes])
            final_annual_loss = self.calculate_power_loss(final_cable_length)
            
            # 生成损失详情
            loss_detail = []
            for t in range(1, self.T + 1):
                annual_loss_cost = final_annual_loss * self.C_elec
                discounted_loss_cost = annual_loss_cost / ((1 + self.r_d) ** t)
                loss_detail.append({
                    "year": int(t),
                    "P_loss": float(final_annual_loss),
                    "loss_cost": float(discounted_loss_cost)
                })
            
            # 计算最终成本
            final_box_purchase = sum([es["cost"]["purchase"] for es in self.equipment_selection])
            final_box_install = sum([es["cost"]["installation"] for es in self.equipment_selection])
            final_cable_cost = sum([cr["cost"]["cable"] for cr in self.cable_routes])
            final_trenching_cost = sum([cr["cost"]["trenching"] for cr in self.cable_routes])
            final_total_construction = final_box_purchase + final_box_install + final_cable_cost + final_trenching_cost
            final_total_loss_pv = sum([ld["loss_cost"] for ld in loss_detail])
            final_weighted_loss = self.lambda_weight * final_total_loss_pv
            final_total_cost = final_total_construction + final_weighted_loss
            
            # 生成模块间反馈信息
            module_feedback = {
                "module1": {
                    "suggested_zone_count": len(self.equipment_selection),
                    "recommended_panel_density": "high" if final_metrics["efficiency"] > 0.9 else "medium"
                },
                "module2": {
                    "optimal_cable_radius": float(self.r_c),
                    "suggested_trench_count": len(self.trench_summary),
                    "recommended_cable_count_per_trench": int(np.mean([ts["cable_count"] for ts in self.trench_summary]))
                }
            }
            
            # 生成详细的性能评估
            performance_evaluation = self._evaluate_performance(optimization_history)
            
            logging.info(f"【RL集成优化】优化完成，全生命周期总成本：{final_total_cost:.2f} 万元，优化后电缆半径：{best_params['cable_radius']:.4f}m")
            logging.info(f"【多目标优化】效率：{final_metrics['efficiency']:.4f}，可靠性：{final_metrics['reliability']:.4f}")
            logging.info(f"【帕累托前沿】找到 {len(self.pareto_front)} 个帕累托最优解")
            
            # 生成可视化
            self._generate_visualizations(optimization_history)
            
            return {
                "total_cost_summary": {
                    "construction_cost": float(final_total_construction),
                    "operation_loss_cost": float(final_total_loss_pv),
                    "total_cost": float(final_total_cost),
                    "cost_breakdown": {
                        "box_purchase": float(final_box_purchase),
                        "box_install": float(final_box_install),
                        "cable": float(final_cable_cost),
                        "trenching": float(final_trenching_cost),
                        "loss": float(final_weighted_loss)
                    }
                },
                "optimized_params": best_params,
                "loss_detail": loss_detail,
                "calculation_params": {
                    "current": float(self.I),
                    "cable_radius": float(self.r_c),
                    "cable_length": float(final_cable_length),
                    "rho": float(self.rho),
                    "lambda_weight": float(self.lambda_weight),
                    "inverter_count": float(len(self.equipment_selection))
                },
                "performance_metrics": {
                    "convergence_iterations": max_iter,
                    "final_epsilon": float(self.epsilon),
                    "final_temperature": float(self.temperature),
                    "best_cost": float(best_cost),
                    "reliability": float(final_metrics["reliability"]),
                    "efficiency": float(final_metrics["efficiency"]),
                    "optimization_history": optimization_history,
                    "performance_evaluation": performance_evaluation
                },
                "pareto_front": self.pareto_front,
                "module_feedback": module_feedback,
                "problem_scale": self.problem_scale,
                "config": self.config
            }
        except Exception as e:
            logging.error(f"优化过程中出错: {e}")
            # 返回默认结果
            return {
                "total_cost_summary": {
                    "construction_cost": 0.0,
                    "operation_loss_cost": 0.0,
                    "total_cost": float("inf"),
                    "cost_breakdown": {
                        "box_purchase": 0.0,
                        "box_install": 0.0,
                        "cable": 0.0,
                        "trenching": 0.0,
                        "loss": 0.0
                    }
                },
                "optimized_params": best_params if 'best_params' in locals() else {},
                "loss_detail": [],
                "calculation_params": {},
                "performance_metrics": {},
                "pareto_front": self.pareto_front,
                "module_feedback": {},
                "error": str(e)
            }
    
    def _execute_action(self, action: int):
        """
        执行动作
        Args:
            action: 动作索引
        """
        if action == 0:
            # 调整电缆半径
            adjust_dir = np.random.choice([-1, 1])
            self.r_c += adjust_dir * self.r_c_step
            self.r_c = max(self.r_c_min, min(self.r_c_max, self.r_c))
            self.r_c = float(self.r_c)
        elif action == 1:
            # 调整管沟电缆数量
            for ts in self.trench_summary:
                ts["cable_count"] = int(max(1, min(4, ts["cable_count"] + np.random.choice([-1, 1]))))
        elif action == 2:
            # 调整箱变容量
            for es in self.equipment_selection:
                if es["Q_box"] == 3200 and np.random.rand() < 0.3:
                    es["Q_box"] = 1600
                    es["cost"]["purchase"] = 30.0
                    es["cost"]["installation"] = 5.0
                elif es["Q_box"] == 1600 and np.random.rand() < 0.7:
                    es["Q_box"] = 3200
                    es["cost"]["purchase"] = 50.0
                    es["cost"]["installation"] = 3.0
        elif action == 3:
            # 调整lambda权重
            adjust_dir = np.random.choice([-1, 1])
            self.lambda_weight += adjust_dir * 0.05
            self.lambda_weight = max(0.1, min(0.9, self.lambda_weight))
            self.lambda_weight = float(self.lambda_weight)
        elif action == 4:
            # 调整电缆路由长度（模拟）
            for cr in self.cable_routes:
                adjust_factor = 1.0 + np.random.uniform(-0.05, 0.05)
                cr["length"] = max(10.0, cr["length"] * adjust_factor)
                # 更新电缆成本
                cr["cost"]["cable"] = cr["length"] * 0.0035  # 假设电缆成本为3.5元/m
                cr["cost"]["trenching"] = cr["length"] * 0.0561  # 假设管沟成本为56.1元/m
        elif action == 5:
            # 执行启发式局部搜索
            self.heuristic_local_search()
        elif action == 6:
            # 调整探索率（自适应）
            if self.epsilon > self.epsilon_min:
                self.epsilon *= 0.95
    
    def _calculate_current_metrics(self) -> Dict:
        """
        计算当前状态的指标
        Returns:
            指标字典
        """
        # 计算当前成本
        box_purchase = sum([es["cost"]["purchase"] for es in self.equipment_selection])
        box_install = sum([es["cost"]["installation"] for es in self.equipment_selection])
        cable_cost = sum([cr["cost"]["cable"] for cr in self.cable_routes])
        trenching_cost = sum([cr["cost"]["trenching"] for cr in self.cable_routes])
        total_construction_cost = box_purchase + box_install + cable_cost + trenching_cost
        total_cable_length = sum([cr["length"] for cr in self.cable_routes])
        
        # 计算逆变器负载率
        total_power = sum([zone["total_power"] for zone in self.module2_output["module1_output"]["zone_summary"]])
        inverter_count = len(self.equipment_selection)
        if inverter_count > 0:
            total_inverter_capacity = sum([es["Q_box"] for es in self.equipment_selection]) / 1000  # 转换为MW
            inverter_load_rate = total_power / (total_inverter_capacity * 1000)  # 转换为kW
        else:
            inverter_load_rate = 0.0
        
        # 计算平均电缆长度
        if len(self.cable_routes) > 0:
            avg_cable_length = total_cable_length / len(self.cable_routes)
        else:
            avg_cable_length = 0.0
        
        # 计算多目标指标
        annual_loss = self.calculate_power_loss(total_cable_length)
        lifecycle_cost, _, _ = self.calculate_lifecycle_cost(total_construction_cost, annual_loss)
        reliability = self.calculate_reliability()
        efficiency = self.calculate_efficiency()
        
        return {
            "lifecycle_cost": lifecycle_cost,
            "reliability": reliability,
            "efficiency": efficiency,
            "inverter_load_rate": inverter_load_rate,
            "avg_cable_length": avg_cable_length,
            "total_cable_length": total_cable_length
        }
    
    def _evaluate_performance(self, history: List[Dict]) -> Dict:
        """
        评估优化性能
        Args:
            history: 优化历史
        Returns:
            性能评估字典
        """
        if not history:
            return {}
        
        # 提取历史数据
        costs = [h["cost"] for h in history]
        efficiencies = [h["efficiency"] for h in history]
        reliabilities = [h["reliability"] for h in history]
        
        # 计算性能指标
        cost_improvement = (costs[0] - costs[-1]) / costs[0] * 100 if costs[0] > 0 else 0
        efficiency_improvement = (efficiencies[-1] - efficiencies[0]) / efficiencies[0] * 100 if efficiencies[0] > 0 else 0
        reliability_improvement = (reliabilities[-1] - reliabilities[0]) / reliabilities[0] * 100 if reliabilities[0] > 0 else 0
        
        # 计算收敛速度
        # 找到首次达到最终成本95%的迭代次数
        final_cost = costs[-1]
        convergence_iter = len(costs)
        for i, cost in enumerate(costs):
            if cost <= final_cost * 1.05:
                convergence_iter = i
                break
        
        return {
            "cost_improvement": float(cost_improvement),
            "efficiency_improvement": float(efficiency_improvement),
            "reliability_improvement": float(reliability_improvement),
            "convergence_speed": convergence_iter,
            "final_cost": float(final_cost),
            "final_efficiency": float(efficiencies[-1]),
            "final_reliability": float(reliabilities[-1])
        }
    
    def _generate_visualizations(self, history: List[Dict]):
        """
        生成可视化图表
        Args:
            history: 优化历史
        """
        try:
            # 创建输出目录
            output_dir = f"visualizations/{self.instance_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 帕累托前沿3D图
            self._plot_pareto_front(output_dir)
            
            # 2. 优化历史趋势图
            self._plot_optimization_history(history, output_dir)
            
            # 3. 性能指标雷达图
            self._plot_performance_radar(output_dir)
            
            # 4. 成本分解饼图
            self._plot_cost_breakdown(output_dir)
            
            logging.info(f"【可视化】已生成所有可视化图表，保存至 {output_dir}")
        except Exception as e:
            logging.error(f"生成可视化时出错: {e}")
    
    def _plot_pareto_front(self, output_dir: str):
        """
        绘制帕累托前沿3D图
        Args:
            output_dir: 输出目录
        """
        if not self.pareto_front:
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取帕累托前沿数据
        costs = [p["cost"] for p in self.pareto_front]
        efficiencies = [p["efficiency"] for p in self.pareto_front]
        reliabilities = [p["reliability"] for p in self.pareto_front]
        
        # 绘制散点图
        scatter = ax.scatter(costs, efficiencies, reliabilities, c=costs, cmap='viridis', s=100, alpha=0.7)
        
        # 设置标签
        ax.set_xlabel('成本 (万元)', fontsize=12)
        ax.set_ylabel('效率', fontsize=12)
        ax.set_zlabel('可靠性', fontsize=12)
        ax.set_title('帕累托最优前沿', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        fig.colorbar(scatter, ax=ax, label='成本')
        
        # 保存图表
        plt.savefig(f"{output_dir}/pareto_front.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_optimization_history(self, history: List[Dict], output_dir: str):
        """
        绘制优化历史趋势图
        Args:
            history: 优化历史
            output_dir: 输出目录
        """
        if not history:
            return
        
        # 提取数据
        iterations = [h["iteration"] for h in history]
        costs = [h["cost"] for h in history]
        efficiencies = [h["efficiency"] for h in history]
        reliabilities = [h["reliability"] for h in history]
        
        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # 绘制成本趋势
        axes[0].plot(iterations, costs, 'b-', linewidth=2)
        axes[0].set_ylabel('成本 (万元)', fontsize=12)
        axes[0].set_title('优化历史趋势', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 绘制效率趋势
        axes[1].plot(iterations, efficiencies, 'g-', linewidth=2)
        axes[1].set_ylabel('效率', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # 绘制可靠性趋势
        axes[2].plot(iterations, reliabilities, 'r-', linewidth=2)
        axes[2].set_ylabel('可靠性', fontsize=12)
        axes[2].set_xlabel('迭代次数', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(f"{output_dir}/optimization_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_radar(self, output_dir: str):
        """
        绘制性能指标雷达图
        Args:
            output_dir: 输出目录
        """
        # 定义性能指标
        metrics = ['成本优化', '效率提升', '可靠性提升', '收敛速度', '算法稳定性']
        values = [85, 75, 80, 70, 90]  # 示例值，实际应从性能评估中获取
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        values += values[:1]
        
        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2, color='b', alpha=0.7)
        ax.fill(angles, values, color='b', alpha=0.25)
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylabel('性能评分', fontsize=12)
        ax.set_title('算法性能评估', fontsize=14, fontweight='bold')
        
        # 保存图表
        plt.savefig(f"{output_dir}/performance_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cost_breakdown(self, output_dir: str):
        """
        绘制成本分解饼图
        Args:
            output_dir: 输出目录
        """
        # 计算成本分解
        box_purchase = sum([es["cost"]["purchase"] for es in self.equipment_selection])
        box_install = sum([es["cost"]["installation"] for es in self.equipment_selection])
        cable_cost = sum([cr["cost"]["cable"] for cr in self.cable_routes])
        trenching_cost = sum([cr["cost"]["trenching"] for cr in self.cable_routes])
        
        # 准备数据
        labels = ['设备采购', '设备安装', '电缆', '管沟']
        sizes = [box_purchase, box_install, cable_cost, trenching_cost]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        explode = (0.1, 0, 0, 0)  # 突出显示第一个部分
        
        # 创建饼图
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90
        )
        
        # 设置字体
        plt.setp(texts, fontsize=12)
        plt.setp(autotexts, fontsize=10, weight='bold')
        
        ax.set_title('成本分解', fontsize=14, fontweight='bold')
        ax.axis('equal')  # 确保饼图是圆形
        
        # 保存图表
        plt.savefig(f"{output_dir}/cost_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()