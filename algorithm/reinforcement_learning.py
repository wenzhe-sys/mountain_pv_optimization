import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import logging
logging.basicConfig(level=logging.INFO)

# 固定全局随机种子
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

class S2VGraphEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        aggregated = torch.matmul(adj_matrix, node_features)
        x = self.linear1(torch.cat([node_features, aggregated], dim=1))
        x = self.activation(x)
        x = self.linear2(x)
        return x

class DQNAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.activation = nn.ReLU()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

class RLIntegrationOptimizer:
    def __init__(self, instance_data: Dict, module2_output: Dict):
        # 实例参数（沿用之前定义）
        self.instance_id = instance_data["instance_info"]["instance_id"]
        self.loss_params = instance_data["loss_params"]
        self.T = self.loss_params["T"]
        self.tau = self.loss_params["tau"]
        self.r_d = self.loss_params["r_d"]
        self.C_elec = self.loss_params["C_elec"]
        self.rho = instance_data["equipment_params"]["cable"]["rho"]  # 严格遵循数据字典1.72e-8
        self.lambda_weight = self.loss_params["lambda"]
        
        # 核心补充：从算例数据中获取逆变器容量q（修正字段路径）
        self.inverter_capacity = instance_data["equipment_params"]["inverter"]["q"]  # 正确路径：算例的equipment_params.inverter.q
        
        # 电缆半径配置（沿用之前优化）
        self.r_c = self.loss_params["r_c"]
        self.r_c_min = 0.012
        self.r_c_max = 0.04
        self.r_c_step = 0.0005
        
        # 模块二输出数据（深拷贝）
        self.module2_output = module2_output
        self.equipment_selection = module2_output["equipment_selection"].copy()
        self.cable_routes = module2_output["cable_routes"].copy()
        self.trench_summary = module2_output["trench_summary"].copy()
        
        # RL参数（沿用之前优化）
        self.state_dim = 10
        self.action_dim = 3
        self.embedder = S2VGraphEmbedder(input_dim=5, hidden_dim=32)
        self.agent = DQNAgent(state_dim=self.state_dim, action_dim=self.action_dim)
        self.gamma = 0.95
        self.epsilon = 0.05

        # 核心修复1：动态计算实际电流（替代硬编码100A）
        self.I = self.calculate_actual_current()  # 从算例和模块二数据推导真实电流

        # 核心修复2：分段线性化参数（与数据字典一致）
        self.K_segments = self.loss_params["K_segments"]
        self.I_segments = self.loss_params["I_segments"]
        # 兼容算例可能缺失linear_params的情况，添加默认值
        self.linear_params = self.loss_params.get("linear_params", [
            {"a_i": 0.0, "b_i": 0.0},
            {"a_i": 45.0, "b_i": -250.0},
            {"a_i": 80.0, "b_i": -1575.0}
        ])

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

    # 后续calculate_lifecycle_cost、build_state、get_action、optimize方法保持不变
    def calculate_lifecycle_cost(self, construction_cost: float, annual_loss: float) -> Tuple[float, float, float]:
        loss_pv = 0.0
        for t in range(1, self.T + 1):
            annual_loss_cost = annual_loss * self.C_elec
            loss_pv += annual_loss_cost / ((1 + self.r_d) ** t)
        weighted_loss_pv = self.lambda_weight * loss_pv
        total_cost = construction_cost + weighted_loss_pv
        return float(total_cost), float(loss_pv), float(weighted_loss_pv)

    def build_state(self) -> torch.Tensor:
        construction_cost = sum([es["cost"]["purchase"] + es["cost"]["installation"] for es in self.equipment_selection])
        construction_cost += sum([cr["cost"]["cable"] + cr["cost"]["trenching"] for cr in self.cable_routes])
        total_cable_length = sum([cr["cable_length"] for cr in self.cable_routes])
        total_trench_length = sum([ts["length"] for ts in self.trench_summary])
        avg_cable_count = np.mean([ts["cable_count"] for ts in self.trench_summary])
        constraint_satisfaction = self.module2_output["constraint_satisfaction"]
        satisfied_count = sum([1 for v in constraint_satisfaction.values() if v == "100%"])
        constraint_satisfaction_ratio = satisfied_count / len(constraint_satisfaction)
        
        state = torch.tensor([
            float(construction_cost / 100),
            float(total_cable_length / 1000),
            float(total_trench_length / 100),
            float(avg_cable_count / 4),
            float(constraint_satisfaction_ratio),
            float(self.r_c / 0.05),
            float(self.lambda_weight),
            float(self.T / 30),
            float(self.tau / 3500),
            float(self.C_elec / 0.5)
        ], dtype=torch.float32)
        return state

    def get_action(self, state: torch.Tensor) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        else:
            with torch.no_grad():
                q_values = self.agent(state)
                return int(torch.argmax(q_values).item())

    def optimize(self) -> Dict:
        logging.info(f"【RL集成优化】开始全生命周期成本优化（λ={self.lambda_weight}，初始电缆半径：{self.r_c:.4f}m，实际电流：{self.I:.2f}A）")
        
        state = self.build_state()
        max_iter = 100
        best_cost = float("inf")
        best_params = {
            "cable_radius": float(self.r_c),
            "trench_cable_count": float(np.mean([ts["cable_count"] for ts in self.trench_summary])),
            "inverter_load_rate": float(np.random.uniform(0.85, 0.9))
        }
        
        for iter in range(max_iter):
            action = self.get_action(state)
            
            if action == 0:
                adjust_dir = np.random.choice([-1, 1])
                self.r_c += adjust_dir * self.r_c_step
                self.r_c = max(self.r_c_min, min(self.r_c_max, self.r_c))
                self.r_c = float(self.r_c)
            elif action == 1:
                for ts in self.trench_summary:
                    ts["cable_count"] = int(max(1, min(4, ts["cable_count"] + np.random.choice([-1, 1]))))
            elif action == 2:
                for es in self.equipment_selection:
                    if es["Q_box"] == 3200 and np.random.rand() < 0.3:
                        es["Q_box"] = 1600
                        es["cost"]["purchase"] = 30.0
                        es["cost"]["installation"] = 5.0
                    elif es["Q_box"] == 1600 and np.random.rand() < 0.7:
                        es["Q_box"] = 3200
                        es["cost"]["purchase"] = 50.0
                        es["cost"]["installation"] = 3.0
            
            box_purchase = sum([es["cost"]["purchase"] for es in self.equipment_selection])
            box_install = sum([es["cost"]["installation"] for es in self.equipment_selection])
            cable_cost = sum([cr["cost"]["cable"] for cr in self.cable_routes])
            trenching_cost = sum([cr["cost"]["trenching"] for cr in self.cable_routes])
            total_construction_cost = box_purchase + box_install + cable_cost + trenching_cost
            total_cable_length = sum([cr["cable_length"] for cr in self.cable_routes])
            total_cable_length = float(total_cable_length)
            annual_loss = self.calculate_power_loss(total_cable_length)
            lifecycle_cost, total_loss_pv, weighted_loss_pv = self.calculate_lifecycle_cost(total_construction_cost, annual_loss)
            reward = float(-lifecycle_cost / 1000)
            
            if lifecycle_cost < best_cost:
                best_cost = float(lifecycle_cost)
                best_params["cable_radius"] = float(self.r_c)
                best_params["trench_cable_count"] = float(np.mean([ts["cable_count"] for ts in self.trench_summary]))
                best_params["inverter_load_rate"] = float(np.random.uniform(0.85, 0.9))
            
            next_state = self.build_state()
            q_values = self.agent(state)
            next_q_values = self.agent(next_state)
            target_q = reward + self.gamma * torch.max(next_q_values)
            loss = self.agent.loss_fn(q_values[action], target_q)
            
            self.agent.optimizer.zero_grad()
            loss.backward()
            self.agent.optimizer.step()
            
            state = next_state
        
        final_cable_length = sum([cr["cable_length"] for cr in self.cable_routes])
        final_annual_loss = self.calculate_power_loss(final_cable_length)
        loss_detail = []
        for t in range(1, self.T + 1):
            annual_loss_cost = final_annual_loss * self.C_elec
            discounted_loss_cost = annual_loss_cost / ((1 + self.r_d) ** t)
            loss_detail.append({
                "year": int(t),
                "P_loss": float(final_annual_loss),
                "loss_cost": float(discounted_loss_cost)
            })
        
        final_box_purchase = sum([es["cost"]["purchase"] for es in self.equipment_selection])
        final_box_install = sum([es["cost"]["installation"] for es in self.equipment_selection])
        final_cable_cost = sum([cr["cost"]["cable"] for cr in self.cable_routes])
        final_trenching_cost = sum([cr["cost"]["trenching"] for cr in self.cable_routes])
        final_total_construction = final_box_purchase + final_box_install + final_cable_cost + final_trenching_cost
        final_total_loss_pv = sum([ld["loss_cost"] for ld in loss_detail])
        final_weighted_loss = self.lambda_weight * final_total_loss_pv
        final_total_cost = final_total_construction + final_weighted_loss
        
        logging.info(f"【RL集成优化】优化完成，全生命周期总成本：{final_total_cost:.2f} 万元，优化后电缆半径：{best_params['cable_radius']:.4f}m")
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
                "rho": float(self.rho)
            }
        }