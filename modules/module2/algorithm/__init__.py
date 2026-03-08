"""
模块二算法包
============
分支定价（Branch-and-Price）求解器，用于设备选型选址 + 电缆共沟优化。

核心组件：
- branch_and_price.py : 主入口，集成所有求解策略
- rmp_solver.py : 受限主问题 LP/MILP 求解
- pricing_subproblem.py : 定价子问题求解器
- column_manager.py : 列管理（聚类 + 路径池）
- bb_tree.py : 分支定界树
- lagrangian.py : 拉格朗日松弛
- heuristics.py : K-Means 启发式回退
- base_components.py : 基础组件（坐标、路径、候选站址）
- result_formatter.py : 结果格式化
"""

from modules.module2.algorithm.branch_and_price import BranchAndPrice
from modules.module2.algorithm.rmp_solver import RMPSolver
from modules.module2.algorithm.pricing_subproblem import PricingSubproblem
from modules.module2.algorithm.column_manager import ColumnManager
from modules.module2.algorithm.bb_tree import BranchAndBoundTree, BBNode
from modules.module2.algorithm.lagrangian import LagrangianRelaxation
from modules.module2.algorithm.heuristics import MatheuristicFallback
from modules.module2.algorithm.result_formatter import ResultFormatter

__all__ = [
    "BranchAndPrice",
    "RMPSolver",
    "PricingSubproblem",
    "ColumnManager",
    "BranchAndBoundTree",
    "BBNode",
    "LagrangianRelaxation",
    "MatheuristicFallback",
    "ResultFormatter",
]
