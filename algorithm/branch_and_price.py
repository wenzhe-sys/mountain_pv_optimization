"""
代理模块：将 algorithm.branch_and_price 的导入转发到实际实现。
实际代码位于 modules/module2/algorithm/branch_and_price.py
"""

from modules.module2.algorithm.branch_and_price import BranchAndPrice

__all__ = ["BranchAndPrice"]
