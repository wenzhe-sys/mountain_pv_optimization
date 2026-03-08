"""
算法包代理
==========
将根目录 algorithm/ 下的导入转发到 modules/module2/algorithm/。
"""

from modules.module2.algorithm.branch_and_price import BranchAndPrice

__all__ = ["BranchAndPrice"]
