"""
algorithm 模块
==============
模块二核心算法实现：分支定价（Branch and Price）框架。

子模块：
- branch_and_price:   分支定价主算法（对外入口）
- candidate_sites:    候选箱变站址生成
- path_generator:     电缆路径生成（曼哈顿路由 + KNN 剪枝）
- arc_flow_milp:      Arc-Flow MILP 模型（PuLP）
- column_generation:  列生成（RMP + 定价子问题）
- matheuristic:       Matheuristic 混合启发式
"""
