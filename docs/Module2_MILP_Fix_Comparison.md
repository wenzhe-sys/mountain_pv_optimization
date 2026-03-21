# 模块二 MILP 修复前后对比（r1-r17）

- 数据来源（修复前）: 历史日志快照提取值（原始日志已清理）
- 数据来源（修复后）: data/results/visualization/benchmark_module2.csv

## 汇总
- 回退次数（按旧日志显式文本提取）: 1 -> 0
- 平均总成本(万元): 154.43 -> 184.15 (变化 +29.72)
- 平均求解时间(s): 0.61 -> 0.50 (变化 -0.11)

说明：修复前日志由 PowerShell 管道写入，部分 "MILP Infeasible, falling back to heuristic" 信息未按算例块稳定落盘；
因此修复前回退次数的显式文本统计偏保守，可将其作为下界参考。

## 明细文件
- data/results/visualization/benchmark_module2_before_after.csv