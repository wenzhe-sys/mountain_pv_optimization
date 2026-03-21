# 模块二总结与模块一/模块三对接文档

## 1. 文档目的
本文件用于阶段性交付，覆盖以下内容：
- 模块二当前实现与验证总结
- 模块二与模块一的数据对接协议
- 模块二与模块三的数据对接协议
- 端到端运行方式与验收清单

## 2. 模块二当前总结

### 2.1 目标与职责
模块二负责电气设备选型选址与电缆共沟优化，输出可被模块三直接消费的 M2-Output 结构。

### 2.2 当前算法实现状态
当前版本已完成并联通以下能力：
- 分支定价主流程：列生成 + 分支定界 + 拉格朗日松弛
- 商业求解器优先选择：GUROBI 优先，其他求解器回退
- 结果格式化与接口校验
- 基准测试脚本与独立约束复核

核心实现位置：
- modules/module2/algorithm/branch_and_price.py
- modules/module2/algorithm/rmp_solver.py
- modules/module2/algorithm/result_formatter.py
- scripts/benchmark_module2.py
- model/model_equipment_cable.py

### 2.3 本轮关键修复
本轮为确保与论文描述一致，完成了以下关键修复：
- 修复 trench 变量建模：beta 改为沟槽数量变量（整数/连续），不再是单纯二元开关
- 修复 fallback 路径激活：仅激活启发式自身路径，不再错误使用整列池
- 增加解状态透明字段：used_fallback、solve_status
- 增加独立约束复核：full_assignment、unique_assignment、substation_capacity、route_nonempty 等
- 修复零长度合法路径误判：长度为 0 的路由允许单点坐标

### 2.4 最新验证结果（GUROBI）
全算例 r1-r17 以 milp 策略运行后：
- solve_status 全部为 optimal
- used_fallback 全部为 False
- constraints_ok_independent 全部为 True

结果文件：
- data/results/visualization/benchmark_module2.csv
- data/results/visualization/benchmark_comparison.png
- test_milp_fallback_summary.csv

## 3. 模块一 -> 模块二 对接

### 3.1 输入来源
- 模块一输出文件：data/results/module1/M1-Output_算例ID.json
- 模块二入口：model/model_equipment_cable.py

### 3.2 模块二对模块一输入的硬校验
模块二在加载模块一输出时执行以下校验：
- 必须存在 instance_id（E101）
- zone_summary 中每个分区 pva_count 必须在 18 到 26（E103）
- inverter_id 必须唯一（E104）

### 3.3 模块二实际使用的模块一字段
- instance_id：算例一致性
- zone_summary：逆变器集合、容量约束与分配目标
- partition_result：计算逆变器重心坐标与路径起点

### 3.4 调用方式
主流程（main.py）中，模块二支持两种输入方式：
- 传 module1_output_path
- 直接传内存对象 module1_output

推荐在端到端流水线中直接传 module1_output，减少中间读写依赖。

## 4. 模块二 -> 模块三 对接

### 4.1 输出落盘位置
- data/results/module2/M2-Output_算例ID.json

### 4.2 模块二输出结构（模块三读取重点）
模块二输出顶层字段：
- instance_id
- module1_output
- equipment_selection
- cable_routes
- trench_summary
- constraint_satisfaction
- total_cost

### 4.3 模块三对模块二输入的硬校验
模块三在 model/model_integration.py 中执行以下校验：
- 顶层字段完整：instance_id、equipment_selection、cable_routes、trench_summary、constraint_satisfaction（E302）
- 共沟约束必须为 100%
- 箱变容量约束必须为 100%

因此，模块二交付给模块三时，至少要保证：
- constraint_satisfaction.共沟约束 = 100%
- constraint_satisfaction.箱变容量 = 100%

## 5. 端到端运行建议

### 5.1 推荐流程
1. 运行模块一，生成 M1-Output
2. 运行模块二，生成 M2-Output
3. 运行模块三，读取 M2-Output 并执行集成优化

### 5.2 复现实验命令（模块二）
建议命令：
venv/Scripts/python.exe scripts/benchmark_module2.py --instances r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 --strategies milp --time-limit 120

## 6. 验收清单

模块一到模块二：
- M1-Output 包含 instance_id 与 zone_summary
- zone_summary.pva_count 全部在 18 到 26
- inverter_id 无重复

模块二自身：
- solve_status 为 optimal
- used_fallback 为 False
- constraints_ok_independent 为 True

模块二到模块三：
- M2-Output 必含模块三要求字段
- constraint_satisfaction 中共沟约束与箱变容量均为 100%

## 7. 风险与后续建议
- 若后续论文需要对比历史结果，建议保留一份固定基线 CSV 作为版本锚点
- 若要进一步提升论文说服力，建议补充：
  - 参数敏感性分析
  - 不同求解器对比（GUROBI 与 CBC）
  - 运行时间分解图（RMP、定价、B&B）
