import json
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.module1.algorithm.benders_decomposition import BendersDecomposition

# 加载算例数据
case_file = "C:\\mountain_pv_optimization\\data\\processed\\PV\\public\\easy\\public_easy_r3.json"
with open(case_file, "r", encoding="utf-8") as f:
    instance_data = json.load(f)

# 初始化求解器（使用DQN）
solver = BendersDecomposition(
    instance_data,
    partition_solver="dqn",  # 使用DQN求解器
    dqn_model_path="C:\\mountain_pv_optimization\\outputs\\checkpoints\\best_model.pt",  # 预训练模型路径
    max_iter=20,  # 最大迭代次数
    epsilon=1.0,  # 收敛阈值
    verbose=True  # 打印详细日志
)

# 执行优化
result = solver.optimize()

# 打印结果摘要
print("\n===== 优化结果摘要 =====")
print(f"算例ID: {result['instance_id']}")
print(f"分区数: {len(result['zone_summary'])}")
print(f"面板分布: {[zone['pva_count'] for zone in result['zone_summary']]}")
if result['zone_summary']:
    print(f"平均周长: {sum(zone['perimeter'] for zone in result['zone_summary']) / len(result['zone_summary']):.2f}m")
else:
    print("平均周长: N/A (无分区)")
print(f"约束满足: {result['constraint_satisfaction']}")
print("=====================")