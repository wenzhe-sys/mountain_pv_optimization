import gurobipy as gp
from gurobipy import GRB

# 禁用 Gurobi 输出日志（新手可先开启，去掉下面这行）
# gp.setParam('OutputFlag', 0)

try:
    # 1. 创建模型
    model = gp.Model("simple_optimization")
    
    # 2. 添加决策变量
    x = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")
    
    # 3. 设置目标函数：max 3x + 5y
    model.setObjective(3*x + 5*y, GRB.MAXIMIZE)
    
    # 4. 添加约束条件
    model.addConstr(2*x + y <= 100, "constraint1")  # 2x + y ≤ 100
    model.addConstr(x + y <= 80, "constraint2")     # x + y ≤ 80
    model.addConstr(x <= 40, "constraint3")         # x ≤ 40
    
    # 5. 求解模型
    model.optimize()
    
    # 6. 输出结果
    if model.Status == GRB.OPTIMAL:
        print("✅ 求解成功！")
        print(f"最优解：x = {x.X:.2f}, y = {y.X:.2f}")
        print(f"最优目标值：{model.ObjVal:.2f}")
    else:
        print(f"❌ 求解失败，状态码：{model.Status}")

except gp.GurobiError as e:
    print(f"Gurobi 错误：{e.errno} - {e.message}")
except Exception as e:
    print(f"其他错误：{str(e)}")
finally:
    # 释放模型资源（好习惯）
    if 'model' in locals():
        model.dispose()