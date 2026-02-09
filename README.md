# 大型山地光伏电站设计建模与算法研究

## 项目简介
聚焦山地光伏电站设计中的三大核心问题：光伏面板切割及分区规划、电气设备选型选址+电缆共沟、建设成本与电力损耗集成优化，通过运筹优化+强化学习算法实现智能化设计。

## 技术栈
- 编程语言：Python 3.9+
- 核心库：PyTorch（强化学习/优化）、NumPy/Pandas（数据处理）、scikit-learn（聚类）、Matplotlib/Plotly（可视化）
- 版本控制：Git + GitHub
- 开发工具：VS Code / Cursor

## 目录结构说明
| 文件夹/文件         | 作用                                                                 |
|---------------------|----------------------------------------------------------------------|
| data/               | 数据文件夹（只读）：存放原始数据、预处理数据、算法结果                  |
| model/              | 模型定义：各模块数学模型编码（约束、目标函数）                        |
| algorithm/          | 算法实现：Benders分解、分支定价、强化学习等核心算法                  |
| utils/              | 工具函数：数据预处理、可视化、指标计算                              |
| tests/              | 测试用例：各模块单元测试、集成测试                                  |
| main.py             | 主程序入口：调用各模块实现端到端优化                                  |
| requirements.txt    | 依赖库清单：安装命令：`pip install -r requirements.txt`              |
| data_dictionary.md  | 数据字典：建模参数、决策变量、约束条件定义                          |

## 协作规则
1. 分支管理：
   - main：主分支，存放最新稳定代码
   - 个人分支：从最新的 main 分支 checkout 出个人分支（如`feature/cutting-partition-youxi`），开发完成后创建 PR 合并回 main

2. 提交规范：
   - 提交信息格式：`[模块名] 操作：具体描述`
   - 示例：`[模块一] 新增：Benders分解主问题编码`、`[工具类] 修复：地形数据网格转换bug`

3. 协作流程：
   - 开发前：拉取 main 最新代码 → 从 main 创建个人分支 → 开发 → 提交 → 推送个人分支 → 发起 PR 到 main → 审核通过后合并
   - 冲突处理：遇到冲突时，优先本地解决，无法解决则联系相关成员协同处理

4. 注意事项：
   - 禁止上传大文件（如原始DEM数据、大尺寸算例），仅上传小样本测试数据；
   - 代码需遵循编码规范（变量命名、函数注释、目录结构）；
   - 每次提交仅包含相关修改，不提交无关文件（参考.gitignore）。

## 环境搭建步骤
1. 克隆仓库：`git clone https://github.com/你的GitHub用户名/mountain-pv-optimization.git`
2. 进入项目目录：`cd mountain_pv_optimization`
3. 创建虚拟环境：`python -m venv venv`
4. 激活虚拟环境：
   - Windows：`venv\Scripts\activate`
   - Mac/Linux：`source venv/bin/activate`
5. 安装依赖：`pip install -r requirements.txt`
6. 打开VS Code：`code .`（终端中执行，直接打开项目）

## 负责人与分工
- 何显文（统筹+模块三）：集成优化模型开发、仓库权限管理
- 游熙+赵毅飞（模块一）：光伏面板切割及分区规划
- 马金灿+孙尚笪（模块二）：电气设备选型选址+电缆共沟