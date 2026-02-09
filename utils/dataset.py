import torch
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data, Dataset

# 解决pandas保存Excel的引擎问题，需提前安装：pip install openpyxl
pd.set_option('io.excel.xlsx.writer', 'openpyxl')
# 设置torch浮点精度，避免数值溢出
torch.set_default_dtype(torch.float32)

# ===================== 关键修改：定义与load_instance.py一致的算例库根路径 =====================
CURRENT_UTILS_DIR = Path(__file__).parent  # utils文件夹绝对路径
PROJECT_ROOT_DIR = CURRENT_UTILS_DIR.parent  # 项目根目录（mountain_pv_optimization）
INSTANCE_LIB_ROOT = PROJECT_ROOT_DIR / "data" / "instance_lib"  # 固定保存到data/instance_lib
# ==============================================================================================


class CMSTDataset(Dataset):
    def __init__(self, data_dir, capacity):
        super().__init__()
        # 转换为绝对路径，解决相对路径混乱问题
        self.data_dir = Path(data_dir).absolute().resolve()  # resolve()消除..路径
        self.capacity = capacity
        
        # 调试：打印数据目录和找到的.dat文件
        print(f"【Dataset调试】数据目录绝对路径：{self.data_dir}")
        self.files = sorted(list(self.data_dir.glob("*.dat")))
        print(f"【Dataset调试】找到的.dat文件数量：{len(self.files)}")
        if self.files:
            print(f"【Dataset调试】找到的文件列表：{[f.name for f in self.files]}")
        
        # 处理文件，捕获异常避免data_list为空
        self.data_list = []
        for f in self.files:
            try:
                data = self._process_file(f)
                self.data_list.append(data)
                print(f"【Dataset调试】成功处理文件：{f.name}")
            except Exception as e:
                print(f"【Dataset调试】处理文件{f.name}失败：{str(e)}")

    def len(self):
        """Dataset基类必需：返回数据集长度"""
        return len(self.data_list)

    def get(self, idx):
        """Dataset基类必需：按索引获取数据"""
        return self.data_list[idx]

    def _get_difficulty(self, n_valid):
        """按有效节点数分级算例难度（符合算例存储规范）"""
        if n_valid <= 50:
            return "easy"
        elif 51 <= n_valid <= 100:
            return "medium"
        else:
            return "hard"

    def _add_standard_fields(self, n_valid):
        """按《数据字典》补充地形/设备/损耗/面板参数，适配三模块输入"""
        # 1. 地形数据（简化算例为平地，全区域可建设）
        grid_size = 10  # 数据字典默认10m×10m网格
        rows = cols = int(np.ceil(np.sqrt(n_valid)))  # 按节点数生成正方形网格
        slope_matrix = np.zeros((rows, cols), dtype=np.float32)  # 坡度=0°（平地）
        buildable_matrix = np.ones((rows, cols), dtype=bool)    # 全区域可建设

        # 2. 设备参数（严格遵循数据字典默认值）
        equipment_params = {
            "inverter": {"q": 320.0, "r": 0.85, "p": int(np.ceil(n_valid / 22))},  # 22块面板/逆变器（符合容量约束）
            "transformer": {
                "Q_box_options": [1600, 3200],  # 仅支持两种箱变容量
                "c_box": {1600: 30.0, 3200: 50.0},  # 购置成本（万元）
                "c_install_box": {1600: 5.0, 3200: 3.0}  # 安装成本（万元）
            },
            "cable": {
                "c1": 15.0, "c2": 35.0, "c3": 200.0,  # 电缆/管沟单位成本
                "rho": 1.72e-8, "r_c": 0.015, "I_max": 200.0  # 电缆电气参数（保留副本，保证设备参数完整）
            },
            "substation": {
                "Q_substation": 50,  # 升压站最大接入逆变器数
                "coord": (rows * grid_size / 2, cols * grid_size / 2)  # 升压站位于网格中心
            }
        }

        # 3. 损耗参数（模块三全生命周期优化必需）【已补充r_c、I_max，满足校验】
        loss_params = {
            "lambda": 0.4, "K_segments": 3, "I_segments": [[0, 20], [20, 35], [35, 50]],
            "T": 25, "tau": 3000, "r_d": 0.08, "C_elec": 0.4,
            "r_c": 0.015, "I_max": 200.0  # 与数据字典一致，满足load_instance校验
        }

        # 4. 面板参数（模块一切割+分区必需）
        pva_params = {
            "D": 12.0, "b": 3.0,  # 面板标准尺寸（长×宽）
            "t_l_options": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],  # 2×整数列切割规格
            "LB": 60.0, "UB": 90.0  # 分区周长上下界（m）
        }

        return {
            "terrain_data": {"grid_size": grid_size, "slope_matrix": slope_matrix.tolist(), "buildable_matrix": buildable_matrix.tolist()},
            "equipment_params": equipment_params,
            "loss_params": loss_params,
            "pva_params": pva_params
        }

    def _add_constraint_labels(self):
        """标注三模块核心约束，与《模块接口协议》完全一致"""
        return {
            "constraint_info": [
                # 模块一：光伏面板切割及分区约束（高优先级为主）
                {"type": "cut_constraint", "value": "2×整数列（2.0/4.0...12.0m）", "priority": "高", "module": "模块一"},
                {"type": "inverter_capacity", "value": [18, 26], "desc": "每分区面板数量", "priority": "高", "module": "模块一"},
                {"type": "connectivity_constraint", "value": True, "desc": "分区内面板连续无孤立", "priority": "高", "module": "模块一"},
                {"type": "perimeter_constraint", "value": [60.0, 90.0], "desc": "分区周长（m）", "priority": "中", "module": "模块一"},
                # 模块二：电气设备+电缆共沟约束（高优先级为主）
                {"type": "trench_max_cables", "value": 4, "desc": "单管沟最大电缆数", "priority": "高", "module": "模块二"},
                {"type": "transformer_capacity", "value": [1600, 3200], "desc": "箱变容量（kVA）", "priority": "高", "module": "模块二"},
                {"type": "substation_capacity", "value": 50, "desc": "升压站最大接入逆变器数", "priority": "中", "module": "模块二"},
                # 模块三：集成优化约束（高优先级）
                {"type": "current_constraint", "value": [0, 200], "desc": "电缆最大允许电流（A）", "priority": "高", "module": "模块三"},
                {"type": "power_loss_constraint", "value": "分段线性化", "desc": "I²R非线性损耗拟合", "priority": "高", "module": "模块三"}
            ]
        }

    def _save_json(self, instance_json, path):
        """按算例规范保存JSON文件，保存到固定路径data/instance_lib"""
        # 解析算例信息
        difficulty = instance_json["instance_info"]["difficulty"]
        instance_id = instance_json["instance_info"]["instance_id"]
        # 构造保存路径：使用固定的INSTANCE_LIB_ROOT，不再用相对路径
        save_dir = INSTANCE_LIB_ROOT / "public" / difficulty
        save_dir.mkdir(parents=True, exist_ok=True)  # 自动创建多级目录
        save_path = save_dir / f"public_{difficulty}_{instance_id}.json"
        # 保存JSON（缩进2，UTF-8编码，确保中文不转义）
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(instance_json, f, indent=2, ensure_ascii=False)
        print(f"【保存】JSON算例文件：{save_path.absolute()}")  # 打印绝对路径，方便验证

    def _save_excel(self, instance_json, path):
        """保存Excel文件，与JSON同路径（data/instance_lib）"""
        # 解析算例信息
        difficulty = instance_json["instance_info"]["difficulty"]
        instance_id = instance_json["instance_info"]["instance_id"]
        # 构造保存路径：使用固定的INSTANCE_LIB_ROOT
        save_dir = INSTANCE_LIB_ROOT / "public" / difficulty
        save_path = save_dir / f"public_{difficulty}_{instance_id}.xlsx"
        # 拆分字段为多个Sheet，避免单Sheet内容过多
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            pd.DataFrame([instance_json["instance_info"]]).to_excel(writer, sheet_name="实例信息", index=False)
            pd.DataFrame(instance_json["constraint_info"]).to_excel(writer, sheet_name="约束标签", index=False)
            pd.DataFrame([instance_json["pva_params"]]).to_excel(writer, sheet_name="面板参数", index=False)
            pd.DataFrame([instance_json["equipment_params"]["inverter"]]).to_excel(writer, sheet_name="逆变器参数", index=False)
            pd.DataFrame([instance_json["equipment_params"]["substation"]]).to_excel(writer, sheet_name="升压站参数", index=False)
            pd.DataFrame([instance_json["loss_params"]]).to_excel(writer, sheet_name="损耗参数", index=False)
            pd.DataFrame(instance_json["dist_matrix"]).to_excel(writer, sheet_name="距离矩阵", index=False)
        print(f"【保存】Excel算例文件：{save_path.absolute()}")  # 打印绝对路径，方便验证

    def _process_file(self, path):
        """核心处理函数：适配单位为米的算例，清洗→标准化→标注→保存"""
        # 1. 读取并解析.dat文件
        txt = path.read_text(encoding="utf-8", errors="ignore")
        # 处理数值粘连问题（通用防护，适配其他开源算例）
        txt = re.sub(r"(\d)1000", r"\1 1000", txt)
        txt = re.sub(r"1000(\d)", r"1000 \1", txt)
        # 提取所有数值并转换为float
        vals = [float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', txt)]
        if not vals:
            raise ValueError(f"文件{path.name}无有效数值，无法解析")

        # 2. 初始解析节点数和容量（按算例格式，前两个值为n和Q）
        n = int(vals[0])
        file_Q = float(vals[1])
        self.capacity = file_Q
        matrix_vals = vals[2:]

        # 3. 补齐/截断矩阵值，保证为n×n维度（通用防护）
        if len(matrix_vals) < n * n:
            print(f"【警告】{path.name}数值不足，自动补0")
            matrix_vals += [0.0] * (n * n - len(matrix_vals))
        else:
            matrix_vals = matrix_vals[:n * n]

        # ========== 核心清洗：适配单位为米，无毫米转米 ==========
        dist_matrix = np.array(matrix_vals, dtype=np.float32).reshape(n, n)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0  # 矩阵对称化（双向路径）
        np.fill_diagonal(dist_matrix, 0.0)  # 替换对角线1000→0（无效值标记）
        dist_matrix[dist_matrix > 500.0] = np.inf  # 过滤异常路径（通用防护）
        # 剔除孤立节点
        valid_nodes = np.where(np.any(dist_matrix != np.inf, axis=1))[0]
        n_valid = len(valid_nodes)
        if n_valid < n:
            dist_matrix = dist_matrix[valid_nodes][:, valid_nodes]
            print(f"【清洗】{path.name}剔除孤立节点：原{n}个→有效{n_valid}个")
        if n_valid == 0:
            raise ValueError(f"文件{path.name}无有效节点，跳过处理")
        # ========================================================

        # 标准化补全+约束标注
        standard_fields = self._add_standard_fields(n_valid)
        constraint_labels = self._add_constraint_labels()

        # 构造符合算例规范的JSON主体
        instance_json = {
            "instance_info": {
                "instance_id": path.stem,  # 算例ID为文件名（无后缀）
                "type": "public",  # 类型为公开简化算例
                "difficulty": self._get_difficulty(n_valid),  # 按节点数分级难度
                "n_nodes": n_valid,
                "unit": "m",  # 明确标注单位为米
                "source": "Luo开源CMST算例（山地光伏适配）",
                "version": "v1.0"
            },
            "terrain_data": standard_fields["terrain_data"],
            "pva_params": standard_fields["pva_params"],
            "equipment_params": standard_fields["equipment_params"],
            "loss_params": standard_fields["loss_params"],
            "dist_matrix": dist_matrix.tolist(),  # 清洗后的距离矩阵（米）
            "constraint_info": constraint_labels["constraint_info"]
        }

        # 保存JSON+Excel双格式文件（保存到固定路径data/instance_lib）
        self._save_json(instance_json, path)
        self._save_excel(instance_json, path)

        # 构造torch_geometric的Data对象（处理np.inf，替换为1e9）
        dist_matrix_torch = dist_matrix.copy()
        # 将np.inf替换为工程大值1e9（避免torch.tensor处理inf报错，代表不可行路径）
        dist_matrix_torch[np.isinf(dist_matrix_torch)] = 1e9
        # 计算归一化系数（排除0和1e9，避免除以0）
        valid_vals = dist_matrix_torch[(dist_matrix_torch != 0) & (dist_matrix_torch != 1e9)]
        max_weight = valid_vals.max() + 1e-5 if len(valid_vals) > 0 else 1.0

        # 构造边索引和边属性
        u, v = np.where(~np.eye(n_valid, dtype=bool))
        edge_index = torch.tensor(np.array([u, v]), dtype=torch.long)
        raw_weights = torch.tensor(dist_matrix_torch[u, v], dtype=torch.float32)
        norm_weights = raw_weights / max_weight  # 边权重归一化，避免数值过大

        # 构造节点需求特征
        demands = torch.ones(n_valid, dtype=torch.float32)
        demands[0] = 0.0  # 根节点需求为0

        # 返回Dataset所需的Data对象（所有字段类型严格校验）
        return Data(
            x_norm=demands.view(-1, 1),          # 归一化节点需求 [n_valid,1]
            edge_index=edge_index,               # 边索引 [2, E]
            edge_attr=norm_weights.view(-1, 1),   # 归一化边权重 [E,1]
            num_nodes=n_valid,                   # 有效节点数（int，非tensor）
            capacity=torch.tensor([self.capacity], dtype=torch.float32),  # 容量 [1]
            demand=demands.view(-1, 1),          # 原始节点需求 [n_valid,1]
            scale_factor=torch.tensor([max_weight], dtype=torch.float32),# 归一化系数 [1]
            dist_matrix=torch.tensor(dist_matrix_torch, dtype=torch.float32) # 距离矩阵 [n_valid,n_valid]
        )


# 测试代码（自动推导data目录，无需手动改路径）
if __name__ == "__main__":
    # 自动获取data目录：utils目录 → 上级目录 → data目录
    CURRENT_DIR = Path(__file__).parent  # utils目录
    DATA_DIR = CURRENT_DIR / "../data"   # 项目根目录的data目录
    # 初始化数据集，默认容量100
    dataset = CMSTDataset(data_dir=DATA_DIR, capacity=100)
    print(f"【测试】数据集加载完成，共{len(dataset)}个算例")
    # 测试获取第一个算例的Data对象
    if len(dataset) > 0:
        sample_data = dataset.get(0)
        print(f"【测试】第一个算例有效节点数：{sample_data.num_nodes}")
        print(f"【测试】第一个算例边数：{sample_data.edge_index.shape[1]}")
        print(f"【测试】第一个算例距离矩阵形状：{sample_data.dist_matrix.shape}")