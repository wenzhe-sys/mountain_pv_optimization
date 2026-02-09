'''
import json
import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

# ===================== 核心修改：修正instance_lib根路径 =====================
# 原路径：项目根/instance_lib → 正确路径：项目根/data/instance_lib
# utils目录 → 项目根目录 → data目录 → instance_lib目录
CURRENT_UTILS_DIR = Path(__file__).parent  # utils文件夹绝对路径
PROJECT_ROOT_DIR = CURRENT_UTILS_DIR.parent  # 项目根目录（mountain_pv_optimization）
INSTANCE_LIB_ROOT = PROJECT_ROOT_DIR / "data" / "instance_lib"  # 正确的算例库根路径
# ===========================================================================

# 必选字段校验清单（基于数据字典，全模块共享）
REQUIRED_FIELDS = [
    "instance_info", "terrain_data", "equipment_params",
    "loss_params", "dist_matrix", "constraint_info"
]
# 子字段校验清单（核心必选）
SUB_REQUIRED_FIELDS = {
    "instance_info": ["instance_id", "type", "difficulty", "n_nodes"],
    "terrain_data": ["grid_size", "slope_matrix", "buildable_matrix"],
    "equipment_params": ["inverter", "transformer", "cable", "substation"],
    "loss_params": ["lambda", "K_segments", "I_segments", "r_c", "I_max"],
    "constraint_info": []  # 仅需非空，无固定子字段
}

def load_instance(instance_id: str) -> Dict:
    """
    按算例ID加载单个标准化算例（JSON格式）
    :param instance_id: 算例ID，如tc40-1/public_easy_tc40-1
    :return: 标准化算例字典
    :raise FileNotFoundError: 算例文件不存在
    :raise ValueError: 算例字段不完整
    """
    # 前置校验：算例库根目录是否存在
    if not INSTANCE_LIB_ROOT.exists():
        raise FileNotFoundError(f"算例库根目录不存在：{INSTANCE_LIB_ROOT.absolute()}")
    
    # 处理ID格式，自动匹配文件（支持tc40-1 / public_easy_tc40-1两种ID）
    target_file_name = f"{instance_id}.json"
    if not instance_id.startswith(("public_", "actual_", "custom_")):
        # 按难度遍历查找public类型的算例
        for difficulty in ["easy", "medium", "hard"]:
            candidate_path = INSTANCE_LIB_ROOT / "public" / difficulty / f"public_{difficulty}_{instance_id}.json"
            if candidate_path.exists():
                target_file_name = candidate_path.name
                break
        else:
            raise FileNotFoundError(f"算例{instance_id}未找到，检查路径：{INSTANCE_LIB_ROOT}/public/[easy/medium/hard]")
    
    # 全局遍历查找目标文件（兼容所有类型/难度）
    file_path = None
    for root, _, files in os.walk(INSTANCE_LIB_ROOT):
        if target_file_name in files:
            file_path = Path(root) / target_file_name
            break
    if not file_path or not file_path.exists():
        raise FileNotFoundError(f"算例JSON文件{target_file_name}未找到，算例库根目录：{INSTANCE_LIB_ROOT.absolute()}")
    
    # 加载并校验算例字段
    with open(file_path, "r", encoding="utf-8") as f:
        instance = json.load(f)
    validate_instance(instance)
    print(f"【算例加载】成功加载{file_path.name}，有效节点数：{instance['instance_info']['n_nodes']}")
    return instance

def load_instance_by_type(
    type: str = "public",
    difficulty: Optional[str] = None
) -> List[Dict]:
    """
    按类型+难度批量加载算例
    :param type: 算例类型，public/actual/custom
    :param difficulty: 难度，easy/medium/hard（仅public类型有效）
    :return: 算例字典列表
    :raise ValueError: 类型/难度不合法
    """
    # 前置校验
    if not INSTANCE_LIB_ROOT.exists():
        raise FileNotFoundError(f"算例库根目录不存在：{INSTANCE_LIB_ROOT.absolute()}")
    if type not in ["public", "actual", "custom"]:
        raise ValueError(f"算例类型{type}不合法，仅支持public/actual/custom")
    
    load_dir = INSTANCE_LIB_ROOT / type
    if not load_dir.exists():
        print(f"【警告】{type}类型算例目录不存在，返回空列表：{load_dir.absolute()}")
        return []
    
    # 按难度过滤（仅public类型）
    if type == "public" and difficulty:
        if difficulty not in ["easy", "medium", "hard"]:
            raise ValueError(f"难度{difficulty}不合法，仅支持easy/medium/hard")
        load_dir = load_dir / difficulty
        if not load_dir.exists():
            print(f"【警告】{type}_{difficulty}目录为空，返回空列表：{load_dir.absolute()}")
            return []
    
    # 加载所有JSON算例并校验
    instance_list = []
    for json_file in load_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            instance = json.load(f)
        if validate_instance(instance, raise_error=False):
            instance_list.append(instance)
            print(f"【批量加载】成功加载：{json_file.name}")
        else:
            print(f"【警告】算例字段不完整，跳过加载：{json_file.name}")
    
    print(f"\n【批量加载汇总】成功加载{type}-{difficulty if difficulty else 'all'}类型算例共{len(instance_list)}个")
    return instance_list

def validate_instance(instance: Dict, raise_error: bool = True) -> bool:
    """
    校验算例字段是否符合数据字典规范
    :param instance: 算例字典
    :param raise_error: 校验失败是否抛出异常（True=抛出，False=返回布尔值）
    :return: 校验结果（True=通过，False=失败）
    :raise ValueError: 字段不完整（raise_error=True时）
    """
    missing_fields = []
    # 一级必选字段校验
    for field in REQUIRED_FIELDS:
        if field not in instance or not instance[field]:
            missing_fields.append(f"一级字段：{field}")
    # 二级必选子字段校验
    for parent_field, sub_fields in SUB_REQUIRED_FIELDS.items():
        if parent_field not in instance:
            continue
        for sub_field in sub_fields:
            if sub_field not in instance[parent_field] or instance[parent_field][sub_field] is None:
                missing_fields.append(f"二级字段：{parent_field}.{sub_field}")
    # 约束信息非空校验
    if "constraint_info" in instance and len(instance["constraint_info"]) == 0:
        missing_fields.append("二级字段：constraint_info（空列表）")
    # 处理校验结果
    if missing_fields:
        err_msg = f"算例字段校验失败，缺失字段：{'; '.join(missing_fields)}"
        if raise_error:
            raise ValueError(err_msg)
        else:
            print(err_msg)
            return False
    return True

def export_instance_to_excel(instance: Dict, save_path: Optional[Path] = None) -> None:
    """
    将算例字典导出为Excel（便于人工查看，补充分段注释）
    :param instance: 算例字典
    :param save_path: 保存路径，默认与JSON同目录
    """
    if save_path is None:
        # 从算例信息中获取路径信息，拼接默认保存路径
        instance_id = instance["instance_info"]["instance_id"]
        difficulty = instance["instance_info"]["difficulty"]
        save_path = INSTANCE_LIB_ROOT / "public" / difficulty / f"{instance_id}_check.xlsx"
    
    # 拆分Sheet存储，与dataset.py的Excel格式一致
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        pd.DataFrame([instance["instance_info"]]).to_excel(writer, sheet_name="实例信息", index=False)
        pd.DataFrame(instance["constraint_info"]).to_excel(writer, sheet_name="约束标签", index=False)
        pd.DataFrame([instance["pva_params"]]).to_excel(writer, sheet_name="面板参数", index=False)
        pd.DataFrame([instance["equipment_params"]["inverter"]]).to_excel(writer, sheet_name="逆变器参数", index=False)
        pd.DataFrame([instance["equipment_params"]["substation"]]).to_excel(writer, sheet_name="升压站参数", index=False)
        pd.DataFrame([instance["loss_params"]]).to_excel(writer, sheet_name="损耗参数", index=False)
        pd.DataFrame(instance["dist_matrix"]).to_excel(writer, sheet_name="距离矩阵", index=False)
    
    print(f"【算例导出】Excel校验文件已保存至：{save_path.absolute()}")

# 测试代码（支持单例加载/批量加载，按需注释）
if __name__ == "__main__":
    try:
        # 测试1：按ID加载单个算例（推荐）
        instance = load_instance("tc40-1")
        
        # 测试2：批量加载public-easy类型所有算例（解开注释即可运行）
        # instance_list = load_instance_by_type(type="public", difficulty="easy")
        
        # 测试3：将加载的算例导出为Excel校验文件（解开注释即可运行）
        # export_instance_to_excel(instance)
        
    except Exception as e:
        print(f"【测试失败】{e}")
'''

import json
import os
from typing import Dict, List, Optional

class InstanceLoader:
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_path = os.path.join(project_root, "data", "processed")

    def load_instance(self, instance_id: str) -> Dict:
        """按算例ID加载标准化JSON算例"""
        # 自动匹配路径（PV/CMST，public/actual）
        instance_file = None
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.startswith(f"public_easy_{instance_id}") and file.endswith(".json"):
                    instance_file = os.path.join(root, file)
                    break
            if instance_file:
                break

        if not instance_file:
            raise FileNotFoundError(f"未找到算例ID：{instance_id}")

        with open(instance_file, "r", encoding="utf-8") as f:
            instance = json.load(f)

        # 加载后自动校验
        self.validate_instance(instance)
        return instance

    def load_instance_by_type(self, type: str = "public", difficulty: str = "easy") -> List[Dict]:
        """按类型+难度批量加载算例"""
        instances = []
        target_path = os.path.join(self.base_path, "PV", type, difficulty)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"路径不存在：{target_path}")

        for file in os.listdir(target_path):
            if file.endswith(".json"):
                with open(os.path.join(target_path, file), "r", encoding="utf-8") as f:
                    instance = json.load(f)
                    if self.validate_instance(instance, silent=True):
                        instances.append(instance)
        return instances

    def validate_instance(self, instance: Dict, silent: bool = False) -> bool:
        """校验算例字段完整性（符合算例处理规范）"""
        required_fields = [
            "instance_info", "pva_list", "terrain_data",
            "pva_params", "equipment_params", "loss_params",
            "dist_matrix", "constraint_info"
        ]

        missing_fields = [f for f in required_fields if f not in instance]
        if missing_fields:
            if not silent:
                raise ValueError(f"算例缺失必选字段：{missing_fields}，错误码E001")
            return False

        # 校验核心参数取值范围
        if instance["equipment_params"]["inverter"]["q"] not in [250, 320, 500]:
            if not silent:
                raise ValueError(f"逆变器容量非法，错误码E002")
            return False

        if instance["loss_params"]["lambda"] < 0.3 or instance["loss_params"]["lambda"] > 0.5:
            if not silent:
                raise ValueError(f"运行成本权重非法，错误码E003")
            return False

        if not silent:
            print(f"【算例校验】{instance['instance_info']['instance_id']} 字段完整，参数合法")
        return True

# 单例模式，供其他模块调用
instance_loader = InstanceLoader()

# 新增顶层函数，支持直接导入调用（核心修复）
def load_instance(instance_id: str) -> Dict:
    return instance_loader.load_instance(instance_id)

def validate_instance(instance: Dict, silent: bool = False) -> bool:
    return instance_loader.validate_instance(instance, silent)

if __name__ == "__main__":
    # 测试加载
    try:
        instance = load_instance("r1")
        print(f"加载成功：{instance['instance_info']['instance_id']}（面板数：{instance['instance_info']['n_nodes']}）")
    except Exception as e:
        print(f"加载失败：{e}")