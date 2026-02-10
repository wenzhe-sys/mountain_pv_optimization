"""
模块一：光伏面板切割及分区规划 -- 统一入口

整合 Benders 分解求解器，提供：
  - 端到端优化入口（数据加载 → 求解 → M1-Output 输出）
  - M1-Output 接口校验（E101-E104 错误码）
  - JSON 文件持久化

依据：《算例处理规范与模块接口协议》4.1 节
"""

import json
import os
import logging
from typing import Dict, Optional

from algorithm.benders_decomposition import BendersDecomposition

logger = logging.getLogger(__name__)


class CuttingPartitionModel:
    def __init__(self, instance_path: str):
        """加载算例并初始化模型"""
        self.instance_path = instance_path
        self.instance_data = self.load_instance()
        self.benders_solver = BendersDecomposition(self.instance_data)
        # 使用相对路径构建结果保存路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # 上两级目录（model目录的父目录）
        self.results_path = os.path.join(project_root, "data", "results", "module1")
        os.makedirs(self.results_path, exist_ok=True)

    def _load_instance(self, path: str) -> Dict:
        """加载算例 JSON 文件。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"【模块一】加载算例: {data['instance_info']['instance_id']} "
                     f"(面板数: {data['instance_info']['n_nodes']})")
        return data

    def run(self, verbose: bool = True, max_iter: int = 10) -> Dict:
        """
        运行模块一端到端优化。

        Args:
            verbose: 是否打印日志
            max_iter: Benders 最大迭代次数

        Returns:
            M1-Output 格式的字典
        """
        solver = BendersDecomposition(
            self.instance_data,
            partition_solver=self.partition_solver,
            max_iter=max_iter,
            verbose=verbose
        )

        output = solver.optimize()

        # 校验输出合规性
        self._validate_output(output)

        # 保存结果
        save_path = os.path.join(self.results_path, f"M1-Output_{self.instance_id}.json")
        output_save = {k: v for k, v in output.items() if k != "optimization_history"}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_save, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"\n  结果文件: {save_path}")

        return output

    def _validate_output(self, output: Dict):
        """
        校验 M1-Output 接口合规性。

        依据：《算例处理规范与模块接口协议》4.1 节校验规则
          E101: 必选字段缺失
          E102: 切割长度不符合整数列约束
          E103: 分区面板数超出 18-26 范围
          E104: ID 重复
        """
        # E101: 必选字段检查
        required_fields = ["instance_id", "cut_result", "partition_result",
                           "zone_summary", "constraint_satisfaction"]
        for field in required_fields:
            if field not in output:
                raise ValueError(f"M1-Output 缺失必选字段: {field}（错误码 E101）")

        # E102: 切割长度检查（spec_l 必须是 2.0 的整数倍）
        valid_specs = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        for material in output["cut_result"]:
            if material.get("is_used"):
                for cut in material.get("cuts", []):
                    spec_l = cut.get("spec_l", 0)
                    if spec_l not in valid_specs:
                        logger.warning(f"切割长度 {spec_l} 不在合法规格中（错误码 E102）")

        # E103: 分区面板数检查
        for zone in output["zone_summary"]:
            pva_count = zone.get("pva_count", 0)
            if not (18 <= pva_count <= 26):
                logger.warning(f"分区 {zone['zone_id']} 面板数 {pva_count} "
                                f"超出 [18,26] 范围（错误码 E103）")

        # E104: ID 唯一性检查
        panel_ids = [p["panel_id"] for p in output["partition_result"]]
        if len(panel_ids) != len(set(panel_ids)):
            logger.warning("面板 ID 存在重复（错误码 E104）")

        zone_ids = [z["zone_id"] for z in output["zone_summary"]]
        if len(zone_ids) != len(set(zone_ids)):
            logger.warning("分区 ID 存在重复（错误码 E104）")

        inverter_ids = [z["inverter_id"] for z in output["zone_summary"]]
        if len(inverter_ids) != len(set(inverter_ids)):
            logger.warning("逆变器 ID 存在重复（错误码 E104）")


def validate_m1_output(output: Dict) -> Dict:
    """
    独立的 M1-Output 校验函数（供其他模块调用）。

    Returns:
        校验结果: {"is_valid": bool, "errors": List[str], "warnings": List[str]}
    """
    errors = []
    warnings = []

    # E101
    for field in ["instance_id", "cut_result", "partition_result",
                   "zone_summary", "constraint_satisfaction"]:
        if field not in output:
            errors.append(f"E101: 缺失字段 {field}")

    if errors:
        return {"is_valid": False, "errors": errors, "warnings": warnings}

    # E102
    valid_specs = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0}
    for mat in output["cut_result"]:
        if mat.get("is_used"):
            for cut in mat.get("cuts", []):
                if cut.get("spec_l") not in valid_specs:
                    errors.append(f"E102: 非法切割长度 {cut.get('spec_l')}")

    # E103
    for zone in output["zone_summary"]:
        n = zone.get("pva_count", 0)
        if not (18 <= n <= 26):
            warnings.append(f"E103: 分区 {zone['zone_id']} 面板数 {n}")

    # E104
    panel_ids = [p["panel_id"] for p in output["partition_result"]]
    if len(panel_ids) != len(set(panel_ids)):
        errors.append("E104: 面板 ID 重复")

    zone_ids = [z["zone_id"] for z in output["zone_summary"]]
    if len(zone_ids) != len(set(zone_ids)):
        errors.append("E104: 分区 ID 重复")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
