import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import xlwt

class PVDataPreprocessor:
    def __init__(self):
        # 项目路径配置（符合算例存储规范）
        self.raw_pv_path = r"C:\mountain_pv_optimization\data\raw\PV\real"
        self.processed_pv_path = r"C:\mountain_pv_optimization\data\processed\PV\public\easy"
        self.summary_path = r"C:\mountain_pv_optimization\data\raw\PV\real\summary.csv"
        
        # 固定参数（严格遵循数据字典）
        self.common_params = {
            "grid_size": 10,
            "slope_max": 25.0,
            "b": 3.0,  # 面板宽度
            "D": 12.0,  # 标准面板长度
            "q": 320.0,  # 逆变器容量
            "r": 0.85,  # 最小负载率
            "Q_box_options": [1600, 3200],
            "c1": 15.0, "c2": 35.0, "c3": 200.0,
            "N_max": 4,  # 单沟最大电缆数
            "Q_substation": 50,
            "LB": 60.0, "UB": 90.0
        }
        
        # 新增：模块三必需的linear_params默认值（与用户验证的一致）
        self.default_linear_params = [
            {"a_i": 0.0, "b_i": 0.0},
            {"a_i": 45.0, "b_i": -250.0},
            {"a_i": 80.0, "b_i": -1575.0}
        ]
        
        # 创建输出目录（确保符合目录结构规范）
        os.makedirs(self.processed_pv_path, exist_ok=True)

    def read_raw_file(self, file_name: str) -> Tuple[List[Dict], Tuple[float, float]]:
        """读取raw目录下的PVA原始文件，修复逆变器行解析逻辑"""
        pva_list = []
        inverter_coord = (0.0, 0.0)
        file_path = os.path.join(self.raw_pv_path, file_name)
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                if "inverter" in line.lower():
                    # 修复逻辑：先提取冒号后的坐标部分，再分割逗号
                    try:
                        # 截取冒号后的内容（格式：" 172460 , 481000"）
                        coord_part = line.split(":")[-1].strip()
                        # 按逗号分割坐标（处理空格）
                        x_str, y_str = [part.strip() for part in coord_part.split(",") if part.strip()]
                        # 毫米转米（符合单位统一规范）
                        x = float(x_str) / 1000.0
                        y = float(y_str) / 1000.0
                        inverter_coord = (x, y)
                        print(f"【解析成功】逆变器坐标：({x:.2f}m, {y:.2f}m)")
                    except Exception as e:
                        raise ValueError(f"逆变器坐标解析失败：{line}，错误信息：{str(e)}")
                else:
                    # 解析PVA坐标（兼容原始格式）
                    try:
                        parts = [part.strip() for part in line.split(",") if part.strip()]
                        if len(parts) < 3:
                            raise ValueError(f"PVA行格式错误（需3列）：{line}")
                        pva_id = f"pva_{parts[0]}"
                        x = float(parts[1]) / 1000.0
                        y = float(parts[2]) / 1000.0
                        pva_list.append({
                            "panel_id": pva_id,
                            "x": x,
                            "y": y
                        })
                    except Exception as e:
                        raise ValueError(f"PVA坐标解析失败：{line}，错误信息：{str(e)}")
        
        # 校验数据完整性（符合算例验证指标）
        if not pva_list:
            raise ValueError(f"文件{file_name}未解析到PVA数据")
        if inverter_coord == (0.0, 0.0):
            raise ValueError(f"文件{file_name}未解析到有效逆变器坐标")
        
        return pva_list, inverter_coord

    def calculate_dist_matrix(self, pva_list: List[Dict], inverter_coord: Tuple[float, float]) -> List[List[float]]:
        """计算距离矩阵（PVA间+PVA到逆变器，符合工程常用曼哈顿距离）"""
        all_points = pva_list + [{"x": inverter_coord[0], "y": inverter_coord[1]}]
        n = len(all_points)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    dist_matrix[i][j] = 0.0
                else:
                    dx = abs(all_points[i]["x"] - all_points[j]["x"])
                    dy = abs(all_points[i]["y"] - all_points[j]["y"])
                    dist_matrix[i][j] = dx + dy  # 曼哈顿距离
        return dist_matrix.tolist()

    def grid_mapping(self, x: float, y: float) -> Tuple[int, int]:
        """坐标映射为网格索引（严格遵循坐标系统一规则）"""
        col = int(round(x / self.common_params["grid_size"]))
        row = int(round(y / self.common_params["grid_size"]))
        return (row, col)

    def generate_terrain_data(self, pva_list: List[Dict]) -> Dict:
        """生成地形数据（默认平地，可扩展山地，符合地形数据格式规范）"""
        xs = [pva["x"] for pva in pva_list]
        ys = [pva["y"] for pva in pva_list]
        max_row = max([self.grid_mapping(x, y)[0] for x, y in zip(xs, ys)]) + 2
        max_col = max([self.grid_mapping(x, y)[1] for x, y in zip(xs, ys)]) + 2
        
        # 坡度矩阵（全0=平地）
        slope_matrix = [[0.0 for _ in range(max_col)] for _ in range(max_row)]
        # 可建设矩阵（全true=无不可建设区域）
        buildable_matrix = [[True for _ in range(max_col)] for _ in range(max_row)]
        
        return {
            "grid_size": self.common_params["grid_size"],
            "slope_matrix": slope_matrix,
            "buildable_matrix": buildable_matrix,
            "rows": max_row,
            "cols": max_col
        }

    def generate_constraint_info(self) -> List[Dict]:
        """生成约束信息（显性化标注，符合算例处理规范）"""
        return [
            {"type": "cut_constraint", "value": "2×整数列（2.0/4.0...12.0m）", "priority": "高", "module": "模块一"},
            {"type": "inverter_capacity", "value": [18, 26], "desc": "每分区面板数量", "priority": "高", "module": "模块一"},
            {"type": "connectivity_constraint", "value": True, "desc": "分区内面板连续无孤立", "priority": "高", "module": "模块一"},
            {"type": "perimeter_constraint", "value": [60.0, 90.0], "desc": "分区周长（m）", "priority": "中", "module": "模块一"},
            {"type": "trench_max_cables", "value": 4, "desc": "单管沟最大电缆数", "priority": "高", "module": "模块二"},
            {"type": "transformer_capacity", "value": [1600, 3200], "desc": "箱变容量（kVA）", "priority": "高", "module": "模块二"},
            {"type": "substation_capacity", "value": 50, "desc": "升压站最大接入逆变器数", "priority": "中", "module": "模块二"},
            {"type": "current_constraint", "value": [0, 200], "desc": "电缆最大允许电流（A）", "priority": "高", "module": "模块三"},
            {"type": "power_loss_constraint", "value": "分段线性化", "desc": "I²R非线性损耗拟合", "priority": "高", "module": "模块三"}
        ]

    def generate_excel(self, instance_data: Dict, save_path: str):
        """生成Excel文件（双格式存储要求，便于人工查看）"""
        workbook = xlwt.Workbook(encoding="utf-8")
        
        # Sheet1: 实例信息
        sheet1 = workbook.add_sheet("实例信息")
        info_keys = list(instance_data["instance_info"].keys())
        for i, key in enumerate(info_keys):
            sheet1.write(i, 0, key)
            sheet1.write(i, 1, str(instance_data["instance_info"][key]))
        
        # Sheet2: PVA信息（新增cut_spec列）
        sheet2 = workbook.add_sheet("PVA信息")
        sheet2.write(0, 0, "panel_id")
        sheet2.write(0, 1, "x(m)")
        sheet2.write(0, 2, "y(m)")
        sheet2.write(0, 3, "grid_coord(row,col)")
        sheet2.write(0, 4, "cut_spec(长度×宽度)")  # 新增列
        for i, pva in enumerate(instance_data["pva_list"]):
            sheet2.write(i+1, 0, pva["panel_id"])
            sheet2.write(i+1, 1, round(pva["x"], 2))
            sheet2.write(i+1, 2, round(pva["y"], 2))
            sheet2.write(i+1, 3, str(pva["grid_coord"]))
            sheet2.write(i+1, 4, str(pva["cut_spec"]))  # 写入cut_spec值
        
        # Sheet3: 约束信息
        sheet3 = workbook.add_sheet("约束信息")
        sheet3.write(0, 0, "type")
        sheet3.write(0, 1, "value")
        sheet3.write(0, 2, "priority")
        sheet3.write(0, 3, "module")
        for i, constraint in enumerate(instance_data["constraint_info"]):
            sheet3.write(i+1, 0, constraint["type"])
            sheet3.write(i+1, 1, str(constraint["value"]))
            sheet3.write(i+1, 2, constraint["priority"])
            sheet3.write(i+1, 3, constraint["module"])
        
        workbook.save(save_path)

    def process_single_file(self, file_name: str):
        """处理单个raw文件，生成JSON+Excel（符合算例预处理流程）"""
        try:
            # 1. 读取原始数据（修复后解析逻辑）
            pva_list, inverter_coord = self.read_raw_file(file_name)
            n_nodes = len(pva_list)
            instance_id = file_name.replace(".txt", "")  # 命名规则：public_easy_[instance_id]
            
            # 2. 补充PVA网格坐标+新增cut_spec字段（符合切割约束）
            for pva in pva_list:
                pva["grid_coord"] = self.grid_mapping(pva["x"], pva["y"])
                # 新增cut_spec：切割规格（长度×宽度），长度取2.0m（2×整数列），宽度取固定参数b=3.0m
                pva["cut_spec"] = [2.0, self.common_params["b"]]
            
            # 3. 生成核心数据（符合算例标准化补充要求）
            terrain_data = self.generate_terrain_data(pva_list)
            dist_matrix = self.calculate_dist_matrix(pva_list, inverter_coord)
            constraint_info = self.generate_constraint_info()
            
            # 4. 构建实例JSON数据（核心优化：补充linear_params字段）
            instance_data = {
                "instance_info": {
                    "instance_id": instance_id,
                    "type": "public",
                    "difficulty": "easy",
                    "n_nodes": n_nodes,
                    "inverter_coord": inverter_coord,
                    "unit": "m",
                    "source": "Luo开源PV算例（山地光伏适配）",
                    "version": "v1.1",  # 版本升级：标识已补充linear_params
                    "desensitization_info": {
                        "is_desensitized": False,
                        "note": "公开简化算例，无敏感信息"
                    }
                },
                "pva_list": pva_list,
                "terrain_data": terrain_data,
                "pva_params": {
                    "D": self.common_params["D"],
                    "b": self.common_params["b"],
                    "t_l_options": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                    "LB": self.common_params["LB"],
                    "UB": self.common_params["UB"]
                },
                "equipment_params": {
                    "inverter": {
                        "q": self.common_params["q"],
                        "r": self.common_params["r"],
                        "p": int(np.ceil(n_nodes / 22))  # 逆变器数量（22块/台）
                    },
                    "transformer": {
                        "Q_box_options": self.common_params["Q_box_options"],
                        "c_box": {"1600": 30.0, "3200": 50.0},
                        "c_install_box": {"1600": 5.0, "3200": 3.0}
                    },
                    "cable": {
                        "c1": self.common_params["c1"],
                        "c2": self.common_params["c2"],
                        "c3": self.common_params["c3"],
                        "rho": 1.72e-08,
                        "r_c": 0.015,
                        "I_max": 200.0
                    },
                    "substation": {
                        "Q_substation": self.common_params["Q_substation"],
                        "coord": [35.0, 35.0]  # 默认升压站坐标（可修改）
                    }
                },
                "loss_params": {
                    "lambda": 0.4,
                    "K_segments": 3,
                    "I_segments": [[0, 20], [20, 35], [35, 50]],
                    # 核心优化：自动补充linear_params字段（与用户验证的参数一致）
                    "linear_params": self.default_linear_params,
                    "T": 25,
                    "tau": 3000,
                    "r_d": 0.08,
                    "C_elec": 0.4,
                    "r_c": 0.015,
                    "I_max": 200.0
                },
                "dist_matrix": dist_matrix,
                "constraint_info": constraint_info
            }
            
            # 5. 保存文件（直接覆盖旧版本，确保所有算例格式统一）
            json_save_path = os.path.join(self.processed_pv_path, f"public_easy_{instance_id}.json")
            excel_save_path = os.path.join(self.processed_pv_path, f"public_easy_{instance_id}.xlsx")
            
            with open(json_save_path, "w", encoding="utf-8") as f:
                json.dump(instance_data, f, ensure_ascii=False, indent=2)
            
            self.generate_excel(instance_data, excel_save_path)
            print(f"【预处理完成】生成文件：")
            print(f"  JSON：{json_save_path}")
            print(f"  Excel：{excel_save_path}")
            print(f"  新增/更新字段：")
            print(f"    - instance_info.version（v1.1，标识已补充linear_params）")
            print(f"    - loss_params.linear_params（分段线性化参数）")
            print(f"    - instance_info.desensitization_info（脱敏状态标注）")
            print(f"    - pva_list.cut_spec（切割规格：[2.0m, 3.0m]）")
        
        except Exception as e:
            print(f"【预处理失败】文件{file_name}：{str(e)}")

    def batch_process(self):
        """批量处理所有raw文件（符合批量处理规范）"""
        try:
            raw_files = [f for f in os.listdir(self.raw_pv_path) if f.endswith(".txt") and not f.startswith("summary")]
            if not raw_files:
                print(f"【警告】未在{self.raw_pv_path}找到待处理的.txt文件")
                return
            
            print(f"【开始批量预处理】共发现 {len(raw_files)} 个文件")
            print(f"【优化说明】自动补充 loss_params.linear_params 字段，覆盖旧算例格式")
            for file in raw_files:
                print(f"\n===== 处理文件：{file} =====")
                self.process_single_file(file)
            
            print(f"\n【批量预处理结束】共处理 {len(raw_files)} 个文件，结果保存至：{self.processed_pv_path}")
            print(f"  所有算例已统一包含 linear_params 字段，可直接用于模块三运行")
        except Exception as e:
            print(f"【批量处理失败】：{str(e)}")

if __name__ == "__main__":
    preprocessor = PVDataPreprocessor()
    preprocessor.batch_process()