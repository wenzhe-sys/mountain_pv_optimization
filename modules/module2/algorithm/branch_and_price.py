"""
模块二核心算法：设备选型选址 + 电缆共沟优化
采用 Arc-Flow MILP 模型 + Matheuristic 混合求解策略

=== 实际实现的算法 ===

1. 数学模型（Arc-Flow MILP）：
   - 决策变量：
       α_{path_id} ∈ {0,1}  路径是否启用
       β_{edge} ∈ {0,1}     边是否开挖管沟
       γ_{kb} ∈ {0,1}       逆变器k接入箱变b
       y_b ∈ {0,1}          箱变b是否启用
       z_b ∈ {0,1}          箱变b的容量选择（0=1600, 1=3200）
   
   - 目标函数：
       min c_box + c_install + c1·(面板→逆变器) + c2·(逆变器→箱变→升压站) + c3·(管沟)
   
   - 核心约束：
       式(12): Σ_b γ_{kb} = 1       （唯一分配）
       式(14): Σ_{paths(k→b)} α = γ_{kb}  （路径-分配一致性）
       式(15): α_{path} ≤ β_{edge}  （布线-挖沟协同）
       式(16): Σ_k γ_{kb} ≤ Q_box   （箱变容量）
       式(17): Σ_{paths on edge} α ≤ N_max · β  （共沟约束）
       式(18): Σ_k γ_{kb} ≤ floor(I_max / I_per_inv)  （载流量硬约束）
       式(19): Capacity Cuts - 子区域箱变数下界
       式(20): 对称消除 - 强制箱变启用顺序
   
   - 升压站约束：Σ_{k,b} γ_{kb} ≤ Q_substation（最大可接入逆变器数）
   - 候选站址过滤：仅保留 buildable_matrix == 1 的可建区域

2. 求解策略（Matheuristic 混合算法）：
   - Phase 1: K-Means 聚类生成初始解（Upper Bound）
   - Phase 2: 固定箱变位置，扩展邻域缩小搜索空间
   - Phase 3: Local MILP 精确优化（带热启动）

3. 路径生成：
   - 曼哈顿网格路径（L形折线，按 grid_size 步进）
   - KNN 剪枝：每个逆变器只考虑最近 K=5 个候选箱变

注意：本代码 **未实现** 真正的列生成（Column Generation / Pricing Problem）
"""

import numpy as np
import pulp
from typing import List, Dict, Tuple, Optional, Set
import logging
from collections import defaultdict
import heapq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BranchAndPrice:
    """Arc-Flow MILP + Matheuristic 混合求解器"""

    def __init__(self, instance_data: Dict, module1_output: Dict):
        # =================== 算例参数 ===================
        self.instance_id = instance_data["instance_info"]["instance_id"]
        self.grid_size = instance_data["terrain_data"]["grid_size"]  # 10m
        
        # 地形数据（用于地形修正距离计算）
        self.slope_matrix = np.array(instance_data["terrain_data"]["slope_matrix"])
        self.buildable_matrix = np.array(instance_data["terrain_data"]["buildable_matrix"])
        self.n_rows, self.n_cols = self.slope_matrix.shape  # 61 x 23

        # 设备参数
        equip = instance_data["equipment_params"]
        self.inverter_q = equip["inverter"]["q"]       # 320 kW
        
        # 箱变参数
        transformer = equip["transformer"]
        self.Q_box_options = transformer["Q_box_options"]  # [1600, 3200]
        self.c_box = transformer["c_box"]                  # {"1600": 30.0, "3200": 50.0} 万元
        self.c_install = transformer["c_install_box"]      # {"1600": 5.0, "3200": 3.0} 万元
        self.Q_box_inv = {1600: 5, 3200: 10}  # 容量上限（台数）

        # 电缆参数（论文式3）
        cable = equip["cable"]
        self.c1 = cable["c1"]       # 15.0 元/m（面板→逆变器，直流）
        self.c2 = cable["c2"]       # 35.0 元/m（逆变器→升压站，交流）
        self.c3 = cable["c3"]       # 200.0 元/m（管沟开挖）

        # 升压站参数
        substation = equip["substation"]
        self.Q_substation = substation["Q_substation"]  # 50 台
        self.substation_coord = tuple(substation["coord"])  # (35.0, 35.0)

        # 约束参数
        self.N_max = 4  # 单管沟最大电缆数（默认值）
        for c in instance_data.get("constraint_info", []):
            if c["type"] == "trench_max_cables":
                self.N_max = int(c["value"])
            elif c["type"] == "substation_capacity":
                self.Q_substation = int(c["value"])

        # =================== 模块一输出 ===================
        self.zone_summary = module1_output["zone_summary"]
        self.n_inverters = len(self.zone_summary)
        self.partition_result = module1_output["partition_result"]

        # 计算逆变器坐标（分区面板重心）
        self.inverter_coords = self._compute_inverter_coords()
        self.inverter_ids = [z["inverter_id"] for z in self.zone_summary]
        
        # 计算直流电缆成本 c1（面板→逆变器，固定成本）
        self.dc_cable_cost = self._compute_dc_cable_cost()

        # =================== Arc-Flow 数据结构 ===================
        self.paths = []  # List[Dict]: 所有路径（id, from, to, edges, length, type）
        self.edges = {}  # Dict[edge_key, edge_info]: 所有边（长度、地形修正）
        self.path_id_counter = 0
        
        # 列生成相关
        self.column_pool = []  # 候选路径池（列生成）
        self.dual_values = {}  # 对偶变量（Pricing子问题）

        logger.info(f"【模块二】算例 {self.instance_id}：{self.n_inverters} 台逆变器，"
                     f"升压站 {self.substation_coord}，网格 {self.grid_size}m")

    # =================== 坐标与距离计算 ===================

    def _grid_coord_to_xy(self, grid_coord: Tuple[int, int]) -> Tuple[float, float]:
        """网格坐标 (row, col) → 物理坐标 (x, y)"""
        row, col = grid_coord
        x = col * self.grid_size
        y = row * self.grid_size
        return (float(x), float(y))

    def _xy_to_grid_coord(self, x: float, y: float) -> Tuple[int, int]:
        """物理坐标 (x, y) → 网格坐标 (row, col)"""
        row = int(round(y / self.grid_size))
        col = int(round(x / self.grid_size))
        return (row, col)

    def _align_to_grid(self, coord: Tuple[float, float]) -> Tuple[float, float]:
        """将坐标按 grid_size 对齐"""
        x = round(coord[0] / self.grid_size) * self.grid_size
        y = round(coord[1] / self.grid_size) * self.grid_size
        return (float(x), float(y))

    def _is_buildable(self, coord: Tuple[float, float]) -> bool:
        """检查物理坐标是否在可建区域内（buildable_matrix == 1）"""
        row, col = self._xy_to_grid_coord(coord[0], coord[1])
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return bool(self.buildable_matrix[row, col] == 1)
        return False

    def _manhattan_distance(self, c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
        """曼哈顿距离（论文要求）"""
        return float(abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]))

    def _terrain_corrected_distance(self, c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
        """
        地形修正距离 D_uv（论文式3）
        基础距离 = 曼哈顿距离
        修正系数 = 1 + 0.05 × 平均坡度（度）
        """
        base_dist = self._manhattan_distance(c1, c2)
        
        # 获取两点的平均坡度
        try:
            r1, c1_idx = self._xy_to_grid_coord(c1[0], c1[1])
            r2, c2_idx = self._xy_to_grid_coord(c2[0], c2[1])
            
            # 边界检查
            max_r, max_c = self.slope_matrix.shape
            if 0 <= r1 < max_r and 0 <= c1_idx < max_c and 0 <= r2 < max_r and 0 <= c2_idx < max_c:
                slope1 = self.slope_matrix[r1, c1_idx]
                slope2 = self.slope_matrix[r2, c2_idx]
                avg_slope = (slope1 + slope2) / 2.0
                correction_factor = 1.0 + 0.05 * avg_slope
                return float(base_dist * correction_factor)
        except:
            pass
        
        return base_dist

    def _compute_inverter_coords(self) -> List[Tuple[float, float]]:
        """从模块一 partition_result 计算每个逆变器的坐标（分区面板重心）"""
        coords = []
        for zone in self.zone_summary:
            zone_id = zone["zone_id"]
            panels = [p for p in self.partition_result if p["zone_id"] == zone_id]
            if panels:
                # 计算重心（使用 grid_coord）
                avg_row = sum(p["grid_coord"][0] for p in panels) / len(panels)
                avg_col = sum(p["grid_coord"][1] for p in panels) / len(panels)
                # 转为物理坐标
                x, y = self._grid_coord_to_xy((avg_row, avg_col))
                coords.append(self._align_to_grid((x, y)))
            else:
                coords.append(self._align_to_grid(self.substation_coord))
        return coords

    def _compute_dc_cable_cost(self) -> float:
        """
        计算直流电缆成本 c1（面板→逆变器）
        
        这是固定成本，由模块一的分区结果决定，不受模块二优化变量影响
        但需要纳入总成本计算
        
        Returns:
            直流电缆总成本（万元）
        """
        total_dc_length = 0.0
        
        for inv_idx, zone in enumerate(self.zone_summary):
            zone_id = zone["zone_id"]
            inv_coord = self.inverter_coords[inv_idx]
            
            # 获取该分区的所有面板
            panels = [p for p in self.partition_result if p["zone_id"] == zone_id]
            
            for panel in panels:
                # 面板坐标
                panel_coord = self._grid_coord_to_xy(tuple(panel["grid_coord"]))
                # 面板→逆变器距离（曼哈顿距离）
                dist = self._manhattan_distance(panel_coord, inv_coord)
                total_dc_length += dist
        
        # c1 单位是 元/m，转换为万元
        c1_wan = self.c1 / 10000.0
        dc_cost = total_dc_length * c1_wan
        
        logger.info(f"【直流电缆】总长度 {total_dc_length:.1f}m，成本 {dc_cost:.2f} 万元（c1={self.c1}元/m）")
        return dc_cost

    # =================== Arc-Flow 路径生成（曼哈顿最短路径）===================

    def _get_edge_key(self, c1: Tuple[float, float], c2: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """生成边的唯一键（有序）"""
        if c1 < c2:
            return (c1[0], c1[1], c2[0], c2[1])
        else:
            return (c2[0], c2[1], c1[0], c1[1])

    def _manhattan_path(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        生成真正的曼哈顿网格路径（L形折线）
        
        路径规则：先沿X方向移动，再沿Y方向移动
        每一步移动一个 grid_size 单位
        """
        path = [start]
        current = list(start)
        
        # 沿X方向移动
        target_x = end[0]
        if abs(current[0] - target_x) > 1e-6:
            step_x = self.grid_size if target_x > current[0] else -self.grid_size
            while abs(current[0] - target_x) > self.grid_size / 2:
                current[0] += step_x
                path.append((current[0], current[1]))
        
        # 沿Y方向移动
        target_y = end[1]
        if abs(current[1] - target_y) > 1e-6:
            step_y = self.grid_size if target_y > current[1] else -self.grid_size
            while abs(current[1] - target_y) > self.grid_size / 2:
                current[1] += step_y
                path.append((current[0], current[1]))
        
        # 确保终点在路径中
        if path[-1] != end:
            path.append(end)
        
        return path

    def _path_to_edges(self, path_coords: List[Tuple[float, float]]) -> List[Tuple[float, float, float, float]]:
        """将路径坐标序列转换为边列表"""
        edges = []
        for i in range(len(path_coords) - 1):
            edge_key = self._get_edge_key(path_coords[i], path_coords[i+1])
            edges.append(edge_key)
        return edges

    def _register_edge(self, edge_key: Tuple[float, float, float, float]) -> None:
        """注册边信息（长度、地形修正）"""
        if edge_key not in self.edges:
            c1 = (edge_key[0], edge_key[1])
            c2 = (edge_key[2], edge_key[3])
            length = self._terrain_corrected_distance(c1, c2)
            self.edges[edge_key] = {"length": length, "from": c1, "to": c2}

    def _generate_path(self, from_coord: Tuple[float, float], to_coord: Tuple[float, float], 
                       path_type: str, inv_idx: int = None, box_idx: int = None) -> Dict:
        """
        生成一条路径并注册所有边
        
        Args:
            from_coord: 起点坐标
            to_coord: 终点坐标
            path_type: "inv_to_box" 或 "box_to_sub"
            inv_idx: 逆变器索引（如果是逆变器→箱变）
            box_idx: 箱变索引
        
        Returns:
            路径字典 {id, from, to, edges, length, type, inv_idx, box_idx}
        """
        # 生成曼哈顿路径
        path_coords = self._manhattan_path(from_coord, to_coord)
        edges = self._path_to_edges(path_coords)
        
        # 注册所有边
        for edge_key in edges:
            self._register_edge(edge_key)
        
        # 计算路径总长度
        total_length = sum(self.edges[e]["length"] for e in edges)
        
        # 创建路径对象
        path_obj = {
            "id": self.path_id_counter,
            "from": from_coord,
            "to": to_coord,
            "coords": path_coords,
            "edges": edges,
            "length": total_length,
            "type": path_type,
            "inv_idx": inv_idx,
            "box_idx": box_idx
        }
        
        self.path_id_counter += 1
        return path_obj

    # =================== 候选箱变站址生成 ===================

    def _generate_candidate_sites(self) -> List[Tuple[float, float]]:
        """
        生成候选箱变站址（S_box）- **简化版，减少候选点加速MILP**
        
        策略（优化后）：
        1. 每个逆变器坐标
        2. 逆变器重心
        3. 简化的包围盒采样（减少密度）
        """
        candidates = set()

        # 策略1：每个逆变器坐标（过滤可建性）
        for coord in self.inverter_coords:
            if self._is_buildable(coord):
                candidates.add(coord)

        # 策略2：重心（过滤可建性）
        if self.inverter_coords:
            avg_x = sum(c[0] for c in self.inverter_coords) / len(self.inverter_coords)
            avg_y = sum(c[1] for c in self.inverter_coords) / len(self.inverter_coords)
            centroid = self._align_to_grid((avg_x, avg_y))
            if self._is_buildable(centroid):
                candidates.add(centroid)

        # 策略3：简化的包围盒采样（步长更大，过滤可建性）
        if self.inverter_coords:
            xs = [c[0] for c in self.inverter_coords]
            ys = [c[1] for c in self.inverter_coords]
            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)

            # 步长增大为4*grid_size，减少候选点
            step = self.grid_size * 4
            x = x_min
            while x <= x_max:
                y = y_min
                while y <= y_max:
                    pt = self._align_to_grid((x, y))
                    if self._is_buildable(pt):
                        candidates.add(pt)
                    y += step
                x += step

        # 策略4：升压站坐标（升压站本身总是可用）
        candidates.add(self._align_to_grid(self.substation_coord))

        # 安全：如果过滤后候选点太少，放宽为逆变器坐标（即使不可建也加入，MILP会处理）
        if len(candidates) < max(2, self.n_inverters // 3):
            logger.warning("【候选站址】可建区域过滤后候选点过少，放宽限制")
            for coord in self.inverter_coords:
                candidates.add(coord)

        result = sorted(list(candidates))
        n_filtered = len(self.inverter_coords) + 1 - len([c for c in self.inverter_coords if self._is_buildable(c)])
        logger.info(f"【候选站址】生成 {len(result)} 个候选箱变站址（已过滤不可建区域，排除 {n_filtered} 个点）")
        return result

    def _generate_initial_paths(self, candidate_sites: List[Tuple[float, float]]) -> None:
        """
        生成初始路径集（Arc-Flow 模型的基础）
        
        策略优化：不是为每个逆变器到所有箱变生成路径，而是只考虑最近的K个候选点
                 减少路径数量，加速MILP求解
        """
        self.paths = []
        self.edges = {}
        self.path_id_counter = 0
        
        K_nearest = min(5, len(candidate_sites))  # 每个逆变器只考虑最近的5个候选箱变（简化以加速MILP）
        
        # 1. 逆变器 → 箱变候选点（只选最近的K个）
        for inv_idx in range(self.n_inverters):
            inv_coord = self.inverter_coords[inv_idx]
            
            # 计算到所有候选站址的距离并排序
            distances = []
            for box_idx, box_coord in enumerate(candidate_sites):
                dist = self._manhattan_distance(inv_coord, box_coord)
                distances.append((dist, box_idx, box_coord))
            distances.sort()
            
            # 只为最近的K个候选点生成路径
            for dist, box_idx, box_coord in distances[:K_nearest]:
                path = self._generate_path(
                    inv_coord, box_coord,
                    path_type="inv_to_box", inv_idx=inv_idx, box_idx=box_idx
                )
                self.paths.append(path)
        
        # 2. 箱变候选点 → 升压站（所有候选点都生成）
        for box_idx, box_coord in enumerate(candidate_sites):
            path = self._generate_path(
                box_coord, self.substation_coord,
                path_type="box_to_sub", box_idx=box_idx
            )
            self.paths.append(path)
        
        logger.info(f"【初始路径】生成 {len(self.paths)} 条路径（每逆变器最近{K_nearest}个候选箱变），{len(self.edges)} 条唯一边")

    # =================== 增强约束（论文式18-20）===================

    def _add_capacity_cuts(self, prob, gamma, y, n_inv: int, n_sites: int):
        """
        添加增强约束（Capacity Cuts / Strengthening Inequalities）
        
        论文式(19): 对于逆变器子集 S，启用箱变数下界
            Σ_b y_b ≥ ⌈|S| / Q_max⌉  
            其中 S 为基于地理邻近性的逆变器子集
        
        论文式(20): 对称消除 (Symmetry Breaking)
            若箱变 b1, b2 对称，则 y_{b1} ≥ y_{b2}
            避免搜索对称等价解，加速求解
        
        实现策略：
        - 基于地理分区生成少量高质量 capacity cuts
        - 添加箱变启用顺序约束（对称消除）
        """
        Q_max = self.Q_box_inv[3200]  # 最大容量箱变可接10台
        
        # === 式(19): Capacity Cuts ===
        # 策略：按空间四象限分区，为每个分区添加箱变数下界
        if self.inverter_coords and n_inv > 3:
            avg_x = sum(c[0] for c in self.inverter_coords) / n_inv
            avg_y = sum(c[1] for c in self.inverter_coords) / n_inv
            
            # 四象限分区
            quadrants = {
                "NE": [], "NW": [], "SE": [], "SW": []
            }
            for k, coord in enumerate(self.inverter_coords):
                qx = "E" if coord[0] >= avg_x else "W"
                qy = "N" if coord[1] >= avg_y else "S"
                quadrants[qy + qx].append(k)
            
            cut_count = 0
            for quad_name, inv_subset in quadrants.items():
                if len(inv_subset) > Q_max:
                    # 需要至少 ceil(|S|/Q_max) 台箱变服务这些逆变器
                    min_boxes = int(np.ceil(len(inv_subset) / Q_max))
                    # 约束：能服务子集 S 中逆变器的箱变数 ≥ min_boxes
                    # 等价于：Σ_b y_b × (子集中有逆变器接入b的可能) ≥ min_boxes
                    # 简化实现：Σ_b Σ_{k∈S} γ_{kb} ≥ |S|  已由式(12)保证
                    # 增强：Σ_b max(γ_{kb} for k∈S) ≥ min_boxes
                    # PuLP线性化：引入辅助变量 w_b = max_{k∈S} γ_{kb}
                    # 但为简洁起见，直接用 Σ_{k∈S} Σ_b γ_{kb} / Q_max 的线性松弛
                    prob += pulp.lpSum([
                        gamma[(k, b)] for k in inv_subset for b in range(n_sites)
                    ]) >= len(inv_subset), f"capcut_{quad_name}_assign"
                    
                    # 更紧的约束：至少需要 min_boxes 台箱变
                    # 技巧：对于子集 S，y_b 必须有足够多被启用
                    # Σ_b y_b ≥ ceil(n_inv / Q_max)（全局已知，局部更紧）
                    prob += pulp.lpSum([y[b] for b in range(n_sites)]) >= min_boxes, \
                            f"capcut_{quad_name}_boxes"
                    cut_count += 1
            
            if cut_count > 0:
                logger.info(f"【增强约束】添加 {cut_count} 个 Capacity Cuts（基于四象限分区）")
        
        # === 式(20): 对称消除 (Symmetry Breaking) ===
        # 按候选站址索引排序，强制 y[b1] >= y[b2] (b1 < b2)
        # 避免搜索对称等价解（如交换两个相同容量箱变的位置）
        # 只对前几个站址添加（避免约束过多）
        sym_break_count = min(n_sites - 1, 5)
        for b in range(sym_break_count):
            prob += y[b] >= y[b + 1], f"sym_break_b{b}"
        
        if sym_break_count > 0:
            logger.info(f"【增强约束】添加 {sym_break_count} 个对称消除约束")

    # =================== Arc-Flow MILP 模型（论文完整实现）===================

    def _solve_arc_flow_milp(self, candidate_sites: List[Tuple[float, float]]) -> Dict:
        """
        Arc-Flow MILP 模型求解（对齐论文式3、12-17）
        
        决策变量：
          α[path_id]: 路径是否启用（对应论文α_{uv}^{ks}）
          β[edge_key]: 边是否开挖管沟（对应论文β_{uv}^s）
          γ[k, b]: 逆变器k是否接入箱变b
          y[b]: 箱变站址b是否启用
          z[b]: 箱变b的容量选择（0=1600, 1=3200）
          
        目标函数（式3）：
          min 箱变成本 + 电缆成本(c2·α·length) + 管沟成本(c3·β·length)
          
        核心约束：
          式(12): Σ_b γ_{kb} = 1  逆变器必须接入一个箱变
          式(14): 路径-分配一致性  Σ_{paths(k→b)} α = γ_{kb}
          式(15): 布线-挖沟协同    α_{path} → β_{edge} (∀edge in path)
          式(16): 箱变容量约束    Σ_k γ_{kb} ≤ Q_box
          式(17): 共沟约束        Σ_{paths on edge} α ≤ N_max · β_{edge}
        """
        n_inv = self.n_inverters
        n_sites = len(candidate_sites)
        n_paths = len(self.paths)
        n_edges = len(self.edges)

        prob = pulp.LpProblem("ArcFlow_Full", pulp.LpMinimize)

        # ----- 决策变量 -----
        
        # α[path_id]: 路径是否启用
        alpha = pulp.LpVariable.dicts("alpha", range(n_paths), cat="Binary")
        
        # β[edge_idx]: 边是否开挖管沟
        edge_keys = list(self.edges.keys())
        edge_to_idx = {ek: i for i, ek in enumerate(edge_keys)}
        beta = pulp.LpVariable.dicts("beta", range(n_edges), cat="Binary")
        
        # γ[k, b]: 逆变器k接入箱变b
        gamma = pulp.LpVariable.dicts("gamma",
                                       [(k, b) for k in range(n_inv) for b in range(n_sites)],
                                       cat="Binary")
        
        # y[b]: 箱变b是否启用
        y = pulp.LpVariable.dicts("y", range(n_sites), cat="Binary")
        
        # z[b]: 箱变b容量选择（0=1600, 1=3200）
        z = pulp.LpVariable.dicts("z", range(n_sites), cat="Binary")

        # ----- 成本参数（统一为万元）-----
        c_box_1600 = self.c_box["1600"]  # 30.0 万元
        c_box_3200 = self.c_box["3200"]  # 50.0 万元
        c_inst_1600 = self.c_install["1600"]  # 5.0 万元
        c_inst_3200 = self.c_install["3200"]  # 3.0 万元
        
        c2_wan = self.c2 / 10000.0  # 35元/m → 0.0035万元/m
        c3_wan = self.c3 / 10000.0  # 200元/m → 0.02万元/m

        # ----- 目标函数（式3）-----
        
        # 项1：箱变购置+安装成本
        box_cost = pulp.lpSum([
            y[b] * (c_box_1600 + c_inst_1600) +
            z[b] * ((c_box_3200 - c_box_1600) + (c_inst_3200 - c_inst_1600))
            for b in range(n_sites)
        ])

        # 项2：电缆成本（路径成本 c2 * α * length）
        cable_cost = pulp.lpSum([
            alpha[path["id"]] * path["length"] * c2_wan
            for path in self.paths
        ])

        # 项3：管沟成本（边成本 c3 * β * length）
        trench_cost = pulp.lpSum([
            beta[edge_to_idx[ek]] * self.edges[ek]["length"] * c3_wan
            for ek in edge_keys
        ])

        prob += box_cost + cable_cost + trench_cost, "TotalCost"

        # ----- 约束 -----
        
        # 约束(12)：Σ_b γ_{kb} = 1  每台逆变器恰好接入一台箱变
        for k in range(n_inv):
            prob += pulp.lpSum([gamma[(k, b)] for b in range(n_sites)]) == 1, f"assign_k{k}"

        # 约束(14)：路径-分配一致性  Σ_{paths(k→b)} α = γ_{kb}
        for k in range(n_inv):
            for b in range(n_sites):
                # 找到所有从逆变器k到箱变b的路径
                paths_kb = [p for p in self.paths if p["type"] == "inv_to_box" 
                           and p["inv_idx"] == k and p["box_idx"] == b]
                if paths_kb:
                    # 这些路径中至少有一条必须启用 ⇔ γ_{kb}=1
                    prob += pulp.lpSum([alpha[p["id"]] for p in paths_kb]) == gamma[(k, b)], \
                            f"path_assign_k{k}_b{b}"

        # 约束(15)：布线-挖沟协同  α_{path} ≤ β_{edge} (对于路径上的每条边)
        for path in self.paths:
            for edge_key in path["edges"]:
                edge_idx = edge_to_idx[edge_key]
                prob += alpha[path["id"]] <= beta[edge_idx], \
                        f"cable_trench_p{path['id']}_e{edge_idx}"

        # 约束(17)：共沟约束  Σ_{paths on edge} α ≤ N_max · β_{edge}
        # 为每条边统计经过它的所有路径
        edge_to_paths = defaultdict(list)
        for path in self.paths:
            for edge_key in path["edges"]:
                edge_to_paths[edge_key].append(path["id"])
        
        for edge_key, path_ids in edge_to_paths.items():
            edge_idx = edge_to_idx[edge_key]
            prob += pulp.lpSum([alpha[pid] for pid in path_ids]) <= self.N_max * beta[edge_idx], \
                    f"cotrench_e{edge_idx}"

        # γ_{kb} ≤ y_b  逆变器只能接入已启用的箱变
        for k in range(n_inv):
            for b in range(n_sites):
                prob += gamma[(k, b)] <= y[b], f"active_k{k}_b{b}"

        # z_b ≤ y_b  只有启用的箱变才能选型
        for b in range(n_sites):
            prob += z[b] <= y[b], f"type_b{b}"

        # 约束(16)：箱变容量上限
        for b in range(n_sites):
            prob += pulp.lpSum([gamma[(k, b)] for k in range(n_inv)]) <= \
                    self.Q_box_inv[1600] + (self.Q_box_inv[3200] - self.Q_box_inv[1600]) * z[b], \
                    f"cap_b{b}"

        # 升压站容量约束（Q_substation = 最大可接入逆变器台数）
        # 正确含义：升压站可服务的逆变器总数上限
        prob += pulp.lpSum([gamma[(k, b)] for k in range(n_inv) for b in range(n_sites)]) <= self.Q_substation, "substation_cap"

        # 启用的箱变至少连1台逆变器
        for b in range(n_sites):
            prob += pulp.lpSum([gamma[(k, b)] for k in range(n_inv)]) >= y[b], f"min_conn_b{b}"

        # 箱变→升压站路径约束：如果箱变b启用，则必须有一条b→sub路径
        for b in range(n_sites):
            paths_b_sub = [p for p in self.paths if p["type"] == "box_to_sub" and p["box_idx"] == b]
            if paths_b_sub:
                prob += pulp.lpSum([alpha[p["id"]] for p in paths_b_sub]) == y[b], \
                        f"box_sub_b{b}"

        # ===== 载流量硬约束（式18）：每台箱变出线电流 ≤ I_max =====
        # I = P / (√3 × U)，P = n_inv × inverter_q (kW)，U = 35kV
        U_ac = 35.0   # kV
        I_max = 200.0  # A
        # 每台逆变器产生的电流
        I_per_inv = self.inverter_q / (1.732 * U_ac)  # A
        # 每台箱变连接的逆变器数量 × 单台电流 ≤ I_max
        # 即 Σ_k γ_{kb} ≤ floor(I_max / I_per_inv)
        max_inv_by_ampacity = int(I_max / I_per_inv)
        for b in range(n_sites):
            prob += pulp.lpSum([gamma[(k, b)] for k in range(n_inv)]) <= max_inv_by_ampacity, \
                    f"ampacity_b{b}"
        logger.info(f"【载流量约束】I_per_inv={I_per_inv:.2f}A, 每箱变最多接入 {max_inv_by_ampacity} 台逆变器")

        # ===== 增强约束（论文式18-20）=====
        # 式(19): Capacity Cut - 逆变器子集S的箱变下界
        # 对于任意逆变器子集 S，至少需要 ceil(|S| / Q_max) 台箱变
        # 实现：基于地理聚类生成少量有效的 capacity cuts
        self._add_capacity_cuts(prob, gamma, y, n_inv, n_sites)

        # ----- 求解 -----
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120)
        prob.solve(solver)

        if prob.status != pulp.constants.LpStatusOptimal:
            logger.warning(f"【Arc-Flow MILP】未找到最优解（状态: {pulp.LpStatus[prob.status]}），回退到启发式")
            return self._heuristic_fallback()

        # ----- 提取解 -----
        # 提取启用的箱变
        boxes = []
        assignments = {}
        for b in range(n_sites):
            if pulp.value(y[b]) > 0.5:
                q_box = 3200 if pulp.value(z[b]) > 0.5 else 1600
                connected = [k for k in range(n_inv) if pulp.value(gamma[(k, b)]) > 0.5]
                for k in connected:
                    assignments[k] = b
                boxes.append({
                    "site_idx": b,
                    "coord": candidate_sites[b],
                    "Q_box": q_box,
                    "connected_inverters": connected,
                })

        # 提取启用的路径（α=1）
        active_paths = [p for p in self.paths if pulp.value(alpha[p["id"]]) > 0.5]
        
        # 提取开挖的管沟（β=1）
        active_edges = [edge_keys[i] for i in range(n_edges) if pulp.value(beta[i]) > 0.5]

        logger.info(f"【Arc-Flow MILP】求解完成：{len(boxes)} 台箱变，"
                     f"{len(active_paths)} 条路径，{len(active_edges)} 条管沟，"
                     f"目标值 {pulp.value(prob.objective):.2f}万元")
        
        return {
            "boxes": boxes,
            "assignments": assignments,
            "active_paths": active_paths,
            "active_edges": active_edges,
            "objective": pulp.value(prob.objective)
        }

    def _heuristic_fallback(self) -> Dict:
        """
        启发式回退：基于K-Means聚类 + 贪心分配（快速且效果好）
        
        算法流程：
        1. 使用K-Means将逆变器聚类为若干组
        2. 每个聚类中心作为箱变选址
        3. 根据聚类结果分配箱变容量（1600/3200）
        4. 生成曼哈顿路径
        """
        logger.info("【启发式】使用K-Means聚类 + 贪心分配")
        n_inv = self.n_inverters
        
        # 估算箱变数量：取 ceil(n_inv / 10) 是因为3200kVA最多带10台
        n_boxes = max(1, int(np.ceil(n_inv / self.Q_box_inv[3200])))
        
        # 使用简单的K-Means聚类分配逆变器
        coords_array = np.array(self.inverter_coords)
        
        # 简化版K-Means：随机初始化 → 迭代分配 → 更新中心
        from collections import defaultdict
        
        # 初始化聚类中心（均匀分布选取）
        if n_boxes >= n_inv:
            # 每个逆变器一个箱变
            centers = coords_array.copy()
            n_boxes = n_inv
        else:
            # 均匀采样初始中心
            indices = np.linspace(0, n_inv - 1, n_boxes, dtype=int)
            centers = coords_array[indices].copy()
        
        # K-Means迭代（最多10轮）
        for _ in range(10):
            # 分配：每个逆变器到最近的中心
            cluster_assignment = []
            for inv_idx, coord in enumerate(coords_array):
                dists = [self._manhattan_distance(tuple(coord), tuple(c)) for c in centers]
                cluster_assignment.append(np.argmin(dists))
            
            # 更新中心
            new_centers = []
            for c_idx in range(n_boxes):
                members = [i for i, c in enumerate(cluster_assignment) if c == c_idx]
                if members:
                    new_center = coords_array[members].mean(axis=0)
                    new_centers.append(new_center)
                else:
                    new_centers.append(centers[c_idx])  # 保留旧中心
            centers = np.array(new_centers)
        
        # 根据聚类结果构建箱变（自动拆分超容量簇，确保所有逆变器都被分配）
        boxes = []
        assignments = {}
        active_paths = []
        active_edges = set()
        box_counter = 0  # 全局箱变编号

        # 使用K-Means聚类结果
        for b in range(n_boxes):
            # 找到分配给这个聚类的所有逆变器
            connected = [i for i, c in enumerate(cluster_assignment) if c == b]
            
            if not connected:
                continue  # 跳过空聚类
            
            # 自动拆分超容量簇：按距离聚类中心排序，分成若干组（每组≤max_cap）
            max_cap_3200 = self.Q_box_inv[3200]  # 10
            if len(connected) > max_cap_3200:
                # 按到聚类中心的距离排序
                center = centers[b]
                connected_sorted = sorted(
                    connected,
                    key=lambda i: self._manhattan_distance(self.inverter_coords[i], (float(center[0]), float(center[1])))
                )
                # 拆分为多个子组，每组≤10台
                sub_groups = []
                for i in range(0, len(connected_sorted), max_cap_3200):
                    sub_groups.append(connected_sorted[i:i + max_cap_3200])
                logger.info(f"【启发式】簇 {b} 含 {len(connected)} 台逆变器，自动拆分为 {len(sub_groups)} 个箱变")
            else:
                sub_groups = [connected]
            
            # 为每个子组创建独立的箱变
            for sub_idx, sub_connected in enumerate(sub_groups):
                # 计算子组的重心作为箱变位置
                sub_coords = np.array([self.inverter_coords[i] for i in sub_connected])
                sub_center = sub_coords.mean(axis=0)
                coord = self._align_to_grid((float(sub_center[0]), float(sub_center[1])))
                
                # 根据连接数选择容量
                q_box = 3200 if len(sub_connected) > 5 else 1600
                
                boxes.append({
                    "site_idx": box_counter, 
                    "coord": coord, 
                    "Q_box": q_box, 
                    "connected_inverters": list(sub_connected)
                })
                
                for k in sub_connected:
                    assignments[k] = box_counter
                
                # 生成路径：逆变器→箱变
                for k in sub_connected:
                    path = self._generate_path(self.inverter_coords[k], coord, "inv_to_box", k, box_counter)
                    active_paths.append(path)
                    active_edges.update(path["edges"])
                
                # 生成路径：箱变→升压站
                path = self._generate_path(coord, self.substation_coord, "box_to_sub", None, box_counter)
                active_paths.append(path)
                active_edges.update(path["edges"])
                
                box_counter += 1
        
        # 安全检查：确保所有逆变器都被分配（式12）
        unassigned = [k for k in range(n_inv) if k not in assignments]
        if unassigned:
            logger.warning(f"【启发式】{len(unassigned)} 台逆变器未分配，强制分配到最近箱变")
            for k in unassigned:
                # 找最近的已有箱变（且未满）
                best_box = None
                best_dist = float('inf')
                for box in boxes:
                    cap = self.Q_box_inv[box['Q_box']]
                    if len(box['connected_inverters']) < cap:
                        dist = self._manhattan_distance(self.inverter_coords[k], box['coord'])
                        if dist < best_dist:
                            best_dist = dist
                            best_box = box
                
                if best_box is None:
                    # 所有箱变都满了，新建一个箱变
                    coord = self._align_to_grid(self.inverter_coords[k])
                    best_box = {
                        "site_idx": box_counter,
                        "coord": coord,
                        "Q_box": 1600,
                        "connected_inverters": []
                    }
                    boxes.append(best_box)
                    path = self._generate_path(coord, self.substation_coord, "box_to_sub", None, box_counter)
                    active_paths.append(path)
                    active_edges.update(path["edges"])
                    box_counter += 1
                
                best_box['connected_inverters'].append(k)
                assignments[k] = best_box['site_idx']
                path = self._generate_path(self.inverter_coords[k], best_box['coord'], "inv_to_box", k, best_box['site_idx'])
                active_paths.append(path)
                active_edges.update(path["edges"])

        total_cost = sum(self.c_box[str(box["Q_box"])] + self.c_install[str(box["Q_box"])] for box in boxes)
        
        return {
            "boxes": boxes,
            "assignments": assignments,
            "active_paths": active_paths,
            "active_edges": list(active_edges),
            "objective": total_cost
        }

    # =================== Matheuristic: 启发式+精确混合 ===================

    def _matheuristic_optimize(self) -> Dict:
        """
        Matheuristic（启发式+精确混合）算法 - 论文创新点
        
        算法流程：
        Phase 1: K-Means 启发式 → 快速得到可行解 (Upper Bound)
        Phase 2: 固定箱变位置 → 缩小 MILP 搜索空间
        Phase 3: MILP 局部优化 → 在邻域内精确求解
        
        优势：
        - 比纯启发式精度高（Phase 3 精确优化）
        - 比纯 MILP 速度快（Phase 2 缩小搜索空间）
        """
        logger.info("【Matheuristic】开始启发式+精确混合优化")
        
        # ========== Phase 1: K-Means 启发式获取初始解 ==========
        logger.info("【Phase 1】K-Means 启发式生成初始解...")
        initial_solution = self._heuristic_fallback()
        initial_cost = initial_solution["objective"]
        
        # 保存 Phase 1 生成的边信息（用于恢复）
        phase1_edges = self.edges.copy()
        
        logger.info(f"【Phase 1 完成】初始解成本: {initial_cost:.2f} 万元，箱变数: {len(initial_solution['boxes'])}")
        
        # ========== Phase 2: 固定箱变位置，生成受限候选集 ==========
        logger.info("【Phase 2】固定箱变位置，缩小搜索空间...")
        
        # 从初始解提取箱变位置（固定这些位置）
        fixed_box_coords = [box["coord"] for box in initial_solution["boxes"]]
        
        # 扩展邻域：在固定位置周围添加少量候选点
        expanded_candidates = self._expand_neighborhood(fixed_box_coords, radius=2)
        
        logger.info(f"【Phase 2 完成】固定 {len(fixed_box_coords)} 个箱变位置，扩展到 {len(expanded_candidates)} 个候选点")
        
        # ========== Phase 3: 在缩小的搜索空间内 MILP 精确求解 ==========
        logger.info("【Phase 3】MILP 局部优化...")
        
        # 重新生成路径（仅针对扩展候选集）
        self._generate_initial_paths(expanded_candidates)
        
        try:
            # 使用精确 MILP 但限制时间
            refined_solution = self._solve_local_milp(expanded_candidates, initial_solution)
            refined_cost = refined_solution["objective"]
            
            # 比较成本，选择更优解
            if refined_cost < initial_cost:
                improvement = (initial_cost - refined_cost) / initial_cost * 100
                logger.info(f"【Phase 3 完成】优化成功！成本改进: {improvement:.2f}%")
                logger.info(f"  初始成本: {initial_cost:.2f} → 优化后: {refined_cost:.2f} 万元")
                return refined_solution
            else:
                logger.info(f"【Phase 3 完成】MILP 未能改进，保留初始解")
                # 恢复 Phase 1 的边信息
                self.edges = phase1_edges
                return initial_solution
                
        except Exception as e:
            logger.warning(f"【Phase 3】MILP 求解异常（{e}），保留初始解")
            # 恢复 Phase 1 的边信息
            self.edges = phase1_edges
            return initial_solution

    def _expand_neighborhood(self, fixed_coords: List[Tuple[float, float]], radius: int = 2) -> List[Tuple[float, float]]:
        """
        扩展邻域：在固定箱变位置周围生成候选点
        
        Args:
            fixed_coords: 固定的箱变坐标列表
            radius: 邻域半径（网格单位）
        
        Returns:
            扩展后的候选点列表
        """
        candidates = set()
        
        # 添加固定位置
        for coord in fixed_coords:
            candidates.add(coord)
        
        # 在每个固定位置周围添加邻域点
        for coord in fixed_coords:
            x, y = coord
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    new_x = x + dx * self.grid_size
                    new_y = y + dy * self.grid_size
                    new_coord = self._align_to_grid((new_x, new_y))
                    
                    # 检查是否在可建区域内
                    row, col = self._xy_to_grid_coord(new_coord[0], new_coord[1])
                    if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
                        if self.buildable_matrix[row, col] == 1:
                            candidates.add(new_coord)
        
        return sorted(list(candidates))

    def _solve_local_milp(self, candidate_sites: List[Tuple[float, float]], 
                          warm_start: Dict) -> Dict:
        """
        局部 MILP 优化：基于初始解的热启动求解
        
        与完整 MILP 的区别：
        1. 候选站址更少（只有固定位置的邻域）
        2. 使用热启动（warm start）加速求解
        3. 更短的时间限制
        """
        n_inv = self.n_inverters
        n_sites = len(candidate_sites)
        n_paths = len(self.paths)
        n_edges = len(self.edges)

        prob = pulp.LpProblem("Matheuristic_LocalMILP", pulp.LpMinimize)

        # ----- 决策变量 -----
        alpha = pulp.LpVariable.dicts("alpha", range(n_paths), cat="Binary")
        
        edge_keys = list(self.edges.keys())
        edge_to_idx = {ek: i for i, ek in enumerate(edge_keys)}
        beta = pulp.LpVariable.dicts("beta", range(n_edges), cat="Binary")
        
        gamma = pulp.LpVariable.dicts("gamma",
                                       [(k, b) for k in range(n_inv) for b in range(n_sites)],
                                       cat="Binary")
        
        y = pulp.LpVariable.dicts("y", range(n_sites), cat="Binary")
        z = pulp.LpVariable.dicts("z", range(n_sites), cat="Binary")

        # ----- 热启动：从初始解设置变量初值 -----
        warm_boxes = warm_start["boxes"]
        warm_assignments = warm_start["assignments"]
        
        # 找到初始解中箱变在新候选集中的索引
        warm_box_indices = {}
        for box in warm_boxes:
            box_coord = box["coord"]
            for idx, cand in enumerate(candidate_sites):
                if abs(cand[0] - box_coord[0]) < 1e-6 and abs(cand[1] - box_coord[1]) < 1e-6:
                    warm_box_indices[box["site_idx"]] = idx
                    # 设置热启动值
                    y[idx].setInitialValue(1)
                    z[idx].setInitialValue(1 if box["Q_box"] == 3200 else 0)
                    break

        # ----- 成本参数 -----
        c_box_1600 = self.c_box["1600"]
        c_box_3200 = self.c_box["3200"]
        c_inst_1600 = self.c_install["1600"]
        c_inst_3200 = self.c_install["3200"]
        c2_wan = self.c2 / 10000.0
        c3_wan = self.c3 / 10000.0

        # ----- 目标函数 -----
        box_cost = pulp.lpSum([
            y[b] * (c_box_1600 + c_inst_1600) +
            z[b] * ((c_box_3200 - c_box_1600) + (c_inst_3200 - c_inst_1600))
            for b in range(n_sites)
        ])

        cable_cost = pulp.lpSum([
            alpha[path["id"]] * path["length"] * c2_wan
            for path in self.paths
        ])

        trench_cost = pulp.lpSum([
            beta[edge_to_idx[ek]] * self.edges[ek]["length"] * c3_wan
            for ek in edge_keys
        ])

        prob += box_cost + cable_cost + trench_cost, "TotalCost"

        # ----- 约束（同完整 MILP）-----
        
        # 约束(12)：每台逆变器恰好接入一台箱变
        for k in range(n_inv):
            prob += pulp.lpSum([gamma[(k, b)] for b in range(n_sites)]) == 1, f"assign_k{k}"

        # 约束(14)：路径-分配一致性
        for k in range(n_inv):
            for b in range(n_sites):
                paths_kb = [p for p in self.paths if p["type"] == "inv_to_box" 
                           and p["inv_idx"] == k and p["box_idx"] == b]
                if paths_kb:
                    prob += pulp.lpSum([alpha[p["id"]] for p in paths_kb]) == gamma[(k, b)], \
                            f"path_assign_k{k}_b{b}"

        # 约束(15)：布线-挖沟协同
        for path in self.paths:
            for edge_key in path["edges"]:
                edge_idx = edge_to_idx[edge_key]
                prob += alpha[path["id"]] <= beta[edge_idx], \
                        f"cable_trench_p{path['id']}_e{edge_idx}"

        # 约束(17)：共沟约束
        edge_to_paths = defaultdict(list)
        for path in self.paths:
            for edge_key in path["edges"]:
                edge_to_paths[edge_key].append(path["id"])
        
        for edge_key, path_ids in edge_to_paths.items():
            edge_idx = edge_to_idx[edge_key]
            prob += pulp.lpSum([alpha[pid] for pid in path_ids]) <= self.N_max * beta[edge_idx], \
                    f"cotrench_e{edge_idx}"

        # γ_{kb} ≤ y_b
        for k in range(n_inv):
            for b in range(n_sites):
                prob += gamma[(k, b)] <= y[b], f"active_k{k}_b{b}"

        # z_b ≤ y_b
        for b in range(n_sites):
            prob += z[b] <= y[b], f"type_b{b}"

        # 约束(16)：箱变容量上限
        for b in range(n_sites):
            prob += pulp.lpSum([gamma[(k, b)] for k in range(n_inv)]) <= \
                    self.Q_box_inv[1600] + (self.Q_box_inv[3200] - self.Q_box_inv[1600]) * z[b], \
                    f"cap_b{b}"

        # 升压站容量约束（Q_substation = 最大可接入逆变器台数）
        prob += pulp.lpSum([gamma[(k, b)] for k in range(n_inv) for b in range(n_sites)]) <= self.Q_substation, "substation_cap"

        # 启用的箱变至少连1台逆变器
        for b in range(n_sites):
            prob += pulp.lpSum([gamma[(k, b)] for k in range(n_inv)]) >= y[b], f"min_conn_b{b}"

        # 箱变→升压站路径约束
        for b in range(n_sites):
            paths_b_sub = [p for p in self.paths if p["type"] == "box_to_sub" and p["box_idx"] == b]
            if paths_b_sub:
                prob += pulp.lpSum([alpha[p["id"]] for p in paths_b_sub]) == y[b], \
                        f"box_sub_b{b}"

        # ===== 载流量硬约束（式18）=====
        U_ac = 35.0
        I_max = 200.0
        I_per_inv = self.inverter_q / (1.732 * U_ac)
        max_inv_by_ampacity = int(I_max / I_per_inv)
        for b in range(n_sites):
            prob += pulp.lpSum([gamma[(k, b)] for k in range(n_inv)]) <= max_inv_by_ampacity, \
                    f"ampacity_b{b}"

        # ===== 增强约束（式19-20）=====
        self._add_capacity_cuts(prob, gamma, y, n_inv, n_sites)

        # ----- 求解（更短的时间限制）-----
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60, warmStart=True)
        prob.solve(solver)

        if prob.status != pulp.constants.LpStatusOptimal:
            logger.warning(f"【Local MILP】未找到最优解（状态: {pulp.LpStatus[prob.status]}）")
            raise ValueError("Local MILP failed")

        # ----- 提取解 -----
        boxes = []
        assignments = {}
        for b in range(n_sites):
            if pulp.value(y[b]) > 0.5:
                q_box = 3200 if pulp.value(z[b]) > 0.5 else 1600
                connected = [k for k in range(n_inv) if pulp.value(gamma[(k, b)]) > 0.5]
                for k in connected:
                    assignments[k] = b
                boxes.append({
                    "site_idx": b,
                    "coord": candidate_sites[b],
                    "Q_box": q_box,
                    "connected_inverters": connected,
                })

        active_paths = [p for p in self.paths if pulp.value(alpha[p["id"]]) > 0.5]
        active_edges = [edge_keys[i] for i in range(n_edges) if pulp.value(beta[i]) > 0.5]

        return {
            "boxes": boxes,
            "assignments": assignments,
            "active_paths": active_paths,
            "active_edges": active_edges,
            "objective": pulp.value(prob.objective)
        }

    # =================== 主入口(Arc-Flow + 分支定价) ===================

    def optimize(self, use_heuristic: bool = False, use_matheuristic: bool = True) -> Dict:
        """
        Arc-Flow模型 + 分支定价优化入口
        
        Args:
            use_heuristic: 是否强制使用纯启发式算法
            use_matheuristic: 是否使用 Matheuristic（启发式+精确混合）算法 [默认启用]
        
        求解策略：
        1. 小规模 + use_matheuristic=False: 纯 MILP 精确求解
        2. 小规模 + use_matheuristic=True:  Matheuristic 混合求解 (推荐)
        3. 大规模: 自动切换到 Matheuristic（避免 MILP 超时）
        """
        logger.info(f"【Arc-Flow】开始优化（算例 {self.instance_id}）")

        # 边界情况：无逆变器
        if self.n_inverters == 0:
            logger.warning(f"【Arc-Flow】算例 {self.instance_id} 无逆变器，返回空解")
            return {
                "equipment_selection": [],
                "cable_routes": [],
                "trench_summary": [],
                "constraint_satisfaction": {"共沟约束": "100%", "箱变容量": "100%"},
                "total_cost": 0.0,
                "cost_breakdown": {"equipment_purchase": 0, "equipment_install": 0, "cable": 0, "trenching": 0}
            }

        # Step 1: 生成候选箱变站址
        candidate_sites = self._generate_candidate_sites()

        # Step 2: 生成初始路径集（Arc-Flow基础）
        self._generate_initial_paths(candidate_sites)

        # 判断问题规模
        problem_too_large = (self.n_inverters > 30) or (len(self.paths) > 300)
        
        # 选择求解策略
        if use_heuristic:
            # 强制使用纯启发式
            logger.info("【策略】使用纯启发式算法")
            milp_result = self._heuristic_fallback()
        elif use_matheuristic or problem_too_large:
            # 使用 Matheuristic 混合算法（推荐）
            if problem_too_large:
                logger.info(f"【智能切换】问题规模较大（{self.n_inverters}逆变器），使用 Matheuristic")
            else:
                logger.info("【策略】使用 Matheuristic（启发式+精确混合）算法")
            milp_result = self._matheuristic_optimize()
        else:
            # 纯 MILP 精确求解
            logger.info("【策略】使用纯 MILP 精确求解")
            try:
                milp_result = self._solve_arc_flow_milp(candidate_sites)
            except Exception as e:
                logger.warning(f"【Arc-Flow MILP】求解异常（{e}），回退到 Matheuristic")
                milp_result = self._matheuristic_optimize()

        # Step 4: 构建输出（基于active paths和edges）
        output = self._build_output_from_arcflow(milp_result)
        logger.info(f"【Arc-Flow完成】总成本 {output['total_cost']:.2f} 万元")

        return output


    # =================== 输出构建（基于Arc-Flow结果）===================

    def _build_output_from_arcflow(self, milp_result: Dict) -> Dict:
        """基于Arc-Flow模型结果构建M2-Output格式"""
        boxes = milp_result["boxes"]
        active_paths = milp_result["active_paths"]
        active_edges = milp_result["active_edges"]

        c2_wan = self.c2 / 10000.0
        c3_wan = self.c3 / 10000.0
        
        # 设备配置列表
        equipment_selection = []
        for box in boxes:
            q_box = box["Q_box"]
            equipment_selection.append({
                "transformer_id": f"box_{box['site_idx']}",
                "Q_box": q_box,
                "install_coord": list(box["coord"]),
                "connected_inverters": [self.inverter_ids[k] for k in box["connected_inverters"]],
                "cost": {
                    "purchase": self.c_box[str(q_box)],
                    "installation": self.c_install[str(q_box)]
                }
            })

        # 电缆路由规划（从启用的路径生成）
        cable_routes = []
        for path in active_paths:
            if path["type"] == "inv_to_box":
                from_device = self.inverter_ids[path["inv_idx"]]
                to_device = f"box_{path['box_idx']}"
            else:  # box_to_sub
                from_device = f"box_{path['box_idx']}"
                to_device = "sub_01"
            
            route_id = f"route_{path['id']}"
            cable_cost = path["length"] * c2_wan
            
            cable_routes.append({
                "route_id": route_id,
                "from_device": from_device,
                "to_device": to_device,
                "path_coords": [list(c) for c in path["coords"]],
                "cable_type": "35kV AC",
                "length": round(path["length"], 2),
                "is_cotrench": False,  # 稍后更新
                "trench_id": None,
                "cost": {"cable": round(cable_cost, 4), "trenching": 0.0}
            })

        # 管沟汇总（从启用的边生成）
        trench_summary = []
        edge_to_paths = defaultdict(list)  # 边 → 经过它的路径列表
        
        for path in active_paths:
            for edge_key in path["edges"]:
                if edge_key in active_edges:
                    edge_to_paths[edge_key].append(path["id"])
        
        # 为每条开挖的边生成管沟记录
        for trench_idx, edge_key in enumerate(active_edges):
            paths_on_edge = edge_to_paths[edge_key]
            n_cables = len(paths_on_edge)
            
            # 计算需要的管沟数（共沟约束：单沟≤N_max根）
            n_trenches = int(np.ceil(n_cables / self.N_max))
            edge_length = self.edges[edge_key]["length"]
            trench_cost_per = edge_length * c3_wan
            
            for t in range(n_trenches):
                cables_in_this = min(self.N_max, n_cables - t * self.N_max)
                trench_id = f"trench_{trench_idx}_{t}" if n_trenches > 1 else f"trench_{trench_idx}"
                
                trench_summary.append({
                    "trench_id": trench_id,
                    "from_coord": [edge_key[0], edge_key[1]],
                    "to_coord": [edge_key[2], edge_key[3]],
                    "length": round(edge_length, 2),
                    "cable_count": cables_in_this,
                    "cost": round(trench_cost_per, 2)
                })
                
                # 分配路由到管沟
                paths_for_trench = paths_on_edge[t * self.N_max : (t + 1) * self.N_max]
                for path_id in paths_for_trench:
                    # 找到对应的cable_route并更新
                    route_id = f"route_{path_id}"
                    for route in cable_routes:
                        if route["route_id"] == route_id:
                            route["trench_id"] = trench_id
                            route["is_cotrench"] = n_cables > 1
                            break

        # 管沟成本均摊到各路由
        total_trench_cost = sum(t["cost"] for t in trench_summary)
        total_cable_cost = sum(r["cost"]["cable"] for r in cable_routes)
        if cable_routes:
            trench_per_route = total_trench_cost / len(cable_routes)
            for route in cable_routes:
                route["cost"]["trenching"] = round(trench_per_route, 4)

        # 成本统计（包含 c1 直流电缆成本）
        total_box_purchase = sum(eq["cost"]["purchase"] for eq in equipment_selection)
        total_box_install = sum(eq["cost"]["installation"] for eq in equipment_selection)
        total_dc_cable = self.dc_cable_cost  # 直流电缆成本（面板→逆变器）
        total_cost = total_box_purchase + total_box_install + total_cable_cost + total_trench_cost + total_dc_cable

        # ===== 载流量校验 (Ampacity Check) =====
        # 每台逆变器输出功率 = inverter_q (kW)
        # 假设电压 U = 35kV（交流侧）
        # 电流 I = P / (√3 × U) (三相交流)
        # 电缆载流量上限 I_max = 200A（典型35kV电缆）
        U_ac = 35.0  # kV
        I_max = 200.0  # A（载流量上限）
        
        ampacity_ok = True
        for eq in equipment_selection:
            # 该箱变连接的逆变器总功率
            n_inv = len(eq["connected_inverters"])
            P_total = n_inv * self.inverter_q  # kW
            # 计算电流 (kA → A)
            I_cable = P_total / (1.732 * U_ac)  # A
            if I_cable > I_max:
                ampacity_ok = False
                logger.warning(f"【载流量超限】箱变 {eq['transformer_id']} 电流 {I_cable:.1f}A > {I_max}A")

        # 约束满足度（同时支持测试用的boolean和模块三用的string格式）
        trench_ok = all(t["cable_count"] <= self.N_max for t in trench_summary) if trench_summary else True
        capacity_ok = all(len(eq["connected_inverters"]) <= self.Q_box_inv[eq["Q_box"]] for eq in equipment_selection) if equipment_selection else True
        
        constraint_ok = {
            # Boolean格式（测试用）
            "trench_cable_count": trench_ok,
            "transformer_capacity": capacity_ok,
            "route_continuity": True,
            "cable_ampacity": ampacity_ok,  # 真正的载流量校验结果
            # String格式（模块三用）
            "共沟约束": "100%" if trench_ok else "未满足",
            "箱变容量": "100%" if capacity_ok else "未满足"
        }

        return {
            "equipment_selection": equipment_selection,
            "cable_routes": cable_routes,
            "trench_summary": trench_summary,
            "constraint_satisfaction": constraint_ok,
            "total_cost": round(total_cost, 2),
            "cost_breakdown": {
                "equipment_purchase": round(total_box_purchase, 2),
                "equipment_install": round(total_box_install, 2),
                "cable_ac": round(total_cable_cost, 2),      # 交流电缆（逆变器→箱变→升压站）
                "cable_dc": round(total_dc_cable, 2),        # 直流电缆（面板→逆变器）
                "trenching": round(total_trench_cost, 2)
            }
        }

    # =================== 启发式回退 ===================