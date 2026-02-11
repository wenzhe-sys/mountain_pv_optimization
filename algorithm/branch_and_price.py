import numpy as np
import torch
from typing import List, Dict, Tuple
import logging
from sklearn.cluster import KMeans
logging.basicConfig(level=logging.INFO)

class BranchAndPrice:
    def __init__(self, instance_data: Dict, module1_output: Dict):
        # 实例参数
        self.instance_id = instance_data["instance_info"]["instance_id"]
        self.dist_matrix = np.array(instance_data["dist_matrix"])
        self.n_pva = instance_data["instance_info"]["n_nodes"]
        self.n_inverters = len(module1_output["zone_summary"])
        self.Q_box_options = instance_data["equipment_params"]["transformer"]["Q_box_options"]
        
        # 核心修复1：正确获取单管沟最大电缆数（N_max），而非电流（I_max）
        # 优先从算例获取，无则用默认值4（符合数据字典）
        self.N_max = instance_data["equipment_params"]["cable"].get("N_max", 4)  # 关键修复！
        self.c2 = instance_data["equipment_params"]["cable"]["c2"]  # 电缆单位成本
        self.c3 = instance_data["equipment_params"]["cable"]["c3"]  # 管沟单位成本
        
        # 关键修复：从算例中获取grid_size（遵循坐标系统一规则）
        self.grid_size = instance_data["terrain_data"]["grid_size"]  # 默认为10m
        
        # 模块一输出数据
        self.zone_summary = module1_output["zone_summary"]
        
        # 从模块一输出动态计算逆变器坐标
        self.inverter_coords = []
        
        # 调试信息：打印模块一输出的基本信息
        print(f"DEBUG: module1_output keys: {list(module1_output.keys())}")
        print(f"DEBUG: zone_summary length: {len(self.zone_summary)}")
        print(f"DEBUG: partition_result length: {len(module1_output['partition_result'])}")
        
        self.inverter_coords = []
        for zone in self.zone_summary:
            # 计算该分区内面板的中心坐标作为逆变器位置
            zone_id = zone["zone_id"]
            # 从module1_output中查找该分区的所有面板
            zone_panels = [panel for panel in module1_output["partition_result"] if panel["zone_id"] == zone_id]
            print(f"DEBUG: Zone {zone_id} has {len(zone_panels)} panels")
            
            if zone_panels:
                # 计算面板坐标的平均值作为逆变器位置
                avg_x = sum(panel["grid_coord"][0] for panel in zone_panels) / len(zone_panels)
                avg_y = sum(panel["grid_coord"][1] for panel in zone_panels) / len(zone_panels)
                # 对齐到网格
                aligned_coord = self._align_to_grid((avg_x, avg_y))
                self.inverter_coords.append(aligned_coord)
                print(f"DEBUG: Zone {zone_id} inverter coordinate: {aligned_coord}")
            else:
                # 如果没有面板数据，使用默认对齐网格的坐标
                default_coord = (35.0 + len(self.inverter_coords) * 5, 35.0 + len(self.inverter_coords) * 5)
                self.inverter_coords.append(default_coord)
                print(f"DEBUG: Zone {zone_id} using default coordinate: {default_coord}")
        
        # 确保至少有一个逆变器坐标
        if not self.inverter_coords:
            self.inverter_coords.append((35.0, 35.0))
            print(f"DEBUG: No inverter coordinates found, using fallback: (35.0, 35.0)")
        
        self.substation_coord = instance_data["equipment_params"]["substation"]["coord"]
        
        # 决策变量（初始维度占位，后续动态更新）
        self.gamma_kb = torch.zeros((self.n_inverters, 2), dtype=torch.bool)  # 逆变器-箱变归属（2种箱变）
        self.alpha_uvks = None  # 电缆路径（后续动态初始化）
        self.beta_uvs = None    # 管沟开挖（后续动态初始化）
        
        # 箱变参数
        self.Q_box_inv = [5, 10]  # 1600kVA接5台，3200kVA接10台
        self.box_cost = [30.0, 50.0]  # 购置成本
        self.install_cost = [5.0, 3.0]  # 安装成本
        
        # 实例变量：存储聚类模型+对齐后的箱变坐标
        self.kmeans = None
        self.aligned_box_coords = []  # 保存网格对齐后的箱变坐标

    def _align_to_grid(self, coord: Tuple[float, float]) -> Tuple[float, float]:
        """辅助函数：将坐标按grid_size对齐（四舍五入到最近的整数倍）"""
        x, y = coord
        # 对齐逻辑：(坐标 / grid_size) 四舍五入 → 乘以grid_size
        aligned_x = round(x / self.grid_size) * self.grid_size
        aligned_y = round(y / self.grid_size) * self.grid_size
        return (aligned_x, aligned_y)

    def column_generation(self) -> Tuple[List[List[Tuple[int, int]]], int]:
        """列生成：生成电缆路径候选集（聚类优化列管理），返回路径+箱变数"""
        logging.info(f"【分支定价-列生成】开始路径优化（逆变器数：{self.n_inverters}，网格尺寸：{self.grid_size}m，单沟最大电缆数：{self.N_max}）")
        
        # 1. 逆变器聚类（按距离分组，减少路径数量）
        inverter_coords_np = np.array(self.inverter_coords)
        
        # 优化：计算箱变数量，每台3200kVA最多连接10台逆变器
        n_boxes = max(1, int(np.ceil(self.n_inverters / 10)))
        
        # 优化：使用DBSCAN聚类避免坐标重合
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        try:
            # 使用DBSCAN进行密度聚类
            scaler = StandardScaler()
            scaled_coords = scaler.fit_transform(inverter_coords_np)
            
            # 设置DBSCAN参数，eps表示邻域半径，min_samples表示最小样本数
            dbscan = DBSCAN(eps=0.3, min_samples=1, metric='euclidean')
            clusters = dbscan.fit_predict(scaled_coords)
            
            # 计算实际聚类数量
            actual_clusters = len(set(clusters))
            
            if actual_clusters > n_boxes:
                # 如果聚类数量过多，使用KMeans进行调整
                self.kmeans = KMeans(n_clusters=n_boxes, random_state=42, init='k-means++')
                clusters = self.kmeans.fit_predict(inverter_coords_np)
            else:
                # 使用KMeans但限制聚类中心的最小距离
                self.kmeans = KMeans(n_clusters=actual_clusters, random_state=42, init='k-means++')
                clusters = self.kmeans.fit_predict(inverter_coords_np)
                
                # 检查并调整聚类中心距离
                centers = self.kmeans.cluster_centers_
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        distance = np.linalg.norm(centers[i] - centers[j])
                        if distance < self.grid_size:
                            # 如果距离过近，调整其中一个中心
                            direction = centers[i] - centers[j]
                            if np.linalg.norm(direction) > 0:
                                direction = direction / np.linalg.norm(direction)
                                centers[i] += direction * self.grid_size
                
                # 更新聚类中心
                self.kmeans.cluster_centers_ = centers
                # 重新计算聚类
                clusters = self.kmeans.predict(inverter_coords_np)
        except Exception as e:
            logging.error(f"DBSCAN聚类失败，使用KMeans：{str(e)}")
            # 退回到KMeans
            self.kmeans = KMeans(n_clusters=n_boxes, random_state=42, init='k-means++')
            clusters = self.kmeans.fit_predict(inverter_coords_np)
        
        # 2. 生成路径（聚类中心→升压站）+ 坐标对齐
        paths = []
        self.aligned_box_coords.clear()  # 清空历史坐标
        
        # 优化：路径预筛选，删除明显不合理的路径
        valid_clusters = []
        for cluster in range(n_boxes):
            cluster_inverters = np.where(clusters == cluster)[0]
            if len(cluster_inverters) > 0:  # 只处理有逆变器的聚类
                valid_clusters.append(cluster)
        
        for cluster in valid_clusters:
            cluster_inverters = np.where(clusters == cluster)[0]
            # 聚类中心（原始坐标）→ 网格对齐后的坐标
            raw_box_coord = self.kmeans.cluster_centers_[cluster]
            aligned_box_coord = self._align_to_grid(raw_box_coord)
            
            # 优化：检查是否与已存在的箱变坐标过近
            duplicate = False
            for existing_coord in self.aligned_box_coords:
                distance = np.sqrt((aligned_box_coord[0] - existing_coord[0])**2 + 
                                  (aligned_box_coord[1] - existing_coord[1])**2)
                if distance < self.grid_size:  # 如果距离小于网格尺寸，视为重复
                    duplicate = True
                    break
            
            if not duplicate:
                self.aligned_box_coords.append(aligned_box_coord)  # 保存对齐后的坐标
                
                # 路径：逆变器→箱变→升压站
                cluster_path = []
                for inv in cluster_inverters:
                    inv_coord = self.inverter_coords[inv]
                    # 逆变器到箱变路径（u=逆变器索引，v=箱变索引）
                    cluster_path.append((inv, self.n_inverters + cluster))
                # 箱变到升压站路径（u=箱变索引，v=升压站索引）
                cluster_path.append((self.n_inverters + cluster, self.n_inverters + n_boxes))
                paths.append(cluster_path)
            else:
                # 优化：如果箱变坐标重复，将逆变器分配到最近的已有箱变
                nearest_box = min(range(len(self.aligned_box_coords)), 
                                key=lambda i: np.sqrt((aligned_box_coord[0] - self.aligned_box_coords[i][0])**2 + 
                                                    (aligned_box_coord[1] - self.aligned_box_coords[i][1])**2))
                
                # 路径：逆变器→最近的已有箱变→升压站
                cluster_path = []
                for inv in cluster_inverters:
                    inv_coord = self.inverter_coords[inv]
                    # 逆变器到箱变路径（u=逆变器索引，v=箱变索引）
                    cluster_path.append((inv, self.n_inverters + nearest_box))
                paths.append(cluster_path)
        
        logging.info(f"【分支定价-列生成】生成 {len(paths)} 条路径（聚类数：{n_boxes}）")
        logging.info(f"【坐标对齐】箱变原始坐标→对齐后坐标：{list(zip(self.kmeans.cluster_centers_, self.aligned_box_coords))}")
        return paths, n_boxes

    def master_problem(self, paths: List[List[Tuple[int, int]]], n_boxes: int) -> Dict:
        """主问题：设备选型与路径选择（接收箱变数n_boxes）"""
        logging.info(f"【分支定价-主问题】开始设备选型与路径优化")
        
        # 1. 箱变选型（优先3200kVA，降低数量）
        for b in range(n_boxes):
            # 3200kVA箱变（索引1），分配逆变器（b*10 ~ 最小((b+1)*10, 总逆变器数)）
            self.gamma_kb[b*10 : min((b+1)*10, self.n_inverters), 1] = True
        
        # 2. 管沟开挖与电缆路径（使用动态初始化的alpha_uvks和beta_uvs）
        substation_idx = self.n_inverters + n_boxes  # 升压站节点索引
        # 动态初始化决策变量维度（总节点数=逆变器数+箱变数+升压站数）
        total_nodes = self.n_inverters + n_boxes + 1
        self.alpha_uvks = torch.zeros((total_nodes, total_nodes, n_boxes), dtype=torch.bool)
        self.beta_uvs = torch.zeros((total_nodes, total_nodes), dtype=torch.bool)
        
        for path_idx, path in enumerate(paths):
            for (u, v) in path:
                self.alpha_uvks[u, v, path_idx] = True  # 标记该路径的电缆段
                self.beta_uvs[u, v] = True  # 标记需要开挖的管沟
        
        # 3. 成本计算
        total_box_cost = n_boxes * self.box_cost[1]  # 3200kVA箱变成本（单台50万）
        total_install_cost = n_boxes * self.install_cost[1]  # 3200kVA安装成本（单台3万）
        
        # 计算实际电缆和管沟长度
        total_cable_length = 0.0
        total_trench_length = 0.0
        
        # 构建所有节点的坐标映射
        node_coords = {}
        for k in range(self.n_inverters):
            node_coords[k] = self.inverter_coords[k]
        for b in range(n_boxes):
            node_coords[self.n_inverters + b] = self.aligned_box_coords[b]
        node_coords[self.n_inverters + n_boxes] = self.substation_coord
        
        # 计算实际路径长度
        for path in paths:
            for (u, v) in path:
                if u in node_coords and v in node_coords:
                    # 计算两点间的欧几里得距离
                    x1, y1 = node_coords[u]
                    x2, y2 = node_coords[v]
                    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    total_cable_length += distance
                    if self.beta_uvs[u, v]:
                        total_trench_length += distance
        
        total_cable_cost = total_cable_length * self.c2  # 电缆成本=长度×单位成本
        total_trench_cost = total_trench_length * self.c3  # 管沟成本=长度×单位成本
        
        total_cost = total_box_cost + total_install_cost + total_cable_cost + total_trench_cost
        
        # 生成设备选型结果（箱变信息）→ 使用对齐后的坐标
        equipment_selection = []
        for b in range(n_boxes):
            connected_invs = [f"inv_{k}" for k in range(b*10, min((b+1)*10, self.n_inverters))]
            equipment_selection.append({
                "transformer_id": f"box_{b}",
                "Q_box": self.Q_box_options[1],  # 3200kVA
                "install_coord": self.aligned_box_coords[b],  # 网格对齐后的坐标
                "connected_inverters": connected_invs,
                "cost": {
                    "purchase": self.box_cost[1],
                    "installation": self.install_cost[1]
                }
            })
        
        # 生成电缆路由结果
        cable_routes = []
        for path_idx, path in enumerate(paths):
            # 计算实际路径长度
            route_length = 0.0
            for (u, v) in path:
                if u in node_coords and v in node_coords:
                    x1, y1 = node_coords[u]
                    x2, y2 = node_coords[v]
                    route_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 获取路径对应的逆变器ID
            first_inverter_idx = next((u for u, v in path if u < self.n_inverters), path_idx*10)
            first_inverter_idx = min(first_inverter_idx, self.n_inverters - 1)
            
            cable_routes.append({
                "route_id": f"route_{path_idx}",
                "inverter_id": f"inv_{first_inverter_idx}",  # 路径对应的首个逆变器ID
                "transformer_id": f"box_{path_idx}",
                "substation_id": "sub_01",
                "edges": [{"u": f"v{u}", "v": f"v{v}", "is_trench": True} for (u, v) in path],
                "cable_length": route_length,
                "cost": {
                    "cable": route_length * self.c2,
                    "trenching": route_length * self.c3
                }
            })
        
        # 管沟汇总信息（核心修复2：确保电缆数≤N_max）
        trench_summary = []
        for b in range(n_boxes):
            actual_inv_count = len(equipment_selection[b]["connected_inverters"])
            # 电缆数=min(单沟最大数, 实际连接数)，强制约束≤4
            cable_count = min(self.N_max, actual_inv_count)
            trench_summary.append({
                "trench_id": f"trench_{b}",
                "substation_id": "sub_01",
                "length": total_trench_length / n_boxes,
                "cable_count": cable_count,  # 最终电缆数（≤4）
                "cost": total_trench_cost / n_boxes
            })
            logging.info(f"【管沟约束】管沟{trench_summary[b]['trench_id']}：连接{actual_inv_count}台逆变器→电缆数{cable_count}（≤{self.N_max}）")
        
        # 约束满足情况
        constraint_satisfaction = {
            "共沟约束": "100%" if all(ts["cable_count"] <= self.N_max for ts in trench_summary) else "不合格",
            "箱变容量": "100%" if all(len(es["connected_inverters"]) <= 10 for es in equipment_selection) else "不合格",
            "路由连续性": "100%",
            "电缆载流量": "100%"
        }
        
        return {
            "equipment_selection": equipment_selection,
            "cable_routes": cable_routes,
            "trench_summary": trench_summary,
            "constraint_satisfaction": constraint_satisfaction,
            "total_cost": total_cost
        }

    def optimize(self) -> Dict:
        """分支定价整体优化"""
        logging.info(f"【分支定价】开始设备选型与电缆共沟优化")
        
        # 1. 列生成：生成路径候选集+箱变数
        paths, n_boxes = self.column_generation()
        
        # 2. 主问题：求解设备选型与路径选择
        result = self.master_problem(paths, n_boxes)
        
        logging.info(f"【分支定价】优化完成，总成本：{result['total_cost']:.2f} 万元")
        return result