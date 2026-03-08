"""
启发式回退与 Matheuristic 策略
用于在 MILP 求解过慢或失败时提供高质量的初始化边界(Upper Bound)解或 Fallback解。
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MatheuristicFallback:
    def __init__(self, inverters: List[Dict], candidate_boxes: List[Tuple[float, float]], 
                 Q_box_inv: Dict[int, int], path_factory):
        self.inverters = inverters
        self.candidate_boxes = candidate_boxes
        self.Q_box_inv = Q_box_inv
        self.path_factory = path_factory
        
    def solve_kmeans_heuristic(self) -> Dict:
        """
        基于 K-Means 的聚类分配启发式算法。
        1. 估算所需箱变数量
        2. K-Means 对逆变器聚类
        3. 对每个簇分配容量最匹配的箱变
        4. 生成路径
        """
        logger.info("Executing K-Means Heuristic Fallback...")
        
        n_inv = len(self.inverters)
        max_cap = max(self.Q_box_inv.values())
        k_clusters = max(1, int(np.ceil(n_inv / max_cap)))
        
        coords = np.array([inv["centroid"] for inv in self.inverters])
        
        # 1. Clustering
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        
        # 2. Assignment
        assignment = {}
        active_boxes = set()
        for c in range(k_clusters):
            cluster_invs = [self.inverters[i] for i in range(n_inv) if labels[i] == c]
            center = kmeans.cluster_centers_[c]
            
            # Find nearest candidate box
            best_dist = float('inf')
            best_box_idx = -1
            for b_idx, box_coord in enumerate(self.candidate_boxes):
                dist = np.linalg.norm(np.array(box_coord) - center)
                if dist < best_dist:
                    best_dist = dist
                    best_box_idx = b_idx
                    
            if best_box_idx != -1:
                active_boxes.add(best_box_idx)
                for inv in cluster_invs:
                    assignment[inv["id"]] = best_box_idx
                    
        # 3. Path Generation
        columns = []
        edges_info = {}
        
        for inv_id, b_idx in assignment.items():
            inv = next(i for i in self.inverters if i["id"] == inv_id)
            box_coord = self.candidate_boxes[b_idx]
            
            # Path Generation handling
            path_dict = self.path_factory.generate_path(
                inv["centroid"], box_coord, path_type="inv_to_box", inv_idx=inv_id, box_idx=b_idx
            )
            
            # Keep original path_dict fields, ensuring needed ones exist
            path_dict["id"] = f"heur_{inv_id}_{b_idx}"
            if "inv_idx" not in path_dict: path_dict["inv_idx"] = inv_id
            if "box_idx" not in path_dict: path_dict["box_idx"] = b_idx
            
            columns.append(path_dict)
            for e in path_dict["edges"]:
                # the edge lengths are maintained in self.path_factory.edges
                if e not in edges_info:
                    edges_info[e] = self.path_factory.edges[e]
            
        return {
            "status": "heuristic_success",
            "assignment": assignment,
            "active_boxes": active_boxes,
            "columns": columns,
            "edges_info": edges_info
        }
