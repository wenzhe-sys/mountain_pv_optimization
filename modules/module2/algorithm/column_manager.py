"""
列管理模块
==========
基于聚类的列（路径）管理策略：
- K-Means / DBSCAN 聚类路径
- 各簇内列生成
- 剔除低潜力路径
- 拉格朗日估计评估检验数
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ColumnManager:
    """
    列管理器 — 管理路径池和聚类列生成

    核心策略：
    1. 路径池维护（活跃路径 + 候选路径）
    2. K-Means 聚类路径，按簇管理
    3. 基于拉格朗日估计剔除低潜力路径
    4. 动态调整列数量
    """

    def __init__(self, n_inverters: int, n_sites: int,
                 max_active_paths: int = 500):
        self.n_inverters = n_inverters
        self.n_sites = n_sites
        self.max_active_paths = max_active_paths

        # 路径池
        self.all_paths: List[Dict] = []
        self.active_path_ids: Set[int] = set()
        self.removed_path_ids: Set[int] = set()

        # 路径索引
        self.paths_by_inverter: Dict[int, List[int]] = defaultdict(list)
        self.paths_by_box: Dict[int, List[int]] = defaultdict(list)

        # 聚类信息
        self.clusters: Dict[int, List[int]] = {}  # cluster_id → [path_ids]
        self.path_cluster: Dict[int, int] = {}  # path_id → cluster_id
        self.n_clusters = 0

        # 统计
        self.stats = {
            "total_added": 0,
            "total_removed": 0,
            "cluster_updates": 0,
        }

    def initialize(self, paths: List[Dict], initial_active_ids: Set[int] = None):
        """初始化路径池"""
        self.all_paths = list(paths)
        self._rebuild_indices()

        if initial_active_ids is not None:
            self.active_path_ids = set(initial_active_ids)
        else:
            # 默认：每个逆变器选最短的路径
            self.active_path_ids = set()
            for inv_idx in range(self.n_inverters):
                inv_paths = self.paths_by_inverter.get(inv_idx, [])
                inv_to_box = [pid for pid in inv_paths
                              if self.all_paths[pid]["type"] == "inv_to_box"]
                if inv_to_box:
                    best = min(inv_to_box, key=lambda p: self.all_paths[p]["length"])
                    self.active_path_ids.add(best)

            # 添加所有 box_to_sub 路径
            for p in self.all_paths:
                if p["type"] == "box_to_sub":
                    self.active_path_ids.add(p["id"])

        logger.info(f"【列管理】初始化：{len(self.all_paths)} 条路径，"
                     f"{len(self.active_path_ids)} 条活跃")

    def add_paths(self, new_paths: List[Dict]):
        """添加新路径到池中"""
        for p in new_paths:
            # 检查是否已存在
            if any(ep["inv_idx"] == p["inv_idx"] and ep["box_idx"] == p["box_idx"]
                   and ep["type"] == p["type"]
                   for ep in self.all_paths if ep["id"] != p["id"]):
                continue

            self.all_paths.append(p)
            pid = p["id"]
            self.active_path_ids.add(pid)
            self.stats["total_added"] += 1

        self._rebuild_indices()

    def get_active_paths(self) -> List[Dict]:
        """获取当前活跃路径"""
        return [p for p in self.all_paths if p["id"] in self.active_path_ids]

    def get_inactive_inv_to_box(self) -> List[Dict]:
        """获取未活跃的 inv_to_box 路径"""
        return [p for p in self.all_paths
                if p["id"] not in self.active_path_ids
                and p["id"] not in self.removed_path_ids
                and p["type"] == "inv_to_box"]

    def cluster_paths(self, n_clusters: int = None):
        """
        对 inv_to_box 路径进行 K-Means 聚类

        聚类特征：(inv_coord_x, inv_coord_y, box_coord_x, box_coord_y, length)
        """
        inv_to_box = [p for p in self.all_paths
                      if p["type"] == "inv_to_box"
                      and p["id"] not in self.removed_path_ids]

        if len(inv_to_box) < 3:
            return

        # 构建特征矩阵
        features = []
        path_ids = []
        for p in inv_to_box:
            f_coord = p["from"]
            t_coord = p["to"]
            features.append([f_coord[0], f_coord[1], t_coord[0], t_coord[1], p["length"]])
            path_ids.append(p["id"])

        features = np.array(features)
        # 归一化
        f_min = features.min(axis=0)
        f_max = features.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-10] = 1.0
        features_norm = (features - f_min) / f_range

        # K-Means
        if n_clusters is None:
            n_clusters = max(2, min(len(inv_to_box) // 5, self.n_inverters))

        n_clusters = min(n_clusters, len(inv_to_box))

        # 简单 K-Means
        labels = self._kmeans(features_norm, n_clusters, max_iter=20)

        # 更新聚类信息
        self.clusters.clear()
        self.path_cluster.clear()
        for idx, pid in enumerate(path_ids):
            cid = int(labels[idx])
            if cid not in self.clusters:
                self.clusters[cid] = []
            self.clusters[cid].append(pid)
            self.path_cluster[pid] = cid

        self.n_clusters = len(self.clusters)
        self.stats["cluster_updates"] += 1
        logger.info(f"【列管理】聚类完成：{self.n_clusters} 个簇")

    def prune_low_potential(self, reduced_costs: Dict[int, float],
                            threshold: float = 0.0, keep_ratio: float = 0.8):
        """
        剔除低潜力路径（检验数过大的路径）

        Parameters
        ----------
        reduced_costs : Dict[int, float]
            路径 ID → 检验数
        threshold : float
            检验数阈值，大于此值的路径可能被剔除
        keep_ratio : float
            每个簇内保留路径的比例
        """
        if not self.clusters:
            return

        removed_count = 0
        for cid, pids in self.clusters.items():
            # 按检验数排序
            scored = [(reduced_costs.get(pid, float('inf')), pid) for pid in pids]
            scored.sort()

            # 每个簇至少保留 1 条
            keep_n = max(1, int(len(scored) * keep_ratio))
            for rc, pid in scored[keep_n:]:
                if rc > threshold and pid in self.active_path_ids:
                    self.active_path_ids.discard(pid)
                    self.removed_path_ids.add(pid)
                    removed_count += 1

        if removed_count > 0:
            self.stats["total_removed"] += removed_count
            logger.info(f"【列管理】剔除 {removed_count} 条低潜力路径")

    def ensure_feasibility(self):
        """确保每个逆变器至少有一条活跃的 inv_to_box 路径"""
        for inv_idx in range(self.n_inverters):
            has_active = False
            for pid in self.paths_by_inverter.get(inv_idx, []):
                if pid in self.active_path_ids and self.all_paths[pid]["type"] == "inv_to_box":
                    has_active = True
                    break

            if not has_active:
                # 从所有路径中找该逆变器的最短路径
                inv_paths = [pid for pid in self.paths_by_inverter.get(inv_idx, [])
                             if self.all_paths[pid]["type"] == "inv_to_box"]
                if inv_paths:
                    best = min(inv_paths, key=lambda p: self.all_paths[p]["length"])
                    self.active_path_ids.add(best)
                    self.removed_path_ids.discard(best)

    def _rebuild_indices(self):
        """重建路径索引"""
        self.paths_by_inverter.clear()
        self.paths_by_box.clear()
        for p in self.all_paths:
            pid = p["id"]
            if p["inv_idx"] is not None:
                self.paths_by_inverter[p["inv_idx"]].append(pid)
            if p["box_idx"] is not None:
                self.paths_by_box[p["box_idx"]].append(pid)

    @staticmethod
    def _kmeans(X: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
        """简单 K-Means 实现（避免 sklearn 依赖）"""
        n = len(X)
        # 均匀采样初始中心
        indices = np.linspace(0, n - 1, k, dtype=int)
        centers = X[indices].copy()
        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # 分配
            for i in range(n):
                dists = np.linalg.norm(X[i] - centers, axis=1)
                labels[i] = np.argmin(dists)

            # 更新中心
            new_centers = np.zeros_like(centers)
            counts = np.zeros(k)
            for i in range(n):
                new_centers[labels[i]] += X[i]
                counts[labels[i]] += 1

            for c in range(k):
                if counts[c] > 0:
                    new_centers[c] /= counts[c]
                else:
                    new_centers[c] = centers[c]

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        return labels
