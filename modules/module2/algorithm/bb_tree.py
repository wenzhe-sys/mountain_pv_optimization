"""
分支定界树模块
==============
管理 Branch-and-Price 的搜索树结构。

支持：
- 节点创建与管理
- 最佳优先搜索（Best-First）
- 变量固定（分支策略）
- 剪枝（基于下界/上界）
- 整数性检查
"""

import heapq
import time
from typing import List, Dict, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


class BBNode:
    """分支定界树节点"""

    __slots__ = [
        "node_id", "parent_id", "depth",
        "fixed_to_one", "fixed_to_zero",
        "lower_bound", "upper_bound",
        "lp_solution", "is_integer", "is_pruned",
        "branch_var", "branch_val",
    ]

    def __init__(self, node_id: int, parent_id: int = -1, depth: int = 0):
        self.node_id = node_id
        self.parent_id = parent_id
        self.depth = depth
        self.fixed_to_one: Set[int] = set()   # y_b 固定为 1
        self.fixed_to_zero: Set[int] = set()  # y_b 固定为 0
        self.lower_bound = -float('inf')
        self.upper_bound = float('inf')
        self.lp_solution: Optional[Dict] = None
        self.is_integer = False
        self.is_pruned = False
        self.branch_var: Optional[int] = None
        self.branch_val: Optional[float] = None

    def __lt__(self, other):
        """用于堆排序（最小下界优先）"""
        return self.lower_bound < other.lower_bound


class BranchAndBoundTree:
    """
    分支定界树

    搜索策略：最佳优先（Best-First） — 始终扩展下界最小的节点
    分支变量：选择最分数的 y_b（箱变启用变量）
    剪枝：当节点下界 ≥ 全局上界时剪枝
    """

    def __init__(self, n_sites: int, time_limit: float = 300.0,
                 gap_tolerance: float = 0.05,
                 max_nodes: int = 200):
        self.n_sites = n_sites
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance
        self.max_nodes = max_nodes

        self._node_counter = 0
        self._heap: List[BBNode] = []  # 优先队列
        self.all_nodes: Dict[int, BBNode] = {}

        # 全局界
        self.global_ub = float('inf')
        self.global_lb = -float('inf')
        self.best_integer_solution: Optional[Dict] = None
        self.incumbent_node_id: Optional[int] = None

        # 统计
        self.stats = {
            "nodes_created": 0,
            "nodes_explored": 0,
            "nodes_pruned": 0,
            "nodes_integer": 0,
            "start_time": None,
        }

        # 搜索历史
        self.search_history: List[Dict] = []

    def create_root(self) -> BBNode:
        """创建根节点"""
        root = BBNode(0)
        self._node_counter = 1
        self.all_nodes[0] = root
        heapq.heappush(self._heap, root)
        self.stats["nodes_created"] = 1
        self.stats["start_time"] = time.time()
        return root

    def get_next_node(self) -> Optional[BBNode]:
        """获取下一个待探索节点（最佳优先）"""
        while self._heap:
            node = heapq.heappop(self._heap)
            if node.is_pruned:
                continue
            # 剪枝检查
            if node.lower_bound >= self.global_ub - 1e-6:
                node.is_pruned = True
                self.stats["nodes_pruned"] += 1
                continue
            return node
        return None

    def process_node_result(self, node: BBNode, lp_result: Optional[Dict]) -> str:
        """
        处理节点求解结果

        Returns
        -------
        str : "integer" | "branch" | "infeasible" | "pruned"
        """
        self.stats["nodes_explored"] += 1

        if lp_result is None or lp_result.get("status") != "Optimal":
            node.is_pruned = True
            return "infeasible"

        lb = lp_result["objective"]
        node.lower_bound = lb
        node.lp_solution = lp_result

        # 更新全局下界
        self._update_global_lb()

        # 剪枝
        if lb >= self.global_ub - 1e-6:
            node.is_pruned = True
            self.stats["nodes_pruned"] += 1
            return "pruned"

        # 整数性检查
        y_vals = lp_result.get("y_values", {})
        is_int = self._check_integrality(y_vals)

        if is_int:
            node.is_integer = True
            self.stats["nodes_integer"] += 1
            if lb < self.global_ub:
                self.global_ub = lb
                self.best_integer_solution = lp_result
                self.incumbent_node_id = node.node_id
                logger.info(f"【B&B】新整数解！UB = {lb:.2f}（节点 {node.node_id}）")
            return "integer"

        return "branch"

    def branch(self, node: BBNode) -> Tuple[Optional[BBNode], Optional[BBNode]]:
        """
        对节点进行分支

        策略：选择最分数的 y_b 变量（离 0.5 最近）
        """
        y_vals = node.lp_solution.get("y_values", {})
        branch_var = self._select_branching_variable(y_vals, node)

        if branch_var is None:
            return None, None

        branch_val = y_vals[branch_var]
        node.branch_var = branch_var
        node.branch_val = branch_val

        # 左子节点：y_b = 0
        left = self._create_child(node)
        left.fixed_to_zero.add(branch_var)

        # 右子节点：y_b = 1
        right = self._create_child(node)
        right.fixed_to_one.add(branch_var)

        logger.debug(f"【B&B】节点 {node.node_id} 分支 y_{branch_var} = {branch_val:.3f}")

        return left, right

    def should_terminate(self) -> bool:
        """检查是否应该终止搜索"""
        # 时间限制
        if self.stats["start_time"]:
            elapsed = time.time() - self.stats["start_time"]
            if elapsed > self.time_limit:
                logger.info(f"【B&B】时间限制 {self.time_limit}s 到达")
                return True

        # 节点数限制
        if self.stats["nodes_explored"] >= self.max_nodes:
            logger.info(f"【B&B】节点数限制 {self.max_nodes} 到达")
            return True

        # Gap 检查
        gap = self.get_gap()
        if gap is not None and gap < self.gap_tolerance:
            logger.info(f"【B&B】Gap {gap:.4f} < {self.gap_tolerance}，收敛")
            return True

        # 队列为空
        if not self._heap:
            return True

        return False

    def get_gap(self) -> Optional[float]:
        """计算当前 optimality gap"""
        if self.global_ub == float('inf') or self.global_lb == -float('inf'):
            return None
        if abs(self.global_ub) < 1e-10:
            return None
        return (self.global_ub - self.global_lb) / abs(self.global_ub)

    def get_summary(self) -> Dict:
        elapsed = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        return {
            "global_lb": self.global_lb,
            "global_ub": self.global_ub,
            "gap": self.get_gap(),
            "nodes_explored": self.stats["nodes_explored"],
            "nodes_pruned": self.stats["nodes_pruned"],
            "nodes_integer": self.stats["nodes_integer"],
            "elapsed_time": elapsed,
            "has_solution": self.best_integer_solution is not None,
        }

    def _create_child(self, parent: BBNode) -> BBNode:
        child = BBNode(self._node_counter, parent.node_id, parent.depth + 1)
        child.fixed_to_one = set(parent.fixed_to_one)
        child.fixed_to_zero = set(parent.fixed_to_zero)
        child.lower_bound = parent.lower_bound
        self._node_counter += 1
        self.all_nodes[child.node_id] = child
        heapq.heappush(self._heap, child)
        self.stats["nodes_created"] += 1
        return child

    def _select_branching_variable(self, y_vals: Dict[int, float],
                                    node: BBNode) -> Optional[int]:
        """选择最分数的 y_b（离 0.5 最近）"""
        best_var = None
        best_frac = 0.0

        for b, val in y_vals.items():
            if b in node.fixed_to_one or b in node.fixed_to_zero:
                continue
            frac = min(val, 1.0 - val)
            if frac > 0.01 and frac > best_frac:
                best_frac = frac
                best_var = b

        return best_var

    def _check_integrality(self, y_vals: Dict[int, float],
                           tol: float = 0.01) -> bool:
        for val in y_vals.values():
            if tol < val < 1.0 - tol:
                return False
        return True

    def _update_global_lb(self):
        """全局下界 = 队列中最小的节点下界"""
        if self._heap:
            # 队列最前面的就是最小下界
            candidates = [n.lower_bound for n in self._heap if not n.is_pruned]
            if candidates:
                self.global_lb = min(candidates)
