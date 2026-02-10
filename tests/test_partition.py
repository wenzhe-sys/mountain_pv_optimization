"""分区子问题单元测试"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_utils import build_adjacency_graph, build_coord_index
from model.partition_sub import PartitionValidator
from algorithm.partition_heuristic import GreedyPartitioner


def make_grid_pva_list(rows, cols):
    """构造 rows×cols 的面板网格列表"""
    pva_list = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            pva_list.append({
                "panel_id": f"pva_{idx}",
                "x": c * 10.0, "y": r * 10.0,
                "grid_coord": [r, c],
                "cut_spec": [6.0, 3.0]
            })
            idx += 1
    return pva_list


class TestPartitionValidator(unittest.TestCase):
    """测试分区约束验证器"""

    def setUp(self):
        # 44 个面板的网格 (适合分2个22面板的分区)
        self.pva_list = make_grid_pva_list(4, 11)
        self.graph = build_adjacency_graph(self.pva_list)
        self.validator = PartitionValidator(
            self.graph, n_inverters=2,
            min_panels=18, max_panels=26,
            perimeter_lb=10.0, perimeter_ub=200.0
        )

    def test_valid_partition(self):
        """均匀分区应满足所有约束"""
        nodes = list(self.graph.nodes())
        zone1 = set(nodes[:22])
        zone2 = set(nodes[22:])
        result = self.validator.validate([zone1, zone2])

        self.assertTrue(result.zone_details[0]["capacity_ok"])
        self.assertTrue(result.zone_details[1]["capacity_ok"])

    def test_oversized_partition(self):
        """超大分区应报容量违规"""
        nodes = list(self.graph.nodes())
        zone1 = set(nodes[:30])
        zone2 = set(nodes[30:])
        result = self.validator.validate([zone1, zone2])

        self.assertFalse(result.zone_details[0]["capacity_ok"])

    def test_wrong_zone_count(self):
        """分区数不等于逆变器数应报违规"""
        nodes = list(self.graph.nodes())
        result = self.validator.validate([set(nodes)])
        self.assertFalse(result.is_feasible)

    def test_duplicate_assignment(self):
        """重复分配面板应报违规"""
        nodes = list(self.graph.nodes())
        zone1 = set(nodes[:22])
        zone2 = set(nodes[20:])  # 与 zone1 重叠 2 个
        result = self.validator.validate([zone1, zone2])
        self.assertFalse(result.is_feasible)


class TestGreedyPartitioner(unittest.TestCase):
    """测试贪心+局部搜索分区器"""

    def setUp(self):
        # 使用真实算例 r1 的数据结构
        import json
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        instance_path = os.path.join(
            project_root, "data", "processed", "PV", "public", "easy", "public_easy_r1.json"
        )
        if os.path.exists(instance_path):
            with open(instance_path, "r", encoding="utf-8") as f:
                self.instance = json.load(f)
            self.graph = build_adjacency_graph(
                self.instance["pva_list"],
                self.instance["terrain_data"]["grid_size"]
            )
            self.n_zones = self.instance["equipment_params"]["inverter"]["p"]
            self.has_data = True
        else:
            self.has_data = False

    def test_partition_produces_correct_zone_count(self):
        """分区数应等于逆变器数"""
        if not self.has_data:
            self.skipTest("算例数据不可用")
        partitioner = GreedyPartitioner(self.graph, self.n_zones)
        result = partitioner.solve()
        self.assertEqual(len(result.zones), self.n_zones)

    def test_all_panels_assigned(self):
        """所有面板应被分配"""
        if not self.has_data:
            self.skipTest("算例数据不可用")
        partitioner = GreedyPartitioner(self.graph, self.n_zones)
        result = partitioner.solve()
        total_assigned = sum(len(z) for z in result.zones)
        self.assertEqual(total_assigned, len(self.graph.nodes))

    def test_no_overlap(self):
        """分区间不应有重叠"""
        if not self.has_data:
            self.skipTest("算例数据不可用")
        partitioner = GreedyPartitioner(self.graph, self.n_zones)
        result = partitioner.solve()
        all_panels = set()
        for zone in result.zones:
            overlap = all_panels & zone
            self.assertEqual(len(overlap), 0, f"分区重叠: {overlap}")
            all_panels |= zone

    def test_connectivity(self):
        """每个分区应保持连通"""
        if not self.has_data:
            self.skipTest("算例数据不可用")
        partitioner = GreedyPartitioner(
            self.graph, self.n_zones, local_search_iters=50
        )
        result = partitioner.solve()
        for i, detail in enumerate(result.zone_details):
            self.assertTrue(detail["is_connected"],
                            f"分区 {i} 连通性违规")


if __name__ == "__main__":
    unittest.main()
