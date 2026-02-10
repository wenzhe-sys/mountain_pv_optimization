"""网格邻接图工具单元测试"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.graph_utils import (
    build_adjacency_graph, check_connectivity, calculate_perimeter_fast,
    build_coord_index, get_boundary_nodes, get_adjacent_external_nodes,
    get_connected_components
)


class TestBuildGraph(unittest.TestCase):
    """测试网格邻接图构建"""

    def setUp(self):
        """构造测试用的面板列表（3×3 网格，缺少中心）"""
        #  布局:  (0,0) (0,1) (0,2)
        #         (1,0)       (1,2)
        #         (2,0) (2,1) (2,2)
        self.pva_list = [
            {"panel_id": f"pva_{i}", "x": c * 10.0, "y": r * 10.0,
             "grid_coord": [r, c], "cut_spec": [2.0, 3.0]}
            for i, (r, c) in enumerate([
                (0, 0), (0, 1), (0, 2),
                (1, 0),         (1, 2),
                (2, 0), (2, 1), (2, 2),
            ])
        ]
        self.graph = build_adjacency_graph(self.pva_list, grid_size=10.0)

    def test_node_count(self):
        self.assertEqual(len(self.graph.nodes), 8)

    def test_adjacency_correct(self):
        """(0,0) 应与 (0,1) 和 (1,0) 相邻，不与 (1,1) 相邻（不存在）"""
        self.assertTrue(self.graph.has_edge("pva_0", "pva_1"))  # (0,0)-(0,1)
        self.assertTrue(self.graph.has_edge("pva_0", "pva_3"))  # (0,0)-(1,0)
        # (0,0) 不与 (0,2) 相邻（距离>1）
        self.assertFalse(self.graph.has_edge("pva_0", "pva_2"))

    def test_node_attributes(self):
        data = self.graph.nodes["pva_0"]
        self.assertEqual(data["row"], 0)
        self.assertEqual(data["col"], 0)
        self.assertIn("power", data)


class TestConnectivity(unittest.TestCase):
    """测试连通性检查"""

    def setUp(self):
        self.pva_list = [
            {"panel_id": f"pva_{i}", "x": c * 10.0, "y": r * 10.0,
             "grid_coord": [r, c], "cut_spec": [2.0, 3.0]}
            for i, (r, c) in enumerate([
                (0, 0), (0, 1), (0, 2),
                (1, 0),         (1, 2),
                (2, 0), (2, 1), (2, 2),
            ])
        ]
        self.graph = build_adjacency_graph(self.pva_list)

    def test_connected_set(self):
        """相邻的面板构成连通集"""
        self.assertTrue(check_connectivity(self.graph, {"pva_0", "pva_1", "pva_2"}))

    def test_disconnected_set(self):
        """不相邻的面板构成非连通集（(0,0) 和 (2,2) 之间缺少中心）"""
        # (0,2) 和 (1,0) 不直接相邻
        components = get_connected_components(self.graph, {"pva_2", "pva_3"})
        # pva_2=(0,2), pva_3=(1,0) 不相邻
        self.assertEqual(len(components), 2)

    def test_single_node(self):
        self.assertTrue(check_connectivity(self.graph, {"pva_0"}))

    def test_empty_set(self):
        self.assertTrue(check_connectivity(self.graph, set()))


class TestPerimeter(unittest.TestCase):
    """测试周长计算"""

    def setUp(self):
        # 2×2 方形分区
        self.pva_list = [
            {"panel_id": f"pva_{i}", "x": c * 10.0, "y": r * 10.0,
             "grid_coord": [r, c], "cut_spec": [2.0, 3.0]}
            for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)])
        ]
        self.graph = build_adjacency_graph(self.pva_list)
        self.coord_index = build_coord_index(self.graph)

    def test_square_perimeter(self):
        """2×2 方形分区的周长 = 2×(2×cut_l + 2×cut_w)"""
        zone = {"pva_0", "pva_1", "pva_2", "pva_3"}
        perimeter = calculate_perimeter_fast(zone, self.graph, self.coord_index)
        # 每个面板: cut_l=2.0, cut_w=3.0
        # 上边2个暴露上边 + 下边2个暴露下边 = 4 × 2.0
        # 左边2个暴露左边 + 右边2个暴露右边 = 4 × 3.0
        expected = 4 * 2.0 + 4 * 3.0  # = 20.0
        self.assertAlmostEqual(perimeter, expected, places=1)

    def test_single_panel_perimeter(self):
        """单个面板周长 = 2×cut_l + 2×cut_w"""
        zone = {"pva_0"}
        perimeter = calculate_perimeter_fast(zone, self.graph, self.coord_index)
        expected = 2 * 2.0 + 2 * 3.0  # = 10.0
        self.assertAlmostEqual(perimeter, expected, places=1)


class TestBoundary(unittest.TestCase):
    """测试边界节点获取"""

    def setUp(self):
        self.pva_list = [
            {"panel_id": f"pva_{i}", "x": c * 10.0, "y": r * 10.0,
             "grid_coord": [r, c], "cut_spec": [2.0, 3.0]}
            for i, (r, c) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
        ]
        self.graph = build_adjacency_graph(self.pva_list)

    def test_boundary_nodes(self):
        zone = {"pva_0", "pva_1", "pva_3", "pva_4"}  # 左半部分
        boundary = get_boundary_nodes(self.graph, zone)
        # pva_1=(0,1) 邻居有 pva_2=(0,2) 在外部 → 边界
        self.assertIn("pva_1", boundary)

    def test_adjacent_external(self):
        zone = {"pva_0", "pva_1"}
        external = get_adjacent_external_nodes(self.graph, zone)
        self.assertIn("pva_2", external)   # (0,2) 与 (0,1) 相邻
        self.assertIn("pva_3", external)   # (1,0) 与 (0,0) 相邻


if __name__ == "__main__":
    unittest.main()
