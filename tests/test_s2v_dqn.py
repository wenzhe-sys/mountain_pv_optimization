"""S2V-DQN 网络和智能体单元测试"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from algorithm.s2v_network import (
    Structure2Vec, QNetwork, S2VDQNModel,
    build_node_features, build_adjacency_matrix, build_context_features
)
from algorithm.dqn_agent import DQNPartitionAgent, ReplayBuffer, PartitionEnv
from utils.graph_utils import build_adjacency_graph


def make_simple_graph():
    """构造简单的 10 节点测试图"""
    pva_list = [
        {"panel_id": f"pva_{i}", "x": c * 10.0, "y": r * 10.0,
         "grid_coord": [r, c], "cut_spec": [6.0, 3.0]}
        for i, (r, c) in enumerate([
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
        ])
    ]
    return build_adjacency_graph(pva_list), pva_list


class TestStructure2Vec(unittest.TestCase):
    """测试 S2V 图嵌入网络"""

    def test_forward_shape(self):
        """前向传播输出形状正确"""
        s2v = Structure2Vec(node_feature_dim=5, hidden_dim=64, n_iterations=4)
        node_features = torch.randn(10, 5)
        adj = torch.eye(10)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[1, 2] = adj[2, 1] = 1.0

        output = s2v(node_features, adj)
        self.assertEqual(output.shape, (10, 64))

    def test_different_iterations(self):
        """不同迭代轮数应产生不同结果"""
        node_features = torch.randn(5, 5)
        adj = torch.ones(5, 5) - torch.eye(5)

        s2v_2 = Structure2Vec(5, 32, n_iterations=2)
        s2v_4 = Structure2Vec(5, 32, n_iterations=4)

        # 使用相同权重初始化
        s2v_4.load_state_dict(s2v_2.state_dict(), strict=False)

        out_2 = s2v_2(node_features, adj)
        out_4 = s2v_4(node_features, adj)

        # 不同轮数应产生不同嵌入
        self.assertFalse(torch.allclose(out_2, out_4))


class TestQNetwork(unittest.TestCase):
    """测试 Q 值网络"""

    def test_forward_shape(self):
        q_net = QNetwork(hidden_dim=64, context_dim=4)
        node_emb = torch.randn(10, 64)
        global_emb = torch.randn(64)
        context = torch.randn(4)

        q_values = q_net(node_emb, global_emb, context)
        self.assertEqual(q_values.shape, (10, 1))


class TestS2VDQNModel(unittest.TestCase):
    """测试完整的 S2V-DQN 模型"""

    def test_forward_with_mask(self):
        """带掩码的前向传播"""
        model = S2VDQNModel(5, 64, 4, 4)
        node_features = torch.randn(10, 5)
        adj = torch.eye(10)
        context = torch.randn(4)
        mask = torch.ones(10, dtype=torch.bool)
        mask[5:] = False  # 后5个节点被掩码

        q_values = model(node_features, adj, context, mask)
        self.assertEqual(q_values.shape, (10,))
        # 被掩码的位置应为 -inf
        self.assertTrue(all(q_values[5:] == float("-inf")))

    def test_get_embeddings(self):
        model = S2VDQNModel(5, 64, 4, 4)
        node_features = torch.randn(10, 5)
        adj = torch.eye(10)

        node_emb, global_emb = model.get_embeddings(node_features, adj)
        self.assertEqual(node_emb.shape, (10, 64))
        self.assertEqual(global_emb.shape, (64,))


class TestBuildFeatures(unittest.TestCase):
    """测试特征构建函数"""

    def test_node_features_shape(self):
        graph, _ = make_simple_graph()
        zones = [{"pva_0", "pva_1"}]
        features = build_node_features(graph, zones, 0)
        self.assertEqual(features.shape, (10, 5))

    def test_adjacency_matrix(self):
        graph, _ = make_simple_graph()
        adj = build_adjacency_matrix(graph)
        self.assertEqual(adj.shape, (10, 10))
        # 对称
        self.assertTrue(torch.allclose(adj, adj.T))

    def test_context_features(self):
        context = build_context_features(
            [{"a", "b"}], {"c"}, n_total=10, max_panels=26, n_zones_target=5
        )
        self.assertEqual(context.shape, (4,))


class TestReplayBuffer(unittest.TestCase):

    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(torch.randn(5, 3), i, 1.0, torch.randn(5, 3), False, torch.ones(5, dtype=torch.bool))
        self.assertEqual(len(buf), 50)
        samples = buf.sample(10)
        self.assertEqual(len(samples), 10)


class TestPartitionEnv(unittest.TestCase):
    """测试分区环境"""

    def test_reset(self):
        graph, _ = make_simple_graph()
        env = PartitionEnv(graph, n_zones=1, min_panels=5, max_panels=10,
                            perimeter_lb=5.0, perimeter_ub=200.0)
        state = env.reset()
        self.assertIn("node_features", state)
        self.assertIn("adjacency", state)
        self.assertIn("action_mask", state)
        self.assertFalse(env.done)

    def test_step_valid(self):
        graph, _ = make_simple_graph()
        env = PartitionEnv(graph, n_zones=1, min_panels=5, max_panels=10,
                            perimeter_lb=5.0, perimeter_ub=200.0)
        env.reset()
        # 选择第一个可用动作
        mask = env._get_action_mask()
        valid = torch.where(mask)[0]
        if len(valid) > 0:
            state, reward, done = env.step(valid[0].item())
            self.assertIsInstance(reward, float)


class TestDQNCheckpoint(unittest.TestCase):
    """测试 checkpoint 保存和加载"""

    def test_save_and_load(self):
        agent = DQNPartitionAgent(device="cpu")
        agent.steps_done = 100
        agent.best_reward = -5.0
        agent.current_epoch = 10

        path = "/tmp/test_ckpt.pt"
        agent.save_checkpoint(path)

        agent2 = DQNPartitionAgent(device="cpu")
        meta = agent2.load_checkpoint(path)

        self.assertEqual(meta["epoch"], 10)
        self.assertAlmostEqual(meta["best_reward"], -5.0)
        self.assertEqual(agent2.steps_done, 100)

        # 清理
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
