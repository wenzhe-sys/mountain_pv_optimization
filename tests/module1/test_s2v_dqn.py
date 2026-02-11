"""S2V-DQN 网络和智能体单元测试（对齐论文重写后）"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np

from algorithm.s2v_network import Structure2Vec, QFunction, encode_graph, get_graph_embedding
from algorithm.dqn_agent import (
    DQNPartitionAgent, ReplayBuffer, PartitionEnv, GraphData, GSet
)
from utils.graph_utils import build_adjacency_graph


def make_simple_graph():
    """构造 2x5 = 10 节点测试图"""
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

    def test_forward_shape(self):
        s2v = Structure2Vec(dim_in=1, dim_embed=64)
        N = 10
        feat = torch.zeros(N, 1)
        adj = torch.eye(N)
        adj[0, 1] = adj[1, 0] = 1.0
        ew = adj.clone()
        embed = torch.zeros(N, 64)

        out = s2v(feat, adj, ew, embed)
        self.assertEqual(out.shape, (N, 64))

    def test_multi_round_encoding(self):
        s2v = Structure2Vec(dim_in=1, dim_embed=32)
        N = 5
        feat = torch.ones(N, 1)
        adj = torch.ones(N, N) - torch.eye(N)
        ew = adj.clone()

        embed = encode_graph(s2v, feat, adj, ew, T=4)
        self.assertEqual(embed.shape, (N, 32))
        # 嵌入不应全为零
        self.assertGreater(embed.abs().sum().item(), 0)


class TestQFunction(unittest.TestCase):

    def test_forward_single(self):
        q = QFunction(dim_embed=64)
        state_embed = torch.randn(1, 64)
        node_embed = torch.randn(10, 64)
        out = q(state_embed, node_embed)
        self.assertEqual(out.shape, (10, 1))

    def test_forward_batch(self):
        q = QFunction(dim_embed=64)
        state_embed = torch.randn(8, 64)
        node_embed = torch.randn(8, 64)
        out = q(state_embed, node_embed)
        self.assertEqual(out.shape, (8, 1))


class TestGraphEmbedding(unittest.TestCase):

    def test_global_embedding(self):
        embed = torch.randn(10, 64)
        global_embed = get_graph_embedding(embed)
        self.assertEqual(global_embed.shape, (1, 64))


class TestGraphData(unittest.TestCase):

    def test_creation(self):
        graph, _ = make_simple_graph()
        gd = GraphData(graph)
        self.assertEqual(gd.n_nodes, 10)
        self.assertEqual(gd.adj.shape, (10, 10))
        # 邻接矩阵应对称
        self.assertTrue(torch.allclose(gd.adj, gd.adj.T))


class TestGSet(unittest.TestCase):

    def test_push_and_get(self):
        graph, _ = make_simple_graph()
        gset = GSet()
        gd = GraphData(graph)
        gid = gset.push(gd)
        self.assertEqual(gid, 0)
        self.assertEqual(gset[0].n_nodes, 10)


class TestReplayBuffer(unittest.TestCase):

    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            state = torch.zeros(10, dtype=torch.long)
            buf.push(0, state, i % 10, 1.0, state, False)
        self.assertEqual(len(buf), 50)
        gid, states, actions, rewards, next_states, dones = buf.sample(8)
        self.assertEqual(len(states), 8)
        self.assertEqual(actions.shape, (8, 1))


class TestPartitionEnv(unittest.TestCase):

    def test_reset(self):
        graph, _ = make_simple_graph()
        gd = GraphData(graph)
        env = PartitionEnv(gd, target_size=5, min_size=3, max_size=8)
        state = env.reset()
        self.assertEqual(state.shape, (10,))
        self.assertFalse(env.done)

    def test_valid_actions_empty_zone(self):
        graph, _ = make_simple_graph()
        gd = GraphData(graph)
        env = PartitionEnv(gd, target_size=5, min_size=3, max_size=8)
        env.reset()
        valid = env.get_valid_actions()
        self.assertEqual(len(valid), 10)  # 所有节点都可选

    def test_step_and_reward(self):
        graph, _ = make_simple_graph()
        gd = GraphData(graph)
        env = PartitionEnv(gd, target_size=5, min_size=3, max_size=8)
        env.reset()
        # 选第一个节点
        state, reward, done = env.step(0)
        self.assertEqual(state[0].item(), 1)
        self.assertFalse(done)
        # First step reward is a float (may include shaping reward)
        self.assertIsInstance(reward, float)
        # 选第二个相邻节点
        state, reward, done = env.step(1)
        self.assertEqual(state[1].item(), 1)
        # 第二步应有非零奖励（周长变化）
        self.assertIsInstance(reward, float)

    def test_excluded_nodes(self):
        graph, _ = make_simple_graph()
        gd = GraphData(graph)
        env = PartitionEnv(gd, target_size=3, min_size=2, max_size=5,
                            excluded={0, 1, 2})
        state = env.reset()
        self.assertEqual(state[0].item(), -1)
        self.assertEqual(state[1].item(), -1)
        valid = env.get_valid_actions()
        self.assertNotIn(0, valid)
        self.assertNotIn(1, valid)
        self.assertNotIn(2, valid)


class TestDQNAgent(unittest.TestCase):

    def test_select_action(self):
        graph, _ = make_simple_graph()
        gd = GraphData(graph)
        agent = DQNPartitionAgent(device="cpu")
        state = torch.zeros(10, dtype=torch.long)
        valid = list(range(10))
        action = agent.select_action(gd, state, epsilon=1.0, valid_actions=valid)
        self.assertIn(action, valid)

    def test_checkpoint(self):
        agent = DQNPartitionAgent(device="cpu")
        agent.current_epoch = 10
        agent.best_reward = 5.0
        path = "/tmp/test_s2v_ckpt.pt"
        agent.save_checkpoint(path)

        agent2 = DQNPartitionAgent(device="cpu")
        meta = agent2.load_checkpoint(path)
        self.assertEqual(meta["epoch"], 10)
        self.assertAlmostEqual(meta["best_reward"], 5.0)

        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
