# S2V-DQN Implementation Analysis
## "Learning Combinatorial Optimization Algorithms over Graphs" (Khalil et al., 2017)

**Date:** 2026-02-09
**Purpose:** Understand working S2V-DQN implementations for adapting to our graph partitioning problem

---

## 1. Reference Implementations Analyzed

| Repository | Language | Stars | Problem | Notes |
|---|---|---|---|---|
| [Hanjun-Dai/graph_comb_opt](https://github.com/Hanjun-Dai/graph_comb_opt) | C++ / Python | 515 | MVC, MaxCut, TSP, SCP | Original paper implementation |
| [jiayuanzhang0/S2V-DQN](https://github.com/jiayuanzhang0/S2V-DQN) | Python/PyTorch | 2 | MaxCut | Clean PyTorch reimplementation |
| [lzhan130/S2V-DQN_pytorch](https://github.com/lzhan130/S2V-DQN_pytorch) | Python/PyTorch | 7 | MVC | PyTorch with n-step returns |
| [tomdbar/eco-dqn](https://github.com/tomdbar/eco-dqn) | Python/PyTorch | 81 | MaxCut | Includes S2V-DQN baseline reimplementation |
| [rvdweerd/GNN-RL-CombOptim](https://github.com/rvdweerd/GNN-RL-CombOptim) | Python/PyTorch | - | MaxCut | Additional reimplementation |

---

## 2. Training Loop Structure

### 2.1 Original Implementation (graph_comb_opt) - MaxCut

The original uses a **C++ backend** with Python orchestration. Key training loop from `code/s2v_maxcut/main.py`:

```python
# ORIGINAL TRAINING LOOP PATTERN (graph_comb_opt)
eps_start = 1.0
eps_end = 0.05
eps_step = 50000.0

for iter in range(max_iter):  # max_iter = 800,000
    # Regenerate training graphs every 5000 iterations
    if iter and iter % 5000 == 0:
        gen_new_graphs(opt)  # generates 1000 new random graphs

    # Linear epsilon decay over 50,000 steps
    eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)

    # Play game episodes every 10 iterations
    if iter % 10 == 0:
        api.lib.PlayGame(10, ctypes.c_double(eps))

    # Validate and save every 300 iterations
    if iter % 300 == 0:
        frac = 0.0
        for idx in range(n_valid):
            frac += api.lib.TestNoStop(idx)
        print('iter', iter, 'eps', eps, 'average cut size: ', frac / n_valid)
        api.SaveModel(model_path)

    # Take snapshot of experience every 1000 iterations
    if iter % 1000 == 0:
        api.TakeSnapshot()

    # Perform one fitted Q-iteration update
    api.lib.Fit()
```

**Key Pattern:** The original does NOT train at every single step. It:
1. Plays episodes in batches (10 games every 10 iterations)
2. Calls `Fit()` (SGD on replay buffer) at every iteration
3. Regenerates training graphs periodically (every 5000 iters)

### 2.2 PyTorch Reimplementation (jiayuanzhang0/S2V-DQN) - MaxCut

```python
# CLEAN PYTORCH TRAINING LOOP
for episode in range(num_episodes + 1):  # num_episodes = 10,000
    num_nodes = random.randint(num_nodes_lower, num_nodes_upper)  # 30-50
    data = graph.generate_graph(num_nodes)
    state = env.reset(data.clone())

    while True:
        # Linear epsilon decay
        epsilon = epsilon_s - (epsilon_s - epsilon_e) * (episode / num_episodes)

        # Encode graph with Structure2Vec
        data.feat = state.unsqueeze(1).float()
        embed = agent.encoding(data, 'policy')
        state_embed = get_state_embed(embed)  # sum of all node embeddings

        # Select action (epsilon-greedy)
        action = agent.select_action(state_embed, embed, epsilon, state)

        # Environment step
        state_next, reward, terminated, truncated, _ = env.step(action)

        # Store transition and train
        agent.ReplayBuffer.push(gid, state, action, reward, state_next, done)
        agent.train()  # <-- trains EVERY step if buffer has enough samples

        state = state_next
        if done:
            break
```

**Key difference:** This trains at EVERY step (after buffer is filled), not periodically.

### 2.3 ECO-DQN's S2V-DQN Baseline (tomdbar/eco-dqn)

```python
# ECO-DQN TRAINING LOOP (most polished implementation)
for timestep in range(nb_steps):  # nb_steps = 2,500,000
    action = agent.act(state, is_training_ready)
    state_next, reward, done, _ = env.step(action)

    replay_buffer.add(state, action, reward, state_next, done)

    if is_training_ready:
        if timestep % update_frequency == 0:  # update_frequency = 32
            transitions = replay_buffer.sample(minibatch_size)  # batch_size = 64
            loss = train_step(transitions)

        if timestep % update_target_frequency == 0:  # every 1000 steps
            target_network.load_state_dict(network.state_dict())
```

---

## 3. Hyperparameters That Work

### 3.1 Comparison Table

| Parameter | Original (graph_comb_opt) | jiayuanzhang0 | ECO-DQN S2V baseline | lzhan130 |
|---|---|---|---|---|
| **Embedding dim** | 64 | 32 | 64 | 64 |
| **S2V iterations (T)** | 3 (max_bp_iter) | 3 | 3 (n_layers) | 5 |
| **Learning rate** | 0.0001 | 0.001 | 0.0001 | 0.01 |
| **Batch size** | 128 | 16 | 64 | 128 |
| **Replay buffer size** | 50,000 | 10,000 | 5,000 | 100,000 |
| **Gamma (discount)** | 1.0 (implied) | 1.0 | 1.0 | 1.0 |
| **Epsilon start** | 1.0 | 0.95 | 1.0 | 1.0 |
| **Epsilon end** | 0.05 | 0.05 | 0.05 | 0.05 |
| **Epsilon decay steps** | 50,000 | 10,000 episodes | 150,000 | linear decay |
| **Total training steps** | 800,000 | 10,000 episodes | 2,500,000 | 100,000 episodes |
| **Target network update** | snapshot every 1000 | soft update (tau=0.05) | hard copy every 1000 | hard copy every 100 |
| **n-step returns** | 1 | 1 | 1 | 5 |
| **Graph size (train)** | 15-20 | 30-50 | 20 | 50-100 |
| **Weight init** | N(0, 0.01) | Xavier uniform | N(0, 0.01) | default |
| **Optimizer** | SGD (momentum=0.9) | Adam | Adam | Adam |
| **Loss function** | MSE | MSE | MSE | MSE |
| **Hidden layer (Q-func)** | 32 (reg_hidden) | - | - | - |

### 3.2 Recommended Hyperparameters (Consensus)

```python
RECOMMENDED_HYPERPARAMS = {
    # Network architecture
    'embedding_dim': 64,          # 32 works for small graphs, 64 is safer
    's2v_iterations': 3,          # T=3 or T=4, more is rarely better
    'weight_init_std': 0.01,      # Important! Small initialization

    # DQN parameters
    'learning_rate': 1e-4,        # 1e-4 is the safest choice
    'gamma': 1.0,                 # CRITICAL: gamma=1 for combinatorial opt
    'batch_size': 64,             # 64-128 range
    'replay_buffer_size': 50000,  # 5000-50000 range

    # Exploration
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_steps': 50000, # Linear decay

    # Target network
    'target_update': 'hard',      # Hard copy every 500-1000 steps
    'target_update_freq': 1000,
    # OR soft update with tau=0.05

    # Training
    'n_step': 1,                  # 1-step is standard, n-step is optional improvement
    'update_frequency': 1,        # Train every step or every N steps
    'replay_start_size': 500,     # Minimum buffer size before training

    # Graph generation
    'graph_type': 'barabasi_albert',  # or erdos_renyi
    'min_nodes': 15,
    'max_nodes': 50,
}
```

---

## 4. Reward Function Design

### 4.1 The Paper's Formulation (Equation 5)

The reward is defined as the **incremental change in objective function**:

```
r(S, v) = c(h(S'), G) - c(h(S), G)
```

where S' = (S, v) is the new state after adding node v.

**This is critical:** The cumulative reward equals the final objective value:
```
R(S_final) = sum of all r(S_t, v_t) = c(h(S_final), G)
```

### 4.2 MaxCut Reward (step-by-step change in cut value)

From jiayuanzhang0's `maxcut_env.py`:

```python
def step(self, action):
    self.state[action] = 1  # Add node to set S

    cost_function_old = self.cost_function
    self.cost_function = calculate_cut_value(self.data, self.state)
    reward = self.cost_function - cost_function_old  # INCREMENTAL reward

    terminated = self.current_step >= self.max_steps
    truncated = reward < 0  # Truncate if adding node decreased cut
    return state_next, reward, terminated, truncated, {}

def calculate_cut_value(data, state):
    edge_index = data.edge_index
    edge_weight = data.edge_weight
    src_feat = state[edge_index[0]]
    tgt_feat = state[edge_index[1]]
    # Edges where one end is in S (state=1) and other is not
    cut_value = edge_weight[(src_feat != tgt_feat)].sum().item()
    return cut_value
```

### 4.3 Key Insight: Dense vs. Sparse Rewards

The ECO-DQN codebase explicitly parameterizes this:

```python
env_args = {
    'reward_signal': RewardSignal.DENSE,  # Step-by-step reward (r_t = c(S_{t+1}) - c(S_t))
    # vs RewardSignal.SINGLE = only terminal reward
    'norm_rewards': True,  # Normalize rewards (helps stability)
}
```

**DENSE rewards are essential for S2V-DQN.** The paper specifically notes that step-by-step rewards
(change in objective) are key to efficient learning, unlike policy gradient methods that only get
terminal rewards.

### 4.4 MaxCut Termination

From the paper (Table 1):
- **MVC:** terminates when all edges are covered
- **MaxCut:** terminates when "cut weight cannot be improved" (i.e., adding any more nodes decreases the cut)
- **TSP:** terminates when tour includes all nodes

In practice, the jiayuanzhang0 implementation terminates MaxCut when:
1. All nodes have been assigned (n-1 steps), OR
2. The reward becomes negative (truncation)

The original paper's formulation is more nuanced - for MaxCut, the agent can stop early.

### 4.5 Adapting for Graph Partitioning

For a balanced graph partitioning problem, the reward should capture:
```python
# Option A: Pure cut minimization with balance penalty
r(S, v) = -(new_edge_cut - old_edge_cut)  # negative because we MINIMIZE cuts
         + balance_bonus                     # bonus for maintaining balance

# Option B: Weighted combination (recommended)
r(S, v) = -alpha * delta_cut + beta * delta_balance

# Option C: Step reward from Khalil formulation
# r(S, v) = c(S', G) - c(S, G)
# where c measures partition quality (higher = better)
```

---

## 5. Experience Replay Structure

### 5.1 What Gets Stored

Each transition in the replay buffer stores:

```python
# jiayuanzhang0 format
(gid, state, action, reward, state_next, done)
# where:
#   gid      = graph ID (needed to retrieve the graph structure for re-encoding)
#   state    = binary vector [0,0,1,0,1,...] indicating which nodes are in S
#   action   = index of node to add
#   reward   = incremental reward (scalar)
#   state_next = new binary vector after adding the node
#   done     = whether episode is finished
```

### 5.2 Graph Storage (GSet)

A critical implementation detail: **the graph structure must be stored separately** because the replay buffer entries reference graphs by ID:

```python
class GSet():
    def __init__(self):
        self.g_list = deque([])

    def push(self, data):
        self.g_list.append(data)
```

During training, when sampling from replay buffer, the graph is retrieved:
```python
data_list = [self.GSet.g_list[gid[i, 0]].clone() for i in range(batch_size)]
```

### 5.3 Buffer Sizes and Sampling

| Implementation | Buffer Size | Min before training | Batch size |
|---|---|---|---|
| graph_comb_opt | 50,000 | ~100 (10 games * 10 steps) | 128 |
| jiayuanzhang0 | 10,000 | batch_size (16) | 16 |
| eco-dqn | 5,000 | 500 | 64 |
| lzhan130 | 100,000 | batch_size (128) | 128 |

### 5.4 N-step Experience Replay

The lzhan130 implementation uses **5-step returns**:

```python
n_step = 5
# ...
if steps_cntr > n_step + 1:
    agent.remember(
        num_nodes, mu, edge_index, edge_w,
        state_steps[-(n_step+1)],      # State from n steps ago
        action_steps[-n_step],          # Action from n steps ago
        [sum(reward_steps[-n_step:])],  # Sum of n rewards
        state_steps[-1],               # Current state
        done
    )
```

The n-step return formula:
```
R_{t,t+n} = sum_{i=0}^{n-1} gamma^i * r(S_{t+i}, v_{t+i})
target = R_{t,t+n} + gamma^n * max_a' Q(S_{t+n}, a')
```

---

## 6. Network Architecture (Structure2Vec)

### 6.1 S2V Embedding Update (Equation 3 from paper)

```python
class S2V(MessagePassing):
    def __init__(self, dim_in, dim_embed):
        super().__init__(aggr='add')
        self.theta1 = nn.Parameter(torch.Tensor(dim_in, dim_embed))   # Node features
        self.theta2 = nn.Parameter(torch.Tensor(dim_embed, dim_embed))  # Neighbor embeddings
        self.theta3 = nn.Parameter(torch.Tensor(dim_embed, dim_embed))  # Edge weights
        self.theta4 = nn.Parameter(torch.Tensor(1, dim_embed))          # Edge weight transform

    def forward(self, feat, edge_index, edge_weight, embed):
        message1 = self.propagate(edge_index, x=embed, ...)  # Aggregate neighbor embeds
        message2 = self.propagate(edge_index, x=embed, ...)  # Aggregate edge weights
        out = F.relu(
            torch.matmul(feat, self.theta1) +      # theta1 * x_v
            torch.matmul(message1, self.theta2) +   # theta2 * sum(mu_u)
            torch.matmul(message2, self.theta3)      # theta3 * sum(relu(theta4 * w(v,u)))
        )
        return out
```

The embedding is computed iteratively T times (typically T=3 or T=4).

### 6.2 Q-Function (Equation 4 from paper)

```python
class QFunction(nn.Module):
    def __init__(self, dim_embed):
        self.theta5 = nn.Parameter(torch.Tensor(2 * dim_embed, 1))
        self.theta6 = nn.Parameter(torch.Tensor(dim_embed, dim_embed))
        self.theta7 = nn.Parameter(torch.Tensor(dim_embed, dim_embed))

    def forward(self, state_embed, embed):
        # state_embed = sum of all node embeddings (graph-level)
        # embed = individual node embedding (action candidate)
        theta7_muv = torch.matmul(embed, self.theta7)
        theta6_state = torch.matmul(state_embed, self.theta6)
        concat = torch.cat((theta6_state, theta7_muv), dim=1)
        q = torch.matmul(F.relu(concat), self.theta5)
        return q
```

### 6.3 State Embedding

The graph-level state embedding is simply the **sum of all node embeddings**:

```python
def get_state_embed(embed):
    return torch.sum(embed, dim=0).unsqueeze(0)
```

---

## 7. Training Convergence

### 7.1 How Many Steps/Episodes Are Needed?

| Implementation | Total Training | Graph Size | Problem | Convergence Point |
|---|---|---|---|---|
| graph_comb_opt | 800,000 iterations | 15-20 nodes | MaxCut | ~200,000-400,000 iters |
| jiayuanzhang0 | 10,000 episodes | 30-50 nodes | MaxCut | ~5,000-8,000 episodes |
| eco-dqn (S2V) | 2,500,000 timesteps | 20 nodes | MaxCut | ~500,000-1,000,000 steps |
| lzhan130 | 100,000 episodes | 50-100 nodes | MVC | ~50,000 episodes |

### 7.2 What Does a Successful Training Curve Look Like?

From the paper (Appendix D.6, Figure D.1):
- **Phase 1 (0-50K steps):** Random exploration, low scores, high variance
- **Phase 2 (50K-200K steps):** Rapid improvement as policy learns basic heuristics
- **Phase 3 (200K-500K steps):** Gradual refinement, approaching near-optimal
- **Phase 4 (500K+ steps):** Plateau with small fluctuations

The reward should **monotonically increase** (for maximization) with decreasing variance.

### 7.3 Validation Frequency

The original checks validation every 300 iterations and saves the model:
```python
if iter % 300 == 0:
    frac = 0.0
    for idx in range(n_valid):  # n_valid = 100
        frac += api.lib.TestNoStop(idx)
    print('average cut size:', frac / n_valid)
```

---

## 8. Common Pitfalls and How to Avoid Them

### 8.1 Gamma Must Be 1.0

**CRITICAL:** For combinatorial optimization, `gamma = 1.0` (no discounting).

Why: The cumulative reward must equal the final objective value. If gamma < 1, later actions
are artificially penalized, which is wrong for problems where we want to maximize the total
objective regardless of when actions are taken.

All successful implementations use gamma = 1.

### 8.2 Small Weight Initialization

The original uses `w_scale = 0.01` (weights initialized from N(0, 0.01)).
The jiayuanzhang0 implementation uses Xavier uniform initialization.

**Both are valid.** Large initial weights can cause the Q-values to explode early in training.

### 8.3 Epsilon Decay Must Be Slow Enough

- Original: decays from 1.0 to 0.05 over 50,000 steps (out of 800,000 total = 6.25% of training)
- ECO-DQN: decays over 150,000 steps (out of 2,500,000 = 6% of training)

**Rule of thumb:** Epsilon should reach its minimum within the first 5-10% of total training steps.
This ensures sufficient exploration early on, then exploitation for fine-tuning.

### 8.4 Graph Regeneration During Training

The original regenerates training graphs every 5,000 iterations:
```python
if iter and iter % 5000 == 0:
    gen_new_graphs(opt)  # generates 1000 new random graphs
```

This is important to prevent **overfitting to specific graph instances**. The agent should learn
a general policy, not memorize solutions for specific graphs.

### 8.5 Action Masking (Only Allow Valid Actions)

During action selection, only unvisited nodes should be candidates:
```python
remaining_nodes = (state == 0).nonzero(as_tuple=True)[0].tolist()

if random_num > epsilon:  # Exploitation
    qs = self.qfunc_policy(state_embed, embed)
    action = remaining_nodes[torch.argmax(qs[remaining_nodes]).item()]
else:  # Exploration
    action = random.choice(remaining_nodes)
```

**NEVER select an already-visited node.** This is essential for correctness.

### 8.6 Target Network Stability

Two approaches, both work:

**Hard update (preferred by original + ECO-DQN):**
```python
if timestep % 1000 == 0:
    target_network.load_state_dict(network.state_dict())
```

**Soft update (used by jiayuanzhang0):**
```python
for param, target_param in zip(policy.parameters(), target.parameters()):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
# tau = 0.05
```

### 8.7 Learning Rate Sensitivity

- **Too high (>1e-3):** Q-values diverge, training is unstable
- **Too low (<1e-5):** Training is too slow, may not converge in reasonable time
- **Sweet spot: 1e-4** is used by both the original and ECO-DQN

The original uses SGD with momentum 0.9, but Adam with lr=1e-4 works equally well.

### 8.8 Batch Training vs. Per-Step Training

The original interleaves game-playing and training:
- Play 10 games every 10 iterations
- Train (Fit) at every iteration

This **decouples data collection from training**, which can improve stability.
In contrast, jiayuanzhang0 trains at every step, which works but may be less stable.

### 8.9 Graph Size Curriculum

From the paper (Appendix D.4, "pre-training"):
> For MVC, where we train the model on graphs with up to 500 nodes, we use the model
> trained on small graphs as initialization for training on larger ones.

**Curriculum learning** (train on small graphs first, then larger ones) significantly helps
for scaling to larger graphs.

### 8.10 The Irreversibility Problem

A fundamental limitation of S2V-DQN: once a node is added to the solution, it cannot be removed.
This means early mistakes are permanent. ECO-DQN addresses this by allowing reversible actions.

For graph partitioning, consider:
- Starting from a random partition and allowing swaps (like ECO-DQN)
- Using the S2V-DQN approach but with a good reward signal that guides early decisions

---

## 9. Adapting to Graph Partitioning

### 9.1 MaxCut as a Proxy for Partitioning

MaxCut is the closest problem to graph partitioning in S2V-DQN:
- MaxCut: maximize edges between S and V\S
- Min-cut partitioning: minimize edges between partitions while maintaining balance

The key difference is the **balance constraint**. In MaxCut, any partition size is valid.
In balanced partitioning, both sides must be roughly equal.

### 9.2 Modifications Needed for Balanced Partitioning

1. **State representation:** Instead of binary (in S or not), use ternary: 
   - 0 = unassigned
   - 1 = partition A
   - 2 = partition B

2. **Termination:** Episode ends when all nodes are assigned

3. **Reward function options:**
   ```python
   # Option A: Incremental cut penalty + balance bonus
   r = -(delta_edge_cut) + lambda * delta_balance_score
   
   # Option B: Only terminal reward (harder to train)
   r = -total_edge_cut * balance_penalty  # only at end
   
   # Option C: Khalil-style incremental (recommended)
   r = c(partition_after) - c(partition_before)
   # where c = -edge_cut + alpha * balance_score
   ```

4. **Action space:** At each step, assign one node to either partition A or partition B
   - This doubles the action space compared to MaxCut
   - Alternative: assign first n/2 nodes to A, rest to B (simpler but less flexible)

### 9.3 Two-Phase Approach (Alternative)

Another strategy for partitioning:
1. Use S2V-DQN exactly like MaxCut to select nodes for set S
2. Force |S| = n/2 by terminating after n/2 selections
3. Remaining nodes go to the other partition

This is simpler and leverages the existing MaxCut framework directly.

---

## 10. Complete Working Example (Minimal S2V-DQN for MaxCut)

Based on jiayuanzhang0's clean implementation, here's the minimal pattern:

```python
# === CONFIG ===
dim_embed = 64
lr = 1e-4
gamma = 1.0
tau = 0.05  # soft update
buffer_size = 50000
batch_size = 64
epsilon_s = 1.0
epsilon_e = 0.05
num_episodes = 10000
T = 4  # S2V iterations
num_nodes_range = (15, 50)

# === TRAINING LOOP ===
for episode in range(num_episodes):
    # Generate random graph
    num_nodes = random.randint(*num_nodes_range)
    data = generate_graph(num_nodes)
    state = env.reset(data)

    while not done:
        # Linear epsilon decay
        epsilon = epsilon_s - (epsilon_s - epsilon_e) * (episode / num_episodes)

        # Compute graph embedding (T iterations of S2V)
        embed = compute_embedding(data, state, T)
        state_embed = embed.sum(dim=0)

        # Epsilon-greedy action selection (only valid actions)
        action = select_action(state_embed, embed, epsilon, valid_actions)

        # Step environment
        state_next, reward, done = env.step(action)

        # Store and train
        buffer.push(graph_id, state, action, reward, state_next, done)
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            loss = compute_td_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            soft_update_target(tau)

        state = state_next

    # Log and validate periodically
    if episode % 100 == 0:
        validate_on_test_graphs()
```

---

## 11. Summary of Key Takeaways

1. **Use gamma = 1.0** - Non-negotiable for combinatorial optimization
2. **Dense rewards** (incremental objective change) >> sparse terminal rewards
3. **Learning rate 1e-4** with Adam is the safest starting point
4. **Embedding dim 64, T=3-4 S2V iterations** is standard
5. **Linear epsilon decay** from 1.0 to 0.05 over ~6% of total training
6. **Replay buffer 5000-50000** with batch size 64-128
7. **Regenerate training graphs** periodically to prevent overfitting
8. **Action masking** is essential - only select unvisited nodes
9. **Weight initialization** should be small (std=0.01 or Xavier)
10. **Training takes 10K-100K episodes** or **500K-2.5M steps** depending on graph size
11. **Validation on held-out graphs** every few hundred iterations
12. **Target network** update every 1000 steps (hard) or with tau=0.05 (soft)

---

## 12. References

- Paper: https://arxiv.org/abs/1704.01665
- Original code: https://github.com/Hanjun-Dai/graph_comb_opt
- PyTorch MaxCut: https://github.com/jiayuanzhang0/S2V-DQN
- PyTorch MVC: https://github.com/lzhan130/S2V-DQN_pytorch
- ECO-DQN (extends S2V-DQN): https://github.com/tomdbar/eco-dqn
- GNN-RL-CombOptim: https://github.com/rvdweerd/GNN-RL-CombOptim
- GP-DQN for partitioning: https://www.nature.com/articles/s41598-025-16768-x
