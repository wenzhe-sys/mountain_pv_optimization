"""
网格邻接图工具模块

将光伏面板阵列(PVA)坐标建模为 NetworkX 图结构，提供：
- 网格邻接图构建（四邻域连接）
- 子图连通性检查（BFS）
- 分区周长计算（基于边界边）
- 分区约束验证工具函数

依据：《数据字典与输入输出格式规范》坐标系统一规则
  - 网格索引 (row, col) 与坐标 (x, y) 映射：x = col × grid_size, y = row × grid_size
  - 设备坐标需对齐网格节点（x/y 为 grid_size 的整数倍）
"""

import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from collections import deque, defaultdict


def build_adjacency_graph(pva_list: List[Dict], grid_size: float = 10.0) -> nx.Graph:
    """
    从面板列表构建网格邻接图（四邻域连接）。

    节点 = 面板 panel_id，属性包含坐标、网格位置、切割规格等。
    边 = 网格上下左右相邻的面板之间的连接。

    Args:
        pva_list: 面板列表，每个元素含 panel_id, x, y, grid_coord, cut_spec
        grid_size: 网格尺寸（米），默认 10m

    Returns:
        NetworkX 无向图，节点为 panel_id，边为四邻域相邻关系
    """
    graph = nx.Graph()

    # 建立 grid_coord -> panel_id 的索引，用于快速查找邻居
    coord_to_panel = {}

    for pva in pva_list:
        panel_id = pva["panel_id"]
        row, col = pva["grid_coord"]
        x, y = pva["x"], pva["y"]

        # 计算等效功率: a_i = t_i × b × P_density
        # 简化为 cut_spec[0] × cut_spec[1] × 单位功率密度
        cut_l = pva.get("cut_spec", [2.0, 3.0])[0]
        cut_w = pva.get("cut_spec", [2.0, 3.0])[1]
        power = cut_l * cut_w * 0.2  # kW，假设功率密度 0.2 kW/m²

        graph.add_node(panel_id, x=x, y=y, row=row, col=col,
                       grid_coord=(row, col), cut_spec=(cut_l, cut_w),
                       power=power, panel_index=len(graph.nodes))

        coord_to_panel[(row, col)] = panel_id

    # 构建邻接关系：基于实际坐标距离，而非严格 ±1 网格
    # 原因：实际数据中相邻面板的网格间距可能 > 1（如间距=2）
    # 策略：找每个面板在行方向和列方向上最近的邻居

    # 按行分组
    row_groups = defaultdict(list)  # row -> [(col, panel_id), ...]
    col_groups = defaultdict(list)  # col -> [(row, panel_id), ...]

    for pva in pva_list:
        panel_id = pva["panel_id"]
        row, col = pva["grid_coord"]
        row_groups[row].append((col, panel_id))
        col_groups[col].append((row, panel_id))

    # 同行中相邻的列（按列排序后连接相邻对）
    for row, panels in row_groups.items():
        panels.sort(key=lambda x: x[0])
        for i in range(len(panels) - 1):
            col1, id1 = panels[i]
            col2, id2 = panels[i + 1]
            col_gap = abs(col2 - col1)
            # 最大允许间距：5 个网格单位（覆盖实际数据中的 2-3 间距）
            if col_gap <= 5 and not graph.has_edge(id1, id2):
                dist = col_gap * grid_size
                graph.add_edge(id1, id2, distance=dist, direction="horizontal")

    # 同列中相邻的行（按行排序后连接相邻对）
    for col, panels in col_groups.items():
        panels.sort(key=lambda x: x[0])
        for i in range(len(panels) - 1):
            row1, id1 = panels[i]
            row2, id2 = panels[i + 1]
            row_gap = abs(row2 - row1)
            if row_gap <= 5 and not graph.has_edge(id1, id2):
                dist = row_gap * grid_size
                graph.add_edge(id1, id2, distance=dist, direction="vertical")

    return graph


def check_connectivity(graph: nx.Graph, node_set: Set[str]) -> bool:
    """
    检查给定节点集合在图中是否构成连通子图（BFS）。

    依据：项目书约束 (8) - 分区内面板空间连续性约束

    Args:
        graph: 完整的邻接图
        node_set: 待检查的节点集合（panel_id 集合）

    Returns:
        True 如果 node_set 中的所有节点在子图中连通
    """
    if len(node_set) <= 1:
        return True

    # 构建子图
    subgraph = graph.subgraph(node_set)

    # BFS 连通性检查
    start = next(iter(node_set))
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        current = queue.popleft()
        for neighbor in subgraph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(node_set)


def get_connected_components(graph: nx.Graph, node_set: Set[str]) -> List[Set[str]]:
    """
    获取节点集合在子图中的所有连通分量。

    Args:
        graph: 完整的邻接图
        node_set: 待分析的节点集合

    Returns:
        连通分量列表，每个分量为 panel_id 集合
    """
    subgraph = graph.subgraph(node_set)
    return [set(comp) for comp in nx.connected_components(subgraph)]


def calculate_perimeter(graph: nx.Graph, zone_nodes: Set[str],
                        grid_size: float = 10.0,
                        panel_length: float = 12.0,
                        panel_width: float = 3.0) -> float:
    """
    计算分区的周长。

    周长 = 分区边界上所有暴露边的长度之和。
    暴露边 = 分区内节点的某个方向上没有同分区邻居。

    依据：项目书公式 (2) / 约束 (11)
      周长 = Σ t_i × (φ_上 + φ_下) + b × (φ_左 + φ_右)
      其中 φ = 1 表示该方向为边界（无同区邻居）

    Args:
        graph: 完整的邻接图
        zone_nodes: 分区内的节点集合（panel_id 集合）
        grid_size: 网格尺寸
        panel_length: 面板切割长度（纵向边长度，对应 t_l）
        panel_width: 面板宽度（横向边长度，对应 b）

    Returns:
        分区周长（米）
    """
    perimeter = 0.0

    for node in zone_nodes:
        node_data = graph.nodes[node]
        row, col = node_data["row"], node_data["col"]
        cut_l = node_data.get("cut_spec", (panel_length, panel_width))[0]
        cut_w = node_data.get("cut_spec", (panel_length, panel_width))[1]

        # 四个方向检查：如果邻居不在同分区内，则该边为边界边
        # 上下方向的边界长度 = 切割长度 t_l
        for dr in [-1, 1]:
            neighbor_coord = (row + dr, col)
            neighbor_id = _find_panel_at(graph, neighbor_coord)
            if neighbor_id is None or neighbor_id not in zone_nodes:
                perimeter += cut_l  # 纵向边界，长度为切割长度

        # 左右方向的边界长度 = 面板宽度 b
        for dc in [-1, 1]:
            neighbor_coord = (row, col + dc)
            neighbor_id = _find_panel_at(graph, neighbor_coord)
            if neighbor_id is None or neighbor_id not in zone_nodes:
                perimeter += cut_w  # 横向边界，长度为面板宽度

    return perimeter


def _find_panel_at(graph: nx.Graph, coord: Tuple[int, int]) -> Optional[str]:
    """在图中查找指定网格坐标处的面板 ID。"""
    for node, data in graph.nodes(data=True):
        if data.get("grid_coord") == coord or (data.get("row"), data.get("col")) == coord:
            return node
    return None


def build_coord_index(graph: nx.Graph) -> Dict[Tuple[int, int], str]:
    """
    构建 (row, col) -> panel_id 的快速索引。

    用于加速 _find_panel_at 的查找，适合在大量调用前预构建。

    Args:
        graph: 邻接图

    Returns:
        坐标到面板 ID 的字典
    """
    index = {}
    for node, data in graph.nodes(data=True):
        coord = (data["row"], data["col"])
        index[coord] = node
    return index


def calculate_perimeter_fast(zone_nodes: Set[str], graph: nx.Graph,
                              coord_index: Dict[Tuple[int, int], str]) -> float:
    """
    基于图边的周长计算。

    周长 = 分区中所有节点的"暴露边"长度之和。
    暴露边 = 图中该节点的某条边连接到分区外部（或该方向无邻居）。

    对于每个节点，计算其在图中有多少邻居在分区内：
    - 不在分区内的邻居方向 → 贡献边界长度
    - 没有邻居的方向 → 也贡献边界长度

    简化计算：每个节点贡献 4 条潜在边界边（上下左右），
    减去与同分区邻居共享的边数 × 对应边长。

    Args:
        zone_nodes: 分区内的节点集合
        graph: 邻接图
        coord_index: 坐标索引

    Returns:
        分区周长（米）
    """
    if not zone_nodes:
        return 0.0

    perimeter = 0.0
    for node in zone_nodes:
        data = graph.nodes[node]
        cut_l, cut_w = data.get("cut_spec", (2.0, 3.0))

        # 每个节点初始有 4 条边界边：上、下各 cut_l，左、右各 cut_w
        node_perimeter = 2 * cut_l + 2 * cut_w

        # 减去与同分区邻居共享的边
        for neighbor in graph.neighbors(node):
            if neighbor in zone_nodes:
                edge_data = graph.edges[node, neighbor]
                direction = edge_data.get("direction", "horizontal")
                if direction == "vertical":
                    node_perimeter -= cut_l  # 纵向共享，减去 cut_l
                else:
                    node_perimeter -= cut_w  # 横向共享，减去 cut_w

        perimeter += max(0, node_perimeter)

    return perimeter


def get_zone_bounding_box(graph: nx.Graph,
                           zone_nodes: Set[str]) -> Tuple[int, int, int, int]:
    """
    获取分区的网格包围盒。

    Returns:
        (min_row, max_row, min_col, max_col)
    """
    rows = [graph.nodes[n]["row"] for n in zone_nodes]
    cols = [graph.nodes[n]["col"] for n in zone_nodes]
    return min(rows), max(rows), min(cols), max(cols)


def get_boundary_nodes(graph: nx.Graph, zone_nodes: Set[str]) -> Set[str]:
    """
    获取分区的边界节点（至少有一个邻居不在分区内的节点）。

    用于局部搜索时的候选交换节点。

    Args:
        graph: 邻接图
        zone_nodes: 分区内的节点集合

    Returns:
        边界节点集合
    """
    boundary = set()
    for node in zone_nodes:
        for neighbor in graph.neighbors(node):
            if neighbor not in zone_nodes:
                boundary.add(node)
                break
    return boundary


def get_adjacent_external_nodes(graph: nx.Graph, zone_nodes: Set[str]) -> Set[str]:
    """
    获取分区外部但与分区相邻的节点（可扩展节点）。

    用于贪心扩展时的候选节点。

    Args:
        graph: 邻接图
        zone_nodes: 当前分区的节点集合

    Returns:
        外部相邻节点集合
    """
    external = set()
    for node in zone_nodes:
        for neighbor in graph.neighbors(node):
            if neighbor not in zone_nodes:
                external.add(neighbor)
    return external


def calculate_total_power(graph: nx.Graph, zone_nodes: Set[str]) -> float:
    """
    计算分区内所有面板的总功率。

    Args:
        graph: 邻接图
        zone_nodes: 分区内的节点集合

    Returns:
        总功率（kW）
    """
    return sum(graph.nodes[n].get("power", 0.0) for n in zone_nodes)


def validate_zone_constraints(graph: nx.Graph, zone_nodes: Set[str],
                               coord_index: Dict[Tuple[int, int], str],
                               min_panels: int = 18, max_panels: int = 26,
                               lb: float = 60.0, ub: float = 90.0) -> Dict:
    """
    验证单个分区是否满足所有约束。

    依据：项目书约束 (5)-(11)
    - 容量约束 (10): min_panels <= 面板数 <= max_panels
    - 连通性约束 (8): 分区内面板空间连续
    - 周长约束 (11): LB <= 周长 <= UB

    Args:
        graph: 邻接图
        zone_nodes: 分区内的节点集合
        coord_index: 坐标索引
        min_panels: 最少面板数（默认 18）
        max_panels: 最多面板数（默认 26）
        lb: 周长下界（默认 60m）
        ub: 周长上界（默认 90m）

    Returns:
        验证结果字典，含 is_valid, violations 列表, 各项指标值
    """
    violations = []
    n_panels = len(zone_nodes)

    # 容量约束
    capacity_ok = min_panels <= n_panels <= max_panels
    if not capacity_ok:
        violations.append(f"容量违规: 面板数={n_panels}, 要求[{min_panels},{max_panels}]")

    # 连通性约束
    connected = check_connectivity(graph, zone_nodes)
    if not connected:
        components = get_connected_components(graph, zone_nodes)
        violations.append(f"连通性违规: {len(components)} 个连通分量")

    # 周长约束
    perimeter = calculate_perimeter_fast(zone_nodes, graph, coord_index)
    perimeter_ok = lb <= perimeter <= ub
    if not perimeter_ok:
        violations.append(f"周长违规: 周长={perimeter:.1f}m, 要求[{lb},{ub}]")

    return {
        "is_valid": len(violations) == 0,
        "violations": violations,
        "n_panels": n_panels,
        "is_connected": connected,
        "perimeter": perimeter,
        "capacity_ok": capacity_ok,
        "perimeter_ok": perimeter_ok,
        "total_power": calculate_total_power(graph, zone_nodes),
    }
