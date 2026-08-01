"""
Sanity-check the graph built by 03_graph_builder.py before spending hours
training on it. Prints basic stats and plots a small sample subgraph.

validate_graph.py
"""

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import subgraph

GRAPH_PATH = "paysim_graph.pt"
SUBGRAPH_SIZE = 20


def load_graph(path: str = GRAPH_PATH):
    return torch.load(path, weights_only=False)


def print_stats(data):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    fraud_count = int(data.y.sum())
    avg_degree = num_edges / num_nodes

    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Fraud nodes: {fraud_count} ({fraud_count / num_nodes:.4%})")
    print(f"Average degree: {avg_degree:.4f}")

    if hasattr(data, "edge_type") and data.edge_type is not None:
        for name, code in [("same-sender", 0), ("same-receiver", 1), ("money-flow", 2)]:
            count = int((data.edge_type == code).sum())
            print(f"  {name} edges: {count} ({count / num_edges:.2%})")


def _bfs_neighborhood(data, seed_node: int, n: int):
    """Expand outward from seed_node following real edges (BFS), stopping at
    n nodes. Unlike padding with arbitrary unrelated nodes, every node
    returned is actually reachable from the seed, so the plot never shows
    fake isolated nodes."""
    row, col = data.edge_index
    visited = [seed_node]
    seen = {seed_node}
    frontier = [seed_node]

    while frontier and len(visited) < n:
        next_frontier = []
        for node in frontier:
            neighbors = torch.cat([col[row == node], row[col == node]]).unique().tolist()
            for nb in neighbors:
                if nb not in seen:
                    seen.add(nb)
                    visited.append(nb)
                    next_frontier.append(nb)
                if len(visited) >= n:
                    break
            if len(visited) >= n:
                break
        frontier = next_frontier

    return torch.tensor(visited[:n], dtype=torch.long)


def plot_subgraph(data, n: int = SUBGRAPH_SIZE):
    # Seed on a fraud node if one exists so the sample plot is actually
    # informative instead of showing an all-normal neighborhood.
    fraud_idx = (data.y == 1).nonzero(as_tuple=True)[0]
    seed_node = int(fraud_idx[0]) if len(fraud_idx) > 0 else 0

    node_ids = _bfs_neighborhood(data, seed_node, n)
    if len(node_ids) < n:
        print(f"Note: seed node's connected neighborhood only has {len(node_ids)} nodes "
              f"(requested {n}) — showing the real component instead of padding with unrelated nodes.")

    sub_edge_index, _ = subgraph(node_ids, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    print(f"Subgraph: {len(node_ids)} nodes, {sub_edge_index.size(1)} directed edge entries")

    g = nx.Graph()
    g.add_nodes_from(range(len(node_ids)))
    g.add_edges_from(sub_edge_index.t().tolist())

    labels = data.y[node_ids]
    colors = ["red" if label == 1 else "steelblue" for label in labels]

    plt.figure(figsize=(9, 7))
    # Larger k spaces nodes further apart so edges between tightly-clustered
    # nodes are actually visible instead of hidden under the node markers.
    pos = nx.spring_layout(g, seed=42, k=1.5 / (len(node_ids) ** 0.5), iterations=100)
    nx.draw(
        g, pos,
        node_color=colors, with_labels=True,
        node_size=350, font_size=8,
        edge_color="gray", width=1.2,
    )
    plt.title(f"Neighborhood around a fraud node ({len(node_ids)} nodes, red = fraud, blue = normal)")
    plt.savefig("subgraph_sample.png", dpi=150, bbox_inches="tight")
    print("Saved subgraph plot to subgraph_sample.png")
    plt.show()


def main():
    data = load_graph()
    print_stats(data)
    plot_subgraph(data)


if __name__ == "__main__":
    main()
