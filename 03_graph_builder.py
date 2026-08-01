"""
Build a transaction graph from cleaned PaySim data (output of 01_load_paysim.py).

Nodes = transactions. An edge is drawn between two transactions if they share
the same sender (nameOrig) within a 24-hour window, or the same receiver
(nameDest) within a 24-hour window. PaySim's `step` column is in hours, so a
24-hour window is simply `abs(step_a - step_b) <= 24`.

Edges are found with a sliding window (two-pointer) scan over each
sender/receiver group sorted by step, NOT a nested loop over all pairs —
each group is O(n) amortized instead of O(n^2).
"""

import pickle

import pandas as pd
import torch
from torch_geometric.data import Data

FULL_FEATURE_COLS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
# PaySim's fraud pattern almost always drains the origin account to ~0, which
# makes oldbalanceOrg/newbalanceOrig a near-deterministic giveaway — any
# tabular model (e.g. XGBoost) can nearly solve fraud detection from those two
# columns alone, which isn't a fair test of whether the GRAPH adds value. Flip
# FEATURE_COLS to REDUCED_FEATURE_COLS (and rebuild the graph + retrain) to
# see how much the graph structure alone can recover once that shortcut is
# removed. Keep this in sync with 07_xgboost_baseline.py's FEATURE_COLS so
# both models are compared on the same feature set.
REDUCED_FEATURE_COLS = ["step", "type", "amount"]

FEATURE_COLS = FULL_FEATURE_COLS
LABEL_COL = "isFraud"


class GraphBuilder:
    def __init__(self, time_window_hours: int = 24):
        self.time_window = time_window_hours

    def _sliding_window_edges(self, node_ids: "list[int]", steps: "list[int]") -> "list[tuple[int, int]]":
        """Connect nodes whose `step` values fall within `time_window` of each
        other, using a two-pointer sliding window over data sorted by step."""
        edges = []
        left = 0
        n = len(steps)
        for right in range(n):
            while steps[right] - steps[left] > self.time_window:
                left += 1
            for k in range(left, right):
                a, b = node_ids[k], node_ids[right]
                edges.append((a, b) if a < b else (b, a))
        return edges

    def _edges_for_grouping_col(self, df: pd.DataFrame, group_col: str) -> set:
        edge_set = set()
        sorted_df = df.sort_values("step")
        for _, group in sorted_df.groupby(group_col, sort=False):
            if len(group) < 2:
                continue
            node_ids = group.index.to_numpy()
            steps = group["step"].to_numpy()
            edge_set.update(self._sliding_window_edges(node_ids, steps))
        return edge_set

    def build(self, df: pd.DataFrame, feature_cols=FEATURE_COLS, label_col=LABEL_COL) -> Data:
        df = df.reset_index(drop=True)

        edge_set = set()
        edge_set |= self._edges_for_grouping_col(df, "nameOrig")
        edge_set |= self._edges_for_grouping_col(df, "nameDest")

        if edge_set:
            src = [a for a, b in edge_set] + [b for a, b in edge_set]
            dst = [b for a, b in edge_set] + [a for a, b in edge_set]
        else:
            src, dst = [], []

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        x = torch.tensor(df[feature_cols].values, dtype=torch.float)
        y = torch.tensor(df[label_col].values, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)


def main():
    with open("paysim_clean.pkl", "rb") as f:
        df = pickle.load(f)

    builder = GraphBuilder(time_window_hours=24)
    data = builder.build(df)

    print(data)
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Fraud nodes: {int(data.y.sum())} / {data.num_nodes}")

    torch.save(data, "paysim_graph.pt")
    print("Saved graph to paysim_graph.pt")


if __name__ == "__main__":
    main()
