"""
FraudGCN — GCN-based node classifier, used purely as a comparison baseline
against FraudGNN (05_gnn_model.py, SAGEConv-based).

GCNConv's message passing is tied to the normalized adjacency of the graph it
was trained on, which is why GCN is the textbook example of a *transductive*
model. This file exists to demonstrate that limitation side-by-side with
GraphSAGE's inductive generalization in 06_train_compare.py — same depth and
hidden size as FraudGNN, so the only variable is the conv layer type.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FraudGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_classes: int = 2,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert num_layers >= 2, "FraudGCN needs at least 2 GCNConv layers"

        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.classifier(x)
        return out
