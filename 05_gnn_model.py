"""
FraudGNN — GraphSAGE-based node classifier for PaySim fraud detection.

Do NOT change this file. Import FraudGNN into the training notebook as-is:

    from gnn_model import FraudGNN
    model = FraudGNN(in_channels=data.num_node_features, hidden_channels=64, num_classes=2)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_classes: int = 2,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert num_layers >= 2, "FraudGNN needs at least 2 SAGEConv layers"

        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.classifier(x)
        return out
