"""
Fused Bi-Level FraudGNN — exact copy of main_gnn_model.py (Colab training
project) needed here to reconstruct the model architecture from the saved
checkpoint's state_dict. Keep these two files in sync if the architecture
changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class BatchNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class RealTimeNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class LambdaNeuralNetwork(nn.Module):
    def __init__(self, hidden_channels: int = 64):
        super().__init__()
        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)

    def forward(self, batch_emb: torch.Tensor, realtime_emb: torch.Tensor, use_batch: bool = True) -> torch.Tensor:
        if not use_batch:
            batch_emb = torch.zeros_like(realtime_emb)
        fused = torch.cat([batch_emb, realtime_emb], dim=1)
        return self.fusion(fused)


class FraudGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.batch_net = BatchNet(in_channels, hidden_channels, dropout)
        self.realtime_net = RealTimeNet(in_channels, hidden_channels, dropout)
        self.lambda_nn = LambdaNeuralNetwork(hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, use_batch: bool = True) -> torch.Tensor:
        realtime_emb = self.realtime_net(x, edge_index)

        if use_batch:
            batch_emb = self.batch_net(x, edge_index)
        else:
            # Real-time inference: BatchNet is never run (no full graph
            # available in production), Lambda substitutes a zero vector.
            batch_emb = torch.zeros_like(realtime_emb)

        fused = self.lambda_nn(batch_emb, realtime_emb, use_batch=use_batch)
        prob = self.classifier(fused)
        return prob.squeeze(-1)
