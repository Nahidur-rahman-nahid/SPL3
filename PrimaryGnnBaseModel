"""
05_gnn_model.py
GraphSentinel — FraudGNN Model Definition
==========================================
Implements a bi-level Graph Neural Network for fraud detection.

Architecture:
  Layer 1 (Structural):  GCNConv  — captures global graph structure
  Layer 2 (Inductive):   SAGEConv — handles unseen nodes at inference time
  Layer 3 (Classification): Linear — outputs fraud probability per node

The bi-level design mirrors the BRIGHT paper's insight:
  - GCNConv propagates fraud signals across the full neighbourhood (batch)
  - SAGEConv aggregates sampled neighbour features inductively (real-time)
  - Together they handle both training-time and inference-time transactions

Authors: Md. Nahidur Rahman Nahid  (BSSE-1429, IIT University of Dhaka)
Paper references:
  BRIGHT  — CIKM 2022  (https://arxiv.org/abs/2205.13084)
  GraphSAGE — NeurIPS 2017 (https://arxiv.org/abs/1706.02216)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (matches feature vector built in 03_graph_builder.py)
# ─────────────────────────────────────────────────────────────────────────────

# PaySim feature vector (7 features per transaction node):
#   0: amount (StandardScaled)
#   1: step / TransactionDT (StandardScaled)
#   2: transaction type (label-encoded int)
#   3: oldbalanceOrg / dist1 proxy (StandardScaled)
#   4: newbalanceOrig (StandardScaled)
#   5: oldbalanceDest (StandardScaled)
#   6: newbalanceDest (StandardScaled)
INPUT_DIM   = 7      # feature vector size per node
HIDDEN_DIM  = 64     # hidden layer dimension
OUTPUT_DIM  = 1      # binary classification (fraud = 1, legit = 0)
DROPOUT     = 0.3    # dropout rate between layers


# ─────────────────────────────────────────────────────────────────────────────
# BATCH NET  —  GCNConv layers (structural, offline training)
# ─────────────────────────────────────────────────────────────────────────────
# GCNConv is transductive: it uses the full normalised adjacency matrix.
# In training (batch jobs on Colab), the entire PaySim graph is known,
# so GCNConv can propagate signals globally across all connected nodes.
# This is the "Batch Net" idea from BRIGHT — used for offline model training.
#
# GCNConv formula per layer:
#   H' = D^(-1/2) * A_hat * D^(-1/2) * H * W
# where A_hat = A + I (adjacency + self-loops), D = degree matrix
# Reference: Kipf & Welling, ICLR 2017

class BatchNet(nn.Module):
    """
    Two-layer GCN for structural fraud signal propagation.
    Used during batch training on the full transaction graph.
    Input:  node features x  [N, INPUT_DIM]
    Output: node embeddings  [N, HIDDEN_DIM]
    """
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn1   = nn.BatchNorm1d(hidden_channels)
        self.bn2   = nn.BatchNorm1d(hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1: aggregate from 1-hop neighbours
        x = self.conv1(x, edge_index)     # [N, HIDDEN_DIM]
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)

        # Layer 2: aggregate from 2-hop neighbours (neighbours of neighbours)
        x = self.conv2(x, edge_index)     # [N, HIDDEN_DIM]
        x = self.bn2(x)
        x = F.relu(x)

        return x                           # [N, HIDDEN_DIM]


# ─────────────────────────────────────────────────────────────────────────────
# REAL-TIME NET  —  SAGEConv layers (inductive, online inference)
# ─────────────────────────────────────────────────────────────────────────────
# SAGEConv is inductive: it learns aggregator FUNCTIONS, not fixed embeddings.
# At inference time, a brand-new transaction TX_999 that never existed during
# training can be scored by fetching its neighbours from Redis and running
# the learned aggregators on that local subgraph.
# This is the "RT Net" idea from BRIGHT — used for real-time online scoring.
#
# SAGEConv formula per layer:
#   h_v = W1 * h_v + W2 * MEAN( h_u for u in N(v) )
# where N(v) is a sampled fixed-size neighbourhood of node v
# Reference: Hamilton et al., NeurIPS 2017

class RealTimeNet(nn.Module):
    """
    Two-layer GraphSAGE for inductive fraud scoring on unseen transactions.
    Used at real-time inference — new transactions scored without retraining.
    Input:  node features x  [N, INPUT_DIM]
    Output: node embeddings  [N, HIDDEN_DIM]
    """
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.sage2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.bn1   = nn.BatchNorm1d(hidden_channels)
        self.bn2   = nn.BatchNorm1d(hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1: aggregate mean of sampled 1-hop neighbours
        x = self.sage1(x, edge_index)     # [N, HIDDEN_DIM]
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)

        # Layer 2: aggregate mean of sampled 2-hop neighbours
        x = self.sage2(x, edge_index)     # [N, HIDDEN_DIM]
        x = self.bn2(x)
        x = F.relu(x)

        return x                           # [N, HIDDEN_DIM]


# ─────────────────────────────────────────────────────────────────────────────
# LAMBDA NEURAL NETWORK  —  combines BatchNet + RealTimeNet embeddings
# ─────────────────────────────────────────────────────────────────────────────
# BRIGHT's key contribution is the Lambda Neural Network (LNN):
# it fuses the batch (structural) embedding with the real-time (inductive)
# embedding using concatenation + linear projection.
#
# During training:    both BatchNet and RealTimeNet receive the full graph
# During inference:   only RealTimeNet receives the local subgraph from Redis
#                     BatchNet embedding is replaced with a zero vector
#
# This means the model degrades gracefully at inference time —
# RealTimeNet alone is sufficient to score a new transaction.

class LambdaNeuralNetwork(nn.Module):
    """
    Fuses batch + real-time embeddings via concatenation and linear projection.
    At inference time, set use_batch=False to use only the inductive net.
    Input:  x [N, INPUT_DIM], edge_index [2, E]
    Output: fused embedding [N, HIDDEN_DIM]
    """
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.batch_net = BatchNet(in_channels, hidden_channels)
        self.rt_net    = RealTimeNet(in_channels, hidden_channels)

        # Project concatenated embedding back to HIDDEN_DIM
        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        self.bn     = nn.BatchNorm1d(hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        use_batch: bool = True
    ) -> torch.Tensor:

        # Always compute inductive embedding (works on any graph/subgraph)
        h_rt = self.rt_net(x, edge_index)               # [N, HIDDEN_DIM]

        if use_batch:
            # Training: use structural embedding as well
            h_batch = self.batch_net(x, edge_index)      # [N, HIDDEN_DIM]
        else:
            # Inference on new transaction: zero out batch embedding
            # Model still works — RT net carries the prediction
            h_batch = torch.zeros_like(h_rt)             # [N, HIDDEN_DIM]

        # Concatenate and project  [N, HIDDEN_DIM*2] → [N, HIDDEN_DIM]
        h_fused = torch.cat([h_batch, h_rt], dim=1)      # [N, HIDDEN_DIM*2]
        h_fused = self.fusion(h_fused)                    # [N, HIDDEN_DIM]
        h_fused = self.bn(h_fused)
        h_fused = F.relu(h_fused)

        return h_fused                                    # [N, HIDDEN_DIM]


# ─────────────────────────────────────────────────────────────────────────────
# FRAUD GNN  —  complete model with classifier head
# ─────────────────────────────────────────────────────────────────────────────

class FraudGNN(nn.Module):
    """
    Complete GraphSentinel fraud detection model.

    Architecture:
        Input features [N, 7]
            ↓
        LambdaNeuralNetwork (GCNConv + SAGEConv bi-level fusion)
            ↓
        Fused embedding [N, 64]
            ↓
        Classifier head (Linear → Sigmoid)
            ↓
        Fraud probability [N, 1]  in range [0, 1]

    Usage:
        Training (full graph):
            model.train()
            out = model(data.x, data.edge_index, use_batch=True)
            loss = criterion(out, data.y.float())

        Inference (real-time, local subgraph from Redis):
            model.eval()
            with torch.no_grad():
                out = model(subgraph.x, subgraph.edge_index, use_batch=False)
                fraud_prob = out[0].item()   # score for the new transaction
    """

    def __init__(
        self,
        in_channels:     int = INPUT_DIM,
        hidden_channels: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.lnn = LambdaNeuralNetwork(in_channels, hidden_channels)

        # Classifier head: embedding → fraud probability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(hidden_channels // 2, OUTPUT_DIM),
            nn.Sigmoid()          # output in [0, 1] — fraud probability
        )

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        use_batch:  bool = True
    ) -> torch.Tensor:
        """
        Args:
            x:          Node feature matrix  [N, in_channels]
            edge_index: Graph edge indices   [2, E]
            use_batch:  True during training, False during real-time inference

        Returns:
            Fraud probabilities              [N, 1]  (values in 0–1)
        """
        h = self.lnn(x, edge_index, use_batch=use_batch)  # [N, HIDDEN_DIM]
        out = self.classifier(h)                           # [N, 1]
        return out

    def predict(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        threshold:  float = 0.75
    ) -> dict:
        """
        Convenience method for inference — returns dict with decision.
        The new transaction must be the FIRST node (index 0) in the subgraph.
        Neighbours are nodes 1..N-1 (fetched from Redis).

        Args:
            x:          Subgraph node features   [N, in_channels]
            edge_index: Subgraph edge indices     [2, E]
            threshold:  Decision threshold        default 0.75

        Returns:
            {
                'fraud_probability': float,   # 0.0 – 1.0
                'decision':          str,      # 'ALLOW' / 'REVIEW' / 'BLOCK'
                'risk_level':        str       # 'LOW' / 'MEDIUM' / 'HIGH' / 'CRITICAL'
            }
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x, edge_index, use_batch=False)  # [N, 1]
        score = probs[0].item()   # score for node 0 (the new transaction)

        # Decision logic
        if score < 0.4:
            decision, risk = 'ALLOW',  'LOW'
        elif score < threshold:
            decision, risk = 'REVIEW', 'MEDIUM'
        elif score < 0.9:
            decision, risk = 'BLOCK',  'HIGH'
        else:
            decision, risk = 'BLOCK',  'CRITICAL'

        return {
            'fraud_probability': round(score, 4),
            'decision':          decision,
            'risk_level':        risk
        }


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTION  —  weighted binary cross-entropy for class imbalance
# ─────────────────────────────────────────────────────────────────────────────
# PaySim fraud rate ≈ 0.13%  →  class imbalance ratio ≈ 770:1
# Without weighting, the model learns to predict "all legitimate" and
# achieves 99.87% accuracy while catching zero fraud.
#
# Solution: assign a higher weight to the minority fraud class.
# pos_weight = (number of legitimate transactions) / (number of fraud transactions)
# This penalises missing a fraud 770× more than a false positive.

def get_loss_function(data: Data) -> nn.BCEWithLogitsLoss:
    """
    Computes class-balanced loss weight from the training data labels.
    Call this once before training and reuse throughout.

    Args:
        data:  PyG Data object with data.y labels (0=legit, 1=fraud)

    Returns:
        nn.BCEWithLogitsLoss with pos_weight set for fraud class
    """
    n_legit = (data.y == 0).sum().float()
    n_fraud = (data.y == 1).sum().float()

    if n_fraud == 0:
        raise ValueError("No fraud samples found in the dataset. Check your labels.")

    pos_weight = n_legit / n_fraud
    print(f"  Class balance — Legitimate: {int(n_legit):,}  "
          f"Fraud: {int(n_fraud):,}  "
          f"pos_weight: {pos_weight:.1f}")

    # NOTE: BCEWithLogitsLoss expects RAW LOGITS (before sigmoid), not probabilities.
    # If using the FraudGNN.forward() which applies Sigmoid, use BCELoss instead.
    # This loss function is for use with a modified forward() without final Sigmoid.
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def get_bce_loss() -> nn.BCELoss:
    """
    Standard BCE loss for use with FraudGNN.forward() which applies Sigmoid.
    Does NOT handle class imbalance — pair with oversampling or SMOTE instead.
    Use get_loss_function() for weighted version.
    """
    return nn.BCELoss()


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:      FraudGNN,
    data:       Data,
    mask:       torch.Tensor,
    threshold:  float = 0.5,
    device:     str   = 'cpu'
) -> dict:
    """
    Evaluate FraudGNN on a given node mask (val_mask or test_mask).
    Returns AUC-ROC, Precision, Recall, F1, and False Positive Rate.

    Args:
        model:      Trained FraudGNN instance
        data:       Full PyG Data object
        mask:       Boolean tensor of nodes to evaluate [N]
        threshold:  Decision threshold for binary prediction
        device:     'cuda' or 'cpu'

    Returns:
        dict with keys: auc_roc, precision, recall, f1, fpr
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        probs = model(data.x, data.edge_index, use_batch=True)  # [N, 1]
        probs = probs.squeeze(1)                                 # [N]

    # Extract only the masked nodes
    y_true  = data.y[mask].cpu().numpy()
    y_score = probs[mask].cpu().numpy()
    y_pred  = (y_score >= threshold).astype(int)

    # Guard: AUC-ROC requires both classes to be present
    if len(np.unique(y_true)) < 2:
        print("  Warning: only one class in evaluation set — AUC-ROC undefined.")
        auc = 0.0
    else:
        auc = roc_auc_score(y_true, y_score)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    # False Positive Rate: FP / (FP + TN)
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        'auc_roc':   round(float(auc),       4),
        'precision': round(float(precision),  4),
        'recall':    round(float(recall),     4),
        'f1':        round(float(f1),         4),
        'fpr':       round(float(fpr),        4),
    }


def print_results_table(results: dict, model_name: str = "FraudGNN") -> None:
    """
    Pretty-print a results dictionary as a formatted table row.
    Useful for building the 3-model comparison table in your thesis.
    """
    print(f"\n{'─'*60}")
    print(f"  Model     : {model_name}")
    print(f"  AUC-ROC   : {results['auc_roc']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print(f"  FPR       : {results['fpr']:.4f}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY — convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    in_channels:     int   = INPUT_DIM,
    hidden_channels: int   = HIDDEN_DIM,
    device:          str   = 'cpu'
) -> FraudGNN:
    """
    Build and return a FraudGNN model, moved to the specified device.

    Args:
        in_channels:     Number of input features per node (default 7)
        hidden_channels: Hidden dimension size (default 64)
        device:          'cuda' if GPU available, else 'cpu'

    Returns:
        FraudGNN model on device

    Example:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = build_model(device=device)
        print(model)
    """
    model = FraudGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  FraudGNN built — {n_params:,} trainable parameters on {device}")
    return model


def save_model(model: FraudGNN, path: str = 'model.pt') -> None:
    """
    Save model weights to disk. Load later with load_model().

    Args:
        model: Trained FraudGNN instance
        path:  File path for saved weights (default 'model.pt')
    """
    torch.save({
        'model_state_dict':  model.state_dict(),
        'in_channels':       INPUT_DIM,
        'hidden_channels':   HIDDEN_DIM,
    }, path)
    print(f"  Model saved to {path}")


def load_model(path: str = 'model.pt', device: str = 'cpu') -> FraudGNN:
    """
    Load a saved FraudGNN from disk.
    Called by ml_service/main.py at FastAPI startup.

    Args:
        path:   Path to saved .pt file
        device: Device to load onto

    Returns:
        FraudGNN model in eval() mode, ready for inference

    Example:
        model = load_model('model.pt', device='cpu')
        result = model.predict(x, edge_index, threshold=0.75)
    """
    checkpoint = torch.load(path, map_location=device)
    model = FraudGNN(
        in_channels=checkpoint['in_channels'],
        hidden_channels=checkpoint['hidden_channels']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Model loaded from {path} — ready for inference on {device}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK  —  run this file directly to verify everything works
# ─────────────────────────────────────────────────────────────────────────────
# Run: python 05_gnn_model.py
# Expected output: model summary + forward pass shape + dummy inference result

if __name__ == '__main__':
    import torch

    print("\n" + "="*60)
    print("  GraphSentinel — FraudGNN Sanity Check")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # ── Step 1: Build model ───────────────────────────────────────────────
    model = build_model(
        in_channels=INPUT_DIM,
        hidden_channels=HIDDEN_DIM,
        device=device
    )
    print(f"\n  Architecture:\n{model}\n")

    # ── Step 2: Create a tiny dummy graph (5 nodes, 4 edges) ─────────────
    # Simulates a fraud ring: TX_0 connected to TX_1,TX_2,TX_3 (same device)
    # TX_0 is the new incoming transaction (node index 0)
    N = 5
    x = torch.randn(N, INPUT_DIM, device=device)  # random features

    # Edges: TX_0 ↔ TX_1, TX_0 ↔ TX_2, TX_0 ↔ TX_3, TX_1 ↔ TX_4
    edge_index = torch.tensor([
        [0, 1, 0, 2, 0, 3, 1, 4],
        [1, 0, 2, 0, 3, 0, 4, 1]
    ], dtype=torch.long, device=device)

    y = torch.tensor([1, 1, 0, 1, 0], dtype=torch.float, device=device)

    # ── Step 3: Training forward pass ────────────────────────────────────
    model.train()
    probs = model(x, edge_index, use_batch=True)
    print(f"  Training forward pass:")
    print(f"    Input  shape: x={list(x.shape)}, edge_index={list(edge_index.shape)}")
    print(f"    Output shape: probs={list(probs.shape)}")
    print(f"    Probabilities: {probs.squeeze().tolist()}")

    # ── Step 4: Real-time inference forward pass ──────────────────────────
    # use_batch=False simulates scoring a brand-new transaction
    model.eval()
    result = model.predict(x, edge_index, threshold=0.75)
    print(f"\n  Real-time inference (node 0 = new transaction):")
    print(f"    Fraud probability : {result['fraud_probability']}")
    print(f"    Decision          : {result['decision']}")
    print(f"    Risk level        : {result['risk_level']}")

    # ── Step 5: Save and reload ───────────────────────────────────────────
    save_model(model, '/tmp/test_model.pt')
    loaded = load_model('/tmp/test_model.pt', device=device)

    with torch.no_grad():
        p1 = model(x, edge_index, use_batch=False)
        p2 = loaded(x, edge_index, use_batch=False)

    assert torch.allclose(p1, p2, atol=1e-6), "Loaded model outputs differ!"
    print(f"\n  Save/load check: PASSED ✓")

    print("\n" + "="*60)
    print("  All checks passed. Ready for training in 06_train_gnn.py")
    print("="*60 + "\n")
