# tests/conftest.py
"""
Shared fixtures and tiny synthetic helpers used across all test modules.
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Any, List, Tuple
from unittest.mock import MagicMock

from epiplexity.model import TTimeProbabilisticModel


# ── Minimal concrete implementation for abstract-class tests ──────────────────

class ConstantModel(TTimeProbabilisticModel):
    """
    Trivial deterministic model: always returns a fixed log-prob
    and a fixed description length. Useful for arithmetic invariant tests.
    """

    def __init__(self, desc_bits: float = 1024.0, log_prob_val: float = -1.0):
        self._desc_bits = desc_bits
        self._lp = log_prob_val

    def description_length_bits(self) -> float:
        return self._desc_bits

    def log_prob(self, x: Any) -> float:
        return self._lp

    def sample(self, n: int = 1) -> List[float]:
        return [0.0] * n

    def raw_bits(self, X: Any) -> float:
        return float(len(X) * 64)  # 64 bits per item as baseline


# ── Minimal PyTorch modules ────────────────────────────────────────────────────

class TinyMLP(nn.Module):
    """2-layer MLP, input_dim → 8 → num_classes."""

    def __init__(self, input_dim: int = 4, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TinyLM(nn.Module):
    """Minimal causal LM: embedding + single linear head."""

    VOCAB_SIZE = 16

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB_SIZE, 8)
        self.head = nn.Linear(8, self.VOCAB_SIZE)

    def forward(self, input_ids, **kwargs):
        # input_ids: (B, T)
        emb = self.embed(input_ids)          # (B, T, 8)
        logits = self.head(emb)              # (B, T, VOCAB_SIZE)
        return MagicMock(logits=logits)      # mimics HF ModelOutput


# ── Minimal graph-like object ──────────────────────────────────────────────────

class TinyGraph:
    """Minimal stand-in for a PyTorch Geometric Data object."""

    def __init__(self, num_nodes: int = 5, num_features: int = 4, num_edges: int = 6):
        self.x = torch.randn(num_nodes, num_features)
        self.edge_index = torch.randint(0, num_nodes, (2, num_edges))
        self.u = None
        self.graph_attr = None

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        return self


class TinyGNN(nn.Module):
    """
    Toy GNN: mean-pools node features, then classifies.
    model(graph) -> logits of shape (1, num_classes).
    """

    def __init__(self, input_dim: int = 4, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, graph):
        pooled = graph.x.mean(dim=0, keepdim=True)  # (1, F)
        return self.fc(pooled)                       # (1, num_classes)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def constant_model():
    return ConstantModel(desc_bits=1024.0, log_prob_val=-2.0)


@pytest.fixture
def tiny_mlp():
    torch.manual_seed(0)
    return TinyMLP(input_dim=4, num_classes=3)


@pytest.fixture
def tiny_lm():
    torch.manual_seed(0)
    return TinyLM()


@pytest.fixture
def tiny_gnn():
    torch.manual_seed(0)
    return TinyGNN(input_dim=4, num_classes=3)


@pytest.fixture
def float_dataset():
    """10 items, each a 4-dim float32 tensor + integer label."""
    torch.manual_seed(1)
    return [(torch.randn(4), i % 3) for i in range(10)]


@pytest.fixture
def graph_dataset():
    """5 synthetic graph objects with labels."""
    torch.manual_seed(2)
    return [(TinyGraph(), i % 3) for i in range(5)]