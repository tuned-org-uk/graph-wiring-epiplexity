# tests/test_gnn.py
"""
Tests for GNNModelAdapter (epiplexity/algorithms/gnn.py).

Verifies:
  - description_length_bits accounts for params + overhead.
  - log_prob handles 1D and 2D logit shapes.
  - log_prob is finite and ≤ 0 for all labels.
  - raw_bits correctly sums node features, edge_index, and global features.
  - sample() raises NotImplementedError.
  - Graphs with optional global features (u / graph_attr) are handled.
  - EpiplexityEngine integration.
"""

import math
import pytest
import torch
import numpy as np

from epiplexity.algorithms.gnn import GNNModelAdapter, GNNEpiplexityConfig
from epiplexity.engine import EpiplexityEngine
from .conftest import TinyGNN, TinyGraph


@pytest.fixture
def gnn_adapter(tiny_gnn):
    cfg = GNNEpiplexityConfig(
        bits_per_param=32,
        overhead_bits=0.0,
        bits_per_node_feature=32,
        bits_per_edge_index=32,
        bits_per_graph_feature=32,
    )
    return GNNModelAdapter(tiny_gnn, cfg)


class TestGNNDescriptionLength:

    def test_description_length_positive(self, gnn_adapter):
        assert gnn_adapter.description_length_bits() > 0

    def test_description_length_reflects_param_count(self, tiny_gnn):
        cfg = GNNEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
        adapter = GNNModelAdapter(tiny_gnn, cfg)
        num_params = sum(p.numel() for p in tiny_gnn.parameters())
        assert adapter.description_length_bits() == pytest.approx(num_params * 32)

    def test_overhead_added(self, tiny_gnn):
        cfg_no = GNNEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
        cfg_ov = GNNEpiplexityConfig(bits_per_param=32, overhead_bits=512.0)
        a_no = GNNModelAdapter(tiny_gnn, cfg_no)
        a_ov = GNNModelAdapter(tiny_gnn, cfg_ov)
        assert a_ov.description_length_bits() == pytest.approx(
            a_no.description_length_bits() + 512.0
        )


class TestGNNLogProb:

    def test_log_prob_finite(self, gnn_adapter, graph_dataset):
        for item in graph_dataset:
            lp = gnn_adapter.log_prob(item)
            assert math.isfinite(lp), f"log_prob not finite: {lp}"

    def test_log_prob_non_positive(self, gnn_adapter, graph_dataset):
        for item in graph_dataset:
            assert gnn_adapter.log_prob(item) <= 1e-6

    def test_log_prob_all_labels(self, gnn_adapter):
        """All class indices must produce valid log-probs for the same graph."""
        g = TinyGraph(num_nodes=4, num_features=4, num_edges=4)
        for label in range(3):
            lp = gnn_adapter.log_prob((g, label))
            assert math.isfinite(lp)
            assert lp <= 1e-6

    def test_log_prob_1d_logit_shape(self, tiny_gnn):
        """Model returning 1D logits (C,) rather than (1, C) must be handled."""

        class GNN1D(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 3)

            def forward(self, graph):
                pooled = graph.x.mean(dim=0)  # (F,) — no keepdim
                return self.fc(pooled)         # (C,)

        cfg = GNNEpiplexityConfig()
        adapter_1d = GNNModelAdapter(GNN1D(), cfg)
        g = TinyGraph()
        lp = adapter_1d.log_prob((g, 0))
        assert math.isfinite(lp)


class TestGNNRawBits:

    def test_raw_bits_node_features(self, gnn_adapter):
        """raw_bits for a dataset of identical graphs scales linearly."""
        g = TinyGraph(num_nodes=5, num_features=4, num_edges=0)
        g.edge_index = torch.zeros(2, 0, dtype=torch.long)  # no edges
        dataset_1 = [(g, 0)]
        dataset_2 = [(g, 0), (g, 1)]
        rb1 = gnn_adapter.raw_bits(dataset_1)
        rb2 = gnn_adapter.raw_bits(dataset_2)
        assert rb2 == pytest.approx(2 * rb1)

    def test_raw_bits_includes_edge_index(self, gnn_adapter):
        """Adding edges increases raw_bits."""
        g_no_edges = TinyGraph(num_nodes=5, num_features=4, num_edges=0)
        g_no_edges.edge_index = torch.zeros(2, 0, dtype=torch.long)
        g_with_edges = TinyGraph(num_nodes=5, num_features=4, num_edges=6)

        rb_no = gnn_adapter.raw_bits([(g_no_edges, 0)])
        rb_with = gnn_adapter.raw_bits([(g_with_edges, 0)])
        assert rb_with > rb_no

    def test_raw_bits_includes_global_u(self, gnn_adapter):
        """Graphs with a 'u' global attribute add to raw_bits."""
        g_no_u = TinyGraph(num_nodes=4, num_features=4, num_edges=4)
        g_with_u = TinyGraph(num_nodes=4, num_features=4, num_edges=4)
        g_with_u.u = torch.randn(8)  # 8 global features

        rb_no_u = gnn_adapter.raw_bits([(g_no_u, 0)])
        rb_with_u = gnn_adapter.raw_bits([(g_with_u, 0)])
        assert rb_with_u > rb_no_u

    def test_raw_bits_includes_graph_attr(self, gnn_adapter):
        """graph_attr attribute is treated the same as u."""
        g_base = TinyGraph(num_nodes=4, num_features=4, num_edges=4)
        g_attr = TinyGraph(num_nodes=4, num_features=4, num_edges=4)
        g_attr.graph_attr = torch.randn(4)

        rb_base = gnn_adapter.raw_bits([(g_base, 0)])
        rb_attr = gnn_adapter.raw_bits([(g_attr, 0)])
        assert rb_attr > rb_base


class TestGNNSample:

    def test_sample_raises_not_implemented(self, gnn_adapter):
        with pytest.raises(NotImplementedError):
            gnn_adapter.sample(n=1)


class TestGNNEngineIntegration:

    def test_engine_mdl_identity(self, tiny_gnn, graph_dataset):
        cfg = GNNEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
        adapter = GNNModelAdapter(tiny_gnn, cfg)
        engine = EpiplexityEngine(adapter, graph_dataset)
        assert engine.mdltotal_bits == pytest.approx(
            engine.structural_bits + engine.total_entropy_bits
        )

    def test_engine_entropy_bits_shape(self, tiny_gnn, graph_dataset):
        cfg = GNNEpiplexityConfig()
        adapter = GNNModelAdapter(tiny_gnn, cfg)
        engine = EpiplexityEngine(adapter, graph_dataset)
        assert engine.entropy_bits.shape == (len(graph_dataset),)