# tests/test_epiplexity_properties.py
"""
Cross-algorithm epiplexity property tests.

These tests go beyond arithmetic correctness and verify the *semantic*
properties of epiplexity as defined in Finzi et al. (2026):

  P1  MDL identity          S_T + H_T == MDL_T
  P2  Compression test      MDL_T < raw_bits  iff the model captures structure
  P3  Non-negativity        H_T(x_i) >= 0  (log_prob <= 0)
  P4  Observer-dependence   larger S_T with more compute/params
  P5  H_T ordering          a model assigns lower H_T to data it explains well
  P6  Per-item consistency  per-item entropy sums to total H_T
  P7  ArrowSpace LGMRF      smooth signals have lower H_T than rough signals
  P8  Epiplexity rank       S_T rank preserved under monotone capacity scaling
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import scipy.sparse as sp
from typing import Any, List
from unittest.mock import MagicMock

from epiplexity.model import TTimeProbabilisticModel
from epiplexity.engine import EpiplexityEngine
from epiplexity.algorithms.torch_classifier import (
    TorchClassifierModelAdapter,
    TorchClassifierEpiplexityConfig,
)
from epiplexity.algorithms.transformer_lm import (
    TransformerLMModelAdapter,
    TransformerLMEpiplexityConfig,
)
from epiplexity.algorithms.gnn import GNNModelAdapter, GNNEpiplexityConfig
from epiplexity.algorithms.arrowspace import ArrowSpaceModelAdapter
from epiplexity.tests.conftest import TinyMLP, TinyLM, TinyGNN, TinyGraph


# ── Minimal ArrowSpaceProbabilisticModel for property tests ───────────────────
# Reproduced inline so the tests have no dependency on the research notebook.

def _elias_gamma_bits(x: int) -> int:
    return 2 * math.floor(math.log2(max(1, x))) + 1


class _MinimalArrowSpaceModel:
    """
    Laplacian-constrained Gaussian MRF (LGMRF) probabilistic model.
    Implements the interface expected by ArrowSpaceModelAdapter.
    """

    def __init__(self, L_F: sp.spmatrix, beta: float = 1.0, gamma: float = 1e-3):
        import scipy.sparse.linalg as spla
        self.F = L_F.shape[0]
        self.L_F = L_F
        self.beta = beta
        self.gamma = gamma
        Q = beta * L_F.tocsc() + gamma * sp.eye(self.F, format="csc")
        self._lu = spla.splu(Q)
        self.Q = Q
        log_det_Q = float(np.sum(np.log(np.abs(self._lu.U.diagonal()))))
        self._log_Z = 0.5 * self.F * np.log(2 * np.pi) - 0.5 * log_det_Q

    def evaluatelogprob(self, x: np.ndarray) -> float:
        Qx = self.Q @ x
        return float(-0.5 * x @ Qx - self._log_Z)

    def descriptionlengthbits(self, C0: int, k: int, b: int = 32) -> float:
        header = sum(_elias_gamma_bits(v) for v in [self.F, C0, k])
        centroid = C0 * self.F * b
        topology = self.F * k * (math.ceil(math.log2(max(2, self.F))) + b)
        params = 64 + 8 + 32
        return float(header + centroid + topology + params)

    def sample(self, n: int = 1) -> np.ndarray:
        z = np.random.default_rng(0).standard_normal((self.F, n))
        return self._lu.solve(z)


def _make_path_laplacian(F: int) -> sp.spmatrix:
    """Combinatorial Laplacian of a path graph on F nodes."""
    row = list(range(F - 1)) + list(range(1, F))
    col = list(range(1, F)) + list(range(F - 1))
    W = sp.csr_matrix((-np.ones(len(row)), (row, col)), shape=(F, F))
    D = sp.diags(np.array(-W.sum(axis=1)).flatten())
    return D + W


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def torch_adapter_and_dataset():
    torch.manual_seed(42)
    model = TinyMLP(input_dim=4, num_classes=3)
    cfg = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
    adapter = TorchClassifierModelAdapter(model, cfg)
    dataset = [(torch.randn(4), i % 3) for i in range(12)]
    return adapter, dataset


@pytest.fixture(scope="module")
def lm_adapter_and_dataset():
    torch.manual_seed(42)
    tiny = TinyLM()
    tok = MagicMock()
    tok.bos_token_id = 1
    tok.eos_token_id = 2
    tok.encode = lambda text, add_special_tokens=False: [ord(c) % TinyLM.VOCAB_SIZE for c in text]
    cfg = TransformerLMEpiplexityConfig(bits_per_param=16, overhead_bits=0.0, max_length=32)
    adapter = TransformerLMModelAdapter(tiny, tok, cfg)
    dataset = ["hello", "world", "abc", "xyz"]
    return adapter, dataset


@pytest.fixture(scope="module")
def gnn_adapter_and_dataset():
    torch.manual_seed(42)
    model = TinyGNN(input_dim=4, num_classes=3)
    cfg = GNNEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
    adapter = GNNModelAdapter(model, cfg)
    dataset = [(TinyGraph(num_nodes=5, num_features=4, num_edges=6), i % 3) for i in range(6)]
    return adapter, dataset


@pytest.fixture(scope="module")
def arrowspace_adapter_and_data():
    F = 16
    L_F = _make_path_laplacian(F)
    inner = _MinimalArrowSpaceModel(L_F, beta=1.0, gamma=0.01)
    adapter = ArrowSpaceModelAdapter(arrowspace_model=inner, C0=32, k=4, b=32)
    # Dataset: N x F float32 array
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, F)).astype(np.float32)
    return adapter, X, inner, F


# ─────────────────────────────────────────────────────────────────────────────
# P1  MDL identity: S_T + H_T == MDL_T
# ─────────────────────────────────────────────────────────────────────────────

class TestP1_MDLIdentity:

    def test_classifier_mdl_identity(self, torch_adapter_and_dataset):
        adapter, dataset = torch_adapter_and_dataset
        engine = EpiplexityEngine(adapter, dataset)
        assert engine.mdltotal_bits == pytest.approx(
            engine.structural_bits + engine.total_entropy_bits, rel=1e-6
        )

    def test_lm_mdl_identity(self, lm_adapter_and_dataset):
        adapter, dataset = lm_adapter_and_dataset
        engine = EpiplexityEngine(adapter, dataset)
        assert engine.mdltotal_bits == pytest.approx(
            engine.structural_bits + engine.total_entropy_bits, rel=1e-6
        )

    def test_gnn_mdl_identity(self, gnn_adapter_and_dataset):
        adapter, dataset = gnn_adapter_and_dataset
        engine = EpiplexityEngine(adapter, dataset)
        assert engine.mdltotal_bits == pytest.approx(
            engine.structural_bits + engine.total_entropy_bits, rel=1e-6
        )

    def test_arrowspace_mdl_identity(self, arrowspace_adapter_and_data):
        adapter, X, _, _ = arrowspace_adapter_and_data
        engine = EpiplexityEngine(adapter, X)
        assert engine.mdltotal_bits == pytest.approx(
            engine.structural_bits + engine.total_entropy_bits, rel=1e-6
        )


# ─────────────────────────────────────────────────────────────────────────────
# P2  Compression test: controlled scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestP2_CompressionTest:

    def test_structural_when_log_prob_near_zero(self):
        """
        A model that assigns near-zero log-prob cost per item makes MDL ≈ S_T.
        If S_T << raw_bits, the compression test must pass.
        """
        class NearPerfectModel(TTimeProbabilisticModel):
            def description_length_bits(self): return 10.0        # tiny S_T
            def log_prob(self, x): return -1e-6                   # near-zero H_T per item
            def sample(self, n=1): return [None] * n
            def raw_bits(self, X): return float(len(X) * 64)      # 64 bits/item baseline

        engine = EpiplexityEngine(NearPerfectModel(), list(range(20)))
        assert engine.mdltotal_bits < engine.raw_bits_total()
        assert engine.compression_ratio() > 1.0

    def test_metadata_when_description_length_dominates(self):
        """
        A model with enormous S_T and negligible H_T still fails the compression test.
        """
        class HeavyModel(TTimeProbabilisticModel):
            def description_length_bits(self): return 1e9
            def log_prob(self, x): return -0.001
            def sample(self, n=1): return [None] * n
            def raw_bits(self, X): return float(len(X) * 64)

        engine = EpiplexityEngine(HeavyModel(), list(range(10)))
        assert engine.mdltotal_bits > engine.raw_bits_total()
        assert engine.compression_ratio() < 1.0

    def test_compression_ratio_positive_for_all_adapters(
        self, torch_adapter_and_dataset, lm_adapter_and_dataset,
        gnn_adapter_and_dataset, arrowspace_adapter_and_data
    ):
        for adapter, dataset in [
            torch_adapter_and_dataset,
            lm_adapter_and_dataset,
            gnn_adapter_and_dataset,
        ]:
            engine = EpiplexityEngine(adapter, dataset)
            assert engine.compression_ratio() > 0, f"Negative compression ratio for {type(adapter).__name__}"

        adapter, X, _, _ = arrowspace_adapter_and_data
        engine = EpiplexityEngine(adapter, X)
        assert engine.compression_ratio() > 0


# ─────────────────────────────────────────────────────────────────────────────
# P3  Non-negativity: H_T(x_i) >= 0 for all items, all adapters
# ─────────────────────────────────────────────────────────────────────────────

class TestP3_NonNegativity:

    @pytest.mark.parametrize("fixture_name", [
        "torch_adapter_and_dataset",
        "lm_adapter_and_dataset",
        "gnn_adapter_and_dataset",
    ])
    def test_entropy_bits_non_negative(self, fixture_name, request):
        adapter, dataset = request.getfixturevalue(fixture_name)
        engine = EpiplexityEngine(adapter, dataset)
        assert np.all(engine.entropy_bits >= -1e-9), (
            f"Negative per-item entropy found in {fixture_name}: "
            f"{engine.entropy_bits[engine.entropy_bits < 0]}"
        )

    def test_arrowspace_entropy_non_negative(self, arrowspace_adapter_and_data):
        adapter, X, _, _ = arrowspace_adapter_and_data
        engine = EpiplexityEngine(adapter, X)
        assert np.all(engine.entropy_bits >= -1e-9)

    def test_entropy_bits_all_finite(self, torch_adapter_and_dataset):
        adapter, dataset = torch_adapter_and_dataset
        engine = EpiplexityEngine(adapter, dataset)
        assert np.all(np.isfinite(engine.entropy_bits))

    def test_structural_bits_positive(self, torch_adapter_and_dataset,
                                       lm_adapter_and_dataset,
                                       gnn_adapter_and_dataset):
        for adapter, dataset in [
            torch_adapter_and_dataset,
            lm_adapter_and_dataset,
            gnn_adapter_and_dataset,
        ]:
            engine = EpiplexityEngine(adapter, dataset)
            assert engine.structural_bits > 0, f"S_T <= 0 for {type(adapter).__name__}"


# ─────────────────────────────────────────────────────────────────────────────
# P4  Observer-dependence: S_T grows monotonically with model capacity
# ─────────────────────────────────────────────────────────────────────────────

class TestP4_ObserverDependence:

    def test_classifier_st_grows_with_bits_per_param(self):
        """
        More bits per parameter → more precise weight description → higher S_T.
        This mirrors the T-scaling S_T(X) ∝ compute budget.
        """
        torch.manual_seed(0)
        model = TinyMLP(input_dim=4, num_classes=3)
        dataset = [(torch.randn(4), i % 3) for i in range(8)]
        st_values = []
        for bpp in [8, 16, 32, 64]:
            cfg = TorchClassifierEpiplexityConfig(bits_per_param=bpp, overhead_bits=0.0)
            adapter = TorchClassifierModelAdapter(model, cfg)
            engine = EpiplexityEngine(adapter, dataset)
            st_values.append(engine.structural_bits)
        assert st_values == sorted(st_values), (
            f"S_T not monotonically increasing with bits_per_param: {st_values}"
        )

    def test_classifier_st_grows_with_model_size(self):
        """
        Larger model (more parameters) → higher S_T for the same bits_per_param.
        """
        torch.manual_seed(0)
        dataset = [(torch.randn(4), i % 3) for i in range(8)]
        cfg = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
        small = TinyMLP(input_dim=4, num_classes=3)  # 4→8→3

        class LargeMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(4, 64), nn.ReLU(),
                    nn.Linear(64, 64), nn.ReLU(),
                    nn.Linear(64, 3),
                )
            def forward(self, x):
                return self.net(x)

        large = LargeMLP()
        st_small = EpiplexityEngine(TorchClassifierModelAdapter(small, cfg), dataset).structural_bits
        st_large = EpiplexityEngine(TorchClassifierModelAdapter(large, cfg), dataset).structural_bits
        assert st_large > st_small, (
            f"Larger model must have higher S_T: small={st_small:.0f} large={st_large:.0f}"
        )

    def test_lm_st_grows_with_bits_per_param(self, lm_adapter_and_dataset):
        torch.manual_seed(0)
        tiny = TinyLM()
        tok = MagicMock()
        tok.bos_token_id = 1
        tok.eos_token_id = 2
        tok.encode = lambda t, add_special_tokens=False: [ord(c) % TinyLM.VOCAB_SIZE for c in t]
        dataset = ["hi", "hello", "world"]
        st_values = []
        for bpp in [4, 8, 16, 32]:
            cfg = TransformerLMEpiplexityConfig(bits_per_param=bpp, overhead_bits=0.0, max_length=32)
            adapter = TransformerLMModelAdapter(tiny, tok, cfg)
            engine = EpiplexityEngine(adapter, dataset)
            st_values.append(engine.structural_bits)
        assert st_values == sorted(st_values)

    def test_arrowspace_st_grows_with_k(self, arrowspace_adapter_and_data):
        """
        Denser graph (higher k) → more topology bits → higher S_T.
        Mirrors the k↑ → PAS grows observation from the notebook.
        """
        _, X, inner, _ = arrowspace_adapter_and_data
        st_values = []
        for k in [2, 4, 8, 16]:
            adapter = ArrowSpaceModelAdapter(inner, C0=32, k=k, b=32)
            engine = EpiplexityEngine(adapter, X)
            st_values.append(engine.structural_bits)
        assert st_values == sorted(st_values), (
            f"S_T not monotonically increasing with k: {st_values}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# P5  H_T ordering: a model assigns lower H_T to data it explains better
# ─────────────────────────────────────────────────────────────────────────────

class TestP5_EntropyOrdering:

    def test_uniform_model_gives_equal_ht_per_item(self):
        """
        A model with exactly uniform logits assigns equal H_T to every item.
        """
        zero_model = TinyMLP(input_dim=4, num_classes=4)
        for p in zero_model.parameters():
            nn.init.zeros_(p)
        cfg = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
        adapter = TorchClassifierModelAdapter(zero_model, cfg)
        dataset = [(torch.randn(4), i % 4) for i in range(8)]
        engine = EpiplexityEngine(adapter, dataset)
        # All per-item entropies should be equal: -log2(1/4) = 2 bits each
        assert np.allclose(engine.entropy_bits, engine.entropy_bits[0], atol=1e-4), (
            f"Uniform model should give equal H_T per item: {engine.entropy_bits}"
        )

    def test_high_confidence_model_gives_lower_ht(self):
        """
        A model that puts all mass on one class gives lower H_T for that class
        than a uniform model.
        """
        # Uniform model: logits all zero → H_T = log2(C) per item
        zero_model = TinyMLP(input_dim=4, num_classes=3)
        for p in zero_model.parameters():
            nn.init.zeros_(p)

        # Biased model: last layer heavily favours class 0
        biased_model = TinyMLP(input_dim=4, num_classes=3)
        for p in biased_model.parameters():
            nn.init.zeros_(p)
        with torch.no_grad():
            biased_model.net[-1].bias[0] = 100.0  # class 0 gets massive logit

        cfg = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
        dataset = [(torch.randn(4), 0) for _ in range(6)]  # all label=0
        engine_uniform = EpiplexityEngine(TorchClassifierModelAdapter(zero_model, cfg), dataset)
        engine_biased  = EpiplexityEngine(TorchClassifierModelAdapter(biased_model, cfg), dataset)
        # The biased model puts near-1 mass on class 0 → H_T ≈ 0 per item
        assert engine_biased.total_entropy_bits < engine_uniform.total_entropy_bits, (
            "High-confidence model should have lower H_T than uniform model"
        )

    def test_lm_longer_sequences_have_higher_total_ht(self, lm_adapter_and_dataset):
        """
        A longer sequence accumulates more per-token entropy terms.
        log P(long) <= log P(short)  →  H_T(long) >= H_T(short).
        """
        adapter, _ = lm_adapter_and_dataset
        short = "hi"
        long  = "hello world"
        ht_short = -adapter.log_prob(short) / math.log(2)
        ht_long  = -adapter.log_prob(long)  / math.log(2)
        assert ht_long >= ht_short, (
            f"H_T(long)={ht_long:.2f} should be >= H_T(short)={ht_short:.2f}"
        )

    def test_arrowspace_smooth_has_lower_ht_than_rough(self, arrowspace_adapter_and_data):
        """
        LGMRF theorem: smooth signals (low Dirichlet energy) have higher probability
        and thus lower H_T than rough signals (high Dirichlet energy).
        """
        adapter, _, inner, F = arrowspace_adapter_and_data
        x_smooth = np.ones(F) / np.sqrt(F)
        x_rough  = np.array([(-1.0)**i for i in range(F)]) / np.sqrt(F)
        ht_smooth = -adapter.log_prob(x_smooth) / math.log(2)
        ht_rough  = -adapter.log_prob(x_rough)  / math.log(2)
        assert ht_smooth < ht_rough, (
            f"Smooth signal must have lower H_T: smooth={ht_smooth:.4f} rough={ht_rough:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# P6  Per-item consistency: entropy_bits sums to total_entropy_bits
# ─────────────────────────────────────────────────────────────────────────────

class TestP6_PerItemConsistency:

    @pytest.mark.parametrize("fixture_name", [
        "torch_adapter_and_dataset",
        "lm_adapter_and_dataset",
        "gnn_adapter_and_dataset",
    ])
    def test_per_item_sum_equals_total(self, fixture_name, request):
        adapter, dataset = request.getfixturevalue(fixture_name)
        engine = EpiplexityEngine(adapter, dataset)
        assert engine.entropy_bits.sum() == pytest.approx(engine.total_entropy_bits, rel=1e-6)

    def test_per_item_count_matches_dataset_length(self, torch_adapter_and_dataset):
        adapter, dataset = torch_adapter_and_dataset
        engine = EpiplexityEngine(adapter, dataset)
        assert len(engine.entropy_bits) == len(dataset)

    def test_arrowspace_per_item_count(self, arrowspace_adapter_and_data):
        adapter, X, _, _ = arrowspace_adapter_and_data
        engine = EpiplexityEngine(adapter, X)
        assert len(engine.entropy_bits) == len(X)

    def test_single_item_dataset(self):
        """Edge case: a dataset of one item."""
        class FixedModel(TTimeProbabilisticModel):
            def description_length_bits(self): return 100.0
            def log_prob(self, x): return -5.0
            def sample(self, n=1): return [None]
            def raw_bits(self, X): return float(len(X) * 64)

        engine = EpiplexityEngine(FixedModel(), [42])
        assert len(engine.entropy_bits) == 1
        assert engine.total_entropy_bits == pytest.approx(engine.entropy_bits[0])
        assert engine.mdltotal_bits == pytest.approx(100.0 + 5.0 / math.log(2))


# ─────────────────────────────────────────────────────────────────────────────
# P7  ArrowSpace LGMRF: Dirichlet energy ↔ H_T monotonicity
# ─────────────────────────────────────────────────────────────────────────────

class TestP7_ArrowSpaceLGMRF:

    def test_dirichlet_energy_order_matches_ht_order(self, arrowspace_adapter_and_data):
        """
        Construct signals with known energy order; confirm H_T tracks energy.
        """
        adapter, _, inner, F = arrowspace_adapter_and_data
        # Eigenvectors of path graph: energy increases with eigenvalue index
        # Use constant (lowest energy) vs alternating (highest energy)
        signals = {
            "constant":    np.ones(F) / np.sqrt(F),
            "alternating": np.array([(-1.0)**i for i in range(F)]) / np.sqrt(F),
        }
        ht = {name: -adapter.log_prob(x) / math.log(2) for name, x in signals.items()}
        assert ht["constant"] < ht["alternating"], (
            f"constant H_T={ht['constant']:.4f} should be < alternating H_T={ht['alternating']:.4f}"
        )

    def test_sample_shapes_from_lgmrf(self, arrowspace_adapter_and_data):
        """Sampled vectors must have the same dimension F as the graph."""
        adapter, _, inner, F = arrowspace_adapter_and_data
        samples = adapter.sample(n=5)
        # ArrowSpaceModelAdapter.sample delegates to inner.sample()
        # which returns shape (F, n)
        assert samples.shape[0] == F

    def test_arrowspace_st_grows_with_C0(self, arrowspace_adapter_and_data):
        """
        More cluster centroids C0 → more centroid bits → higher S_T.
        """
        _, X, inner, _ = arrowspace_adapter_and_data
        st_values = []
        for C0 in [8, 16, 32, 64]:
            adapter = ArrowSpaceModelAdapter(inner, C0=C0, k=4, b=32)
            engine = EpiplexityEngine(adapter, X)
            st_values.append(engine.structural_bits)
        assert st_values == sorted(st_values), (
            f"S_T must grow monotonically with C0: {st_values}"
        )

    def test_arrowspace_ht_non_negative_on_samples(self, arrowspace_adapter_and_data):
        """All sampled items from the LGMRF must also have non-negative H_T."""
        adapter, _, inner, F = arrowspace_adapter_and_data
        samples = inner.sample(n=10)          # shape (F, 10)
        for i in range(samples.shape[1]):
            x = samples[:, i]
            ht = -adapter.log_prob(x) / math.log(2)
            assert ht >= -1e-9, f"Negative H_T on sampled vector {i}: {ht}"


# ─────────────────────────────────────────────────────────────────────────────
# P8  Epiplexity rank: S_T rank preserved under capacity scaling
# ─────────────────────────────────────────────────────────────────────────────

class TestP8_EpiplexityRank:

    def test_classifier_st_rank_stable_across_bpp(self):
        """
        Given two models (small, large), their S_T rank should be the same
        regardless of the bits_per_param setting: S_T(large) > S_T(small) always.
        """
        torch.manual_seed(7)
        dataset = [(torch.randn(4), i % 3) for i in range(6)]
        small = TinyMLP(input_dim=4, num_classes=3)

        class BigMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(4, 128), nn.ReLU(),
                    nn.Linear(128, 3),
                )
            def forward(self, x): return self.net(x)

        big = BigMLP()
        for bpp in [8, 16, 32]:
            cfg = TorchClassifierEpiplexityConfig(bits_per_param=bpp, overhead_bits=0.0)
            st_s = EpiplexityEngine(TorchClassifierModelAdapter(small, cfg), dataset).structural_bits
            st_b = EpiplexityEngine(TorchClassifierModelAdapter(big,   cfg), dataset).structural_bits
            assert st_b > st_s, (
                f"At bpp={bpp}: S_T(big)={st_b:.0f} should > S_T(small)={st_s:.0f}"
            )

    def test_gnn_st_rank_stable(self, gnn_adapter_and_dataset):
        """
        A GNN with more parameters must always have higher S_T than a smaller one,
        regardless of bits-per-parameter.
        """
        torch.manual_seed(7)
        _, dataset = gnn_adapter_and_dataset
        small = TinyGNN(input_dim=4, num_classes=3)

        class BigGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 3)
                )
            def forward(self, graph):
                pooled = graph.x.mean(dim=0, keepdim=True)
                return self.fc(pooled)

        big = BigGNN()
        for bpp in [16, 32]:
            cfg = GNNEpiplexityConfig(bits_per_param=bpp, overhead_bits=0.0)
            st_s = EpiplexityEngine(GNNModelAdapter(small, cfg), dataset).structural_bits
            st_b = EpiplexityEngine(GNNModelAdapter(big,   cfg), dataset).structural_bits
            assert st_b > st_s
