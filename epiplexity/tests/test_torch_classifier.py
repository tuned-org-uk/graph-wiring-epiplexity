# tests/test_torch_classifier.py
"""
Tests for TorchClassifierModelAdapter (epiplexity/algorithms/torch_classifier.py).

Verifies:
  - description_length_bits reflects param count and config.
  - log_prob returns a finite non-positive float (log of probability ≤ 0).
  - log_prob selects the correct class channel.
  - raw_bits is proportional to input dimensionality.
  - sample() raises NotImplementedError.
  - EpiplexityEngine integration produces consistent MDL.
"""

import math
import pytest
import torch
import torch.nn as nn

from epiplexity.algorithms.torch_classifier import (
    TorchClassifierModelAdapter,
    TorchClassifierEpiplexityConfig,
)
from epiplexity.engine import EpiplexityEngine
from tests.conftest import TinyMLP


@pytest.fixture
def adapter(tiny_mlp, float_dataset):
    cfg = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
    return TorchClassifierModelAdapter(tiny_mlp, cfg)


class TestTorchClassifierDescriptionLength:

    def test_description_length_is_positive(self, adapter, tiny_mlp):
        assert adapter.description_length_bits() > 0

    def test_description_length_scales_with_params(self, tiny_mlp):
        cfg32 = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
        cfg16 = TorchClassifierEpiplexityConfig(bits_per_param=16, overhead_bits=0.0)
        a32 = TorchClassifierModelAdapter(tiny_mlp, cfg32)
        a16 = TorchClassifierModelAdapter(tiny_mlp, cfg16)
        assert a32.description_length_bits() == pytest.approx(2 * a16.description_length_bits())

    def test_overhead_bits_added(self, tiny_mlp):
        cfg_no_overhead = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
        cfg_overhead = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=1024.0)
        a_no = TorchClassifierModelAdapter(tiny_mlp, cfg_no_overhead)
        a_ov = TorchClassifierModelAdapter(tiny_mlp, cfg_overhead)
        assert a_ov.description_length_bits() == pytest.approx(
            a_no.description_length_bits() + 1024.0
        )

    def test_description_length_matches_manual_count(self, tiny_mlp):
        cfg = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
        adapter = TorchClassifierModelAdapter(tiny_mlp, cfg)
        num_params = sum(p.numel() for p in tiny_mlp.parameters())
        assert adapter.description_length_bits() == pytest.approx(num_params * 32)


class TestTorchClassifierLogProb:

    def test_log_prob_is_finite(self, adapter, float_dataset):
        for item in float_dataset:
            lp = adapter.log_prob(item)
            assert math.isfinite(lp), f"log_prob not finite: {lp}"

    def test_log_prob_is_non_positive(self, adapter, float_dataset):
        """log of a probability must be ≤ 0."""
        for item in float_dataset:
            assert adapter.log_prob(item) <= 1e-6  # allow float epsilon

    def test_log_prob_sums_to_at_most_zero_per_class(self, tiny_mlp):
        """With uniform logits the model assigns equal prob to all classes;
        log_prob for any class must equal -log(num_classes)."""
        # Build a model whose weights are zero → uniform logits
        zero_model = TinyMLP(input_dim=4, num_classes=3)
        for p in zero_model.parameters():
            nn.init.zeros_(p)

        cfg = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
        adapter_zero = TorchClassifierModelAdapter(zero_model, cfg)
        x = torch.zeros(4)
        for label in range(3):
            lp = adapter_zero.log_prob((x, label))
            assert lp == pytest.approx(-math.log(3), abs=1e-5)

    def test_log_prob_varies_across_labels(self, tiny_mlp, float_dataset):
        """For a non-trivial model, different labels should yield different log-probs."""
        cfg = TorchClassifierEpiplexityConfig()
        adapter = TorchClassifierModelAdapter(tiny_mlp, cfg)
        x, _ = float_dataset[0]
        lps = {label: adapter.log_prob((x, label)) for label in range(3)}
        # At least two classes should differ
        assert len(set(round(v, 8) for v in lps.values())) > 1


class TestTorchClassifierRawBits:

    def test_raw_bits_proportional_to_dataset_size(self, tiny_mlp, float_dataset):
        cfg = TorchClassifierEpiplexityConfig(input_bits_per_float=32)
        adapter = TorchClassifierModelAdapter(tiny_mlp, cfg)
        half = float_dataset[: len(float_dataset) // 2]
        full = float_dataset
        rb_half = adapter.raw_bits(half)
        rb_full = adapter.raw_bits(full)
        assert rb_full == pytest.approx(2 * rb_half)

    def test_raw_bits_scales_with_input_precision(self, tiny_mlp, float_dataset):
        cfg32 = TorchClassifierEpiplexityConfig(input_bits_per_float=32)
        cfg16 = TorchClassifierEpiplexityConfig(input_bits_per_float=16)
        a32 = TorchClassifierModelAdapter(tiny_mlp, cfg32)
        a16 = TorchClassifierModelAdapter(tiny_mlp, cfg16)
        assert a32.raw_bits(float_dataset) == pytest.approx(2 * a16.raw_bits(float_dataset))


class TestTorchClassifierSample:

    def test_sample_raises_not_implemented(self, adapter):
        with pytest.raises(NotImplementedError):
            adapter.sample(n=1)


class TestTorchClassifierEngineIntegration:

    def test_engine_mdl_identity(self, tiny_mlp, float_dataset):
        cfg = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=0.0)
        adapter = TorchClassifierModelAdapter(tiny_mlp, cfg)
        engine = EpiplexityEngine(adapter, float_dataset)
        assert engine.mdltotal_bits == pytest.approx(
            engine.structural_bits + engine.total_entropy_bits
        )

    def test_engine_compression_ratio_is_positive(self, tiny_mlp, float_dataset):
        cfg = TorchClassifierEpiplexityConfig()
        adapter = TorchClassifierModelAdapter(tiny_mlp, cfg)
        engine = EpiplexityEngine(adapter, float_dataset)
        assert engine.compression_ratio() > 0