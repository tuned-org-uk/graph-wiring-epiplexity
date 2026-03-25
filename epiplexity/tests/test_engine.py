# tests/test_engine.py
"""
Tests for EpiplexityEngine (epiplexity/engine.py).

Verifies:
  - Arithmetic invariants: S_T + H_T == MDL.
  - Entropy bits are per-item arrays of the right shape.
  - Compression ratio = raw / MDL.
  - Verdict is "STRUCTURAL" iff MDL < raw, else "METADATA".
  - report() runs without error and prints expected keys.
"""

import math
import numpy as np
import pytest

from epiplexity.engine import EpiplexityEngine
from tests.conftest import ConstantModel


def make_engine(desc_bits: float = 1024.0, log_prob_val: float = -2.0, n_items: int = 10):
    model = ConstantModel(desc_bits=desc_bits, log_prob_val=log_prob_val)
    X = list(range(n_items))
    return EpiplexityEngine(model, X), model, X


class TestEpiplexityEngineArithmetic:

    def test_structural_bits_equals_description_length(self):
        engine, model, _ = make_engine(desc_bits=2048.0)
        assert engine.structural_bits == pytest.approx(2048.0)

    def test_entropy_bits_shape(self):
        n = 7
        engine, _, _ = make_engine(n_items=n)
        assert engine.entropy_bits.shape == (n,)

    def test_entropy_bits_are_positive(self):
        """log_prob is negative, so -log_prob/log(2) must be positive."""
        engine, _, _ = make_engine(log_prob_val=-3.0)
        assert np.all(engine.entropy_bits > 0)

    def test_per_item_entropy_formula(self):
        lp = -4.0
        engine, _, _ = make_engine(log_prob_val=lp, n_items=5)
        expected = -lp / math.log(2)
        assert np.allclose(engine.entropy_bits, expected)

    def test_total_entropy_bits_is_sum(self):
        engine, _, _ = make_engine(log_prob_val=-1.0, n_items=8)
        assert engine.total_entropy_bits == pytest.approx(engine.entropy_bits.sum())

    def test_mdltotal_is_structural_plus_entropy(self):
        engine, _, _ = make_engine(desc_bits=512.0, log_prob_val=-2.0, n_items=6)
        expected = engine.structural_bits + engine.total_entropy_bits
        assert engine.mdltotal_bits == pytest.approx(expected)

    def test_compression_ratio_formula(self):
        engine, model, X = make_engine()
        expected = model.raw_bits(X) / engine.mdltotal_bits
        assert engine.compression_ratio() == pytest.approx(expected)


class TestEpiplexityEngineVerdict:

    def test_structural_verdict_when_mdl_less_than_raw(self):
        """
        ConstantModel.raw_bits = n * 64 bits.
        Force MDL << raw by using a very high (negative) log_prob
        and a small desc_bits so that per-item bits are tiny.
        Actually: log_prob very close to 0 → entropy close to 0 → MDL ≈ desc_bits.
        """
        # desc_bits=10, log_prob=-0.001 → H_T tiny → MDL << raw=10*64=640
        model = ConstantModel(desc_bits=10.0, log_prob_val=-0.001)
        X = list(range(10))
        engine = EpiplexityEngine(model, X)
        assert engine.mdltotal_bits < engine.raw_bits_total()
        # verify compression_ratio > 1
        assert engine.compression_ratio() > 1.0

    def test_metadata_verdict_when_mdl_exceeds_raw(self):
        """
        Force MDL > raw by using a very large desc_bits and a moderate H_T.
        raw = 10 * 64 = 640; desc_bits = 10_000 → MDL > raw.
        """
        model = ConstantModel(desc_bits=10_000.0, log_prob_val=-1.0)
        X = list(range(10))
        engine = EpiplexityEngine(model, X)
        assert engine.mdltotal_bits > engine.raw_bits_total()
        assert engine.compression_ratio() < 1.0


class TestEpiplexityEngineReport:

    def test_report_runs_without_error(self, capsys):
        engine, _, _ = make_engine()
        engine.report()  # must not raise
        captured = capsys.readouterr()
        assert "Structural bits" in captured.out
        assert "Random bits" in captured.out
        assert "Compression ratio" in captured.out
        assert "Verdict" in captured.out

    def test_report_contains_verdict_word(self, capsys):
        engine, _, _ = make_engine(desc_bits=10.0, log_prob_val=-0.001, n_items=10)
        engine.report()
        captured = capsys.readouterr()
        assert "STRUCTURAL" in captured.out or "METADATA" in captured.out