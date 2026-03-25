# tests/test_transformer_lm.py
"""
Tests for TransformerLMModelAdapter (epiplexity/algorithms/transformer_lm.py).

Uses a tiny mock tokenizer and TinyLM (no HuggingFace download required).

Verifies:
  - _to_token_ids handles strings and pre-tokenised lists.
  - BOS/EOS wrapping works correctly.
  - max_length truncation is respected.
  - log_prob is finite and ≤ 0.
  - A degenerate 1-token sequence returns 0.0.
  - description_length_bits scales with param count and bits_per_param.
  - raw_bits correctly distinguishes str vs list inputs.
  - sample() returns token-id lists.
"""

import math
import pytest
import torch
from unittest.mock import MagicMock

from epiplexity.algorithms.transformer_lm import (
    TransformerLMModelAdapter,
    TransformerLMEpiplexityConfig,
)
from epiplexity.engine import EpiplexityEngine
from tests.conftest import TinyLM


# ── Minimal mock tokenizer ──────────────────────────────────────────────────────

def make_mock_tokenizer(vocab_size: int = 16, bos_id: int = 1, eos_id: int = 2):
    tok = MagicMock()
    tok.bos_token_id = bos_id
    tok.eos_token_id = eos_id
    # encode a string by mapping each character to its ord % vocab_size
    tok.encode = lambda text, add_special_tokens=False: [
        ord(c) % vocab_size for c in text
    ]
    return tok


@pytest.fixture
def lm_adapter(tiny_lm):
    tokenizer = make_mock_tokenizer()
    cfg = TransformerLMEpiplexityConfig(
        bits_per_param=16,
        overhead_bits=0.0,
        max_length=32,
    )
    return TransformerLMModelAdapter(tiny_lm, tokenizer, cfg)


@pytest.fixture
def text_dataset():
    return ["hi", "hello world", "abc"]


class TestTokenisation:

    def test_string_input_uses_tokenizer(self, lm_adapter):
        ids = lm_adapter._to_token_ids("ab")
        # "a"=97%16=1 → BOS prepended + EOS appended by fixture tokenizer
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_bos_prepended(self, lm_adapter):
        ids = lm_adapter._to_token_ids("x")
        assert ids[0] == lm_adapter.config.bos_token_id

    def test_eos_appended(self, lm_adapter):
        ids = lm_adapter._to_token_ids("x")
        assert ids[-1] == lm_adapter.config.eos_token_id

    def test_max_length_truncation(self, lm_adapter):
        long_text = "a" * 100
        ids = lm_adapter._to_token_ids(long_text)
        assert len(ids) <= lm_adapter.config.max_length

    def test_pretokenised_list_accepted(self, lm_adapter):
        token_list = [3, 5, 7, 9]
        ids = lm_adapter._to_token_ids(token_list)
        # original tokens should appear in the middle (wrapped by BOS/EOS)
        assert 3 in ids and 9 in ids

    def test_no_bos_eos_when_disabled(self, tiny_lm):
        tokenizer = make_mock_tokenizer()
        cfg = TransformerLMEpiplexityConfig(
            bits_per_param=16, overhead_bits=0.0,
            max_length=32, bos_token_id=None, eos_token_id=None,
        )
        # Override tokenizer token ids so they don't bleed in
        tokenizer.bos_token_id = None
        tokenizer.eos_token_id = None
        adapter = TransformerLMModelAdapter(tiny_lm, tokenizer, cfg)
        token_list = [3, 5, 7]
        ids = adapter._to_token_ids(token_list)
        assert ids == [3, 5, 7]


class TestLogProb:

    def test_log_prob_is_finite(self, lm_adapter, text_dataset):
        for text in text_dataset:
            lp = lm_adapter.log_prob(text)
            assert math.isfinite(lp), f"log_prob not finite for '{text}': {lp}"

    def test_log_prob_is_non_positive(self, lm_adapter, text_dataset):
        for text in text_dataset:
            assert lm_adapter.log_prob(text) <= 1e-6

    def test_degenerate_single_token_returns_zero(self, lm_adapter):
        """A sequence with only BOS (1 token after truncation to length 1) returns 0."""
        lp = lm_adapter.log_prob([])  # empty → just BOS/EOS but too short after tokenise
        # After wrapping a single-item list: [bos, item, eos] ≥ 3 tokens → log_prob normal.
        # Test the explicit degenerate branch instead: pass a single token id without BOS/EOS.
        cfg = TransformerLMEpiplexityConfig(
            bits_per_param=16, overhead_bits=0.0,
            bos_token_id=None, eos_token_id=None, max_length=32,
        )
        tok = MagicMock()
        tok.bos_token_id = None
        tok.eos_token_id = None
        tok.encode = lambda text, add_special_tokens=False: [3]  # single token
        adapter_single = TransformerLMModelAdapter(lm_adapter.model, tok, cfg)
        lp = adapter_single.log_prob("x")  # → [3], length < 2 → returns 0.0
        assert lp == pytest.approx(0.0)

    def test_longer_text_has_lower_log_prob(self, lm_adapter):
        """
        log P(long) ≤ log P(short) because each token contributes a negative term.
        """
        short_lp = lm_adapter.log_prob("hi")
        long_lp = lm_adapter.log_prob("hello world")
        assert long_lp <= short_lp


class TestDescriptionLength:

    def test_description_length_positive(self, lm_adapter):
        assert lm_adapter.description_length_bits() > 0

    def test_description_length_scales_with_bits_per_param(self, tiny_lm):
        tok = make_mock_tokenizer()
        cfg8 = TransformerLMEpiplexityConfig(bits_per_param=8, overhead_bits=0.0, max_length=32)
        cfg32 = TransformerLMEpiplexityConfig(bits_per_param=32, overhead_bits=0.0, max_length=32)
        a8 = TransformerLMModelAdapter(tiny_lm, tok, cfg8)
        a32 = TransformerLMModelAdapter(tiny_lm, tok, cfg32)
        assert a32.description_length_bits() == pytest.approx(4 * a8.description_length_bits())


class TestRawBits:

    def test_raw_bits_strings_use_8bits_per_char(self, lm_adapter):
        dataset = ["ab", "cde"]
        # "ab" = 2 * 8 = 16, "cde" = 3 * 8 = 24 → total 40
        rb = lm_adapter.raw_bits(dataset)
        assert rb == pytest.approx(40.0)

    def test_raw_bits_token_lists_use_32bits_per_token(self, lm_adapter):
        dataset = [[1, 2, 3], [4, 5]]   # 3 + 2 = 5 tokens × 32 bits = 160
        rb = lm_adapter.raw_bits(dataset)
        assert rb == pytest.approx(160.0)


class TestSample:

    def test_sample_returns_list_of_token_lists(self, lm_adapter):
        samples = lm_adapter.sample(n=2, max_length=5)
        assert isinstance(samples, list)
        assert len(samples) == 2
        for seq in samples:
            assert isinstance(seq, list)
            assert all(isinstance(t, int) for t in seq)


class TestEngineIntegration:

    def test_engine_mdl_identity(self, lm_adapter, text_dataset):
        engine = EpiplexityEngine(lm_adapter, text_dataset)
        assert engine.mdltotal_bits == pytest.approx(
            engine.structural_bits + engine.total_entropy_bits
        )