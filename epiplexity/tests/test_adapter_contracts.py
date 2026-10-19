import numpy as np
import pytest

from epiplexity.algorithms.arrowspace import ArrowSpaceModelAdapter
from epiplexity.algorithms.gnn import GNNModelAdapter, GNNEpiplexityConfig
from epiplexity.algorithms.torch_classifier import (
    TorchClassifierModelAdapter, TorchClassifierEpiplexityConfig,
)
from epiplexity.algorithms.transformer_lm import TransformerLMModelAdapter
from epiplexity.engine import EpiplexityEngine


def test_classifier_batch_matches_single_item_scores(tiny_mlp, float_dataset):
    adapter = TorchClassifierModelAdapter(tiny_mlp, TorchClassifierEpiplexityConfig())
    expected = [adapter.log_prob(x) for x in float_dataset]
    np.testing.assert_allclose(adapter.batch_log_prob(float_dataset), expected, rtol=1e-6)
    # A conditional label code cannot compress the input feature bytes.
    result = EpiplexityEngine(adapter, float_dataset).result
    assert result.score_kind == "conditional"
    assert result.compression_ratio is None


def test_gnn_reports_conditional_scores(tiny_gnn, graph_dataset):
    result = EpiplexityEngine(
        GNNModelAdapter(tiny_gnn, GNNEpiplexityConfig()), graph_dataset
    ).result
    assert result.score_kind == "conditional"
    assert result.verdict == "NOT_COMPARABLE"


def test_lm_marks_prefix_conditioning_and_truncation():
    assert TransformerLMModelAdapter.score_kind == "conditional"
    assert TransformerLMModelAdapter.baseline_comparable is False


def test_arrowspace_does_not_treat_density_as_probability_mass():
    class Gaussian:
        def descriptionlengthbits(self, *args):
            return 64

        def evaluatelogprob(self, x):
            return 1.0

    result = EpiplexityEngine(
        ArrowSpaceModelAdapter(Gaussian(), C0=1, k=1), np.zeros((2, 3))
    ).result
    assert result.score_kind == "density"
    assert result.data_bits < 0
    assert result.verdict == "NOT_COMPARABLE"


def test_batch_classifier_rejects_negative_and_out_of_range_labels(tiny_mlp):
    import torch
    adapter = TorchClassifierModelAdapter(tiny_mlp, TorchClassifierEpiplexityConfig())
    for label in [-1, 3]:
        with pytest.raises(ValueError, match="label"):
            adapter.batch_log_prob([(torch.zeros(4), label)])
