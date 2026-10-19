"""Analytic fixtures for log estimators; also the reference for native ports."""

import json
import math

import numpy as np
import pytest

from epiplexity.estimators import PrequentialEstimator, RequentialEstimator


def test_prequential_integrates_irregular_token_intervals_in_bits():
    # Areas: 2*(4+2)/2 + 3*(2+1)/2 = 10.5 bits.
    result = PrequentialEstimator.estimate([0, 2, 5], [4, 2, 1], loss_unit="bits")
    assert result.model_bits == pytest.approx(5.5)
    assert result.data_bits == pytest.approx(5)
    assert result.total_bits == pytest.approx(10.5)
    assert result.observation_count == 5
    assert result.method == "prequential"
    assert json.loads(result.to_json())["schema_version"] == "1.0"


def test_nats_conversion_and_explicit_reference_loss():
    result = PrequentialEstimator.estimate(
        [0, 2, 5], np.array([4, 2, 1]) * math.log(2), final_loss=2 * math.log(2)
    )
    assert result.model_bits == pytest.approx(0.5)
    assert result.data_bits == pytest.approx(10)


def test_worsening_loss_is_not_clamped_into_positive_structure():
    result = PrequentialEstimator.estimate([0, 10], [1, 3], loss_unit="bits")
    assert result.model_bits == -10
    assert result.total_bits == 20
    assert "negative_model_bits" in result.diagnostics


def test_small_structure_is_not_lost_when_residual_is_large():
    result = PrequentialEstimator.estimate(
        [0, 1], [1e16 + 2, 1e16], loss_unit="bits"
    )
    assert result.model_bits == 1


def test_token_count_is_validated_before_float_rounding():
    with pytest.raises(ValueError):
        PrequentialEstimator.estimate([0, 2**53 + 1], [2, 1])
    with pytest.raises(ValueError):
        RequentialEstimator.estimate([0.1], [1], final_loss=1, evaluation_tokens=2**53 + 1)


@pytest.mark.parametrize("tokens,losses", [
    ([1, 2], [2, 1]), ([0, 0], [2, 1]), ([0, 2, 1], [3, 2, 1]),
    ([0], [1]), ([0, 1], [1]), ([0, 1], [1, float("nan")]),
    ([0, 1], [1, -1]), ([0, float("inf")], [1, 1]),
    ([[0, 1]], [[1, 1]]), ([0, 1.5], [2, 1]),
])
def test_prequential_rejects_invalid_logs(tokens, losses):
    with pytest.raises(ValueError):
        PrequentialEstimator.estimate(tokens, losses)


def test_requential_weights_each_step_and_keeps_evaluation_tokens_separate():
    result = RequentialEstimator.estimate(
        [0.5, 0.25, 0], [4, 8, 3], final_loss=2, evaluation_tokens=7,
        loss_unit="bits",
    )
    assert result.model_bits == 4
    assert result.data_bits == 14
    assert result.total_bits == 18
    assert result.observation_count == 7
    assert result.method == "requential"


def test_requential_nats_conversion():
    result = RequentialEstimator.estimate(
        [math.log(2)], [4], final_loss=math.log(2), evaluation_tokens=7
    )
    assert result.model_bits == pytest.approx(4)
    assert result.data_bits == pytest.approx(7)


@pytest.mark.parametrize("kl,counts", [
    ([], []), ([0.1], [0]), ([-0.1], [2]), ([float("inf")], [2]),
    ([0.1, 0.2], [2]), ([0.1], [1.5]),
])
def test_requential_rejects_invalid_logs(kl, counts):
    with pytest.raises(ValueError):
        RequentialEstimator.estimate(kl, counts, final_loss=1, evaluation_tokens=10)


@pytest.mark.parametrize("unit", ["bytes", "nat", ""])
def test_unknown_units_are_rejected(unit):
    with pytest.raises(ValueError, match="loss_unit"):
        PrequentialEstimator.estimate([0, 1], [2, 1], loss_unit=unit)


def test_json_rejects_nonfinite_provenance():
    result = PrequentialEstimator.estimate([0, 1], [2, 1])
    with pytest.raises(ValueError):
        result.to_json(metadata={"compute": float("nan")})
