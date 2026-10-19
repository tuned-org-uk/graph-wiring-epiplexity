import json
import math

import numpy as np
import pytest

from epiplexity.engine import EpiplexityEngine
from epiplexity.tests.conftest import ConstantModel


class VectorModel(ConstantModel):
    def log_prob(self, x):
        return -float(x) * math.log(2)

    def batch_log_prob(self, batch):
        return -np.asarray(batch, dtype=float) * math.log(2)


def test_one_pass_stream_matches_eager_and_does_not_store_items():
    model = VectorModel(desc_bits=4)
    eager = EpiplexityEngine(model, [1, 2, 3, 4, 5])
    batches = (batch for batch in [[1, 2], [], [3], [4, 5]])
    streamed = EpiplexityEngine.from_batches(model, batches)
    assert streamed.total_entropy_bits == pytest.approx(15)
    assert streamed.mdltotal_bits == eager.mdltotal_bits
    assert streamed.raw_bits_total() == eager.raw_bits_total()
    assert streamed.result.observation_count == 5
    assert streamed.X is None
    with pytest.raises(ValueError, match="not retained"):
        _ = streamed.entropy_bits
    assert list(batches) == []


def test_batches_are_scored_before_requesting_next_chunk():
    class OnlineModel(VectorModel):
        seen = 0

        def batch_log_prob(self, batch):
            self.seen += len(batch)
            return super().batch_log_prob(batch)

    model = OnlineModel()

    def batches():
        for i in range(50):
            assert model.seen == i
            yield [i]

    result = EpiplexityEngine.from_batches(model, batches())
    assert result.result.observation_count == 50


def test_generator_constructor_and_retained_values():
    engine = EpiplexityEngine(VectorModel(), iter([1, 2, 3]), batch_size=2)
    np.testing.assert_allclose(engine.entropy_bits, [1, 2, 3])
    values = engine.entropy_bits
    values[0] = 100
    assert engine.entropy_bits[0] == pytest.approx(1)


def test_scalar_fallback_and_empty_stream():
    engine = EpiplexityEngine.from_batches(ConstantModel(), iter([[1], [2, 3]]))
    assert engine.total_entropy_bits == pytest.approx(3 / math.log(2))
    empty = EpiplexityEngine.from_batches(ConstantModel(desc_bits=0), iter([]))
    assert empty.result.observation_count == 0
    assert empty.to_dict()["compression_ratio"] is None
    json.loads(empty.to_json())


def test_result_snapshots_model_and_provenance():
    model = ConstantModel(desc_bits=4)
    metadata = {"seed": 3407, "config": {"precision": 32}}
    engine = EpiplexityEngine(model, [1, 2], metadata=metadata)
    model._desc_bits = 100
    metadata["config"]["precision"] = 8
    output = json.loads(engine.to_json(include_per_item=True))
    assert output["model_bits"] == 4
    assert output["metadata"]["config"]["precision"] == 32
    assert len(output["entropy_bits"]) == 2
    assert output["method"] == "static"


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [0, 0], [[0]], [0.1]])
def test_invalid_batch_outputs_are_rejected(values):
    class InvalidModel(ConstantModel):
        def batch_log_prob(self, batch):
            return values

    with pytest.raises(ValueError):
        EpiplexityEngine.from_batches(InvalidModel(), [[1]])


@pytest.mark.parametrize("bits", [-1, float("nan"), float("inf")])
def test_invalid_description_and_raw_size_are_rejected(bits):
    with pytest.raises(ValueError):
        EpiplexityEngine(ConstantModel(desc_bits=bits), [1])
    with pytest.raises(ValueError):
        EpiplexityEngine(ConstantModel(), [1], raw_bits=bits)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_batch_size(size):
    with pytest.raises(ValueError):
        EpiplexityEngine(ConstantModel(), [1], batch_size=size)


def test_density_is_allowed_but_has_no_discrete_compression_verdict():
    class Density(ConstantModel):
        score_kind = "density"

    engine = EpiplexityEngine(Density(log_prob_val=2), [1])
    assert engine.total_entropy_bits < 0
    assert engine.to_dict()["verdict"] == "NOT_COMPARABLE"
    assert engine.to_dict()["compression_ratio"] is None


def test_conditional_score_requires_a_matching_baseline():
    class Conditional(ConstantModel):
        score_kind = "conditional"
        baseline_comparable = False

    engine = EpiplexityEngine(Conditional(), [1])
    assert engine.to_dict()["verdict"] == "NOT_COMPARABLE"
    matched = EpiplexityEngine(
        Conditional(), [1], raw_bits=10, baseline_comparable=True
    )
    assert matched.to_dict()["verdict"] == "METADATA"


def test_raw_override_supports_adapters_without_an_additive_baseline():
    class NoRaw(VectorModel):
        def raw_bits(self, X):
            raise AssertionError("caller supplied a whole-corpus baseline")

    engine = EpiplexityEngine.from_batches(NoRaw(), [[1], [2]], raw_bits=27)
    assert engine.raw_bits_total() == 27
