from dataclasses import FrozenInstanceError
from importlib.resources import files
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from epiplexity import CodingResult, PrequentialEstimator, RequentialEstimator


def test_portable_golden_vectors_and_wire_schema():
    vectors = json.loads((Path(__file__).parent / "fixtures/coding-v1.json").read_text())
    schema = json.loads(files("epiplexity").joinpath("schemas/result-v1.json").read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for case in vectors["cases"]:
        estimator = PrequentialEstimator if case["method"] == "prequential" else RequentialEstimator
        result = estimator.estimate(**case["input"])
        output = json.loads(result.to_json())
        validator.validate(output)
        for key, expected in case["expected"].items():
            assert output[key] == pytest.approx(
                expected, rel=vectors["tolerance"]["relative"], abs=vectors["tolerance"]["absolute"]
            ), case["name"]


@pytest.mark.parametrize("changes", [
    {"method": "unknown"}, {"score_kind": "unknown"}, {"model_bits": -1},
    {"model_bits": float("nan")}, {"data_bits": float("inf")},
    {"data_bits": -1}, {"observation_count": 1.2}, {"raw_bits": -1},
    {"baseline_comparable": "yes"},
])
def test_result_rejects_invalid_values(changes):
    values = dict(method="static", model_bits=10, data_bits=20, observation_count=5)
    with pytest.raises(ValueError):
        CodingResult(**(values | changes))


def test_result_is_immutable_and_metadata_is_copied():
    result = CodingResult("static", 10, 20, 5, raw_bits=60, baseline_comparable=True)
    assert result.compression_ratio == 2
    assert result.verdict == "STRUCTURAL"
    with pytest.raises(FrozenInstanceError):
        result.model_bits = 100
    metadata = {"config": {"precision": 32}}
    output = result.to_dict(metadata=metadata)
    output["metadata"]["config"]["precision"] = 16
    assert metadata["config"]["precision"] == 32


def test_density_and_zero_code_have_json_null_ratios():
    for result in [CodingResult("static", 0, 0, 0, raw_bits=0, baseline_comparable=True),
                   CodingResult("static", 10, -1, 1, raw_bits=60,
                                score_kind="density", baseline_comparable=True)]:
        assert json.loads(result.to_json())["compression_ratio"] is None
        assert result.verdict == "NOT_COMPARABLE"
