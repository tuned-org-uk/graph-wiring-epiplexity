"""Framework-free result and wire format shared by all estimators."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import math
from typing import Any, Mapping


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _count(value: int, name: str) -> int:
    number = _nonnegative(value, name)
    if number != int(number) or number > 2**53 or value != int(number):
        raise ValueError(f"{name} must be an integer in [0, 2**53]")
    return int(number)


@dataclass(frozen=True)
class CodingResult:
    """A code-length estimate, not a proof of the MDL-optimal epiplexity.

    All numeric outputs use bits. ``density`` scores are differential scores,
    not discrete code lengths. A baseline is comparable only if it encodes
    the same targets with the same conditioning and precision.
    """

    method: str
    model_bits: float
    data_bits: float
    observation_count: int
    raw_bits: float | None = None
    score_kind: str = "discrete"
    baseline_comparable: bool = False

    def __post_init__(self):
        if self.method not in {"static", "prequential", "requential"}:
            raise ValueError("unknown estimation method")
        if self.score_kind not in {"discrete", "conditional", "density"}:
            raise ValueError("unknown score_kind")
        object.__setattr__(self, "model_bits", _finite(self.model_bits, "model_bits"))
        object.__setattr__(self, "data_bits", _finite(self.data_bits, "data_bits"))
        object.__setattr__(self, "observation_count", _count(
            self.observation_count, "observation_count"
        ))
        if self.method != "prequential" and self.model_bits < 0:
            raise ValueError("model_bits must be non-negative for this method")
        if self.score_kind != "density" and self.data_bits < 0:
            raise ValueError("discrete data_bits must be non-negative")
        if not isinstance(self.baseline_comparable, bool):
            raise ValueError("baseline_comparable must be a bool")
        if self.raw_bits is not None:
            object.__setattr__(self, "raw_bits", _nonnegative(self.raw_bits, "raw_bits"))
        _finite(self.total_bits, "total_bits")

    @property
    def total_bits(self) -> float:
        return self.model_bits + self.data_bits

    @property
    def diagnostics(self) -> tuple[str, ...]:
        notes = []
        if self.model_bits < 0:
            notes.append("negative_model_bits")
        if self.score_kind == "density":
            notes.append("density_not_discrete_code")
        if self.raw_bits is not None and not self.baseline_comparable:
            notes.append("baseline_not_comparable")
        if self.total_bits <= 0:
            notes.append("nonpositive_total_bits")
        return tuple(notes)

    @property
    def compression_ratio(self) -> float | None:
        if (self.raw_bits is None or not self.baseline_comparable
                or self.score_kind == "density" or self.model_bits < 0
                or self.total_bits <= 0):
            return None
        ratio = self.raw_bits / self.total_bits
        return ratio if math.isfinite(ratio) else None

    @property
    def verdict(self) -> str:
        ratio = self.compression_ratio
        if ratio is None:
            return "NOT_COMPARABLE"
        return "STRUCTURAL" if ratio > 1 else "METADATA"

    def to_dict(self, *, metadata: Mapping[str, Any] | None = None) -> dict:
        """Return JSON-safe data; caller-supplied provenance is copied.

        Record dataset/model digests, config, seed and compute units in metadata.
        The library does not guess these values or hash model weights implicitly.
        """
        try:
            library_version = version("epiplexity")
        except PackageNotFoundError:
            library_version = "uninstalled"
        payload = {
            "schema_version": "1.0",
            "library_version": library_version,
            "method": self.method,
            "score_kind": self.score_kind,
            "unit": "bits",
            "model_bits": self.model_bits,
            "data_bits": self.data_bits,
            "total_bits": self.total_bits,
            "observation_count": self.observation_count,
            "raw_bits": self.raw_bits,
            "baseline_comparable": self.baseline_comparable,
            "compression_ratio": self.compression_ratio,
            "verdict": self.verdict,
            "diagnostics": list(self.diagnostics),
            "metadata": dict(metadata) if metadata is not None else {},
        }
        return json.loads(json.dumps(payload, allow_nan=False))

    def to_json(self, *, metadata: Mapping[str, Any] | None = None) -> str:
        return json.dumps(self.to_dict(metadata=metadata), allow_nan=False, sort_keys=True)
