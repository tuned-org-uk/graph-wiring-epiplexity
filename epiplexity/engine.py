"""Batch and streaming evaluation of a fitted probabilistic model."""

from itertools import islice
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np

from epiplexity.model import TTimeProbabilisticModel
from epiplexity.results import CodingResult, _count, _nonnegative


class _Sum:
    """Neumaier compensated summation with constant storage."""

    def __init__(self):
        self.total = 0.0
        self.correction = 0.0

    def add(self, value):
        updated = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - updated) + value
        else:
            self.correction += (value - updated) + self.total
        self.total = updated

    def value(self):
        return self.total + self.correction


def _batches(items, size):
    if isinstance(items, np.ndarray):
        for start in range(0, len(items), size):
            yield items[start:start + size]
    else:
        iterator = iter(items)
        while batch := list(islice(iterator, size)):
            yield batch

class EpiplexityEngine:
    """Static two-part code accounting with optional per-item retention.

    The constructor retains X for compatibility. Use ``from_batches`` for
    bounded-memory ingestion. Results snapshot scores and model cost; changing
    model weights later cannot update only half of the reported decomposition.
    """

    def __init__(
        self, model: TTimeProbabilisticModel, X: Any, *, batch_size: int = 1024,
        store_per_item: bool = True, raw_bits: float | None = None,
        baseline_comparable: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        size = _count(batch_size, "batch_size")
        if not size:
            raise ValueError("batch_size must be positive")
        self.X = X
        # Preserve whole-dataset baseline semantics for existing sized inputs.
        if raw_bits is None and hasattr(X, "__len__"):
            raw_bits = model.raw_bits(X)
        self._evaluate(model, _batches(X, size), store_per_item, raw_bits,
                       baseline_comparable, metadata)

    @classmethod
    def from_batches(
        cls, model: TTimeProbabilisticModel, batches: Iterable, *,
        store_per_item: bool = False, raw_bits: float | None = None,
        baseline_comparable: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EpiplexityEngine":
        """Consume sized batches once; retain no inputs.

        Without a whole-corpus ``raw_bits`` override, ``model.raw_bits(batch)``
        must be additive across batches. Memory is O(largest batch), or O(N)
        when ``store_per_item=True``. Empty batches are ignored.
        """
        engine = cls.__new__(cls)
        engine.X = None
        engine._evaluate(model, batches, store_per_item, raw_bits,
                         baseline_comparable, metadata)
        return engine

    def _evaluate(self, model, batches, store_per_item, raw_bits, comparable, metadata):
        self.model = model
        if not isinstance(store_per_item, bool):
            raise ValueError("store_per_item must be a bool")
        self._metadata_json = json.dumps(dict(metadata or {}), allow_nan=False)
        model_bits = _nonnegative(model.description_length_bits(), "model_bits")
        kind = getattr(model, "score_kind", "discrete")
        if kind not in {"discrete", "conditional", "density"}:
            raise ValueError("unknown score_kind")
        if comparable is None:
            comparable = getattr(model, "baseline_comparable", True)
        if raw_bits is not None:
            raw_bits = _nonnegative(raw_bits, "raw_bits")
        score_batch = getattr(model, "batch_log_prob", None)
        chunks = [] if store_per_item else None
        entropy_sum, raw_sum = _Sum(), _Sum()
        count = 0
        for batch in batches:
            n = len(batch)
            if n == 0:
                continue
            logs = np.asarray(
                score_batch(batch) if score_batch else [model.log_prob(x) for x in batch],
                dtype=np.float64,
            )
            if logs.shape != (n,) or not np.all(np.isfinite(logs)):
                raise ValueError("batch_log_prob must return one finite log score per item")
            if kind != "density" and np.any(logs > 0):
                raise ValueError("discrete log probabilities must be <= 0")
            bits = -logs / math.log(2)
            entropy_sum.add(math.fsum(bits))
            if raw_bits is None:
                raw_sum.add(_nonnegative(model.raw_bits(batch), "raw_bits"))
            count += n
            if chunks is not None:
                chunks.append(bits)
        self._entropy_bits = (
            np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        ) if chunks is not None else None
        self._result = CodingResult(
            "static", model_bits, entropy_sum.value(), count,
            raw_sum.value() if raw_bits is None else raw_bits, kind, comparable,
        )

    @property
    def result(self) -> CodingResult:
        return self._result

    @property
    def structural_bits(self) -> float:
        """
        Epiplexity proxy S_T(X): model/program bits.
        """
        return self.result.model_bits

    @property
    def entropy_bits(self) -> np.ndarray:
        """
        Per-item time-bounded entropy H_T(x_i) in bits.
        """
        if self._entropy_bits is None:
            raise ValueError("per-item scores were not retained; use store_per_item=True")
        return self._entropy_bits.copy()

    @property
    def total_entropy_bits(self) -> float:
        """
        H_T(X): sum of per-item random-information bits.
        """
        return self.result.data_bits

    @property
    def mdltotal_bits(self) -> float:
        """
        Two-part MDL code length: S_T(X) + H_T(X).
        """
        return self.result.total_bits

    def raw_bits_total(self) -> float:
        """
        Baseline uncompressed size of X in bits.
        """
        return self.result.raw_bits

    def compression_ratio(self) -> float | None:
        """
        Legacy arithmetic raw / total. This is not a validity certificate.

        Prefer result.compression_ratio, which requires comparable encodings.
        A zero denominator returns None instead of infinity.
        """
        if self.mdltotal_bits == 0:
            return None
        return self.raw_bits_total() / self.mdltotal_bits

    def to_dict(self, *, include_per_item: bool = False) -> dict:
        output = self.result.to_dict(metadata=json.loads(self._metadata_json))
        if include_per_item:
            output["entropy_bits"] = self.entropy_bits.tolist()
        return output

    def to_json(self, *, include_per_item: bool = False) -> str:
        return json.dumps(self.to_dict(include_per_item=include_per_item),
                          allow_nan=False, sort_keys=True)

    def report(self) -> None:
        raw_kb = self.raw_bits_total() / (8 * 1024)
        st_kb = self.structural_bits / (8 * 1024)
        ht_kb = self.total_entropy_bits / (8 * 1024)
        mdl_kb = self.mdltotal_bits / (8 * 1024)
        cr = self.result.compression_ratio
        print("Epiplexity report")
        print("-----------------")
        print(f"  Structural bits S_T     : {st_kb:8.2f} KB")
        print(f"  Random bits H_T         : {ht_kb:8.2f} KB")
        print(f"  Total MDL S_T + H_T     : {mdl_kb:8.2f} KB")
        print(f"  Raw size                : {raw_kb:8.2f} KB")
        ratio = f"{cr:8.2f}x" if cr is not None else "not comparable"
        print(f"  Compression ratio       : {ratio}")
        print(f"  Verdict                 : {self.result.verdict}")
        print(f"  Method / score kind     : static / {self.result.score_kind}")
