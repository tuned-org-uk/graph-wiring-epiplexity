"""Log-consuming estimates without training-framework dependencies.

Losses and KL values must be means per scored token, not batch sums. Token
counts refer to scored targets (exclude masked/padded positions). These are
numerical estimates; log ingestion alone does not construct a decodable code.
"""

import math
from typing import Sequence

import numpy as np

from epiplexity.results import CodingResult, _count, _nonnegative


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain finite non-negative numbers")
    return array


def _bit_scale(loss_unit: str) -> float:
    if loss_unit == "nats":
        return 1 / math.log(2)
    if loss_unit == "bits":
        return 1.0
    raise ValueError("loss_unit must be 'nats' or 'bits'")


def _token_counts(values: Sequence[float], name: str) -> np.ndarray:
    counts = _vector(values, name)
    # Validate original integers too: float64 can round 2**53 + 1 down to 2**53.
    for value in values:
        _count(value, name)
    return counts


class PrequentialEstimator:
    """Trapezoid loss-curve estimate over cumulative scored tokens.

    Include a zero-token initial loss. No unobserved prefix is extrapolated.
    ``final_loss`` defaults to the final logged loss, in ``loss_unit``. It must
    describe the same target distribution as the curve. Negative model-bit
    estimates are retained and marked, not silently clamped.
    """

    @staticmethod
    def estimate(
        tokens: Sequence[int],
        losses: Sequence[float],
        *,
        loss_unit: str = "nats",
        final_loss: float | None = None,
    ) -> CodingResult:
        scale = _bit_scale(loss_unit)
        times = _token_counts(tokens, "tokens")
        values = _vector(losses, "losses")
        if len(times) != len(values) or len(times) < 2:
            raise ValueError("tokens and losses must have the same length, at least two")
        if times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError("tokens must start at zero and strictly increase")
        reference = values[-1] if final_loss is None else _nonnegative(final_loss, "final_loss")
        # Explicit trapezoid rule avoids NumPy trapz/trapezoid version differences.
        differences = values - reference
        model_bits = math.fsum(
            float(width) * (float(left) / 2 + float(right) / 2)
            for width, left, right in zip(np.diff(times), differences[:-1], differences[1:])
        ) * scale
        residual = float(times[-1]) * float(reference) * scale
        return CodingResult("prequential", model_bits, residual, int(times[-1]))


class RequentialEstimator:
    """Token-weighted teacher-to-student KL estimate.

    Supply pre-update KL(P_teacher || P_student) on teacher-generated data.
    ``token_counts`` gives the number of scored student targets at each step;
    the evaluation corpus size is separate. The caller is responsible for the
    teacher/student schedule and the validity of the distillation protocol.
    """

    @staticmethod
    def estimate(
        kl_values: Sequence[float],
        token_counts: Sequence[int],
        *,
        final_loss: float,
        evaluation_tokens: int,
        loss_unit: str = "nats",
    ) -> CodingResult:
        scale = _bit_scale(loss_unit)
        divergences = _vector(kl_values, "kl_values")
        counts = _token_counts(token_counts, "token_counts")
        if len(divergences) != len(counts) or np.any(counts <= 0):
            raise ValueError("each KL value must have a positive token count")
        n_eval = _count(evaluation_tokens, "evaluation_tokens")
        if n_eval == 0:
            raise ValueError("evaluation_tokens must be positive")
        loss = _nonnegative(final_loss, "final_loss")
        model_bits = math.fsum(float(kl) * float(n) for kl, n in zip(divergences, counts)) * scale
        return CodingResult("requential", model_bits, loss * n_eval * scale, n_eval)
