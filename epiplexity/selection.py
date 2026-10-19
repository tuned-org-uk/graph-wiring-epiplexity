"""Select among supplied candidates using explicit compute measurements."""

from dataclasses import dataclass
from typing import Iterable

from epiplexity.results import CodingResult, _nonnegative


@dataclass(frozen=True)
class Candidate:
    """One measured run on an identified target and coding protocol.

    ``comparison_key`` must identify dataset, split, target encoding,
    conditioning, and scoring protocol. ``compute_unit`` must identify the
    resource and scope (for example, training_flops or cpu_scoring_seconds).
    A precision setting is not a compute measurement.
    """

    name: str
    result: CodingResult
    compute: float
    compute_unit: str
    comparison_key: str

    def __post_init__(self):
        for name in ("name", "compute_unit", "comparison_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "compute", _nonnegative(self.compute, "compute"))
        if self.result.score_kind == "density" or self.result.model_bits < 0:
            raise ValueError("candidate must have non-negative discrete code estimates")


def pareto_frontier(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Return strict improvements in total bits as compute increases.

    Ties prefer lower compute, then lexicographic name. This is a frontier of
    the supplied candidates, not optimisation over all time-bounded programs.
    Different estimator methods must be analysed in separate frontiers.
    """
    runs = list(candidates)
    if not runs:
        return []
    signature = lambda c: (c.comparison_key, c.compute_unit, c.result.method,
                           c.result.score_kind, c.result.observation_count)
    expected = signature(runs[0])
    if any(signature(c) != expected for c in runs):
        raise ValueError("comparison requires matching targets, units, methods and counts")
    if len({c.name for c in runs}) != len(runs):
        raise ValueError("candidate names must be unique")
    frontier = []
    best = float("inf")
    for run in sorted(runs, key=lambda c: (c.compute, c.result.total_bits, c.name)):
        if run.result.total_bits < best:
            frontier.append(run)
            best = run.result.total_bits
    return frontier


def select_under_budget(candidates: Iterable[Candidate], budget: float) -> Candidate | None:
    """Return the shortest supplied code within budget, or None if none fits."""
    limit = _nonnegative(budget, "budget")
    eligible = [run for run in pareto_frontier(candidates) if run.compute <= limit]
    return eligible[-1] if eligible else None
