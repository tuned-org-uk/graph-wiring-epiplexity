import pytest

from epiplexity.results import CodingResult
from epiplexity.selection import Candidate, pareto_frontier, select_under_budget


def candidate(name, compute, model, data, key="corpus-a/tokens-v1"):
    return Candidate(name, CodingResult("static", model, data, 100),
                     compute, "training_flops", key)


def test_frontier_uses_total_code_not_largest_model_or_lowest_loss():
    runs = [candidate("large", 30, 80, 5), candidate("small", 10, 10, 90),
            candidate("best", 20, 20, 60), candidate("dominated", 25, 25, 60)]
    assert [r.name for r in pareto_frontier(runs)] == ["small", "best"]
    assert select_under_budget(runs, 9) is None
    assert select_under_budget(runs, 10).name == "small"
    assert select_under_budget(runs, 100).name == "best"


def test_equal_costs_and_totals_have_stable_ties():
    runs = [candidate("b", 10, 5, 5), candidate("a", 10, 5, 5),
            candidate("later", 20, 5, 5)]
    assert [r.name for r in pareto_frontier(runs)] == ["a"]
    assert pareto_frontier([]) == []


def test_incomparable_targets_are_rejected():
    with pytest.raises(ValueError, match="comparison"):
        pareto_frontier([candidate("a", 1, 1, 1), candidate("b", 2, 1, 1, "other")])


@pytest.mark.parametrize("change", [
    {"compute_unit": "seconds"},
    {"result": CodingResult("static", 1, 1, 101)},
    {"result": CodingResult("requential", 1, 1, 100)},
])
def test_frontier_rejects_mixed_units_counts_and_methods(change):
    from dataclasses import replace
    a = candidate("a", 1, 1, 1)
    b = replace(candidate("b", 2, 1, 1), **change)
    with pytest.raises(ValueError, match="comparison"):
        pareto_frontier([a, b])


@pytest.mark.parametrize("compute", [-1, float("nan"), float("inf")])
def test_invalid_compute(compute):
    with pytest.raises(ValueError):
        candidate("bad", compute, 1, 1)


def test_density_scores_cannot_enter_a_discrete_code_frontier():
    with pytest.raises(ValueError):
        Candidate("density", CodingResult("static", 1, -1, 10, score_kind="density"),
                  1, "seconds", "a")


def test_negative_estimates_are_not_selected_as_shorter_model_codes():
    with pytest.raises(ValueError):
        Candidate("bad", CodingResult("prequential", -1, 2, 10), 1, "seconds", "a")


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError):
        select_under_budget([], -1)
