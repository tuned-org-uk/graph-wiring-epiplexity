"""Code accounting and log-based epiplexity estimates; NumPy-only core."""

from epiplexity.engine import EpiplexityEngine
from epiplexity.estimators import PrequentialEstimator, RequentialEstimator
from epiplexity.model import TTimeProbabilisticModel
from epiplexity.results import CodingResult
from epiplexity.selection import Candidate, pareto_frontier, select_under_budget

__version__ = "0.6.1"

__all__ = [
    "Candidate", "CodingResult", "EpiplexityEngine", "PrequentialEstimator",
    "RequentialEstimator", "TTimeProbabilisticModel", "pareto_frontier",
    "select_under_budget", "__version__",
]
