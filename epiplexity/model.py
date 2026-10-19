from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TTimeProbabilisticModel(ABC):
    """
    Interface for fitted-model code accounting.

    This interface alone does not enforce a time bound, normalisation, or
    MDL-optimal model selection. Density scores require a discrete encoding
    before comparison with raw storage sizes.
    """

    score_kind = "discrete"  # discrete, conditional, or density
    baseline_comparable = True

    def batch_log_prob(self, X: Any) -> np.ndarray:
        """Return one natural-log score per item; override to vectorise."""
        return np.asarray([self.log_prob(x) for x in X], dtype=np.float64)

    @abstractmethod
    def description_length_bits(self) -> float:
        """
        Return a candidate description length (in bits) of this model/program,
        including hyperparameters, initialisation, and any learned state.
        """
        ...

    @abstractmethod
    def log_prob(self, x: Any) -> float:
        """
        Return log P(x) in natural logs.
        x can be a numpy array, token sequence, graph object, etc.
        """
        ...

    @abstractmethod
    def sample(self, n: int = 1) -> Any:
        """
        Draw n samples from P. Shape/type is model-specific.
        """
        ...

    @abstractmethod
    def raw_bits(self, X: Any) -> float:
        """
        Baseline uncompressed size of dataset X in bits.
        """
        ...
