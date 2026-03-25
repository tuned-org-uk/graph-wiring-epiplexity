from abc import ABC, abstractmethod
from typing import Any, Protocol
import numpy as np


class TTimeProbabilisticModel(ABC):
    """
    Abstract epiplexity interface for arbitrary algorithms.
    Matches the Definition 7 + MDL two-part code pattern in your notebook.
    """

    @abstractmethod
    def description_length_bits(self) -> float:
        """
        Return S_T(X): description length (in bits) of this model/program,
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
