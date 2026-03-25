# epiplexity_torch_classifier.py

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from epiplexity.model import TTimeProbabilisticModel


@dataclass
class TorchClassifierEpiplexityConfig:
    """
    Configuration for encoding a PyTorch classifier as a T-time model.

    - bits_per_param: how many bits to budget per learned weight (e.g. 32).
    - overhead_bits: extra bits for architecture, optimiser, training recipe.
    - input_bits_per_float: raw baseline bits per input feature (default 32).
    """
    bits_per_param: int = 32
    overhead_bits: float = 8_192.0  # 1 KB architecture metadata, tweak as needed
    input_bits_per_float: int = 32


class TorchClassifierModelAdapter(TTimeProbabilisticModel):
    """
    T-time probabilistic model wrapper for a generic PyTorch classifier.

    The model defines a distribution P(y | x) over discrete labels;
    we lift this to a distribution over (x, y) pairs by treating x as given
    and coding only the labels, using the classifier's predicted probabilities.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TorchClassifierEpiplexityConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        # Pre-compute the number of parameters for S_T(X).
        self._num_params = sum(p.numel() for p in self.model.parameters())

    def description_length_bits(self) -> float:
        """
        S_T(X): bits needed to specify the classifier under a given precision budget.
        """
        param_bits = self._num_params * self.config.bits_per_param
        return float(param_bits + self.config.overhead_bits)

    def log_prob(self, xy: Tuple[torch.Tensor, int]) -> float:
        """
        log P(y | x) for a single (x, y) pair.

        xy: (input_tensor, label_index)
        """
        x, y = xy
        x = x.to(self.device).unsqueeze(0)  # add batch dim
        y = int(y)

        with torch.no_grad():
            logits = self.model(x)  # shape (1, C)
            log_probs = torch.log_softmax(logits, dim=-1)  # natural log
            lp = log_probs[0, y].item()
        return float(lp)

    def sample(self, n: int = 1):
        """
        Sampling requires a prior over inputs; for simplicity we only sample labels
        y ~ P(y) under a synthetic prior (e.g., average softmax over some reference set).
        This is often not needed for epiplexity use-cases.
        """
        raise NotImplementedError(
            "Sampling (x, y) from a generic classifier is application-specific."
        )

    def raw_bits(self, X: Sequence[Tuple[torch.Tensor, int]]) -> float:
        """
        Baseline size: treat each input tensor as float32 (or config.input_bits_per_float).
        """
        total_bits = 0.0
        for x, _ in X:
            num_floats = x.numel()
            total_bits += num_floats * self.config.input_bits_per_float
        return float(total_bits)