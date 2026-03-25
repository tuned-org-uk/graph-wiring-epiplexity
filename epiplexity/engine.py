from typing import Any

import numpy as np

from epiplexity.model import TTimeProbabilisticModel

class EpiplexityEngine:
    """
    Generic epiplexity & time-bounded entropy calculator for any
    TTimeProbabilisticModel implementation.
    """

    def __init__(self, model: TTimeProbabilisticModel, X: Any):
        self.model = model
        self.X = X

        # Compute per-item log-probabilities and entropies (in bits).
        self._log_probs = np.array([self.model.log_prob(x) for x in X], dtype=float)
        self._entropy_bits = -self._log_probs / np.log(2.0)

    @property
    def structural_bits(self) -> float:
        """
        Epiplexity proxy S_T(X): model/program bits.
        """
        return float(self.model.description_length_bits())

    @property
    def entropy_bits(self) -> np.ndarray:
        """
        Per-item time-bounded entropy H_T(x_i) in bits.
        """
        return self._entropy_bits

    @property
    def total_entropy_bits(self) -> float:
        """
        H_T(X): sum of per-item random-information bits.
        """
        return float(self._entropy_bits.sum())

    @property
    def mdltotal_bits(self) -> float:
        """
        Two-part MDL code length: S_T(X) + H_T(X).
        """
        return self.structural_bits + self.total_entropy_bits

    def raw_bits_total(self) -> float:
        """
        Baseline uncompressed size of X in bits.
        """
        return float(self.model.raw_bits(self.X))

    def compression_ratio(self) -> float:
        """
        raw / MDL: >1 means non-trivial structure.
        """
        return self.raw_bits_total() / self.mdltotal_bits

    def report(self) -> None:
        raw_kb = self.raw_bits_total() / (8 * 1024)
        st_kb = self.structural_bits / (8 * 1024)
        ht_kb = self.total_entropy_bits / (8 * 1024)
        mdl_kb = self.mdltotal_bits / (8 * 1024)
        cr = self.compression_ratio()
        print("Epiplexity report")
        print("-----------------")
        print(f"  Structural bits S_T     : {st_kb:8.2f} KB")
        print(f"  Random bits H_T         : {ht_kb:8.2f} KB")
        print(f"  Total MDL S_T + H_T     : {mdl_kb:8.2f} KB")
        print(f"  Raw size                : {raw_kb:8.2f} KB")
        print(f"  Compression ratio       : {cr:8.2f}x")
        verdict = "STRUCTURAL" if self.mdltotal_bits < self.raw_bits_total() else "METADATA"
        print(f"  Verdict                 : {verdict}")