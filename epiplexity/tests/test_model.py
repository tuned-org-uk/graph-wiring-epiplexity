# tests/test_model.py
"""
Tests for the TTimeProbabilisticModel ABC (epiplexity/model.py).

Verifies:
  - The ABC cannot be instantiated directly.
  - A concrete implementation satisfies all abstract contracts.
  - Return types are correct for every abstract method.
"""

import pytest
from epiplexity.model import TTimeProbabilisticModel
from tests.conftest import ConstantModel


class TestAbstractInterface:

    def test_cannot_instantiate_abstract_class(self):
        """TTimeProbabilisticModel must not be directly instantiatable."""
        with pytest.raises(TypeError):
            TTimeProbabilisticModel()

    def test_missing_one_method_raises(self):
        """A class that omits any abstract method must raise TypeError on init."""

        class Incomplete(TTimeProbabilisticModel):
            def description_length_bits(self): return 1.0
            def log_prob(self, x): return -1.0
            def sample(self, n=1): return []
            # raw_bits deliberately omitted

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_is_instantiatable(self):
        m = ConstantModel()
        assert isinstance(m, TTimeProbabilisticModel)

    def test_description_length_returns_float(self):
        m = ConstantModel(desc_bits=512.0)
        result = m.description_length_bits()
        assert isinstance(result, float)
        assert result == 512.0

    def test_log_prob_returns_float(self):
        m = ConstantModel(log_prob_val=-3.5)
        result = m.log_prob("anything")
        assert isinstance(result, float)
        assert result == pytest.approx(-3.5)

    def test_sample_returns_sequence(self):
        m = ConstantModel()
        s = m.sample(n=5)
        assert len(s) == 5

    def test_raw_bits_positive(self):
        m = ConstantModel()
        X = list(range(10))
        rb = m.raw_bits(X)
        assert isinstance(rb, float)
        assert rb > 0