# epiplexity

A generic tool for measuring **epiplexity** (Finzi et al., 2026) -- the structural information learnable by a computationally bounded observer -- for *any* algorithm wrapped as a T-time probabilistic model.

References, paper and notebooks [available here](https://github.com/tuned-org-uk/graph-wiring-epiplexity).

---

## Table of Contents

1. [Background](#background)
2. [Installation](#installation)
3. [Package layout](#package-layout)
4. [Core concepts](#core-concepts)
5. [How to define your own adapter](#how-to-define-your-own-adapter)
6. [Built-in adapters](#built-in-adapters)
7. [Interpreting results](#interpreting-results)
8. [Epiplexity properties and their tests](#epiplexity-properties-and-their-tests)
9. [Running the test suite](#running-the-test-suite)
10. [Full worked example: scikit-learn classifier](#full-worked-example)

---

## Background

Classical Shannon entropy and Kolmogorov complexity both assume unlimited computation.
**Epiplexity** captures what a *bounded* observer can actually learn, splitting dataset X into:

    MDL_T(X)  =  S_T(X)          +  H_T(X)
                 epiplexity           time-bounded entropy
                 (structural bits)    (random bits)

| Symbol   | Name                     | Meaning |
|----------|--------------------------|---------|
| S_T(X)   | **Epiplexity**           | Bits to describe the model program; learnable structure |
| H_T(X)   | **Time-bounded entropy** | Per-item irreducible noise for a bounded observer |
| MDL_T(X) | Two-part MDL code        | Total description length under time-bound T |

**Compression test:** if `MDL_T(X) < n_raw` the model captures real structure, not metadata.

> Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson -- *"From Entropy to Epiplexity: Rethinking
> Information for Computationally Bounded Intelligence"*, arXiv:2601.03220, 2026.

---

## Installation

```bash
pip install -e .
pip install transformers    # optional: LM adapter
pip install torch_geometric # optional: GNN adapter
```

---

## Package layout

```
epiplexity/
|-- model.py                  # TTimeProbabilisticModel ABC
|-- engine.py                 # EpiplexityEngine (MDL calculator)
`-- algorithms/
    |-- arrowspace.py         # ArrowSpace LGMRF adapter
    |-- torch_classifier.py   # PyTorch classifier adapter
    |-- transformer_lm.py     # Auto-regressive LM adapter
    `-- gnn.py                # Graph neural network adapter

tests/
|-- conftest.py                      # Shared fixtures, tiny synthetic models
|-- test_model.py                    # ABC contract tests
|-- test_engine.py                   # Engine arithmetic + verdict tests
|-- test_torch_classifier.py         # TorchClassifierModelAdapter tests
|-- test_transformer_lm.py           # TransformerLMModelAdapter tests
|-- test_gnn.py                      # GNNModelAdapter tests
`-- test_epiplexity_properties.py    # Cross-algorithm semantic property tests (P1-P8)
```

---

## Core concepts

### TTimeProbabilisticModel (epiplexity/model.py)

```python
from abc import ABC, abstractmethod
from typing import Any

class TTimeProbabilisticModel(ABC):

    @abstractmethod
    def description_length_bits(self) -> float:
        # S_T(X) proxy: bits to describe this program/model
        ...

    @abstractmethod
    def log_prob(self, x: Any) -> float:
        # log P(x) in natural log. Must be <= 0.
        ...

    @abstractmethod
    def sample(self, n: int = 1) -> Any:
        # Draw n samples from P. May raise NotImplementedError.
        ...

    @abstractmethod
    def raw_bits(self, X: Any) -> float:
        # Uncompressed bit-size of dataset X (the 'do nothing' baseline).
        ...
```

### EpiplexityEngine (epiplexity/engine.py)

```python
engine = EpiplexityEngine(adapter, dataset)

engine.structural_bits      # S_T(X)            float
engine.entropy_bits         # H_T(x_i) per item np.ndarray
engine.total_entropy_bits   # H_T(X)            float
engine.mdltotal_bits        # S_T + H_T
engine.raw_bits_total()     # uncompressed baseline
engine.compression_ratio()  # raw / MDL  (>1 means structural)
engine.report()             # prints formatted summary
```

---

## How to define your own adapter

### Step 1 -- Subclass TTimeProbabilisticModel

```python
from epiplexity.model import TTimeProbabilisticModel

class MyAlgorithmAdapter(TTimeProbabilisticModel):
    def __init__(self, my_model, config):
        self.model  = my_model
        self.config = config
```

---

### Step 2 -- description_length_bits -> S_T(X)

**S_T(X)** is the epiplexity proxy: bits to fully specify the algorithm's program.

What to include:

| Component          | Example |
|--------------------|---------|
| Learned parameters | num_params x bits_per_param |
| Architecture       | Layer sizes, graph topology |
| Hyperparameters    | Learning rate, cluster count |
| Training seed      | ~64 bits |

**Elias gamma coding** for non-negative integers (prefix-free):

```python
import math
def elias_gamma_bits(x: int) -> int:
    return 2 * math.floor(math.log2(max(1, x))) + 1
```

**Precision table:**

| Setting      | bits_per_param |
|--------------|---------------|
| float64      | 64 |
| float32      | 32 |
| float16/fp16 | 16 |
| int8 quant   | 8  |

Example:

```python
def description_length_bits(self) -> float:
    num_params  = sum(p.numel() for p in self.model.parameters())
    param_bits  = num_params * self.config.bits_per_param
    arch_bits   = elias_gamma_bits(self.config.hidden_dim)
    overhead    = 64  # seed + flags
    return float(param_bits + arch_bits + overhead)
```

> Observer-dependence (Property P4): S_T should grow monotonically with
> bits_per_param and model size -- more compute means more structure encoded.

---

### Step 3 -- log_prob -> per-item H_T contribution

Return `log P(x)` in **natural logarithm** for a single item.  The engine converts to bits:

    H_T(x_i) = -log2 P(x_i) = -log P(x_i) / log(2)

This must be <= 0 since P(x) is a probability in (0, 1].

Probabilistic interpretation guide:

| Algorithm type        | P(x)                            | Implementation |
|-----------------------|---------------------------------|----------------|
| Classifier            | P(y|x) via softmax              | `log_softmax(logits)[y]` |
| Language model        | prod_t P(x_t | x_{<t})         | sum of token log-probs |
| GNN                   | P(y|G) via softmax              | `log_softmax(logits)[y]` |
| ArrowSpace (LGMRF)    | N(0, Q^{-1})                    | `-0.5 * x^T Q x - log Z` |
| Density estimator     | direct model output             | `model.log_prob(x)` |
| Autoencoder           | reconstruction likelihood       | `-0.5 * ||x - x_hat||^2 / sigma^2` |

Example for a classifier:

```python
def log_prob(self, x) -> float:
    features, label = x
    with torch.no_grad():
        logits = self.model(features.unsqueeze(0))
        lp = torch.log_softmax(logits, dim=-1)[0, int(label)]
    return float(lp)
```

> Tip: for non-probabilistic algorithms, treat the score function f(x) as negative energy:
> P(x) proportional to exp(-f(x)), with normalisation estimated from the dataset.

---

### Step 4 -- raw_bits -> uncompressed baseline

Return the total bit-size under a "do nothing" encoding.

| Data type      | Baseline |
|----------------|----------|
| Float32 tensor | N x F x 32 bits |
| Float16 tensor | N x F x 16 bits |
| Raw text       | 8 x num_chars bits |
| Graph nodes    | num_nodes x F x 32 bits |
| Graph edges    | 2 x num_edges x 32 bits |

```python
def raw_bits(self, X) -> float:
    return float(sum(x.numel() * 32 for x, _ in X))
```

---

### Step 5 -- sample (optional)

Required by Definition 7 for a formally valid T-time probabilistic model, but not used by
EpiplexityEngine.  For discriminative models, `NotImplementedError` is acceptable:

```python
def sample(self, n: int = 1):
    raise NotImplementedError("Sampling is task-specific for discriminative models.")
```

---

### Step 6 -- Run EpiplexityEngine

```python
from epiplexity.engine import EpiplexityEngine

adapter = MyAlgorithmAdapter(my_trained_model, my_config)
engine  = EpiplexityEngine(adapter, my_dataset)
engine.report()
```

Expected output:

```
Epiplexity report
-----------------
  Structural bits S_T     :   512.00 KB
  Random bits H_T         :   128.00 KB
  Total MDL S_T + H_T     :   640.00 KB
  Raw size                :  3000.00 KB
  Compression ratio       :     4.69x
  Verdict                 : STRUCTURAL
```

---

## Built-in adapters

### PyTorch classifier

```python
from epiplexity.algorithms.torch_classifier import (
    TorchClassifierModelAdapter, TorchClassifierEpiplexityConfig,
)
cfg     = TorchClassifierEpiplexityConfig(bits_per_param=32, overhead_bits=8192.0)
adapter = TorchClassifierModelAdapter(model, cfg)
engine  = EpiplexityEngine(adapter, dataset)  # dataset: list of (Tensor, int)
```

- S_T = num_params x bits_per_param + overhead
- H_T(x_i) = -log2 P(y_i|x_i) under softmax

### Transformer language model

```python
from epiplexity.algorithms.transformer_lm import (
    TransformerLMModelAdapter, TransformerLMEpiplexityConfig,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
cfg       = TransformerLMEpiplexityConfig(bits_per_param=16, max_length=2048)
adapter   = TransformerLMModelAdapter(model, tokenizer, cfg)
engine    = EpiplexityEngine(adapter, ["text item 1", "text item 2"])
```

- S_T = num_params x bits_per_param + overhead
- H_T(x_i) = sum of per-token negative log-probs

### Graph neural network

```python
from epiplexity.algorithms.gnn import GNNModelAdapter, GNNEpiplexityConfig
cfg     = GNNEpiplexityConfig(bits_per_param=32, overhead_bits=16384.0)
adapter = GNNModelAdapter(model, cfg)
engine  = EpiplexityEngine(adapter, graph_dataset)  # list of (graph_obj, int)
# graph_obj must have .x (node features) and .edge_index
```

### ArrowSpace spectral LGMRF

```python
from epiplexity.algorithms.arrowspace import ArrowSpaceModelAdapter
adapter = ArrowSpaceModelAdapter(arrowspace_model, C0=200, k=16, b=32)
# arrowspace_model must expose:
#   .evaluatelogprob(x)           natural log
#   .descriptionlengthbits(C0,k,b)
#   .sample(n)                    shape (F, n)
engine  = EpiplexityEngine(adapter, X)  # X: np.ndarray (N, F)
```

---

## Interpreting results

| Metric                 | What it tells you |
|------------------------|-------------------|
| structural_bits (S_T)  | Information absorbed into weights. Grows with capacity and compute budget. |
| total_entropy_bits (H_T) | Irreducible noise; decreases as the model improves on the data. |
| compression_ratio > 1  | Algorithm compresses data -> passes the structural content test. |
| compression_ratio < 1  | Model too large, data too small, or data is effectively random noise. |
| Verdict: STRUCTURAL    | MDL_T < raw_bits. The algorithm found learnable structure. |
| Verdict: METADATA      | MDL_T >= raw_bits. The algorithm is not compressive for this data. |

**bits_per_param is the T-budget knob.** Lower = more restricted observer. Higher = richer
observer. S_T grows monotonically -- this is the observer-dependence property (P4).

---

## Epiplexity properties and their tests

`tests/test_epiplexity_properties.py` verifies eight formal properties across all adapters:

| Property | Test class | What is verified |
|----------|------------|------------------|
| P1 MDL identity         | TestP1_MDLIdentity         | S_T + H_T == MDL_T for all four adapters |
| P2 Compression test     | TestP2_CompressionTest     | STRUCTURAL / METADATA verdict in controlled scenarios |
| P3 Non-negativity       | TestP3_NonNegativity       | H_T(x_i) >= 0, all items finite |
| P4 Observer-dependence  | TestP4_ObserverDependence  | S_T grows monotonically with bits_per_param and model size |
| P5 H_T ordering         | TestP5_EntropyOrdering     | Confident models have lower H_T; smooth < rough (ArrowSpace) |
| P6 Per-item consistency | TestP6_PerItemConsistency  | entropy_bits.sum() == total_entropy_bits |
| P7 ArrowSpace LGMRF     | TestP7_ArrowSpaceLGMRF     | Dirichlet energy order matches H_T; S_T grows with k and C0 |
| P8 Epiplexity rank      | TestP8_EpiplexityRank      | S_T rank between small/large models is stable under capacity scaling |

---

## Running the test suite

```bash
# Full suite
pytest tests/ -v

# Only property tests
pytest tests/test_epiplexity_properties.py -v

# A single property class
pytest tests/test_epiplexity_properties.py::TestP4_ObserverDependence -v

# With coverage
pytest tests/ --cov=epiplexity --cov-report=term-missing
```

---

## Full worked example: scikit-learn classifier

The same six-step pattern works for any probability-scoring algorithm.

```python
# examples/sklearn_rf_adapter.py
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from epiplexity.model import TTimeProbabilisticModel
from epiplexity.engine import EpiplexityEngine


def elias_gamma_bits(x: int) -> int:
    return 2 * math.floor(math.log2(max(1, x))) + 1


class RFConfig:
    bits_per_split: int   = 48    # log2(F) for feature index + 32 for threshold
    overhead_bits: float  = 64.0


class RandomForestAdapter(TTimeProbabilisticModel):

    def __init__(self, model: RandomForestClassifier, config: RFConfig):
        self.model  = model
        self.config = config

    def description_length_bits(self) -> float:
        n_trees     = self.model.n_estimators
        total_nodes = sum(t.tree_.node_count for t in self.model.estimators_)
        avg_nodes   = total_nodes / n_trees
        param_bits  = n_trees * avg_nodes * self.config.bits_per_split
        header_bits = elias_gamma_bits(n_trees) + elias_gamma_bits(int(avg_nodes))
        return float(param_bits + header_bits + self.config.overhead_bits)

    def log_prob(self, xy) -> float:
        x, y  = xy
        proba = self.model.predict_proba([x])[0]
        p     = float(np.clip(proba[int(y)], 1e-12, 1.0))
        return math.log(p)   # natural log

    def sample(self, n: int = 1):
        raise NotImplementedError

    def raw_bits(self, X) -> float:
        return float(sum(len(x) * 32 for x, _ in X))


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X_raw, y_raw = make_classification(
        n_samples=500, n_features=20, n_informative=10, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2)

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)

    engine = EpiplexityEngine(RandomForestAdapter(rf, RFConfig()), list(zip(X_test, y_test)))
    engine.report()
    print(f"S_T = {engine.structural_bits / (8*1024):.2f} KB   "
          f"H_T mean = {engine.entropy_bits.mean():.2f} bits/item")
```

> Note: a 50-tree forest typically has large S_T relative to a small test set, producing
> Verdict: METADATA.  Increase dataset size or reduce `bits_per_split` to reflect a more
> compressed tree representation (e.g. pruned or quantised splits) to pass the compression test.
