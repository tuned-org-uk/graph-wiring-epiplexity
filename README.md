# epiplexity

> Measure any algorithm's **epiplexity** (Finzi et al., 2026) -- the structural information
> learnable by a computationally bounded observer -- for *any* algorithm wrapped as a
> T-time probabilistic model.

This repository brings together three artefacts around the epiplexity measure:

| | Component | What it is | Where |
|---|-----------|------------|------|
| 1 | **Library** | `epiplexity` -- a lean, dependency-light Python package that implements the `EpiplexityEngine` and the `TTimeProbabilisticModel` ABC, plus ready-made adapters for PyTorch classifiers, transformer LMs, GNNs, and ArrowSpace. | [`epiplexity/`](epiplexity) |
| 2 | **Notebooks** | Two reproducible Jupyter notebooks that apply the library end-to-end: a synthetic structural-information study and a full CVE-1999--2025 case study achieving a ~38.4x compression ratio. | [`notebooks/`](notebooks) |
| 3 | **Paper** | *"Epiplexity: A Measure on Graph Wiring"* -- the accompanying paper (LaTeX source + PDF). | [`paper/`](paper) |

> **The library is the entry point.** The notebooks demonstrate it on real data, and the
> paper (to follow) formalises the theory behind the numbers the engine reports.

References, paper and notebooks [available here](https://github.com/tuned-org-uk/graph-wiring-epiplexity).

---

## Table of Contents

1. [Background](#background)
2. [Installation](#installation)
3. [Notebooks](#notebooks)
4. [Paper](#paper)
5. [Package layout](#package-layout)
6. [Core concepts](#core-concepts)
7. [How to define your own adapter](#how-to-define-your-own-adapter)
8. [Built-in adapters](#built-in-adapters)
9. [Interpreting results](#interpreting-results)
10. [Epiplexity properties and their tests](#epiplexity-properties-and-their-tests)
11. [Running the test suite](#running-the-test-suite)
12. [Full worked example: scikit-learn classifier](#full-worked-example)

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

The core library is intentionally lean: a plain install pulls in **only NumPy**.
Heavy ML dependencies are opt-in through [PEP 508 extras](https://peps.python.org/pep-0508/):

```bash
# Lean core only: engine + model ABC + ArrowSpace adapter (NumPy only)
pip install epiplexity

# PyTorch classifier / GNN adapters
pip install epiplexity[torch]

# HuggingFace transformer LM adapter
pip install epiplexity[transformers]

# Full notebook stack (sentence-transformers + torch + transformers)
pip install epiplexity[notebooks]

# Everything needed to run the test suite
pip install epiplexity[dev]
```

From source:

```bash
uv sync                 # or: pip install -e .
uv sync --extra notebooks   # to also run the notebooks
```

| Extra | Adds | Use it for |
|-------|------|------------|
| *(none)* | `numpy` | Engine, ABC, ArrowSpace adapter |
| `[torch]` | `torch` | `torch_classifier`, `gnn` adapters |
| `[transformers]` | `transformers` | `transformer_lm` adapter |
| `[notebooks]` | `sentence-transformers` + `[torch,transformers]` | Reproducing the notebooks |
| `[dev]` | `pytest`, `scipy`, `[torch,transformers]` | Running the test suite |

---

## Notebooks

Two reproducible Jupyter notebooks apply the library end-to-end. They live in
[`notebooks/`](notebooks) (see [`notebooks/README.md`](notebooks/README.md) for the
full abstract and run instructions).

| Notebook | Description |
|----------|-------------|
| [`00_arrowspace_epiplexity_structural_information.ipynb`](notebooks/00_arrowspace_epiplexity_structural_information.ipynb) | Synthetic structural-information study: builds an ArrowSpace LGMRF and walks through every epiplexity diagnostic (S_T, H_T, compression, eigenmaps). |
| [`01_arrowspace_cve1999_2025_epiplexity_check_v3.ipynb`](notebooks/01_arrowspace_cve1999_2025_epiplexity_check_v3.ipynb) | Full CVE-1999--2025 corpus case study. Achieves a **~38.4x compression ratio** over raw float32 storage, passing all three structural-information diagnostic tests. |

To run them locally:

```bash
uv venv .venv && uv sync --extra notebooks
# or: pip install -e .[notebooks]
jupyter lab notebooks/
```

The notebooks have their own `pyproject.toml` (in `notebooks/`) pinning the analysis
stack (pandas, pyarrow, scikit-learn, scipy, matplotlib, plotly, seaborn). Sample data
lives in [`samples/`](samples) and rendered outputs in [`output/`](output) (use the `v3`
plots). Legacy notebook revisions are kept under `notebooks/legacy/`.

---

## Paper

The accompanying paper, *"Epiplexity: A Measure on Graph Wiring"*, is in
[`paper/`](paper):

- [`paper/Epiplexity_A_measure_on_Graph_Wiring.pdf`](paper/Epiplexity_A_measure_on_Graph_Wiring.pdf) -- current PDF
- [`paper/Epiplexity_A_measure_on_Graph_Wiring.tex`](paper/Epiplexity_A_measure_on_Graph_Wiring.tex) -- LaTeX source

The paper formalises the theory the engine implements and reports the empirical results
from the notebooks. **It is the authoritative reference for the definitions behind
`S_T(X)`, `H_T(X)`, and the compression test**; the library is its executable companion.

> Citation (to follow the camera-ready revision):
>
> Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson -- *"From Entropy to Epiplexity: Rethinking
> Information for Computationally Bounded Intelligence"*, arXiv:2601.03220, 2026.

---

## Package layout

```
epiplexity/                     # the library (Python package)
|-- model.py                    # TTimeProbabilisticModel ABC
|-- engine.py                   # EpiplexityEngine (MDL calculator)
|-- algorithms/
|   |-- arrowspace.py           # ArrowSpace LGMRF adapter        [core, numpy-only]
|   |-- torch_classifier.py      # PyTorch classifier adapter      [extra: torch]
|   |-- transformer_lm.py       # Auto-regressive LM adapter      [extra: transformers]
|   `-- gnn.py                  # Graph neural network adapter     [extra: torch]
`-- tests/                      # pytest suite (109 tests)

notebooks/                      # reproducible case studies (Jupyter)
|-- 00_arrowspace_epiplexity_structural_information.ipynb
|-- 01_arrowspace_cve1999_2025_epiplexity_check_v3.ipynb
|-- legacy/                     # earlier notebook revisions
|-- pyproject.toml              # analysis-stack pins (pandas, pyarrow, ...)
`-- README.md                   # abstract + run instructions

paper/                          # accompanying paper (LaTeX + PDF)
|-- Epiplexity_A_measure_on_Graph_Wiring.tex
`-- Epiplexity_A_measure_on_Graph_Wiring.pdf

samples/                        # sample data (download as per samples/.gitkeep)
output/                         # rendered plots (use the v3 figures)


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

> Requires `pip install epiplexity[torch]`.

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

> Requires `pip install epiplexity[transformers]`.

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

> Requires `pip install epiplexity[torch]`.

```python
from epiplexity.algorithms.gnn import GNNModelAdapter, GNNEpiplexityConfig
cfg     = GNNEpiplexityConfig(bits_per_param=32, overhead_bits=16384.0)
adapter = GNNModelAdapter(model, cfg)
engine  = EpiplexityEngine(adapter, graph_dataset)  # list of (graph_obj, int)
# graph_obj must have .x (node features) and .edge_index
```

### ArrowSpace spectral LGMRF

> Core adapter -- available with a plain `pip install epiplexity` (NumPy only).

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

> Requires `pip install epiplexity[dev]` (pulls in `pytest`, `scipy`, `torch`, `transformers`).

```bash
# Full suite (109 tests across the library + packaging contract)
pytest -v

# Only the cross-algorithm property tests (P1-P8)
pytest epiplexity/tests/test_epiplexity_properties.py -v

# A single property class
pytest epiplexity/tests/test_epiplexity_properties.py::TestP4_ObserverDependence -v

# The 0.5.0 packaging / leanness contract tests
pytest epiplexity/tests/test_packaging.py -v

# With coverage
pytest --cov=epiplexity --cov-report=term-missing
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
