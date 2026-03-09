# Epiplexity And Graph Wiring: An Empirical Study for the design of a generic algorithm


Full paper in `paper/` and code in `notebooks/`. In `output/` the plotted diagrams, consider only `v3`.

Download the data as in `samples/.gitkeep` to run the full pipeline.

## Abstract

> Having introduced *Graph Wiring*, a technique that leverages the Graph Laplacian computed in feature space to provide semantically-aware search over high-dimensional vector corpora, and *MRR-Top0* as a topology-aware retrieval metric for evaluating it; this paper proceeds to demonstrate formally and empirically that the feature-space Laplacian produced by ArrowSpace carries *structural information* in the sense of epiplexity finzi2026epiplexity, so that it is not plain graph metadata but a reusable context-bound semantic artifact.
Using the CVE-1999--2025 vulnerability corpus as a case study, we instantiate ArrowSpace as a spectral vector-search engine, wrap its feature-space Laplacian in a Laplacian-constrained Gaussian Markov Random Field (LGMRF), and evaluate the resulting two-part Minimum
Description Length (MDL) code. The model achieves *a compression ratio of~38.4x over raw float32 storage, passes all three structural-information diagnostic tests*. Furthermore *it is demonstrated that the same Laplacian object as computed by Graph Wiring supports six distinct algorithm families (search, classification, anomaly detection, diffusion, dimensionality
reduction, and data valuation) without additional learning*.
The two accompanying Jupyter notebooks are intended as a reproducible reference pattern for applying epiplexity measurement in algorithm design for large-scale data engineering for LLMs and ML operations.

## Run the notebooks
To install locally, us `uv`:
```
uv venv .venv
uv sync
```
Or simply in your virtual environment:
```
pip install .
```

## v1

**🔴 Critical bug in §13 (Spectral Anomaly Guard):** The z-scores of 30.96σ and 66.68σ are artifacts — the code compares raw Python-computed Rayleigh quotients (values ~4–9) against ArrowSpace's taumode-compressed lambdas (range ). These are two *incompatible scales*. The ArrowSpace Rust engine applies a bounded sigmoid map `E/(E+τ)` plus optional dispersion mixing before storing lambdas, which the Python `model.rayleigh_quotient()` doesn't replicate. The fix is either to use native `rayleigh_pop[item_idx]` for the z-score, or to reimplement the full taumode pipeline in Python.[^1]

**🟡 Three moderate issues:**

- The 38.4× compression ratio is real but inflated by comparing against raw float32 — adding a `zlib` baseline and a diagonal Gaussian baseline would give a fairer picture
- The H_T spread is only ~5 bits across 314K items because `γ=1e-3` makes the fixed `log Z` term dominate — a β/γ sweep would find better discrimination
- Two of the three SOTA citations compare feature-space (F×F) results against item-space (N×N) benchmarks, which are incommensurable domains

## v3

Fixed the above problems.

