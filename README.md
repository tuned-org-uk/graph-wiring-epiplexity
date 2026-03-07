```
uv venv .venv
us sync
```

## v1

**🔴 Critical bug in §13 (Spectral Anomaly Guard):** The z-scores of 30.96σ and 66.68σ are artifacts — the code compares raw Python-computed Rayleigh quotients (values ~4–9) against ArrowSpace's taumode-compressed lambdas (range ). These are two *incompatible scales*. The ArrowSpace Rust engine applies a bounded sigmoid map `E/(E+τ)` plus optional dispersion mixing before storing lambdas, which the Python `model.rayleigh_quotient()` doesn't replicate. The fix is either to use native `rayleigh_pop[item_idx]` for the z-score, or to reimplement the full taumode pipeline in Python.[^1]

**🟡 Three moderate issues:**

- The 38.4× compression ratio is real but inflated by comparing against raw float32 — adding a `zlib` baseline and a diagonal Gaussian baseline would give a fairer picture
- The H_T spread is only ~5 bits across 314K items because `γ=1e-3` makes the fixed `log Z` term dominate — a β/γ sweep would find better discrimination
- Two of the three SOTA citations compare feature-space (F×F) results against item-space (N×N) benchmarks, which are incommensurable domains

## v2

