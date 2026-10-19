# Changelog

## 0.6.1

- Explicit inclusion rules for wheel and source archives.
- Tests, fixtures, examples, notebooks, research assets, and local build files
  are excluded from both distributions. Tests remain in the Git repository.
- Archive-content regression checks verify the actual files prepared for PyPI.
- Package and lockfile versions updated to 0.6.1.

## 0.6.0

### Added

- NumPy-only prequential and requential estimators with explicit input units,
  scored-token counts, finite-value checks, and signed-estimate diagnostics.
- Immutable `CodingResult`, versioned JSON output, caller-supplied provenance,
  and a packaged result JSON Schema.
- One-pass `EpiplexityEngine.from_batches()` with aggregate-only storage by
  default, optional per-item retention, and compensated accumulation.
- Optional adapter batch scoring; vectorised PyTorch classifier inference.
- Candidate selection under measured compute budgets and deterministic Pareto
  frontiers with comparison checks.
- JSON CLI through `epiplexity` and `python -m epiplexity`.
- Portable golden vectors and a [migration and port guide](docs/v0.6.md).

### Changed

- Engine results snapshot description cost, scores, baseline, and provenance.
  Mutating the model after evaluation no longer changes the reported model cost.
- Checked reports distinguish discrete, conditional, and density scores.
  Built-in adapters with unmatched baselines report `NOT_COMPARABLE`.
- Legacy arithmetic `compression_ratio()` remains available; a zero denominator
  returns `None`. Per-item arrays are copies and are unavailable in aggregate mode.
- Invalid non-finite scores and negative model or raw costs fail explicitly.
- The `transformers` extra now includes its PyTorch runtime dependency.
- Public core APIs are importable from `epiplexity` without loading ML frameworks.
- Package and lockfile versions are 0.6.0; tests remain outside the wheel.

### Method limits

Static description cost is not guaranteed to bound the model component of the
MDL optimum. Precision is not a compute budget. Log-based methods are estimates
under caller-supplied protocols. Density scores need a discrete encoding before
comparison with storage bytes. The release provides a Python reference for a
future native implementation, not a Rust or C backend.
