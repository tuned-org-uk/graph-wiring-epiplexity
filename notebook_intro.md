Here's the complete Jupyter notebook. It contains **21 cells** (10 executable code + 11 theory markdown) and is fully self-contained — every class and function is defined from scratch using only standard Python scientific libraries.

**What's inside, section by section:**

- **§1 — Epiplexity \& MDL Theory** — the Finzi et al. equations in markdown with a toy numerical illustration of the $S_T / H_T$ split across three hypothetical dataset types
- **§2 — ArrowSpace as a Prefix-Free Program** — `compute_description_length()` computing $|P_\text{AS}|$ in bits at four scales up to LLM-scale (1M items, 4096d), with Elias gamma self-delimiting codes for integers
- **§3 — LGMRF Probabilistic Wrapper** — the full `ArrowSpaceProbabilisticModel` class satisfying all three Definition 7 requirements (evaluate, sample, normalise) in polynomial time; sanity checks proving smooth signals get higher $P$ than rough ones
- **§4 — MDL Toolkit** — `ArrowSpaceMDLToolkit` computing two-part MDL for any dataset, with a live compression test across three synthetic datasets (structured clusters, pure noise, smooth low-rank manifold)
- **§5 — Diagnostic Tests** — `StructuralInformationDiagnostics` running all three tests (compression ratio, spectral gap $\lambda_2$, Rayleigh coefficient of variation) with verdicts
- **§6 — Six Algorithm Classes** — `ArrowSpaceMultiClassEngine` demonstrating search, label propagation, anomaly detection, heat-flow diffusion, Laplacian eigenmaps, and data valuation all on the same $L_F$
- **§7 — LLM Data Engineering Toolset** — four production tools: epiplexity-style data selection, spectral anomaly guard ($z$-score on Rayleigh quotient), spectral drift monitor (fingerprint diffing), and quality gate
- **§8 — Observer-Dependence** — sweeping $k$ from 2 to 10 to show $|P_\text{AS}| \uparrow$ and $\bar{H}_T \downarrow$ as compute budget grows, mirroring epiplexity's $T$-scaling
- **§9 — Five Visualisations** — all Plotly charts generated in one cell
