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

## Theoretical backgorund summary

> what is the relation between Entropy and Complexity in Information Theory and Algorithmic information theory (Shannon/Kolmogorov)? and how Epiplexity try to reframe these definitions?

Entropy and complexity coincide only in special averaged senses: Shannon entropy measures average unpredictability under a probabilistic model, while Kolmogorov complexity measures description length of individual objects, and epiplexity reframes both as *compute‑relative* notions that split “information” into time‑bounded noise vs. learnable structure.[^1][^2][^3][^4][^5][^6]

### Shannon vs. Kolmogorov: core relation

- Shannon entropy $H(X)$ is the expected number of bits needed to encode draws from a random variable $X$ under an optimal code for a given distribution.[^4]
- Kolmogorov complexity $K(x)$ is the length of the shortest program that outputs a specific finite string $x$ on a universal Turing machine.[^4]
- For any *fixed* computable distribution, the expected Kolmogorov complexity of samples equals the Shannon entropy up to an additive constant; equivalently, Shannon entropy is the average of Kolmogorov complexity over typical realizations.[^2]
- There is a deep structural parallel: any linear information inequality that holds for Shannon entropy also holds (up to logarithmic slack) for Kolmogorov complexities, and vice versa.[^7][^5]

So entropy is model‑based and distribution‑level, whereas Kolmogorov complexity is model‑free and sequence‑level; they align only after taking expectations under a computable source.[^2][^4]

### Entropy vs. (algorithmic) complexity

- High Shannon entropy means high *unpredictability*: a fair coin sequence has maximal entropy per bit, but a *particular* long coin‑flip string is algorithmically simple if you just list it, and has no compressible regularities beyond the source law.[^4]
- High Kolmogorov complexity means no short description: random sequences are maximally complex in this sense, but so are many highly structured objects that simply lack short programs given the chosen universal machine.[^4]
- Complexity science often wants “organized structure at the edge of chaos,” which lies *between* low‑entropy order and high‑entropy noise; neither Shannon entropy nor bare $K(x)$ cleanly isolates that regime.[^8][^9][^10]

Thus, both classical measures conflate two different things: noise‑like unpredictability and learnable structure, especially for computationally bounded observers.[^9][^3][^6]

### Time‑bounded notions and their role

- Time‑bounded Kolmogorov complexity $K^t(x)$ restricts programs to run within a time budget $t$; its expectation again matches Shannon entropy under suitable computability assumptions, linking average time‑bounded description length to entropy.[^2]
- For a finite compute budget, what you cannot feasibly exploit appears as “effective randomness,” even if there is latent simple structure requiring more time to uncover.[^3][^2]

This moves us toward viewing entropy and complexity as *observer‑relative* rather than absolute: they depend on which patterns are computationally tractable.

### How epiplexity reframes things

Epiplexity (from “epi‑” + “complexity”) is proposed as an information measure for *computationally bounded* agents, explicitly splitting information into two components.[^6][^1][^3]

- Relative to a given compute budget, they define:
    - **Time‑bounded entropy**: the part of the data that remains effectively unpredictable or uncompressible for that observer—what still looks like noise under the best algorithms they can afford.[^1][^3][^6]
    - **Epiplexity**: the amount of *learnable structural information* that the observer can in fact extract and reuse—roughly, the budget‑relative, algorithmically exploitable complexity of the data.[^3][^6][^1]

Key reframings:

- Classical Shannon entropy and Kolmogorov complexity implicitly assume unbounded computation, so they do not distinguish between “structure I could in principle learn” and “structure I can actually learn with my resources.”[^6][^3]
- Epiplexity makes this distinction explicit by conditioning on a computational class: as you increase compute, some apparent epiplexity collapses into lower time‑bounded entropy when you discover simpler generative mechanisms.[^1][^3]
- They define **epiplexity emergence**: in multi‑step processes (e.g. cellular automata), low‑compute observers see increasing structural complexity over time, whereas high‑compute observers can simulate or invert the dynamics cheaply and thus perceive less emergent complexity.[^3]

In the ML context, epiplexity tracks the reusable circuit depth a model must internalize to perform a task, rather than just the unpredictability of labels: factorisations or tasks that force higher epiplexity empirically correlate with better out‑of‑distribution generalization.[^1][^3]

### Conceptual summary

- Shannon entropy: average unpredictability under a known distribution; good for coding and inference, blind to computational limits.[^9][^4]
- Kolmogorov complexity: shortest description of individual objects; connects to entropy in expectation but remains uncomputable and agnostic to resource bounds.[^5][^2][^4]
- Epiplexity: compute‑relative structural complexity, defined together with time‑bounded entropy to separate what looks like noise from what can be learned by a bounded observer, aiming to capture the “useful” part of complexity that drives emergent structure and generalization.[^6][^3][^1]

<div align="center">⁂</div>

[^1]: https://arxiviq.substack.com/p/from-entropy-to-epiplexity-rethinking

[^2]: https://sqigmath.tecnico.ulisboa.pt/pub/SoutoA/11-TMSA-entropy.pdf

[^3]: https://www.alphaxiv.org/overview/2601.03220

[^4]: https://en.wikipedia.org/wiki/Entropy_(information_theory)

[^5]: https://www.lirmm.fr/~ashen/mathtext/inequal/final.pdf

[^6]: https://www.youtube.com/watch?v=opGMkWbzM88

[^7]: https://www.sciencedirect.com/science/article/pii/S002200009991677X

[^8]: https://pubmed.ncbi.nlm.nih.gov/35895715/

[^9]: https://www.pnas.org/doi/10.1073/pnas.2119089119

[^10]: https://arxiv.org/pdf/math/0107067.pdf

## ChangeLog
### v1

**🔴 Critical bug in §13 (Spectral Anomaly Guard):** The z-scores of 30.96σ and 66.68σ are artifacts — the code compares raw Python-computed Rayleigh quotients (values ~4–9) against ArrowSpace's taumode-compressed lambdas (range ). These are two *incompatible scales*. The ArrowSpace Rust engine applies a bounded sigmoid map `E/(E+τ)` plus optional dispersion mixing before storing lambdas, which the Python `model.rayleigh_quotient()` doesn't replicate. The fix is either to use native `rayleigh_pop[item_idx]` for the z-score, or to reimplement the full taumode pipeline in Python.[^1]

**🟡 Three moderate issues:**

- The 38.4× compression ratio is real but inflated by comparing against raw float32 — adding a `zlib` baseline and a diagonal Gaussian baseline would give a fairer picture
- The H_T spread is only ~5 bits across 314K items because `γ=1e-3` makes the fixed `log Z` term dominate — a β/γ sweep would find better discrimination
- Two of the three SOTA citations compare feature-space (F×F) results against item-space (N×N) benchmarks, which are incommensurable domains

### v3

Fixed the above problems.

