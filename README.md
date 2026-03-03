# Polynomial PSLQ Bounds — Bivariate and Trivariate

Systematic computational search for polynomial relations among fundamental transcendental constants, using the PSLQ integer relation algorithm with arbitrary-precision arithmetic.

**Companion code for:** M. Narcisi, *Computational Bounds on Polynomial Relations Among Fundamental Transcendental Constants: Bivariate, Trivariate, and Statistical Evidence*, arXiv:XXXX.XXXXX (2026).

## Main results

**1,811 PSLQ runs. Zero non-trivial relations found.**

No polynomial relation was found among any of the 19 bivariate pairs or 10 trivariate triples drawn from {π, e, γ, ζ(3), G, Ω, ζ(5), A, ln 2}.

### Bivariate exclusion bounds (top 3 pairs)

| Pair    | Max degree (‖c‖∞ ≤ 100) | Max degree (‖c‖∞ ≤ 10⁶) |
|---------|:------------------------:|:------------------------:|
| (π, γ)  | 40                       | 32                       |
| (e, γ)  | 40                       | 32                       |
| (π, e)  | 39                       | 32                       |

### Trivariate exclusion bounds (top 4 triples)

| Triple            | Max degree (all tiers) | ‖m‖₂ ≥          |
|-------------------|:----------------------:|:----------------:|
| (π, e, γ)         | 6                      | 1.05 × 10⁸      |
| (π, e, ζ(3))      | 6                      | 1.06 × 10⁸      |
| (π, γ, ζ(3))      | 6                      | 1.19 × 10⁸      |
| (e, γ, ζ(3))      | 6                      | 1.05 × 10⁸      |

### Extended analyses

- **Near-miss analysis:** 138 vectors extracted from PSLQ's B-matrix; mean λ = 0.806, no anomalous cancellation (λ < 0.5).
- **Statistical comparison:** KS, Anderson–Darling, Mann–Whitney tests vs. 30 pseudo-generic baseline pairs; all p > 0.05.
- **Exclusion frontiers:** Visualized in the (degree, height) plane for all 19 pairs.
- **Conjecture connections:** Each bound mapped to Schanuel's conjecture, conjectured irrationality of γ, or odd zeta independence.

Full results in the paper and in [`results/deep_v2/REPORT_DEEP_V2.md`](results/deep_v2/REPORT_DEEP_V2.md).

## Requirements

- Python ≥ 3.10
- [mpmath](https://mpmath.org/) ≥ 1.3.0
- [gmpy2](https://github.com/aleaxit/gmpy) ≥ 2.1 (provides fast GMP-backed arithmetic)
- [SymPy](https://www.sympy.org/) ≥ 1.12
- numpy, scipy, matplotlib

```bash
python3 -m venv env && source env/bin/activate
pip install mpmath sympy gmpy2 numpy scipy matplotlib
```

## Quick start

```bash
# Estimate feasible degrees and runtimes (no computation)
python3 run_deep_search_v2.py --estimate

# Calibrate the time model on your hardware
python3 run_deep_search_v2.py --benchmark

# Run the full bivariate search (default: 1 hour per pair)
python3 run_deep_search_v2.py --max-hours-per-pair 1.0

# Run all extended analyses in parallel (near-miss + baseline + trivariate)
python3 run_parallel_v2.py

# Run a single pair with extended budget
python3 run_deep_search_v2.py --pair pi+euler_gamma --max-hours-per-pair 4.0

# Resume after interruption (uses checkpoints)
python3 run_deep_search_v2.py --resume --max-hours-per-pair 1.0
```

## How it works

For each pair of constants (α, β) and each target degree *d*, the code:

1. Builds the monomial vector **v** = (1, α, β, α², αβ, β², …, β^d) of dimension N(d) = (d+1)(d+2)/2.
2. Computes **v** at precision p = 2·N·D decimal digits, where D = ⌈log₁₀(M+1)⌉ and M is the coefficient bound (safety factor s = 2.0 over the theoretical minimum).
3. Calls `mpmath.pslq(v, maxcoeff=M, maxsteps=0)`.
4. If PSLQ returns `None`: no relation with ‖c‖∞ ≤ M exists at this degree. This is a rigorous bound.
5. If PSLQ returns a vector: the candidate is verified at 1.5× precision and checked for triviality.

The search proceeds incrementally from degree 9 upward, with three coefficient tiers (10², 10⁴, 10⁶) at each degree, until the time budget is exhausted. Trivariate searches use the same approach with 3-variable monomials.

## Project structure

| File | Description |
|------|-------------|
| **Core bivariate search** | |
| `run_deep_search_v2.py` | Entry point with argument parsing |
| `deep_engine.py` | Core incremental PSLQ search engine |
| `precision_manager.py` | Dynamic precision scaling and time estimation |
| `checkpoint.py` | Checkpoint/resume for long-running searches |
| `bound_calculator.py` | Rigorous bound formatting |
| `pslq_bounds_fast.py` | Patched PSLQ with H-matrix and B-matrix extraction |
| `constants.py` | High-precision constant computation with caching |
| `config.py` | Search parameters and configuration |
| **Extended analyses** | |
| `near_miss_collector.py` | Near-miss vector extraction from B-matrix |
| `random_baseline.py` | Pseudo-generic baseline generation |
| `statistical_analysis.py` | KS, AD, MW hypothesis tests and diagnostic plots |
| `monomials_trivariate.py` | Trivariate monomial generation |
| `run_trivariate_search.py` | Trivariate PSLQ search engine |
| `run_parallel_v2.py` | Parallel orchestrator (10 workers) |
| **Output** | |
| `results/deep_v2/` | Bivariate bounds, paper, reports |
| `results/near_misses/` | Near-miss analysis results |
| `results/statistics/` | Statistical analysis, plots |
| `results/trivariate/` | Trivariate exclusion bounds |
| `results/frontiers/` | Exclusion frontier plots (19 individual + comparative) |

## Hardware used in the paper

- CPU: Intel Core i7-12700F (12 cores, 20 threads)
- RAM: 64 GB DDR5
- OS: Debian 13

Total computation: ~24 hours wall-clock (10 parallel workers), ~160 CPU-hours, 1,811 PSLQ runs.

## Citation

```bibtex
@article{Narcisi2026polynomial,
  author  = {Narcisi, Marco},
  title   = {Computational Bounds on Polynomial Relations Among
             Fundamental Transcendental Constants: Bivariate,
             Trivariate, and Statistical Evidence},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Code: [MIT License](LICENSE)

Paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
