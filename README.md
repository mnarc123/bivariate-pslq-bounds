# Bivariate PSLQ Bounds

Systematic computational search for bivariate polynomial relations $P(\alpha, \beta) = 0$ among pairs of fundamental transcendental constants, using the PSLQ integer relation algorithm.

**Companion code for:** M. Narcisi, *Computational Bounds on Bivariate Polynomial Relations Between Fundamental Transcendental Constants*, arXiv:XXXX.XXXXX (2026).

## Main result

No non-trivial bivariate polynomial relation was found among any of the 19 tested pairs drawn from {π, e, γ, ζ(3), G, Ω, ζ(5), A, ln 2}. The strongest exclusion bounds are:

| Pair    | Max degree (‖c‖∞ ≤ 100) | Max degree (‖c‖∞ ≤ 10⁶) |
|---------|:------------------------:|:------------------------:|
| (π, γ)  | 40                       | 32                       |
| (e, γ)  | 40                       | 32                       |
| (π, e)  | 39                       | 32                       |

Full results for all 19 pairs are in the paper and in [`results/deep_v2/REPORT_DEEP_V2.md`](results/deep_v2/REPORT_DEEP_V2.md).

## Requirements

- Python ≥ 3.10
- [mpmath](https://mpmath.org/) ≥ 1.3.0
- [gmpy2](https://github.com/aleaxit/gmpy) ≥ 2.1 (provides fast GMP-backed arithmetic)
- [SymPy](https://www.sympy.org/) ≥ 1.12
- numpy, scipy

```bash
python3 -m venv env && source env/bin/activate
pip install mpmath sympy gmpy2 numpy scipy
```

## Quick start

```bash
# Estimate feasible degrees and runtimes (no computation)
python3 run_deep_search_v2.py --estimate

# Calibrate the time model on your hardware
python3 run_deep_search_v2.py --benchmark

# Run the full search (default: 1 hour per pair)
python3 run_deep_search_v2.py --max-hours-per-pair 1.0

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

The search proceeds incrementally from degree 9 upward, with three coefficient tiers (10², 10⁴, 10⁶) at each degree, until the time budget is exhausted.

## Project structure

| File | Description |
|------|-------------|
| `run_deep_search_v2.py` | Entry point with argument parsing |
| `deep_engine.py` | Core incremental PSLQ search engine |
| `precision_manager.py` | Dynamic precision scaling and time estimation |
| `checkpoint.py` | Checkpoint/resume for long-running searches |
| `bound_calculator.py` | Rigorous bound formatting |
| `constants.py` | High-precision constant computation with caching |
| `config.py` | Search parameters and configuration |
| `monomials.py` | Multivariate monomial generator |
| `pslq_search.py` | PSLQ wrapper with validation |
| `validator.py` | Independent verification (double precision, SymPy) |
| `results/` | Output directory (reports, logs, checkpoints) |

## Hardware used in the paper

- CPU: Intel Core i7-12700F (12 cores, 20 threads)
- RAM: 64 GB DDR5
- OS: Debian 13

Total computation time: 8 hours 27 minutes (1331 PSLQ runs).

## Citation

```bibtex
@article{Narcisi2026bivariate,
  author  = {Narcisi, Marco},
  title   = {Computational Bounds on Bivariate Polynomial Relations
             Between Fundamental Transcendental Constants},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Code: [MIT License](LICENSE)

Paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

[![DOI](https://zenodo.org/badge/1161208505.svg)](https://doi.org/10.5281/zenodo.18686238)
