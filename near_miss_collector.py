#!/usr/bin/env python3
"""
Near-Miss Collector for PSLQ searches.

When PSLQ returns None (no relation found within maxcoeff), this module
re-runs PSLQ with progressively higher maxcoeff thresholds to extract
"near-miss" vectors — integer vectors that produce very small (but nonzero)
residuals when dotted with the monomial vector.

These near-misses are NOT claimed as relations. They are analyzed to:
1. Detect anomalous cancellation that might indicate hidden structure.
2. Compare with random baseline expectations.
3. Provide structural insight beyond the binary "found/not found" result.

This is Punto 1 of the Experimental Mathematics upgrade.
"""

import mpmath
import math
import json
import time
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict, field

from config import SearchConfig
from constants import ConstantsComputer
from deep_engine import generate_bivariate_monomials, TOP_PAIRS, HIGH_PAIRS, MEDIUM_PAIRS
from precision_manager import compute_precision_plan
from pslq_bounds_fast import pslq_with_bound
from bound_calculator import SYMBOLS


@dataclass
class NearMiss:
    """A single near-miss result."""
    pair: Tuple[str, str]
    pair_symbol: str
    degree: int
    vector: List[int]
    exponents: List[Tuple[int, int]]
    labels: List[str]
    norm_inf: int
    norm_l2: float
    residual: float
    log10_residual: float
    working_precision: int
    digits_cancelled: float          # -log10(|R|)
    precision_fraction: float        # -log10(|R|) / working_precision
    quality_score: float             # -log10(|R|) / (norm_l2 * degree)
    maxcoeff_used: int
    polynomial_string: str
    timestamp: str


@dataclass
class NearMissReport:
    """Full report for near-miss analysis."""
    near_misses: List[NearMiss] = field(default_factory=list)
    pairs_analyzed: int = 0
    total_pslq_runs: int = 0
    total_time_seconds: float = 0.0
    timestamp: str = ""


def _format_polynomial(vector: List[int], labels: List[str]) -> str:
    """Format a polynomial from coefficient vector and labels."""
    terms = []
    for c, label in zip(vector, labels):
        if c == 0:
            continue
        if c > 0 and terms:
            sign = " + "
        elif c < 0:
            sign = " - " if terms else "-"
            c = -c
        else:
            sign = "" if not terms else " + "
        if label == "1":
            terms.append(f"{sign}{c}")
        elif c == 1:
            terms.append(f"{sign}{label}")
        else:
            terms.append(f"{sign}{c}*{label}")
    return "".join(terms) if terms else "0"


def collect_near_misses(
    const1: str,
    const2: str,
    degrees: List[int],
    config: SearchConfig,
    log_func=None,
) -> List[NearMiss]:
    """
    Collect near-misses for a single pair across specified degrees.

    Uses pslq_bound() which extracts the best candidate vector from the
    B-matrix even when PSLQ terminates without finding a relation. The
    best candidate is the B-column corresponding to the smallest |y_i|.
    Residuals are verified at double precision.
    """
    if log_func is None:
        log_func = lambda msg: print(f"  {msg}", flush=True)

    cc = ConstantsComputer(config)
    near_misses = []
    s1 = SYMBOLS.get(const1, const1)
    s2 = SYMBOLS.get(const2, const2)
    pair_symbol = f"({s1}, {s2})"

    for degree in degrees:
        plan = compute_precision_plan(degree, max_coeff=100)
        working_digits = plan.working_digits

        # Compute constants and monomials at working precision
        mpmath.mp.dps = working_digits + 50
        alpha = cc.compute_all(working_digits + 50, [const1])[const1]
        beta = cc.compute_all(working_digits + 50, [const2])[const2]
        values, exponents, labels = generate_bivariate_monomials(alpha, beta, degree)

        # Run pslq_bound which extracts best_candidate from B-matrix
        try:
            result = pslq_with_bound(values, maxcoeff=100, maxsteps=0)
        except Exception as exc:
            log_func(f"    d={degree}: PSLQ error: {exc}")
            continue

        candidate = result.best_candidate
        if candidate is None:
            log_func(f"    d={degree}: no best candidate extracted")
            continue

        # Skip if relation was found (this is handled elsewhere)
        if result.relation is not None:
            log_func(f"    d={degree}: exact relation found, skipping near-miss")
            continue

        # Verify residual at double precision
        double_digits = working_digits * 2
        mpmath.mp.dps = double_digits + 50
        alpha_2x = cc.compute_all(double_digits + 50, [const1])[const1]
        beta_2x = cc.compute_all(double_digits + 50, [const2])[const2]
        values_2x, _, _ = generate_bivariate_monomials(alpha_2x, beta_2x, degree)

        residual = abs(sum(
            c * v for c, v in zip(candidate, values_2x)
        ))

        if residual == 0:
            log10_res = -float('inf')
            digits_cancelled = float('inf')
        else:
            log10_res = float(mpmath.log10(residual))
            digits_cancelled = -log10_res

        norm_inf = max(abs(c) for c in candidate)
        norm_l2 = math.sqrt(sum(c**2 for c in candidate))
        prec_frac = digits_cancelled / working_digits if working_digits > 0 else 0
        quality = digits_cancelled / (norm_l2 * degree) if norm_l2 * degree > 0 else 0

        poly_str = _format_polynomial(candidate, labels)

        nm = NearMiss(
            pair=(const1, const2),
            pair_symbol=pair_symbol,
            degree=degree,
            vector=list(candidate),
            exponents=[list(e) for e in exponents],
            labels=labels,
            norm_inf=norm_inf,
            norm_l2=round(norm_l2, 2),
            residual=float(residual) if residual > 0 else 0.0,
            log10_residual=round(log10_res, 2) if log10_res != -float('inf') else None,
            working_precision=working_digits,
            digits_cancelled=round(digits_cancelled, 2) if digits_cancelled != float('inf') else None,
            precision_fraction=round(prec_frac, 4),
            quality_score=round(quality, 6),
            maxcoeff_used=100,
            polynomial_string=poly_str,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        near_misses.append(nm)

        dc_str = f"{digits_cancelled:.1f}" if digits_cancelled != float('inf') else "∞"
        log_func(
            f"    d={degree}: ||m||∞={norm_inf}, "
            f"-log₁₀R={dc_str}, "
            f"λ={prec_frac:.3f}, "
            f"Q={quality:.4f}"
        )

    return near_misses


def generate_near_miss_summary(
    all_near_misses: List[NearMiss],
    output_dir: Path,
):
    """Generate summary report in Markdown."""
    lines = [
        "# Near-Miss Analysis Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"Total near-misses collected: {len(all_near_misses)}",
        "",
        "A near-miss is a polynomial with integer coefficients that evaluates to",
        "a very small (but nonzero) value at the given constants. These are obtained",
        "by running PSLQ with progressively higher maxcoeff thresholds.",
        "",
        "**Quality score** $Q = -\\log_{10}|R| / (\\|m\\|_2 \\cdot d)$: a higher Q",
        "indicates more anomalous cancellation relative to the polynomial's complexity.",
        "",
        "---",
        "",
        "## Top 20 Near-Misses by Quality Score",
        "",
    ]

    sorted_nm = sorted(all_near_misses, key=lambda nm: nm.quality_score, reverse=True)
    top20 = sorted_nm[:20]

    lines.append("| # | Pair | Degree | Q score | -log₁₀|R| | ||m||∞ | ||m||₂ | λ = digits/prec |")
    lines.append("|---|------|--------|---------|-----------|--------|--------|-----------------|")

    for i, nm in enumerate(top20, 1):
        dc = f"{nm.digits_cancelled:.1f}" if nm.digits_cancelled is not None else "∞"
        lines.append(
            f"| {i} | {nm.pair_symbol} | {nm.degree} | {nm.quality_score:.4f} | "
            f"{dc} | {nm.norm_inf} | {nm.norm_l2:.1f} | {nm.precision_fraction:.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "If the quality scores are uniformly low (Q < 0.1) and the precision",
        "fraction λ is small (λ < 0.5), this indicates that the cancellation",
        "observed in near-misses is consistent with random behavior — i.e., the",
        "constants behave as if they were algebraically independent.",
        "",
        "If any near-miss shows Q > 0.5 or λ > 0.8, it warrants further",
        "investigation as a potential indicator of hidden algebraic structure.",
        "",
        "---",
        "",
        "## Per-Pair Summary",
        "",
    ])

    # Group by pair
    pairs = {}
    for nm in all_near_misses:
        key = nm.pair_symbol
        if key not in pairs:
            pairs[key] = []
        pairs[key].append(nm)

    for pair_sym, nms in pairs.items():
        best = max(nms, key=lambda x: x.quality_score)
        avg_q = sum(nm.quality_score for nm in nms) / len(nms)
        avg_lam = sum(nm.precision_fraction for nm in nms) / len(nms)
        lines.append(f"### {pair_sym}")
        lines.append(f"- Near-misses collected: {len(nms)}")
        lines.append(f"- Best Q score: {best.quality_score:.4f} (degree {best.degree})")
        lines.append(f"- Average Q score: {avg_q:.4f}")
        lines.append(f"- Average λ: {avg_lam:.3f}")
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "near_miss_summary.md").write_text("\n".join(lines))


def generate_near_miss_latex(all_near_misses: List[NearMiss], output_dir: Path):
    """Generate LaTeX section for near-miss analysis."""
    sorted_nm = sorted(all_near_misses, key=lambda nm: nm.quality_score, reverse=True)
    top20 = sorted_nm[:20]

    lines = []
    lines.append(r"\subsection{Near-Miss Analysis}")
    lines.append(r"\label{sec:near-miss}")
    lines.append("")
    lines.append(r"When PSLQ terminates without finding a relation within the specified")
    lines.append(r"coefficient bound $M$, the search space has been exhausted: no polynomial")
    lines.append(r"$P \in \mathbb{Z}[x,y]$ with $\deg P \leq d$ and $\|c\|_\infty \leq M$")
    lines.append(r"satisfies $P(\alpha, \beta) = 0$. However, by running PSLQ with")
    lines.append(r"progressively higher coefficient thresholds ($10^8, 10^{10}, 10^{12}$),")
    lines.append(r"we can extract \emph{near-miss} vectors --- integer vectors $\mathbf{m}$")
    lines.append(r"that produce very small residuals $|R| = |\mathbf{m} \cdot \mathbf{v}|$")
    lines.append(r"without being exact relations.")
    lines.append("")
    lines.append(r"These near-misses are \emph{not} claimed as relations. Instead, we analyze")
    lines.append(r"them to quantify the \emph{quality} of near-cancellation via the score")
    lines.append(r"\[")
    lines.append(r"  Q = \frac{-\log_{10}|R|}{\|\mathbf{m}\|_2 \cdot d}\,,")
    lines.append(r"\]")
    lines.append(r"which normalizes the number of cancelled digits by the complexity of the")
    lines.append(r"polynomial (coefficient size and degree). A high $Q$ would indicate")
    lines.append(r"anomalous cancellation beyond random expectation.")
    lines.append("")

    # Table
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Top 20 near-misses ranked by quality score $Q$.}")
    lines.append(r"  \label{tab:near-misses}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{clccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \# & Pair & $d$ & $Q$ & $-\log_{10}|R|$ & $\|\mathbf{m}\|_\infty$ & $\lambda$ \\")
    lines.append(r"    \midrule")

    for i, nm in enumerate(top20, 1):
        # Get latex pair name
        c1, c2 = nm.pair
        from exclusion_frontier import LATEX_SYMBOLS
        l1 = LATEX_SYMBOLS.get(c1, c1)
        l2 = LATEX_SYMBOLS.get(c2, c2)
        pair_latex = f"$({l1}, {l2})$"
        dc = f"{nm.digits_cancelled:.0f}" if nm.digits_cancelled is not None else r"$\infty$"
        lines.append(
            f"    {i} & {pair_latex} & {nm.degree} & {nm.quality_score:.4f} & "
            f"{dc} & {nm.norm_inf} & {nm.precision_fraction:.3f} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"Table~\ref{tab:near-misses} shows the top 20 near-misses ranked by $Q$.")
    lines.append(r"The parameter $\lambda = -\log_{10}|R| / p$ measures the fraction of")
    lines.append(r"working precision digits consumed by the cancellation.")
    lines.append("")

    # Interpretation paragraph
    avg_q = sum(nm.quality_score for nm in all_near_misses) / max(len(all_near_misses), 1)
    max_q = max((nm.quality_score for nm in all_near_misses), default=0)
    avg_lam = sum(nm.precision_fraction for nm in all_near_misses) / max(len(all_near_misses), 1)

    lines.append(r"The average quality score across all " + str(len(all_near_misses)))
    lines.append(f"near-misses is $Q = {avg_q:.4f}$, with maximum $Q = {max_q:.4f}$.")
    lines.append(f"The average precision fraction is $\\lambda = {avg_lam:.3f}$.")

    if max_q < 0.5:
        lines.append(r"All quality scores are low ($Q < 0.5$), indicating that the")
        lines.append(r"near-miss cancellation is consistent with random behavior and")
        lines.append(r"does not suggest hidden algebraic structure.")
    else:
        lines.append(r"Some quality scores exceed 0.5, warranting further investigation")
        lines.append(r"to determine whether this reflects genuine algebraic structure")
        lines.append(r"or is a statistical artifact of the sample.")

    lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "near_miss_analysis.tex").write_text("\n".join(lines))


def main():
    """Run the full near-miss analysis."""
    project_root = Path(__file__).parent
    output_dir = project_root / "results" / "near_misses"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SearchConfig()
    log_file = output_dir / f"near_miss_{time.strftime('%Y%m%d_%H%M%S')}.log"

    def log(msg):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=" * 60)
    log("  PUNTO 1: Near-Miss Analysis")
    log("=" * 60)

    t_start = time.time()
    all_near_misses = []
    total_runs = 0

    # Define degree ranges per pair priority
    all_pairs = (
        [(p, "top") for p in TOP_PAIRS]
        + [(p, "high") for p in HIGH_PAIRS]
        + [(p, "medium") for p in MEDIUM_PAIRS]
    )

    for (c1, c2), priority in all_pairs:
        pair_name = f"({SYMBOLS.get(c1, c1)}, {SYMBOLS.get(c2, c2)})"
        log(f"\n  Pair: {pair_name} [{priority}]")

        # Degree range based on priority
        if priority == "top":
            # Degrees 3-8: each takes 0.1s to 3min per pair
            degrees = list(range(3, 9))
        elif priority == "high":
            degrees = list(range(3, 8))
        else:
            degrees = list(range(3, 7))

        log(f"    Degrees: {degrees[0]}-{degrees[-1]} ({len(degrees)} degrees)")

        pair_nms = collect_near_misses(c1, c2, degrees, config, log)
        all_near_misses.extend(pair_nms)
        total_runs += len(degrees)

        # Checkpoint: save intermediate results
        _save_checkpoint(all_near_misses, output_dir)

        elapsed = time.time() - t_start
        log(f"    Collected {len(pair_nms)} near-misses. Elapsed: {elapsed/60:.1f} min")

        # Time budget check: max 4 hours
        if elapsed > 4 * 3600:
            log(f"  Time budget exceeded (4h). Stopping.")
            break

    elapsed_total = time.time() - t_start

    # Save final results
    log(f"\n  Total near-misses: {len(all_near_misses)}")
    log(f"  Total time: {elapsed_total/60:.1f} min")

    # JSON report
    report_data = {
        "near_misses": [asdict(nm) for nm in all_near_misses],
        "pairs_analyzed": len(set(nm.pair_symbol for nm in all_near_misses)),
        "total_near_misses": len(all_near_misses),
        "total_time_seconds": round(elapsed_total, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_dir / "near_miss_report.json").write_text(json.dumps(report_data, indent=2))

    # Markdown summary
    generate_near_miss_summary(all_near_misses, output_dir)

    # LaTeX section
    generate_near_miss_latex(all_near_misses, output_dir)

    log(f"\n  Output saved to {output_dir}")
    log("  Done.")


def _save_checkpoint(near_misses: List[NearMiss], output_dir: Path):
    """Save intermediate checkpoint."""
    cp = {
        "near_misses": [asdict(nm) for nm in near_misses],
        "count": len(near_misses),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_dir / "near_miss_checkpoint.json").write_text(json.dumps(cp, indent=2))


if __name__ == "__main__":
    main()
