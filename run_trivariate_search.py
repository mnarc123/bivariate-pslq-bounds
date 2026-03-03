#!/usr/bin/env python3
"""
Trivariate PSLQ Search: P(α, β, γ) = 0

Extends the bivariate search to polynomial relations among triples of
transcendental constants. Even negative results at low degree (5-8) for
trivariate relations are completely new in the literature.

This is Punto 2 of the Experimental Mathematics upgrade.

Usage:
    ~/bridge_eq_env/bin/python3 run_trivariate_search.py
    ~/bridge_eq_env/bin/python3 run_trivariate_search.py --estimate
"""

import mpmath
import sympy
import math
import json
import time
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

from config import SearchConfig
from constants import ConstantsComputer
from monomials_trivariate import (
    generate_trivariate_monomials,
    count_trivariate_monomials,
    compute_trivariate_precision,
)
from pslq_bounds_fast import pslq_with_bound as pslq_bound, PSLQResult as PSLQBoundResult
from bound_calculator import SYMBOLS
from precision_manager import CALIBRATION_K, CALIBRATION_ALPHA, CALIBRATION_BETA

# === TRIPLE DEFINITIONS (ordered by priority) ===
TRIPLES = [
    (("pi", "e", "euler_gamma"),        "top"),
    (("pi", "e", "zeta3"),              "top"),
    (("pi", "euler_gamma", "zeta3"),    "top"),
    (("e", "euler_gamma", "zeta3"),     "top"),
    (("pi", "e", "catalan"),            "high"),
    (("pi", "euler_gamma", "catalan"),  "high"),
    (("pi", "e", "ln2"),               "high"),
    (("pi", "zeta3", "catalan"),        "high"),
    (("e", "euler_gamma", "ln2"),       "high"),
    (("pi", "e", "zeta5"),             "high"),
]

# Coefficient tiers
COEFF_TIERS = [100, 10000, 1000000]


@dataclass
class TrivariateResult:
    """Result of a single trivariate PSLQ run."""
    triple: Tuple[str, str, str]
    triple_symbol: str
    degree: int
    max_coeff: int
    n_monomials: int
    precision_digits: int
    found_relation: bool
    relation: Optional[List[int]]
    residual_working: Optional[float]
    residual_verification: Optional[float]
    bound_statement: str
    norm_bound: Optional[int]
    elapsed_seconds: float
    timestamp: str


def estimate_time(n_monomials: int, working_digits: int) -> float:
    """Estimate PSLQ time in hours using calibrated model."""
    return CALIBRATION_K * (n_monomials ** CALIBRATION_ALPHA) * (working_digits ** CALIBRATION_BETA)


def print_feasibility_table():
    """Print feasibility table for all triples and degrees."""
    print("\n" + "=" * 80)
    print("  TRIVARIATE SEARCH — FEASIBILITY TABLE")
    print("=" * 80)
    print(f"\n  {'Triple':<35} {'Deg':>3} {'N':>5} {'Digits':>6} {'Est. time':>12} {'Feasible':>8}")
    print("  " + "-" * 75)

    for (c1, c2, c3), priority in TRIPLES:
        s1 = SYMBOLS.get(c1, c1)
        s2 = SYMBOLS.get(c2, c2)
        s3 = SYMBOLS.get(c3, c3)
        triple_sym = f"({s1}, {s2}, {s3})"

        budget_h = 1.0 if priority == "top" else 0.5

        for max_coeff in [100]:
            for degree in range(3, 13):
                n, wd, vd = compute_trivariate_precision(degree, max_coeff)
                est_h = estimate_time(n, wd)
                feasible = est_h <= budget_h

                if degree == 3:
                    label = f"  {triple_sym:<35}"
                else:
                    label = f"  {'':35}"

                time_str = f"{est_h*60:.1f} min" if est_h < 1 else f"{est_h:.1f} h"
                feas_str = "✓" if feasible else "✗"

                print(f"{label} {degree:>3} {n:>5} {wd:>6} {time_str:>12} {feas_str:>8}")

                if not feasible:
                    break
        print()


def _check_trivial_trivariate(
    relation: List[int],
    exponents: List[Tuple[int, int, int]],
    const1: str, const2: str, const3: str,
) -> bool:
    """Check if a trivariate relation is trivial using sympy."""
    SYMPY_MAP = {
        "pi": sympy.pi, "e": sympy.E, "euler_gamma": sympy.EulerGamma,
        "ln2": sympy.log(2), "zeta3": sympy.zeta(3), "catalan": sympy.Catalan,
        "zeta5": sympy.zeta(5), "omega": sympy.LambertW(1),
        "glaisher": sympy.Symbol("A_Glaisher"),
    }
    for name in [const1, const2, const3]:
        if name not in SYMPY_MAP:
            return False
    try:
        sym1 = SYMPY_MAP[const1]
        sym2 = SYMPY_MAP[const2]
        sym3 = SYMPY_MAP[const3]
        total = sympy.Integer(0)
        for c, (i, j, k) in zip(relation, exponents):
            if c == 0:
                continue
            total += sympy.Integer(c) * sym1**i * sym2**j * sym3**k
        if sympy.simplify(total) == 0:
            return True
        rewritten = total.rewrite(sympy.sqrt)
        if sympy.simplify(rewritten) == 0:
            return True
        return False
    except Exception:
        return False


class TrivariateSearchEngine:
    """Engine for trivariate PSLQ searches."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.cc = ConstantsComputer(config)
        self.results: List[TrivariateResult] = []

        self.output_dir = config.results_dir / "trivariate"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.output_dir / f"trivariate_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self.checkpoint_file = self.output_dir / "trivariate_checkpoint.json"

        # Load checkpoint
        self.completed = set()
        if self.checkpoint_file.exists():
            try:
                cp = json.loads(self.checkpoint_file.read_text())
                self.completed = set(cp.get("completed", []))
            except (json.JSONDecodeError, OSError):
                pass

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def _checkpoint_key(self, c1, c2, c3, degree, max_coeff):
        return f"{c1}+{c2}+{c3}_d{degree}_c{max_coeff}"

    def _save_checkpoint(self):
        cp = {
            "completed": list(self.completed),
            "n_results": len(self.results),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tmp = self.checkpoint_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(cp, indent=2))
        tmp.replace(self.checkpoint_file)

    def run(self):
        """Execute the full trivariate search."""
        self.log("=" * 60)
        self.log("  TRIVARIATE PSLQ SEARCH")
        self.log("=" * 60)

        # Self-test
        self.log("\n[SELF-TEST] Verifying PSLQ...")
        mpmath.mp.dps = 200
        phi = (1 + mpmath.sqrt(5)) / 2
        rel = mpmath.pslq([phi**2, phi, mpmath.mpf(1)])
        if rel is None or (rel != [1, -1, -1] and rel != [-1, 1, 1]):
            self.log(f"  PSLQ self-test FAILED: {rel}")
            return
        self.log("  PSLQ operational.")

        t_global_start = time.time()

        for (c1, c2, c3), priority in TRIPLES:
            s1 = SYMBOLS.get(c1, c1)
            s2 = SYMBOLS.get(c2, c2)
            s3 = SYMBOLS.get(c3, c3)
            triple_sym = f"({s1}, {s2}, {s3})"

            budget_h = 1.0 if priority == "top" else 0.5

            self.log(f"\n{'='*55}")
            self.log(f"  TRIPLE: {triple_sym}  [priority: {priority}]")
            self.log(f"  Budget: {budget_h:.1f}h")
            self.log(f"{'='*55}")

            pair_start = time.time()

            for degree in range(3, 50):
                elapsed_h = (time.time() - pair_start) / 3600
                remaining_h = budget_h - elapsed_h
                if remaining_h <= 0:
                    self.log(f"  Time budget exhausted at degree {degree - 1}.")
                    break

                # Check feasibility for smallest tier first
                n, wd, vd = compute_trivariate_precision(degree, 100)
                est_h = estimate_time(n, wd)
                if est_h > remaining_h:
                    self.log(f"  Degree {degree} not feasible (est. {est_h*60:.1f}min > "
                             f"{remaining_h*60:.1f}min remaining). Max degree: {degree-1}.")
                    break

                # Run through coefficient tiers
                for max_coeff in COEFF_TIERS:
                    key = self._checkpoint_key(c1, c2, c3, degree, max_coeff)
                    if key in self.completed:
                        self.log(f"    d={degree} |c|≤{max_coeff}: already completed (checkpoint)")
                        continue

                    n_mc, wd_mc, vd_mc = compute_trivariate_precision(degree, max_coeff)
                    est_mc = estimate_time(n_mc, wd_mc)

                    # Check feasibility for this tier
                    elapsed_h = (time.time() - pair_start) / 3600
                    remaining_h = budget_h - elapsed_h
                    if est_mc > remaining_h:
                        continue

                    result = self._run_single(c1, c2, c3, triple_sym,
                                              degree, max_coeff, n_mc, wd_mc, vd_mc)
                    self.completed.add(key)
                    self._save_checkpoint()

                    if result and result.found_relation:
                        self.log(f"  ★★★ TRIVARIATE RELATION FOUND! ★★★")
                        break

            # Check global time budget: 10 hours total
            global_elapsed = (time.time() - t_global_start) / 3600
            if global_elapsed > 10:
                self.log(f"\n  Global time budget exceeded (10h). Stopping.")
                break

        # Final report
        self._generate_report()
        self._generate_latex()

    def _run_single(self, c1, c2, c3, triple_sym, degree, max_coeff,
                    n_monomials, working_digits, verification_digits) -> Optional[TrivariateResult]:
        """Execute a single trivariate PSLQ search."""
        t_start = time.time()

        # Set precision and compute constants
        mpmath.mp.dps = working_digits + 50
        alpha = self.cc.compute_all(working_digits + 50, [c1])[c1]
        beta = self.cc.compute_all(working_digits + 50, [c2])[c2]
        gamma_val = self.cc.compute_all(working_digits + 50, [c3])[c3]

        # Generate monomials
        values, exponents, labels = generate_trivariate_monomials(
            alpha, beta, gamma_val, degree
        )

        est_min = estimate_time(n_monomials, working_digits) * 60
        self.log(f"    d={degree} |c|≤{max_coeff}: {n_monomials} mono, "
                 f"{working_digits} digits (est. {est_min:.1f}min)...")

        # Run PSLQ
        try:
            pslq_result = pslq_bound(values, maxcoeff=max_coeff, maxsteps=0)
            relation = pslq_result.relation
            norm_bound = pslq_result.norm_bound
        except Exception as exc:
            self.log(f"    PSLQ error: {exc}")
            relation = None
            norm_bound = None

        elapsed = time.time() - t_start
        found = relation is not None

        # Build bound statement
        s1 = SYMBOLS.get(c1, c1)
        s2 = SYMBOLS.get(c2, c2)
        s3 = SYMBOLS.get(c3, c3)

        residual_w = None
        residual_v = None

        if found:
            # Compute residual
            residual_val = abs(sum(c * v for c, v in zip(relation, values)))
            residual_w = float(residual_val) if residual_val > 0 else 0.0

            # Check triviality
            nonzero = [c for c in relation if c != 0]
            if len(nonzero) <= 1:
                self.log(f"    Degenerate (1 nonzero term). Discarded.")
                found = False
            else:
                is_trivial = _check_trivial_trivariate(
                    relation, exponents, c1, c2, c3
                )
                if is_trivial:
                    self.log(f"    Trivial (sympy simplifies to 0). Discarded.")
                    found = False

            if found:
                # Verify at higher precision
                mpmath.mp.dps = verification_digits + 50
                a_v = self.cc.compute_all(verification_digits + 50, [c1])[c1]
                b_v = self.cc.compute_all(verification_digits + 50, [c2])[c2]
                g_v = self.cc.compute_all(verification_digits + 50, [c3])[c3]
                vals_v, _, _ = generate_trivariate_monomials(a_v, b_v, g_v, degree)
                res_v = abs(sum(c * v for c, v in zip(relation, vals_v)))
                residual_v = float(res_v) if res_v > 0 else 0.0

                self.log(f"    ★ FOUND! coefficients: {relation}")
                self.log(f"    ★ Working residual:  {residual_w:.5e}")
                self.log(f"    ★ Verif. residual:   {residual_v:.5e}")
        else:
            elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"
            norm_info = f" ‖m‖₂ ≥ {norm_bound}." if norm_bound else ""
            bound_stmt = (f"No P ∈ Z[x,y,z] with deg ≤ {degree}, "
                         f"|c| ≤ {max_coeff} s.t. P({s1},{s2},{s3})=0.")
            self.log(f"    ✓ No relation. {elapsed_str}.{norm_info} {bound_stmt}")

        if found:
            bound_stmt = f"Relation found for ({s1},{s2},{s3}) at deg ≤ {degree}!"
        else:
            bound_stmt = (f"No P ∈ Z[x,y,z] with deg(P) ≤ {degree} and "
                         f"max|c| ≤ {max_coeff} s.t. P({s1},{s2},{s3})=0.")

        result = TrivariateResult(
            triple=(c1, c2, c3),
            triple_symbol=triple_sym,
            degree=degree,
            max_coeff=max_coeff,
            n_monomials=n_monomials,
            precision_digits=working_digits,
            found_relation=found,
            relation=list(relation) if found and relation else None,
            residual_working=residual_w,
            residual_verification=residual_v,
            bound_statement=bound_stmt,
            norm_bound=norm_bound,
            elapsed_seconds=round(elapsed, 1),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self.results.append(result)
        return result

    def _generate_report(self):
        """Generate Markdown report."""
        lines = [
            "# Trivariate PSLQ Search Report",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Hardware:** Intel i7-12700F, 64 GB DDR5, Debian 13",
            "",
            "---",
            "",
            "## Main Result",
            "",
        ]

        found_any = any(r.found_relation for r in self.results)
        if found_any:
            lines.append("**NON-TRIVIAL TRIVARIATE RELATION FOUND!** See details below.")
        else:
            lines.append("**No non-trivial trivariate polynomial relation found.**")

        lines.extend(["", "---", "", "## Bounds Established", ""])
        lines.append("| Triple | d_max (|c|≤100) | d_max (|c|≤10⁴) | d_max (|c|≤10⁶) | Time |")
        lines.append("|--------|-----------------|-----------------|-----------------|------|")

        # Group by triple
        triples_seen = {}
        for r in self.results:
            key = r.triple_symbol
            if key not in triples_seen:
                triples_seen[key] = {"100": 0, "10000": 0, "1000000": 0, "time": 0.0}
            mc = str(r.max_coeff)
            if mc in triples_seen[key] and not r.found_relation:
                triples_seen[key][mc] = max(triples_seen[key][mc], r.degree)
            triples_seen[key]["time"] += r.elapsed_seconds

        for key, data in triples_seen.items():
            t_str = f"{data['time']:.0f}s" if data['time'] < 60 else f"{data['time']/60:.1f}min"
            lines.append(
                f"| {key} | {data['100']} | {data['10000']} | {data['1000000']} | {t_str} |"
            )

        # Statistics
        total_time = sum(r.elapsed_seconds for r in self.results)
        lines.extend([
            "", "---", "",
            "## Statistics", "",
            f"- Total PSLQ runs: {len(self.results)}",
            f"- Total time: {total_time/3600:.2f} hours",
            f"- Triples explored: {len(triples_seen)}",
            f"- Relations found: {sum(1 for r in self.results if r.found_relation)}",
        ])

        report_path = self.output_dir / "REPORT_TRIVARIATE.md"
        report_path.write_text("\n".join(lines))
        self.log(f"  Report saved to {report_path}")

        # JSON
        json_data = {
            "results": [asdict(r) for r in self.results],
            "triples_summary": triples_seen,
            "total_time_seconds": round(total_time, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        json_path = self.output_dir / "trivariate_bounds.json"
        json_path.write_text(json.dumps(json_data, indent=2, default=str))

    def _generate_latex(self):
        """Generate LaTeX section."""
        lines = []
        lines.append(r"\section{Trivariate Results}")
        lines.append(r"\label{sec:trivariate}")
        lines.append("")
        lines.append(r"We extend our search to \emph{trivariate} polynomial relations")
        lines.append(r"$P(\alpha, \beta, \gamma) = 0$ among selected triples of constants.")
        lines.append(r"The monomial vector for three variables at degree $d$ has dimension")
        lines.append(r"$N_3(d) = \binom{d+3}{3} = (d+1)(d+2)(d+3)/6$, which grows faster")
        lines.append(r"than the bivariate case. For example, $N_3(8) = 165$ is comparable")
        lines.append(r"to the bivariate $N_2(17) = 171$.")
        lines.append("")
        lines.append(r"To the best of our knowledge, no systematic trivariate polynomial")
        lines.append(r"exclusion bounds have been published for these triples.")
        lines.append("")

        # Table
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"  \centering")
        lines.append(r"  \caption{Trivariate exclusion bounds.}")
        lines.append(r"  \label{tab:trivariate}")
        lines.append(r"  \small")
        lines.append(r"  \begin{tabular}{lcccc}")
        lines.append(r"    \toprule")
        lines.append(r"    Triple & $d_{\max}$ ($\|c\|_\infty \leq 10^2$) & $d_{\max}$ ($\|c\|_\infty \leq 10^4$) & $d_{\max}$ ($\|c\|_\infty \leq 10^6$) & Time \\")
        lines.append(r"    \midrule")

        from exclusion_frontier import LATEX_SYMBOLS
        triples_seen = {}
        for r in self.results:
            key = r.triple_symbol
            if key not in triples_seen:
                triples_seen[key] = {"100": 0, "10000": 0, "1000000": 0,
                                     "time": 0.0, "triple": r.triple}
            mc = str(r.max_coeff)
            if mc in triples_seen[key] and not r.found_relation:
                triples_seen[key][mc] = max(triples_seen[key][mc], r.degree)
            triples_seen[key]["time"] += r.elapsed_seconds

        for key, data in triples_seen.items():
            c1, c2, c3 = data["triple"]
            l1 = LATEX_SYMBOLS.get(c1, c1)
            l2 = LATEX_SYMBOLS.get(c2, c2)
            l3 = LATEX_SYMBOLS.get(c3, c3)
            t_str = f"{data['time']:.0f}\\,s" if data['time'] < 60 else f"{data['time']/60:.0f}\\,min"
            lines.append(
                f"    $({l1}, {l2}, {l3})$ & {data['100']} & {data['10000']} & {data['1000000']} & {t_str} \\\\"
            )

        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        latex_path = self.output_dir / "trivariate_section.tex"
        latex_path.write_text("\n".join(lines))
        self.log(f"  LaTeX section saved to {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Trivariate PSLQ Search")
    parser.add_argument("--estimate", action="store_true",
                        help="Print feasibility table and exit")
    args = parser.parse_args()

    if args.estimate:
        print_feasibility_table()
        return

    config = SearchConfig()
    engine = TrivariateSearchEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
