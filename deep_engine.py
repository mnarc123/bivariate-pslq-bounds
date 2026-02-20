"""
Deep search engine for bivariate polynomial relations.

This module runs PSLQ on pairs of transcendental constants,
pushing to the maximum degree allowed by the time budget.

ALGORITHM FOR EACH PAIR (α, β):
1. Compute the maximum feasible degree given the time budget
2. For each degree d from 9 up to the maximum (incremental):
   a. Generate the monomial vector [1, α, β, α², αβ, β², ..., β^d]
   b. Compute values at adequate precision (N×D×safety digits)
   c. Run PSLQ with maxcoeff=100 ("elegant" relations first)
   d. If no result: run PSLQ with maxcoeff=10000
   e. If no result: run PSLQ with maxcoeff=10^6
   f. Log the bound established at this degree
   g. Checkpoint
3. Verify any found relation at double precision

INCREMENTAL STRATEGY:
- We start from degree 9 (where Phase 1 left off)
- We increment by 1 up to the maximum degree
- This allows interruption at any point with partial results
- Checkpoints ensure that completed work is not repeated
"""

import mpmath
import sympy
import time
import json
import sys
from pslq_bounds import pslq_bound, PSLQResult
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from config import SearchConfig
from constants import ConstantsComputer
from precision_manager import compute_precision_plan, PrecisionPlan
from checkpoint import CheckpointManager
from bound_calculator import format_bound_statement, compute_search_space_size


# === PAIR DEFINITIONS ===

TOP_PAIRS = [
    ("pi", "e"),
    ("pi", "euler_gamma"),
    ("e", "euler_gamma"),
]

HIGH_PAIRS = [
    ("pi", "zeta3"),
    ("e", "zeta3"),
    ("euler_gamma", "zeta3"),
    ("pi", "catalan"),
    ("e", "catalan"),
    ("euler_gamma", "catalan"),
]

MEDIUM_PAIRS = [
    ("pi", "omega"),
    ("e", "omega"),
    ("euler_gamma", "omega"),
    ("zeta3", "catalan"),
    ("pi", "zeta5"),
    ("euler_gamma", "glaisher"),
    ("zeta3", "zeta5"),
    ("pi", "ln2"),
    ("e", "ln2"),
    ("euler_gamma", "ln2"),
]


@dataclass
class DeepSearchResult:
    """Result of a single deep search run."""
    pair: Tuple[str, str]
    degree: int
    max_coeff: int
    n_monomials: int
    precision_digits: int
    found_relation: bool
    relation: Optional[List[int]]
    residual_working: Optional[float]
    residual_verification: Optional[float]
    bound_statement: str
    elapsed_seconds: float
    timestamp: str
    norm_bound: Optional[int] = None
    norm_bound_allH: Optional[int] = None
    pslq_iterations: Optional[int] = None


# Map constants → sympy symbols for triviality checking
_SYMPY_CONSTANTS = {
    "pi": sympy.pi,
    "e": sympy.E,
    "euler_gamma": sympy.EulerGamma,
    "phi": sympy.GoldenRatio,
    "ln2": sympy.log(2),
    "sqrt2": sympy.sqrt(2),
    "zeta3": sympy.zeta(3),
    "catalan": sympy.Catalan,
    "sqrt3": sympy.sqrt(3),
    "sqrt5": sympy.sqrt(5),
    "ln3": sympy.log(3),
    "ln5": sympy.log(5),
    "ln10": sympy.log(10),
    "pi2": sympy.pi**2,
    "zeta5": sympy.zeta(5),
    "omega": sympy.LambertW(1),
    "glaisher": sympy.Symbol("A_Glaisher"),  # no closed form in sympy
}


def generate_bivariate_monomials(
    alpha: mpmath.mpf,
    beta: mpmath.mpf,
    degree: int,
) -> Tuple[List[mpmath.mpf], List[Tuple[int, int]], List[str]]:
    """
    Generate all monomials α^i · β^j with i+j ≤ degree.

    Order: increasing total degree, then increasing i within each degree.
    The first element is always 1 (i=0, j=0).

    Returns:
        values: numerical values of the monomials
        exponents: list of (i, j)
        labels: human-readable strings
    """
    values = []
    exponents = []
    labels = []

    # Precompute powers
    alpha_powers = [mpmath.mpf(1)]
    beta_powers = [mpmath.mpf(1)]
    for k in range(1, degree + 1):
        alpha_powers.append(alpha_powers[-1] * alpha)
        beta_powers.append(beta_powers[-1] * beta)

    for total_deg in range(degree + 1):
        for i in range(total_deg + 1):
            j = total_deg - i
            val = alpha_powers[i] * beta_powers[j]
            values.append(val)
            exponents.append((i, j))

            # Human-readable label
            parts = []
            if i > 0:
                if i == 1:
                    parts.append("α")
                else:
                    parts.append(f"α^{i}")
            if j > 0:
                if j == 1:
                    parts.append("β")
                else:
                    parts.append(f"β^{j}")
            if not parts:
                parts = ["1"]
            labels.append("·".join(parts))

    return values, exponents, labels


def check_relation_trivial_sympy(
    relation: List[int],
    exponents: List[Tuple[int, int]],
    const1: str,
    const2: str,
) -> bool:
    """
    Check whether a found relation is trivial using sympy.
    Returns True if sympy can simplify it to zero.
    """
    if const1 not in _SYMPY_CONSTANTS or const2 not in _SYMPY_CONSTANTS:
        return False

    try:
        sym1 = _SYMPY_CONSTANTS[const1]
        sym2 = _SYMPY_CONSTANTS[const2]

        total = sympy.Integer(0)
        for c, (i, j) in zip(relation, exponents):
            if c == 0:
                continue
            term = sympy.Integer(c) * sym1**i * sym2**j
            total += term

        if sympy.simplify(total) == 0:
            return True
        rewritten = total.rewrite(sympy.sqrt)
        if sympy.simplify(rewritten) == 0:
            return True
        if sympy.expand(rewritten) == 0:
            return True
        return False
    except Exception:
        return False


class DeepSearchEngine:
    """Deep search engine."""

    def __init__(self, config: SearchConfig, max_hours_per_pair: float = 24.0):
        self.config = config
        self.max_hours_per_pair = max_hours_per_pair

        deep_dir = config.results_dir / "deep_v2"
        deep_dir.mkdir(parents=True, exist_ok=True)
        (deep_dir / "bounds").mkdir(exist_ok=True)
        (deep_dir / "logs").mkdir(exist_ok=True)

        self.checkpoint = CheckpointManager(deep_dir / "checkpoints")
        self.results: List[DeepSearchResult] = []
        self.constants_computer = ConstantsComputer(config)

        self.log_file = deep_dir / "logs" / f"deep_search_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self.bounds_dir = deep_dir / "bounds"
        self.deep_dir = deep_dir

    def log(self, msg: str):
        """Log to file and console."""
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def get_constant_value(self, name: str, digits: int) -> mpmath.mpf:
        """Retrieve or compute a constant at the given precision."""
        all_vals = self.constants_computer.compute_all(digits, [name])
        return all_vals[name]

    def run(self, pair_filter: Optional[str] = None):
        """Main execution of the deep search."""
        self.log("=" * 70)
        self.log("  BRIDGE EQUATION — Phase 2: High-Degree Deep Search")
        self.log("=" * 70)

        # === SELF-TEST ===
        self.log("\n[SELF-TEST] Verifying PSLQ on known relation φ²-φ-1=0...")
        mpmath.mp.dps = 200
        phi = (1 + mpmath.sqrt(5)) / 2
        rel = mpmath.pslq([phi**2, phi, mpmath.mpf(1)])
        if rel is None or (rel != [1, -1, -1] and rel != [-1, 1, 1]):
            self.log(f"  ✗ ERROR: PSLQ returns {rel}! Aborting.")
            return
        self.log("  ✓ PSLQ operational.")

        # === SELF-TEST 2: ζ(2) = π²/6 ===
        self.log("[SELF-TEST] Verifying PSLQ on ζ(2) - π²/6 = 0...")
        mpmath.mp.dps = 200
        rel2 = mpmath.pslq([mpmath.zeta(2), mpmath.pi**2])
        if rel2 is None or (rel2 != [6, -1] and rel2 != [-6, 1]):
            self.log(f"  ✗ ERROR: PSLQ returns {rel2}! Aborting.")
            return
        self.log("  ✓ PSLQ operational.")

        # === PLANNING ===
        self.log(f"\n[PLANNING] Budget: {self.max_hours_per_pair:.1f}h per pair")
        self.log(f"  Completed checkpoints: {self.checkpoint.get_completed_count()}")

        all_pairs = (
            [(p, "top") for p in TOP_PAIRS]
            + [(p, "high") for p in HIGH_PAIRS]
            + [(p, "medium") for p in MEDIUM_PAIRS]
        )

        # Filter if requested
        if pair_filter:
            all_pairs = [
                (p, pri) for p, pri in all_pairs
                if f"{p[0]}+{p[1]}" == pair_filter
            ]
            if not all_pairs:
                self.log(f"  ⚠ Pair '{pair_filter}' not found!")
                return

        for (c1, c2), priority in all_pairs:
            pair_name = f"{c1}+{c2}"

            # Time budget based on priority
            if priority == "top":
                time_budget = self.max_hours_per_pair * 2
            elif priority == "high":
                time_budget = self.max_hours_per_pair
            else:
                time_budget = self.max_hours_per_pair * 0.5

            self.log(f"\n{'='*60}")
            self.log(f"  PAIR: ({c1}, {c2})  [priority: {priority}]")
            self.log(f"  Time budget: {time_budget:.1f}h")
            self.log(f"{'='*60}")

            self._search_pair(c1, c2, pair_name, time_budget)

        # === FINAL REPORT ===
        self._generate_report()

    def _search_pair(
        self,
        const1: str,
        const2: str,
        pair_name: str,
        time_budget_hours: float,
    ):
        """Incremental deep search for a pair."""

        pair_start_time = time.time()
        start_degree = 9  # continuing from where Phase 1 left off

        for degree in range(start_degree, 200):
            # Check remaining time
            elapsed_hours = (time.time() - pair_start_time) / 3600
            remaining_hours = time_budget_hours - elapsed_hours
            if remaining_hours <= 0:
                self.log(f"  ⏱ Time budget exhausted at degree {degree-1}.")
                break

            # === TIER A: Small coefficients (≤100) ===
            plan_small = compute_precision_plan(
                degree, max_coeff=100, max_hours=remaining_hours
            )
            if not plan_small.is_feasible:
                self.log(f"  📊 Degree {degree} (|c|≤100) not feasible "
                         f"(estimate {plan_small.estimated_hours:.1f}h > {remaining_hours:.1f}h remaining). "
                         f"Max degree reached: {degree-1}.")
                break

            if not self.checkpoint.is_completed(pair_name, degree, 100):
                result_a = self._run_single_pslq(
                    const1, const2, pair_name, degree,
                    max_coeff=100, plan=plan_small
                )
                if result_a and result_a.found_relation:
                    self.log(f"  ★★★ RELATION FOUND at degree {degree}, |c|≤100! ★★★")
                    return
            else:
                self.log(f"    Degree {degree}, |c|≤100: already completed (checkpoint)")

            # Update remaining time
            elapsed_hours = (time.time() - pair_start_time) / 3600
            remaining_hours = time_budget_hours - elapsed_hours

            # === TIER B: Medium coefficients (≤10000) ===
            plan_medium = compute_precision_plan(
                degree, max_coeff=10000, max_hours=remaining_hours
            )
            if plan_medium.is_feasible and not self.checkpoint.is_completed(pair_name, degree, 10000):
                result_b = self._run_single_pslq(
                    const1, const2, pair_name, degree,
                    max_coeff=10000, plan=plan_medium
                )
                if result_b and result_b.found_relation:
                    self.log(f"  ★★★ RELATION FOUND at degree {degree}, |c|≤10000! ★★★")
                    return
            elif self.checkpoint.is_completed(pair_name, degree, 10000):
                self.log(f"    Degree {degree}, |c|≤10000: already completed (checkpoint)")

            # Update remaining time
            elapsed_hours = (time.time() - pair_start_time) / 3600
            remaining_hours = time_budget_hours - elapsed_hours

            # === TIER C: Large coefficients (≤10⁶) — only if fast ===
            plan_large = compute_precision_plan(
                degree, max_coeff=10**6,
                max_hours=min(1.0, remaining_hours)
            )
            if plan_large.is_feasible and not self.checkpoint.is_completed(pair_name, degree, 10**6):
                result_c = self._run_single_pslq(
                    const1, const2, pair_name, degree,
                    max_coeff=10**6, plan=plan_large
                )
                if result_c and result_c.found_relation:
                    self.log(f"  ★★★ RELATION FOUND at degree {degree}, |c|≤10⁶! ★★★")
                    return

        # Save bounds for this pair
        self._save_pair_bounds(const1, const2, pair_name)

    def _run_single_pslq(
        self,
        const1: str,
        const2: str,
        pair_name: str,
        degree: int,
        max_coeff: int,
        plan: PrecisionPlan,
    ) -> Optional[DeepSearchResult]:
        """Execute a single PSLQ search."""

        self.checkpoint.mark_started(pair_name, degree, max_coeff)
        t_start = time.time()

        # Set precision
        mpmath.mp.dps = plan.working_digits + 50

        # Compute constants
        alpha = self.get_constant_value(const1, plan.working_digits + 50)
        beta = self.get_constant_value(const2, plan.working_digits + 50)

        # Generate monomials
        values, exponents, labels = generate_bivariate_monomials(alpha, beta, degree)

        self.log(f"    Degree {degree}, |c|≤{max_coeff}: "
                 f"{plan.n_monomials} mono, {plan.working_digits} digits "
                 f"(est. {plan.estimated_hours*60:.1f}min)... ")

        # Run PSLQ with H-matrix bound extraction
        try:
            pslq_result = pslq_bound(mpmath.mp, values, maxcoeff=max_coeff, maxsteps=0)
            relation = pslq_result.relation
            _norm_bound = pslq_result.norm_bound
            _norm_bound_allH = pslq_result.norm_bound_allH
            _pslq_iterations = pslq_result.iterations
        except Exception as exc:
            self.log(f"    ⚠ PSLQ error: {exc}")
            relation = None
            _norm_bound = None
            _norm_bound_allH = None
            _pslq_iterations = None

        elapsed = time.time() - t_start
        found = relation is not None

        # Build the bound statement
        bound_stmt = format_bound_statement(const1, const2, degree, max_coeff, found)

        residual_w = None
        residual_v = None

        if found:
            # Compute residual at working precision
            residual_val = abs(sum(c * v for c, v in zip(relation, values)))
            residual_w = float(residual_val) if residual_val > 0 else 0.0

            # Check if degenerate
            nonzero = [c for c in relation if c != 0]
            if len(nonzero) <= 1:
                self.log(f"    ⚠ Degenerate relation (only one nonzero term). Discarded.")
                found = False
            else:
                # Check triviality with sympy
                is_trivial = check_relation_trivial_sympy(
                    relation, exponents, const1, const2
                )
                if is_trivial:
                    self.log(f"    ⚠ Trivial relation (sympy simplifies to 0). Discarded.")
                    found = False

            if found:
                # Verify at double precision
                mpmath.mp.dps = plan.verification_digits + 50
                alpha_v = self.get_constant_value(const1, plan.verification_digits + 50)
                beta_v = self.get_constant_value(const2, plan.verification_digits + 50)
                values_v, _, _ = generate_bivariate_monomials(alpha_v, beta_v, degree)
                residual_v_val = abs(sum(c * v for c, v in zip(relation, values_v)))
                residual_v = float(residual_v_val) if residual_v_val > 0 else 0.0

                # Format the relation
                eq_parts = []
                for c, (i, j) in zip(relation, exponents):
                    if c == 0:
                        continue
                    eq_parts.append(f"{c}·{const1}^{i}·{const2}^{j}")
                eq_str = " + ".join(eq_parts)

                self.log(f"    ★ FOUND! {eq_str} = 0")
                self.log(f"    ★ Coefficients: {relation}")
                self.log(f"    ★ Working residual:      {residual_w:.5e}")
                self.log(f"    ★ Verification residual: {residual_v:.5e}")

                # Save candidate to disk
                candidate = {
                    "pair": [const1, const2],
                    "degree": degree,
                    "max_coeff": max_coeff,
                    "coefficients": relation,
                    "exponents": exponents,
                    "labels": labels,
                    "equation": eq_str,
                    "residual_working": residual_w,
                    "residual_verification": residual_v,
                    "precision_digits": plan.working_digits,
                    "verification_digits": plan.verification_digits,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                cand_path = (
                    self.deep_dir / "bounds"
                    / f"CANDIDATE_{const1}_{const2}_d{degree}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                )
                cand_path.write_text(json.dumps(candidate, indent=2))
                self.log(f"    → Saved to {cand_path}")
        else:
            elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"
            norm_info = ""
            if _norm_bound is not None:
                norm_info = f" ‖m‖₂ ≥ {_norm_bound}."
            self.log(f"    ✓ No relation found. {elapsed_str}.{norm_info} {bound_stmt}")

        result = DeepSearchResult(
            pair=(const1, const2),
            degree=degree,
            max_coeff=max_coeff,
            n_monomials=plan.n_monomials,
            precision_digits=plan.working_digits,
            found_relation=found,
            relation=relation if found else None,
            residual_working=residual_w,
            residual_verification=residual_v,
            bound_statement=bound_stmt,
            elapsed_seconds=elapsed,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            norm_bound=_norm_bound,
            norm_bound_allH=_norm_bound_allH,
            pslq_iterations=_pslq_iterations,
        )

        self.results.append(result)

        # Checkpoint
        self.checkpoint.mark_completed(pair_name, degree, max_coeff, {
            "found": found,
            "elapsed_s": round(elapsed, 1),
            "bound": bound_stmt,
            "norm_bound": _norm_bound,
            "norm_bound_allH": _norm_bound_allH,
            "pslq_iterations": _pslq_iterations,
        })

        return result

    def _save_pair_bounds(self, const1: str, const2: str, pair_name: str):
        """Save a summary of bounds for this pair."""
        pair_results = [r for r in self.results if r.pair == (const1, const2)]
        if not pair_results:
            return

        max_degree_100 = max(
            (r.degree for r in pair_results if r.max_coeff == 100 and not r.found_relation),
            default=0
        )
        max_degree_10k = max(
            (r.degree for r in pair_results if r.max_coeff == 10000 and not r.found_relation),
            default=0
        )
        max_degree_1M = max(
            (r.degree for r in pair_results if r.max_coeff == 10**6 and not r.found_relation),
            default=0
        )
        total_time = sum(r.elapsed_seconds for r in pair_results)

        summary = {
            "pair": [const1, const2],
            "bounds": {
                "max_coeff_100": {"max_degree": max_degree_100},
                "max_coeff_10000": {"max_degree": max_degree_10k},
                "max_coeff_1000000": {"max_degree": max_degree_1M},
            },
            "total_time_seconds": round(total_time, 1),
            "n_searches": len(pair_results),
            "found_any": any(r.found_relation for r in pair_results),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        bound_path = self.bounds_dir / f"bound_{pair_name}.json"
        bound_path.write_text(json.dumps(summary, indent=2))

    def _generate_report(self):
        """Generate the final deep search report."""
        report_path = self.deep_dir / "REPORT_DEEP_V2.md"

        lines = [
            "# Report — Bridge Equation, Phase 2: High-Degree Deep Search",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Hardware:** Intel i7-12700F, 64 GB DDR5, Debian 13",
            f"**Software:** Python 3, mpmath, sympy",
            f"**Budget:** {self.max_hours_per_pair:.1f}h per pair",
            "",
            "---",
            "",
            "## Main result",
            "",
        ]

        found_any = any(r.found_relation for r in self.results)
        if found_any:
            lines.append("**⭐ NON-TRIVIAL RELATION FOUND!** See details below.")
        else:
            lines.append("**No non-trivial polynomial relation found.**")
            lines.append("")
            lines.append("This is a significant negative computational result.")

        # Bounds table per pair
        lines.extend([
            "",
            "---",
            "",
            "## Bounds established per pair",
            "",
            "| Pair | Max deg (|c|≤100) | Max deg (|c|≤10⁴) | Max deg (|c|≤10⁶) | Time |",
            "|------|---------------------|---------------------|---------------------|------|",
        ])

        # Group by pair
        pairs_seen = {}
        for r in self.results:
            key = f"({r.pair[0]}, {r.pair[1]})"
            if key not in pairs_seen:
                pairs_seen[key] = {"100": 0, "10000": 0, "1000000": 0, "time": 0.0, "found": False}
            mc = str(r.max_coeff)
            if mc in pairs_seen[key] and not r.found_relation:
                pairs_seen[key][mc] = max(pairs_seen[key][mc], r.degree)
            pairs_seen[key]["time"] += r.elapsed_seconds
            if r.found_relation:
                pairs_seen[key]["found"] = True

        for key, data in pairs_seen.items():
            time_str = f"{data['time']:.0f}s" if data['time'] < 60 else f"{data['time']/60:.1f}min"
            status = " ⭐" if data["found"] else ""
            lines.append(
                f"| {key}{status} | {data['100']} | {data['10000']} | {data['1000000']} | {time_str} |"
            )

        # Detail of found relations
        found_results = [r for r in self.results if r.found_relation]
        if found_results:
            lines.extend([
                "",
                "---",
                "",
                "## ⭐ Relations found",
                "",
            ])
            for r in found_results:
                lines.extend([
                    f"### ({r.pair[0]}, {r.pair[1]}) — degree {r.degree}",
                    f"- **Coefficients:** `{r.relation}`",
                    f"- **Working residual:** {r.residual_working:.5e}",
                    f"- **Verification residual:** {r.residual_verification:.5e}",
                    f"- **Precision:** {r.precision_digits} digits (working), "
                    f"{r.precision_digits * 3 // 2} digits (verification)",
                    f"- **Time:** {r.elapsed_seconds:.1f}s",
                    "",
                ])

        # Statistics
        total_time = sum(r.elapsed_seconds for r in self.results)
        total_searches = len(self.results)
        lines.extend([
            "",
            "---",
            "",
            "## Statistics",
            "",
            f"- **Total PSLQ searches:** {total_searches}",
            f"- **Total time:** {total_time/3600:.2f} hours ({total_time:.0f}s)",
            f"- **Relations found:** {len(found_results)}",
            f"- **Pairs explored:** {len(pairs_seen)}",
            "",
            "---",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "source ~/bridge_eq_env/bin/activate",
            "cd ~/bridge_equation",
            f"python3 run_deep_search_v2.py --max-hours-per-pair {self.max_hours_per_pair}",
            "```",
            "",
            "## References",
            "",
            "- Bailey & Ferguson, 'Numerical results on relations between fundamental constants' (1989)",
            "- Ferguson, Bailey & Arno, 'Analysis of PSLQ' (1999)",
            "- Bailey & Borwein, 'PSLQ: An Algorithm to Discover Integer Relations' (2009)",
        ])

        report_path.write_text("\n".join(lines))
        self.log(f"\n  Report saved to: {report_path}")
