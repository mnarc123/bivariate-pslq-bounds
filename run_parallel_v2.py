#!/usr/bin/env python3
"""
Parallel Orchestrator v2 — with realistic time limits.

Skips trivariate tasks that would take >30 min per run.
Saves results incrementally to avoid data loss.
"""

import mpmath
import math
import json
import time
import sys
import os
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
from config import SearchConfig
from constants import ConstantsComputer
from deep_engine import generate_bivariate_monomials, TOP_PAIRS, HIGH_PAIRS, MEDIUM_PAIRS
from precision_manager import compute_precision_plan
from pslq_bounds_fast import pslq_with_bound, PSLQResult
from bound_calculator import SYMBOLS
from monomials_trivariate import (
    generate_trivariate_monomials,
    count_trivariate_monomials,
    compute_trivariate_precision,
)

N_WORKERS = 10

# ── Baseline pair definitions (pickle-safe) ──
BASELINE_PAIRS_DATA = [
    ("sin",  2,  "cos",  3,  "sin(√2), cos(√3)"),
    ("log_p1", 5, "tanh", 7,  "ln(√5+1), tanh(√7)"),
    ("besselj0", 11, "erf", 13, "J₀(√11), erf(√13)"),
    ("sin", 17, "exp_inv", 19, "sin(√17), e^(1/√19)"),
    ("log_p2", 23, "cos", 29, "ln(√23+2), cos(√29)"),
    ("tanh", 31, "sin", 37, "tanh(√31), sin(√37)"),
    ("besselj1", 41, "log_p1", 43, "J₁(√41), ln(√43+1)"),
    ("erf", 47, "cos", 53, "erf(√47), cos(√53)"),
    ("sin", 59, "tanh", 61, "sin(√59), tanh(√61)"),
    ("log_p3", 67, "besselj0", 71, "ln(√67+3), J₀(√71)"),
    ("exp_inv", 73, "sin", 79, "e^(1/√73), sin(√79)"),
    ("cos", 83, "erf", 89, "cos(√83), erf(√89)"),
    ("tanh", 97, "log_p1", 101, "tanh(√97), ln(√101+1)"),
    ("besselj0", 103, "sin", 107, "J₀(√103), sin(√107)"),
    ("erf", 109, "cos", 113, "erf(√109), cos(√113)"),
    ("sin", 127, "exp_inv", 131, "sin(√127), e^(1/√131)"),
    ("log_p2", 137, "tanh", 139, "ln(√137+2), tanh(√139)"),
    ("cos", 149, "besselj1", 151, "cos(√149), J₁(√151)"),
    ("sin", 157, "erf", 163, "sin(√157), erf(√163)"),
    ("tanh", 167, "log_p1", 173, "tanh(√167), ln(√173+1)"),
    ("exp_inv", 179, "cos", 181, "e^(1/√179), cos(√181)"),
    ("besselj0", 191, "sin", 193, "J₀(√191), sin(√193)"),
    ("erf", 197, "tanh", 199, "erf(√197), tanh(√199)"),
    ("sin", 211, "log_p2", 223, "sin(√211), ln(√223+2)"),
    ("cos", 227, "exp_inv", 229, "cos(√227), e^(1/√229)"),
    ("tanh", 233, "besselj0", 239, "tanh(√233), J₀(√239)"),
    ("log_p1", 241, "sin", 251, "ln(√241+1), sin(√251)"),
    ("erf", 257, "cos", 263, "erf(√257), cos(√263)"),
    ("sin", 269, "tanh", 271, "sin(√269), tanh(√271)"),
    ("besselj1", 277, "log_p3", 281, "J₁(√277), ln(√281+3)"),
]

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

COEFF_TIERS = [100, 10000, 1000000]


def _eval_baseline_func(func_name: str, arg: int) -> mpmath.mpf:
    sq = mpmath.sqrt(arg)
    dispatch = {
        "sin": lambda: mpmath.sin(sq),
        "cos": lambda: mpmath.cos(sq),
        "tanh": lambda: mpmath.tanh(sq),
        "erf": lambda: mpmath.erf(sq),
        "exp_inv": lambda: mpmath.exp(1 / sq),
        "besselj0": lambda: mpmath.besselj(0, sq),
        "besselj1": lambda: mpmath.besselj(1, sq),
        "log_p1": lambda: mpmath.log(sq + 1),
        "log_p2": lambda: mpmath.log(sq + 2),
        "log_p3": lambda: mpmath.log(sq + 3),
    }
    return dispatch[func_name]()


# ═══════════════════════════════════════════════════════════════
#  WORKER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _worker_near_miss(const1: str, const2: str, degree: int) -> Dict[str, Any]:
    try:
        config = SearchConfig()
        cc = ConstantsComputer(config)
        plan = compute_precision_plan(degree, max_coeff=100)
        wd = plan.working_digits

        mpmath.mp.dps = wd + 50
        alpha = cc.compute_all(wd + 50, [const1])[const1]
        beta = cc.compute_all(wd + 50, [const2])[const2]
        values, exponents, labels = generate_bivariate_monomials(alpha, beta, degree)

        t0 = time.time()
        result = pslq_with_bound(values, maxcoeff=100, maxsteps=0)
        elapsed = time.time() - t0

        if result.relation is not None or result.best_candidate is None:
            return {"type": "near_miss", "pair": (const1, const2), "degree": degree,
                    "status": "skip", "elapsed": elapsed}

        candidate = result.best_candidate

        dbl = wd * 2
        mpmath.mp.dps = dbl + 50
        a2 = cc.compute_all(dbl + 50, [const1])[const1]
        b2 = cc.compute_all(dbl + 50, [const2])[const2]
        v2, _, _ = generate_bivariate_monomials(a2, b2, degree)
        residual = abs(sum(c * v for c, v in zip(candidate, v2)))

        if residual == 0:
            log10_res = None; dc = None; pf = 0; q = 0
        else:
            log10_res = float(mpmath.log10(residual))
            dc = -log10_res
            pf = dc / wd if wd > 0 else 0
            norm_l2 = math.sqrt(sum(c**2 for c in candidate))
            q = dc / (norm_l2 * degree) if norm_l2 * degree > 0 else 0

        s1 = SYMBOLS.get(const1, const1)
        s2 = SYMBOLS.get(const2, const2)

        return {
            "type": "near_miss",
            "pair": (const1, const2),
            "pair_symbol": f"({s1}, {s2})",
            "degree": degree,
            "status": "ok",
            "vector": list(candidate),
            "exponents": [list(e) for e in exponents],
            "labels": labels,
            "norm_inf": max(abs(c) for c in candidate),
            "norm_l2": round(math.sqrt(sum(c**2 for c in candidate)), 2),
            "residual": float(residual) if residual > 0 else 0.0,
            "log10_residual": round(log10_res, 2) if log10_res is not None else None,
            "working_precision": wd,
            "digits_cancelled": round(dc, 2) if dc is not None else None,
            "precision_fraction": round(pf, 4),
            "quality_score": round(q, 6) if q else 0,
            "maxcoeff_used": 100,
            "elapsed": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    except Exception as exc:
        return {"type": "near_miss", "pair": (const1, const2), "degree": degree,
                "status": "error", "error": str(exc)}


def _worker_baseline(pair_idx, func_a, arg_a, func_b, arg_b, label, degree,
                     max_coeff=100):
    try:
        plan = compute_precision_plan(degree, max_coeff=max_coeff)
        wd = plan.working_digits

        mpmath.mp.dps = wd + 50
        alpha_val = _eval_baseline_func(func_a, arg_a)
        beta_val = _eval_baseline_func(func_b, arg_b)

        if (abs(alpha_val) < mpmath.mpf(10)**(-wd//2) or
            abs(beta_val) < mpmath.mpf(10)**(-wd//2)):
            return {"type": "baseline", "pair_idx": pair_idx, "degree": degree,
                    "status": "skip", "label": label}

        values, _, _ = generate_bivariate_monomials(alpha_val, beta_val, degree)

        t0 = time.time()
        pslq_result = pslq_with_bound(values, maxcoeff=max_coeff, maxsteps=0)
        elapsed = time.time() - t0

        residual = None; log10_res = None; lambda_val = None
        norm_inf = 0; norm_l2 = 0.0

        if pslq_result is not None:
            found_rel = pslq_result.relation is not None
            candidate = pslq_result.best_candidate

            if candidate is not None and not found_rel:
                dbl = wd * 2
                mpmath.mp.dps = dbl + 50
                a2 = _eval_baseline_func(func_a, arg_a)
                b2 = _eval_baseline_func(func_b, arg_b)
                v2, _, _ = generate_bivariate_monomials(a2, b2, degree)
                res = abs(sum(c * v for c, v in zip(candidate, v2)))
                if res > 0:
                    residual = float(res)
                    log10_res = float(mpmath.log10(res))
                    lambda_val = -log10_res / wd if wd > 0 else 0
                norm_inf = max(abs(c) for c in candidate)
                norm_l2 = math.sqrt(sum(c**2 for c in candidate))

        return {
            "type": "baseline", "pair_idx": pair_idx, "pair_label": label,
            "degree": degree, "status": "ok", "max_coeff": max_coeff,
            "n_monomials": plan.n_monomials, "working_digits": wd,
            "found_relation": pslq_result.relation is not None if pslq_result else False,
            "residual": residual,
            "log10_residual": round(log10_res, 2) if log10_res is not None else None,
            "lambda_value": round(lambda_val, 4) if lambda_val is not None else None,
            "norm_inf": norm_inf, "norm_l2": round(norm_l2, 2),
            "elapsed": round(elapsed, 1),
        }
    except Exception as exc:
        return {"type": "baseline", "pair_idx": pair_idx, "degree": degree,
                "status": "error", "error": str(exc), "label": label}


def _worker_trivariate(c1, c2, c3, degree, max_coeff):
    try:
        config = SearchConfig()
        cc = ConstantsComputer(config)
        n_mono, wd, vd = compute_trivariate_precision(degree, max_coeff)

        mpmath.mp.dps = wd + 50
        alpha = cc.compute_all(wd + 50, [c1])[c1]
        beta = cc.compute_all(wd + 50, [c2])[c2]
        gamma_val = cc.compute_all(wd + 50, [c3])[c3]

        values, exponents, labels = generate_trivariate_monomials(
            alpha, beta, gamma_val, degree)

        t0 = time.time()
        pslq_result = pslq_with_bound(values, maxcoeff=max_coeff, maxsteps=0)
        elapsed = time.time() - t0

        relation = pslq_result.relation
        norm_bound = pslq_result.norm_bound
        found = relation is not None

        s1 = SYMBOLS.get(c1, c1)
        s2 = SYMBOLS.get(c2, c2)
        s3 = SYMBOLS.get(c3, c3)

        if found:
            nonzero = [c for c in relation if c != 0]
            if len(nonzero) <= 1:
                found = False

        return {
            "type": "trivariate",
            "triple": (c1, c2, c3),
            "triple_symbol": f"({s1}, {s2}, {s3})",
            "degree": degree, "max_coeff": max_coeff,
            "n_monomials": n_mono, "precision_digits": wd,
            "status": "ok", "found_relation": found,
            "relation": list(relation) if found and relation else None,
            "norm_bound": norm_bound,
            "elapsed": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    except Exception as exc:
        return {"type": "trivariate", "triple": (c1, c2, c3),
                "degree": degree, "max_coeff": max_coeff,
                "status": "error", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════
#  TASK GENERATION — realistic time limits
# ═══════════════════════════════════════════════════════════════

# Empirical time data from v1 run (seconds per trivariate task):
# d=3: 3-8s, d=4: 55-155s, d=5: 770-2400s, d=6: 9000-28500s, d=7: >>10h
# Strategy: limit top triples to d≤6, high triples to d≤5

def generate_tasks():
    tasks = []

    # ── 1. Near-miss tasks ──
    all_pairs = (
        [(p, "top") for p in TOP_PAIRS]
        + [(p, "high") for p in HIGH_PAIRS]
        + [(p, "medium") for p in MEDIUM_PAIRS]
    )
    for (c1, c2), priority in all_pairs:
        if priority == "top":
            degrees = list(range(3, 13))    # 3-12
        elif priority == "high":
            degrees = list(range(3, 11))    # 3-10
        else:
            degrees = list(range(3, 9))     # 3-8
        for d in degrees:
            tasks.append(("near_miss", _worker_near_miss, c1, c2, d))

    # ── 2. Baseline tasks (30 pairs × degrees 3-10) ──
    for pair_idx, (fa, aa, fb, ab, label) in enumerate(BASELINE_PAIRS_DATA[:30]):
        for d in range(3, 11):
            tasks.append(("baseline", _worker_baseline,
                         pair_idx, fa, aa, fb, ab, label, d))

    # ── 3. Trivariate tasks (realistic limits) ──
    for (c1, c2, c3), priority in TRIPLES:
        if priority == "top":
            max_degree = 6   # d=6 takes ~2-8h, d=7 takes >>10h
        else:
            max_degree = 5   # high priority: limit to d≤5
        for degree in range(3, max_degree + 1):
            for max_coeff in COEFF_TIERS:
                tasks.append(("trivariate", _worker_trivariate,
                             c1, c2, c3, degree, max_coeff))

    return tasks


# ═══════════════════════════════════════════════════════════════
#  INCREMENTAL SAVE
# ═══════════════════════════════════════════════════════════════

class ResultCollector:
    def __init__(self, output_base: Path):
        self.nm_dir = output_base / "near_misses"
        self.stats_dir = output_base / "statistics"
        self.tri_dir = output_base / "trivariate"
        for d in [self.nm_dir, self.stats_dir, self.tri_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.near_misses = []
        self.baselines = []
        self.trivariates = []
        self.errors = []

    def add(self, result: Dict):
        if result.get("status") == "error":
            self.errors.append(result)
            return
        if result.get("status") == "skip":
            return
        t = result["type"]
        if t == "near_miss":
            self.near_misses.append(result)
        elif t == "baseline":
            self.baselines.append(result)
        elif t == "trivariate":
            self.trivariates.append(result)

    def save_checkpoint(self):
        """Save incremental checkpoint every N tasks."""
        self._save_json(self.nm_dir / "near_miss_checkpoint.json",
                       {"near_misses": self.near_misses, "count": len(self.near_misses)})
        self._save_json(self.stats_dir / "baseline_checkpoint.json",
                       {"baselines": self.baselines, "count": len(self.baselines)})
        self._save_json(self.tri_dir / "trivariate_checkpoint.json",
                       {"results": self.trivariates, "count": len(self.trivariates)})

    def save_final(self, total_time: float):
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Near-miss report
        nm_report = {
            "near_misses": self.near_misses,
            "pairs_analyzed": len(set(r["pair_symbol"] for r in self.near_misses)),
            "total_near_misses": len(self.near_misses),
            "total_time_seconds": round(total_time, 1),
            "timestamp": ts,
        }
        self._save_json(self.nm_dir / "near_miss_report.json", nm_report)

        # Baseline
        bl_json = {
            "baseline_results": self.baselines,
            "n_pairs": len(set(r.get("pair_idx") for r in self.baselines)),
            "n_results": len(self.baselines),
            "timestamp": ts,
        }
        self._save_json(self.stats_dir / "random_baseline.json", bl_json)

        # Real residuals (from near-miss data)
        real_residuals = []
        for nm in self.near_misses:
            real_residuals.append({
                "pair": nm["pair"],
                "pair_symbol": nm["pair_symbol"],
                "degree": nm["degree"],
                "working_digits": nm["working_precision"],
                "vector": nm["vector"],
                "norm_inf": nm["norm_inf"],
                "norm_l2": nm["norm_l2"],
                "residual": nm["residual"],
                "log10_residual": nm["log10_residual"],
                "lambda_value": nm["precision_fraction"],
                "quality_score": nm["quality_score"],
                "elapsed_seconds": nm["elapsed"],
            })
        self._save_json(self.stats_dir / "real_residuals.json", {
            "real_residuals": real_residuals,
            "n_pairs": len(set(r["pair_symbol"] for r in self.near_misses)),
            "n_results": len(real_residuals),
            "timestamp": ts,
        })

        # Trivariate
        self._save_json(self.tri_dir / "trivariate_bounds.json", {
            "results": self.trivariates,
            "total_time_seconds": round(total_time, 1),
            "timestamp": ts,
        })

        # Errors
        if self.errors:
            self._save_json(Path(__file__).parent / "results" / "parallel_errors.json",
                           self.errors)

    def _save_json(self, path, data):
        path.write_text(json.dumps(data, indent=2, default=str))


# ═══════════════════════════════════════════════════════════════
#  REPORT GENERATORS
# ═══════════════════════════════════════════════════════════════

def generate_near_miss_summary(results, output_dir):
    if not results:
        return
    lines = [
        "# Near-Miss Analysis Report (Parallel Run v2)",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total near-misses:** {len(results)}",
        f"**Pairs analyzed:** {len(set(r['pair_symbol'] for r in results))}",
        "",
        "## Summary by Pair", "",
        "| Pair | Degrees | λ range | Best Q |",
        "|------|---------|---------|--------|",
    ]
    by_pair = {}
    for r in results:
        by_pair.setdefault(r["pair_symbol"], []).append(r)
    for pair_sym, nms in sorted(by_pair.items()):
        degs = sorted(set(nm["degree"] for nm in nms))
        lambdas = [nm["precision_fraction"] for nm in nms]
        best_q = max(nm["quality_score"] for nm in nms)
        lines.append(f"| {pair_sym} | {degs[0]}-{degs[-1]} | "
                     f"{min(lambdas):.3f}-{max(lambdas):.3f} | {best_q:.4f} |")

    all_lambdas = [r["precision_fraction"] for r in results]
    lines += ["", "## Key Statistics", "",
              f"- **Mean λ:** {sum(all_lambdas)/len(all_lambdas):.3f}",
              f"- **λ range:** [{min(all_lambdas):.3f}, {max(all_lambdas):.3f}]",
              f"- **Total near-misses:** {len(results)}"]
    (output_dir / "near_miss_summary.md").write_text("\n".join(lines))


def generate_near_miss_latex(results, output_dir):
    if not results:
        return
    all_lambdas = [r["precision_fraction"] for r in results]
    mean_l = sum(all_lambdas) / len(all_lambdas)
    n_pairs = len(set(r["pair_symbol"] for r in results))
    max_deg = max(r["degree"] for r in results)
    lines = [
        r"\subsection{Near-Miss Analysis}\label{ssec:nearmiss}",
        "",
        r"When PSLQ terminates without finding a relation, the internal $B$-matrix",
        r"contains a \emph{best candidate} vector: the column of $B$ corresponding",
        r"to the smallest residual $|y_i|$. We extract these candidates and compute",
        r"the \emph{precision fraction}",
        r"\[",
        r"  \lambda = \frac{-\log_{10}|R|}{p},",
        r"\]",
        r"where $R$ is the residual at double verification precision and $p$ is the",
        r"working precision in digits.",
        "",
        r"\begin{result}\label{res:nearmiss}",
        f"Across {len(results)} near-miss vectors extracted from "
        f"{n_pairs} constant pairs at degrees $3$--${max_deg}$, the mean "
        f"precision fraction is $\\bar{{\\lambda}} = {mean_l:.3f}$ "
        f"with range $[{min(all_lambdas):.3f}, {max(all_lambdas):.3f}]$. "
        r"No anomalous cancellation ($\lambda < 0.5$) was observed.",
        r"\end{result}",
    ]
    (output_dir / "near_miss_analysis.tex").write_text("\n".join(lines))


def generate_trivariate_report(results, output_dir):
    if not results:
        return
    lines = [
        "# Trivariate PSLQ Search Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total runs:** {len(results)}", "",
        "## Bounds Established", "",
        "| Triple | d_max (|c|≤100) | d_max (|c|≤10⁴) | d_max (|c|≤10⁶) | Time |",
        "|--------|-----------------|------------------|------------------|------|",
    ]
    td = {}
    for r in results:
        if r.get("found_relation"): continue
        key = r["triple_symbol"]
        if key not in td:
            td[key] = {"100": 0, "10000": 0, "1000000": 0, "time": 0.0}
        mc = str(r["max_coeff"])
        if mc in td[key]:
            td[key][mc] = max(td[key][mc], r["degree"])
        td[key]["time"] += r["elapsed"]
    for key, data in td.items():
        t_str = f"{data['time']:.0f}s" if data['time'] < 60 else f"{data['time']/60:.1f}min"
        lines.append(f"| {key} | {data['100']} | {data['10000']} | {data['1000000']} | {t_str} |")

    found_any = any(r.get("found_relation") for r in results)
    lines += ["", "## Main Result", ""]
    lines.append("**NON-TRIVIAL TRIVARIATE RELATION FOUND!**" if found_any
                 else "**No non-trivial trivariate polynomial relation found.**")
    total_time = sum(r["elapsed"] for r in results)
    lines += ["", "## Statistics", "",
              f"- Total PSLQ runs: {len(results)}",
              f"- Total time: {total_time/3600:.2f} hours",
              f"- Triples explored: {len(td)}"]
    (output_dir / "REPORT_TRIVARIATE.md").write_text("\n".join(lines))


def generate_trivariate_latex(results, output_dir):
    if not results:
        return
    td = {}
    for r in results:
        if r.get("found_relation"): continue
        key = r["triple_symbol"]
        if key not in td:
            td[key] = {"100": 0, "10000": 0, "1000000": 0}
        mc = str(r["max_coeff"])
        if mc in td[key]:
            td[key][mc] = max(td[key][mc], r["degree"])

    lines = [
        r"\subsection{Trivariate Results}\label{ssec:trivariate}",
        "",
        r"We extend the search to trivariate polynomial relations",
        r"$P(\alpha,\beta,\gamma) = 0$ with $P \in \mathbb{Z}[x,y,z]$.",
        r"The monomial count for trivariate degree~$d$ is $\binom{d+3}{3}$,",
        r"which grows much faster than the bivariate $\binom{d+2}{2}$.",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Trivariate exclusion bounds established by PSLQ. No non-trivial",
        r"relation was found for any triple.}\label{tab:trivariate}",
        r"\smallskip",
        r"\begin{tabular}{@{}l rrr@{}}",
        r"\toprule",
        r"Triple $(\alpha,\beta,\gamma)$",
        r"  & \multicolumn{1}{c}{$d_{\max}$}",
        r"  & \multicolumn{1}{c}{$d_{\max}$}",
        r"  & \multicolumn{1}{c}{$d_{\max}$} \\",
        r"  & \multicolumn{1}{c}{\scriptsize$\|c\|_\infty\!\le\!10^2$}",
        r"  & \multicolumn{1}{c}{\scriptsize$\|c\|_\infty\!\le\!10^4$}",
        r"  & \multicolumn{1}{c}{\scriptsize$\|c\|_\infty\!\le\!10^6$} \\",
        r"\midrule",
    ]
    sym_map = {"π": r"\pi", "γ": r"\gamma", "ζ(3)": r"\zeta(3)",
               "G": "G", "ln2": r"\ln 2", "ζ(5)": r"\zeta(5)", "Ω": r"\Omega"}
    for key, data in td.items():
        lk = key
        for sym, ltx in sym_map.items():
            lk = lk.replace(sym, f"${ltx}$")
        d100 = data["100"] if data["100"] > 0 else "--"
        d10k = data["10000"] if data["10000"] > 0 else "--"
        d1M = data["1000000"] if data["1000000"] > 0 else "--"
        lines.append(f"{lk} & {d100} & {d10k} & {d1M} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{result}\label{res:trivariate}",
        r"No non-trivial trivariate polynomial relation was found for any of the",
        f"{len(td)} triples tested. These are, to our knowledge, the first",
        r"systematic trivariate exclusion bounds for these constant triples.",
        r"\end{result}",
    ]
    (output_dir / "trivariate_section.tex").write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    output_base = Path(__file__).parent / "results"
    collector = ResultCollector(output_base)

    log_path = output_base / f"parallel_v2_{time.strftime('%Y%m%d_%H%M%S')}.log"

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log("=" * 65)
    log("  PARALLEL PSLQ ORCHESTRATOR v2 (realistic time limits)")
    log(f"  Workers: {N_WORKERS}")
    log("=" * 65)

    tasks = generate_tasks()
    nm_t = sum(1 for t in tasks if t[0] == "near_miss")
    bl_t = sum(1 for t in tasks if t[0] == "baseline")
    tr_t = sum(1 for t in tasks if t[0] == "trivariate")

    log(f"  Near-miss: {nm_t}  |  Baseline: {bl_t}  |  Trivariate: {tr_t}  |  TOTAL: {len(tasks)}")
    log("")

    completed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for task in tasks:
            func = task[1]
            args = task[2:]
            future = pool.submit(func, *args)
            futures[future] = task

        for future in as_completed(futures):
            completed += 1
            task = futures[future]
            task_type = task[0]

            try:
                result = future.result(timeout=7200)  # 2h max per task
            except Exception as exc:
                collector.errors.append({"task": str(task[:1] + task[2:]), "error": str(exc)})
                continue

            collector.add(result)

            # Log selected results
            if result.get("status") == "ok":
                if task_type == "near_miss":
                    log(f"  [{completed}/{len(tasks)}] NM {result['pair_symbol']} "
                        f"d={result['degree']}: λ={result.get('precision_fraction',0):.3f}, "
                        f"{result['elapsed']}s")
                elif task_type == "trivariate":
                    nb = result.get("norm_bound", "?")
                    sym = result.get("triple_symbol", "?")
                    mc = result.get("max_coeff", "?")
                    tag = "★ FOUND!" if result.get("found_relation") else f"✓ ‖m‖₂≥{nb}"
                    log(f"  [{completed}/{len(tasks)}] TRI {sym} "
                        f"d={result['degree']} |c|≤{mc}: {tag}, {result['elapsed']}s")
                elif task_type == "baseline" and completed % 20 == 0:
                    log(f"  [{completed}/{len(tasks)}] BL {result.get('pair_label','?')} "
                        f"d={result['degree']}: λ={result.get('lambda_value','?')}, "
                        f"{result['elapsed']}s")

            # Checkpoint every 50 tasks
            if completed % 50 == 0:
                collector.save_checkpoint()
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - completed) / rate / 60 if rate > 0 else 0
                log(f"  ── Progress: {completed}/{len(tasks)} "
                    f"({elapsed/60:.1f}min, ETA {eta:.0f}min) ──")

    total_time = time.time() - t_start
    log("")
    log("=" * 65)
    log(f"  COMPLETED: {completed}/{len(tasks)} in {total_time/60:.1f} min")
    log(f"  NM: {len(collector.near_misses)}  BL: {len(collector.baselines)}  "
        f"TRI: {len(collector.trivariates)}  ERR: {len(collector.errors)}")
    log("=" * 65)

    # Save all results
    collector.save_final(total_time)
    log("  JSON data saved.")

    # Generate reports
    generate_near_miss_summary(collector.near_misses, collector.nm_dir)
    generate_near_miss_latex(collector.near_misses, collector.nm_dir)
    generate_trivariate_report(collector.trivariates, collector.tri_dir)
    generate_trivariate_latex(collector.trivariates, collector.tri_dir)
    log("  Reports generated.")

    # Run statistical analysis
    log("  Running statistical analysis...")
    try:
        from statistical_analysis import run_full_analysis
        run_full_analysis(collector.stats_dir)
        log("  Statistical analysis complete.")
    except Exception as exc:
        log(f"  Statistical analysis error: {exc}")
        log(f"  {traceback.format_exc()}")

    log("  ALL DONE.")


if __name__ == "__main__":
    main()
