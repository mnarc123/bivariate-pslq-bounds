#!/usr/bin/env python3
"""
BRIDGE EQUATION — Phase 2: High-Degree Deep Search

Runs PSLQ on pairs of transcendental constants up to the maximum
degree allowed by the time budget, with checkpoint/resume.

Usage:
    python3 run_deep_search_v2.py                          # default 24h per pair
    python3 run_deep_search_v2.py --max-hours-per-pair 48  # 48h per pair
    python3 run_deep_search_v2.py --pair pi+e              # single pair only
    python3 run_deep_search_v2.py --estimate               # time estimates only
    python3 run_deep_search_v2.py --benchmark              # calibrate time estimates
"""

import argparse
import sys
import time
import mpmath
from config import SearchConfig
from deep_engine import DeepSearchEngine
from precision_manager import (
    compute_precision_plan, find_max_feasible_degree,
    calibrate_from_benchmarks, CALIBRATION_K, CALIBRATION_ALPHA, CALIBRATION_BETA,
)


def run_benchmark():
    """
    Run a PSLQ benchmark on (π,e) at increasing degrees to calibrate
    the time estimation model.
    """
    print("\n=== PSLQ BENCHMARK on (π, e) ===\n")
    print(f"{'Degree':>8} {'N mono':>8} {'Digits':>8} {'Time':>12} {'Result':>12}")
    print("-" * 54)

    benchmark_data = []

    for d in [10, 12, 15, 18, 20]:
        n = (d + 1) * (d + 2) // 2
        # Precision: N * 2 * safety=2 for |c|≤100
        digits = n * 2 * 2
        mpmath.mp.dps = digits + 50

        alpha = mpmath.pi
        beta = mpmath.e

        # Generate monomials
        ap = [mpmath.mpf(1)]
        bp = [mpmath.mpf(1)]
        for k in range(1, d + 1):
            ap.append(ap[-1] * alpha)
            bp.append(bp[-1] * beta)

        vals = []
        for td in range(d + 1):
            for i in range(td + 1):
                j = td - i
                vals.append(ap[i] * bp[j])

        t0 = time.time()
        r = mpmath.pslq(vals, maxcoeff=100)
        t1 = time.time()
        elapsed = t1 - t0

        result_str = "None" if r is None else str(r)[:30]
        time_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"
        print(f"{d:>8} {n:>8} {digits:>8} {time_str:>12} {result_str:>12}")

        benchmark_data.append((n, digits, elapsed))

    # Calibrate
    print("\n=== CALIBRATION ===")
    print(f"  Before: k={CALIBRATION_K:.2e}, α={CALIBRATION_ALPHA:.2f}, β={CALIBRATION_BETA:.2f}")
    calibrate_from_benchmarks(benchmark_data)
    from precision_manager import CALIBRATION_K as K, CALIBRATION_ALPHA as A, CALIBRATION_BETA as B
    print(f"  After:  k={K:.2e}, α={A:.2f}, β={B:.2f}")

    # Show calibrated estimates
    print("\n=== CALIBRATED ESTIMATES ===\n")
    print(f"{'Degree':>8} {'N mono':>10} {'Digits':>8} {'Est. time':>16}")
    print("-" * 46)
    for d in [10, 15, 20, 25, 30, 40, 50]:
        plan = compute_precision_plan(d, max_coeff=100)
        if plan.estimated_hours < 1/60:
            time_str = f"{plan.estimated_hours*3600:.1f}s"
        elif plan.estimated_hours < 1:
            time_str = f"{plan.estimated_hours*60:.1f} min"
        elif plan.estimated_hours < 24:
            time_str = f"{plan.estimated_hours:.1f}h"
        else:
            time_str = f"{plan.estimated_hours/24:.1f} days"
        print(f"{d:>8} {plan.n_monomials:>10} {plan.working_digits:>8} {time_str:>16}")

    return benchmark_data


def estimate_only():
    """Print an estimate of achievable degrees and times."""
    print("\n=== ESTIMATED ACHIEVABLE DEGREES ===\n")
    print(f"{'Budget (hours)':>14} {'|c|≤100':>12} {'|c|≤10⁴':>12} {'|c|≤10⁶':>12}")
    print("-" * 54)
    for hours in [1, 4, 12, 24, 48, 96, 168]:
        d100 = find_max_feasible_degree(100, hours)
        d10k = find_max_feasible_degree(10000, hours)
        d1M = find_max_feasible_degree(10**6, hours)
        print(f"{hours:>12}h  degree {d100:>4}  degree {d10k:>4}  degree {d1M:>4}")

    print("\n=== DETAIL FOR (π, e), |c|≤100 ===\n")
    print(f"{'Degree':>8} {'Monomials':>10} {'Digits':>8} {'Est. time':>16}")
    print("-" * 46)
    for d in [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
        plan = compute_precision_plan(d, max_coeff=100)
        if plan.estimated_hours < 1/60:
            time_str = f"{plan.estimated_hours*3600:.1f}s"
        elif plan.estimated_hours < 1:
            time_str = f"{plan.estimated_hours*60:.1f} min"
        elif plan.estimated_hours < 24:
            time_str = f"{plan.estimated_hours:.1f}h"
        else:
            time_str = f"{plan.estimated_hours/24:.1f} days"
        feas = "✓" if plan.is_feasible else "✗"
        print(f"{d:>8} {plan.n_monomials:>10} {plan.working_digits:>8} {time_str:>16}  {feas}")


def main():
    parser = argparse.ArgumentParser(
        description="Bridge Equation — Phase 2: Deep Search"
    )
    parser.add_argument(
        "--max-hours-per-pair", type=float, default=24.0,
        help="Maximum hours per pair of constants (default: 24)"
    )
    parser.add_argument(
        "--pair", type=str, default=None,
        help="Search only this pair (e.g. 'pi+e', 'pi+euler_gamma')"
    )
    parser.add_argument(
        "--estimate", action="store_true",
        help="Print time estimates only, do not execute"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark to calibrate time estimates"
    )
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        return

    if args.estimate:
        estimate_only()
        return

    config = SearchConfig()
    engine = DeepSearchEngine(config, max_hours_per_pair=args.max_hours_per_pair)

    t_start = time.time()
    engine.run(pair_filter=args.pair)
    t_total = time.time() - t_start

    hours = t_total / 3600
    print(f"\nSearch completed in {hours:.2f} hours ({t_total:.0f}s).")


if __name__ == "__main__":
    main()
