"""
Multi-level precision management.

STRATEGY: For each pair and degree, compute the minimum required precision,
then add a safety margin. Uses a "staircase" strategy:
first test at moderate precision, then increase if needed.
"""

import math
import mpmath
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PrecisionPlan:
    """Precision plan for a PSLQ search."""
    n_monomials: int
    max_coeff_digits: int       # D: log10 of the max coefficient norm
    working_digits: int         # working digits for PSLQ
    verification_digits: int    # digits for independent verification
    is_feasible: bool           # True if computation is feasible in reasonable time
    estimated_hours: float      # estimated time in hours


# Global calibration parameter — calibrated from real benchmark
# on i7-12700F with mpmath PSLQ, 2026-02-18.
# Model: T(hours) ≈ k · N^alpha · P^beta
# where N = n_monomials, P = working_digits
# Benchmark (π,e): degree 10-20, error <1% on all data points.
CALIBRATION_K = 2.575159e-10
CALIBRATION_ALPHA = 1.362
CALIBRATION_BETA = 1.5


def compute_precision_plan(
    degree: int,
    max_coeff: int = 100,
    safety_factor: float = 2.0,
    max_hours: float = 48.0,
) -> PrecisionPlan:
    """
    Compute the optimal precision plan for a pair at a given degree.

    Args:
        degree: maximum polynomial degree
        max_coeff: maximum coefficient norm
        safety_factor: safety multiplier on precision
        max_hours: maximum allowed time in hours

    Returns:
        PrecisionPlan with all computed parameters
    """
    n_monomials = (degree + 1) * (degree + 2) // 2
    max_coeff_digits = max(1, math.ceil(math.log10(max_coeff + 1)))

    # Minimum precision: N × D (Bailey's rule)
    min_digits = n_monomials * max_coeff_digits

    # Working precision with safety margin
    working_digits = int(min_digits * safety_factor)

    # Verification precision: at least 1.5x the working precision
    verification_digits = int(working_digits * 1.5)

    # Time estimate
    estimated_hours = (
        CALIBRATION_K
        * (n_monomials ** CALIBRATION_ALPHA)
        * (working_digits ** CALIBRATION_BETA)
    )

    is_feasible = estimated_hours <= max_hours

    return PrecisionPlan(
        n_monomials=n_monomials,
        max_coeff_digits=max_coeff_digits,
        working_digits=working_digits,
        verification_digits=verification_digits,
        is_feasible=is_feasible,
        estimated_hours=estimated_hours,
    )


def find_max_feasible_degree(
    max_coeff: int = 100,
    max_hours: float = 24.0,
    safety_factor: float = 2.0,
) -> int:
    """
    Find the maximum feasible degree for a bivariate pair.
    Uses binary search on the degree.
    """
    lo, hi = 8, 200
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        plan = compute_precision_plan(mid, max_coeff, safety_factor, max_hours)
        if plan.is_feasible:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def calibrate_from_benchmarks(benchmark_data: list):
    """
    Calibrate parameters k, alpha, beta from benchmark data.

    benchmark_data: list of (n_monomials, working_digits, elapsed_seconds)
    """
    global CALIBRATION_K, CALIBRATION_ALPHA, CALIBRATION_BETA

    if len(benchmark_data) < 2:
        # With a single data point, adjust only k while keeping alpha and beta
        if benchmark_data:
            n, p, t = benchmark_data[0]
            t_hours = t / 3600.0
            CALIBRATION_K = t_hours / (n ** CALIBRATION_ALPHA * p ** CALIBRATION_BETA)
        return

    # With multiple points, log-linear fit: log(T) = log(k) + alpha*log(N) + beta*log(P)
    # Simplification: if P is constant across benchmarks, fit only on N
    import numpy as np

    log_n = np.array([math.log(d[0]) for d in benchmark_data])
    log_p = np.array([math.log(d[1]) for d in benchmark_data])
    log_t = np.array([math.log(d[2] / 3600.0) for d in benchmark_data])

    # Fit: log_t = c0 + c1*log_n + c2*log_p
    A = np.column_stack([np.ones(len(log_n)), log_n, log_p])
    try:
        result = np.linalg.lstsq(A, log_t, rcond=None)
        c0, c1, c2 = result[0]
        CALIBRATION_K = math.exp(c0)
        CALIBRATION_ALPHA = c1
        CALIBRATION_BETA = c2
    except Exception:
        # Fallback: use only the largest data point
        n, p, t = max(benchmark_data, key=lambda x: x[0])
        t_hours = t / 3600.0
        CALIBRATION_K = t_hours / (n ** CALIBRATION_ALPHA * p ** CALIBRATION_BETA)
