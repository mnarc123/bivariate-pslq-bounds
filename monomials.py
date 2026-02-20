"""
Multivariate monomial generator for PSLQ search.

Given a list of constants [c₁, c₂, ..., cₘ] and a maximum degree d,
generates all monomials c₁^a₁ · c₂^a₂ · ... · cₘ^aₘ with Σaᵢ ≤ d.

NOTE: inclusion of the constant term "1" (all exponents zero)
is ESSENTIAL — without it, affine relations (with a constant term) cannot be found.
"""

from itertools import combinations_with_replacement
from typing import List, Tuple, Dict
import mpmath


def generate_exponent_tuples(n_vars: int, max_degree: int) -> List[Tuple[int, ...]]:
    """
    Generate all exponent tuples (a₁, ..., aₙ) with Σaᵢ ≤ max_degree.

    Uses the "stars and bars" method for efficient enumeration.

    Returns:
        List of tuples, each of length n_vars.
        The first tuple is always (0, 0, ..., 0) — the constant term "1".
    """
    tuples = []
    for total_degree in range(max_degree + 1):
        # Generate all partitions of total_degree into n_vars non-negative parts
        for combo in _partitions(total_degree, n_vars):
            tuples.append(combo)
    return tuples


def _partitions(total: int, n_parts: int) -> List[Tuple[int, ...]]:
    """Generate all weak compositions of 'total' into 'n_parts' parts."""
    if n_parts == 1:
        return [(total,)]
    result = []
    for i in range(total + 1):
        for rest in _partitions(total - i, n_parts - 1):
            result.append((i,) + rest)
    return result


def compute_monomial_values(
    constant_values: Dict[str, mpmath.mpf],
    constant_names: List[str],
    max_degree: int,
    max_monomials: int = 80
) -> Tuple[List[mpmath.mpf], List[Tuple[int, ...]], List[str]]:
    """
    Compute the numerical values of all monomials.

    Args:
        constant_values: dictionary name -> mpf value
        constant_names: names of constants to use (fixed order)
        max_degree: maximum degree
        max_monomials: maximum number of monomials (to control PSLQ runtime)

    Returns:
        - values: list of mpf monomial values
        - exponents: list of corresponding exponent tuples
        - labels: list of human-readable strings (e.g. "π²·e·γ")

    NOTE: The first value is always 1 (constant term).
    """
    n_vars = len(constant_names)
    all_exponents = generate_exponent_tuples(n_vars, max_degree)

    if len(all_exponents) > max_monomials:
        # Strategy: prioritize low degrees and "mixed" monomials
        # (products of different constants, which are more likely unexplored)
        all_exponents = _prioritize_exponents(all_exponents, max_monomials)

    values = []
    labels = []
    for exp_tuple in all_exponents:
        val = mpmath.mpf(1)
        label_parts = []
        for i, (name, exp) in enumerate(zip(constant_names, exp_tuple)):
            if exp > 0:
                val *= constant_values[name] ** exp
                if exp == 1:
                    label_parts.append(name)
                else:
                    label_parts.append(f"{name}^{exp}")
        if not label_parts:
            label_parts = ["1"]
        values.append(val)
        labels.append("·".join(label_parts))

    return values, all_exponents, labels


def _prioritize_exponents(
    exponents: List[Tuple[int, ...]], max_count: int
) -> List[Tuple[int, ...]]:
    """
    Prioritize monomials: low degree first, then "mixed" monomials
    (involving more variables), then high degree.
    """
    def sort_key(exp_tuple):
        total_deg = sum(exp_tuple)
        n_nonzero = sum(1 for e in exp_tuple if e > 0)
        max_single = max(exp_tuple) if exp_tuple else 0
        # Priority: low degree, many variables involved, low exponents
        return (total_deg, -n_nonzero, max_single)

    sorted_exp = sorted(exponents, key=sort_key)
    return sorted_exp[:max_count]


def count_monomials(n_vars: int, max_degree: int) -> int:
    """
    Compute the total number of monomials C(n_vars + max_degree, max_degree).
    Useful for estimating the search space size.
    """
    from math import comb
    return comb(n_vars + max_degree, max_degree)
