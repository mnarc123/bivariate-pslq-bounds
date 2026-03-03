"""
Trivariate monomial generator for PSLQ search.

Generates the monomial vector for three variables:
  v = (1, α, β, γ, α², αβ, αγ, β², βγ, γ², ..., γ^d)

with dimension N₃(d) = C(d+3, 3) = (d+1)(d+2)(d+3)/6.

Reference dimensions:
  d=5:  N=56     d=6:  N=84     d=7:  N=120
  d=8:  N=165    d=9:  N=220    d=10: N=286
  d=12: N=455
"""

import mpmath
import math
from typing import List, Tuple, Dict


def count_trivariate_monomials(degree: int) -> int:
    """
    Compute the number of monomials for 3 variables at total degree ≤ d.
    N₃(d) = C(d+3, 3) = (d+1)(d+2)(d+3)/6
    """
    return (degree + 1) * (degree + 2) * (degree + 3) // 6


def generate_trivariate_monomials(
    alpha: mpmath.mpf,
    beta: mpmath.mpf,
    gamma: mpmath.mpf,
    degree: int,
) -> Tuple[List[mpmath.mpf], List[Tuple[int, int, int]], List[str]]:
    """
    Generate all monomials α^i · β^j · γ^k with i+j+k ≤ degree.

    Order: increasing total degree, then lexicographic within each degree.
    The first element is always 1 (i=0, j=0, k=0).

    Returns:
        values: numerical values of the monomials
        exponents: list of (i, j, k) tuples
        labels: human-readable strings
    """
    values = []
    exponents = []
    labels = []

    # Precompute powers
    alpha_powers = [mpmath.mpf(1)]
    beta_powers = [mpmath.mpf(1)]
    gamma_powers = [mpmath.mpf(1)]
    for n in range(1, degree + 1):
        alpha_powers.append(alpha_powers[-1] * alpha)
        beta_powers.append(beta_powers[-1] * beta)
        gamma_powers.append(gamma_powers[-1] * gamma)

    for total_deg in range(degree + 1):
        for i in range(total_deg + 1):
            for j in range(total_deg - i + 1):
                k = total_deg - i - j
                val = alpha_powers[i] * beta_powers[j] * gamma_powers[k]
                values.append(val)
                exponents.append((i, j, k))

                # Human-readable label
                parts = []
                if i > 0:
                    parts.append(f"α^{i}" if i > 1 else "α")
                if j > 0:
                    parts.append(f"β^{j}" if j > 1 else "β")
                if k > 0:
                    parts.append(f"γ^{k}" if k > 1 else "γ")
                if not parts:
                    parts = ["1"]
                labels.append("·".join(parts))

    return values, exponents, labels


def compute_trivariate_precision(
    degree: int,
    max_coeff: int = 100,
    safety_factor: float = 2.0,
) -> Tuple[int, int, int]:
    """
    Compute working and verification precision for trivariate search.

    Returns:
        (n_monomials, working_digits, verification_digits)
    """
    n = count_trivariate_monomials(degree)
    D = max(1, math.ceil(math.log10(max_coeff + 1)))
    working_digits = int(n * D * safety_factor)
    verification_digits = int(working_digits * 1.5)
    return n, working_digits, verification_digits
