"""
Bridge Equation project configuration.
All search parameters are centralized here.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

@dataclass
class SearchConfig:
    """PSLQ search parameters."""

    # === PRECISION ===
    # Decimal digit precision for constant computation.
    # RULE: To find a relation with maximum coefficient norm D
    # in a vector of dimension N, at least N*log10(D) digits are needed.
    # With N=50 and D=1000, ~150 digits are required. We use a 3x safety margin.
    working_precision: int = 500  # working decimal digits
    verification_precision: int = 1000  # digits for independent verification

    # === CONSTANTS TO INCLUDE ===
    # Level 1: most fundamental constants (priority search)
    constants_level_1: List[str] = field(default_factory=lambda: [
        "pi", "e", "euler_gamma", "phi",  # golden ratio
        "ln2", "sqrt2",
    ])

    # Level 2: important analytic constants
    constants_level_2: List[str] = field(default_factory=lambda: [
        "zeta3",          # Apéry's constant ζ(3)
        "catalan",        # Catalan's constant G
        "sqrt3", "sqrt5",
        "ln3", "ln5", "ln10",
        "pi2",            # π² (treated as independent constant for efficiency)
    ])

    # Level 3: more exotic constants (secondary search)
    constants_level_3: List[str] = field(default_factory=lambda: [
        "zeta5",          # ζ(5)
        "khinchin",       # Khinchin's constant K₀
        "glaisher",       # Glaisher–Kinkelin constant A
        "omega",          # Omega constant Ω (solution of x*e^x = 1)
        "feigenbaum_d",   # First Feigenbaum constant δ
        "feigenbaum_a",   # Second Feigenbaum constant α
        "meissel_mertens", # Meissel–Mertens constant M
        "twin_prime",     # Twin prime constant C₂
    ])

    # === SEARCH DEGREES ===
    # Maximum polynomial degree in the constants.
    # Degree 1 = linear relations (well explored, unlikely to find new ones)
    # Degree 2 = quadratic relations (poorly explored for mixed combinations)
    # Degree 3 = cubic relations (nearly uncharted territory)
    # Degree 4+ = computationally expensive, only for small subsets
    max_degree: int = 4

    # === COEFFICIENTS ===
    # Integer coefficient norm threshold.
    # Relations with large coefficients are less mathematically "interesting".
    # Bailey typically uses a threshold of ~10^6 for significant results.
    max_coefficient_norm: int = 10**6

    # === VALIDITY CRITERIA ===
    # A relation is considered a "candidate" if the residual is < 10^(-threshold)
    # when computed at precision=working_precision.
    # EMPIRICAL RULE (Bailey): a "drop" of 20+ orders of magnitude in the residual
    # relative to background noise almost certainly indicates a true relation.
    residual_threshold_log10: int = -50  # |residual| < 10^(-50) at 500 digits

    # For verification: recomputation at double precision
    verification_residual_threshold_log10: int = -100  # |residual| < 10^(-100) at 1000 digits

    # === SEARCH STRATEGY ===
    # Maximum number of constants per single PSLQ search
    # (complexity is polynomial in N but practical runtime grows rapidly)
    max_constants_per_search: int = 5

    # Maximum number of monomials in the PSLQ vector
    # (beyond ~100 it becomes very slow even at 500 digits)
    max_monomials_per_vector: int = 80

    # === PATHS ===
    project_root: Path = Path.home() / "bridge_equation"
    results_dir: Path = field(default_factory=lambda: Path.home() / "bridge_equation" / "results")
    cache_dir: Path = field(default_factory=lambda: Path.home() / "bridge_equation" / "cache")
    log_dir: Path = field(default_factory=lambda: Path.home() / "bridge_equation" / "results" / "logs")

    def __post_init__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "candidates").mkdir(exist_ok=True)
        (self.results_dir / "verified").mkdir(exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
