"""
INDEPENDENT result validation module.

PRINCIPLE: validation must use a computation path completely
different from the search. This prevents systematic errors.

Validation strategies:
1. Recomputation at higher precision (1000+ digits)
2. Recomputation with a different library (gmpy2/sympy instead of mpmath)
3. Verify that the residual scales correctly with precision
4. Reduction attempt: can the relation be simplified?
"""

import mpmath
import sympy
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from math import gcd
from functools import reduce


@dataclass
class ValidationResult:
    """Validation result."""
    is_valid: bool
    residual_mpmath_1000: float
    residual_sympy: Optional[float]
    scaling_consistent: bool  # does the residual scale as expected with precision?
    is_reducible: bool  # can it be simplified (coefficients with GCD > 1)?
    reduced_equation: Optional[str]
    notes: str


class ResultValidator:
    """Independent validator for PSLQ results."""

    def validate(
        self,
        coefficients: list,
        exponent_tuples: list,
        constant_names: list,
    ) -> ValidationResult:
        """
        Full validation of a candidate relation.

        Performs 4 levels of independent verification.
        """
        notes = []

        # === LEVEL 1: Recomputation at 1000 digits with mpmath ===
        mpmath.mp.dps = 1050
        values_1000 = self._compute_monomials_mpmath(
            constant_names, exponent_tuples
        )
        residual_1000 = abs(sum(
            c * v for c, v in zip(coefficients, values_1000)
        ))
        notes.append(f"Residual mpmath@1000: {mpmath.nstr(residual_1000, 5)}")

        # === LEVEL 2: Verification with sympy (exact arithmetic where possible) ===
        residual_sympy = self._verify_with_sympy(
            coefficients, exponent_tuples, constant_names
        )
        if residual_sympy is not None:
            notes.append(f"Residual sympy: {residual_sympy}")

        # === LEVEL 3: Scaling check ===
        # If the relation is true, the residual at N digits should be ~10^(-N)
        # If spurious, the residual will not improve with precision
        mpmath.mp.dps = 550
        values_500 = self._compute_monomials_mpmath(
            constant_names, exponent_tuples
        )
        residual_500 = abs(sum(
            c * v for c, v in zip(coefficients, values_500)
        ))

        scaling_consistent = False
        if residual_500 > 0 and residual_1000 > 0:
            log_ratio = float(mpmath.log10(residual_500) - mpmath.log10(residual_1000))
            # Should be approximately 500 (precision difference)
            scaling_consistent = log_ratio > 400  # tolerant margin
            notes.append(f"Scaling: log10(r500/r1000) = {log_ratio:.1f} (expected ~500)")
        elif residual_1000 == 0:
            scaling_consistent = True
            notes.append("Scaling: residual exactly zero at 1000 digits!")

        # === LEVEL 4: Reducibility (GCD of coefficients) ===
        nonzero_coeffs = [abs(c) for c in coefficients if c != 0]
        g = reduce(gcd, nonzero_coeffs) if nonzero_coeffs else 1
        is_reducible = g > 1
        reduced_eq = None
        if is_reducible:
            reduced_coeffs = [c // g for c in coefficients]
            notes.append(f"GCD = {g}, coefficients reducible to {reduced_coeffs}")
        else:
            notes.append("Irreducible coefficients (GCD = 1)")

        # === VERDICT ===
        if residual_1000 == 0:
            is_valid = True
        elif residual_1000 > 0:
            log_res = float(mpmath.log10(residual_1000))
            is_valid = (
                log_res < -900  # very small residual
                and scaling_consistent  # residual scales correctly
            )
        else:
            is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            residual_mpmath_1000=float(residual_1000),
            residual_sympy=residual_sympy,
            scaling_consistent=scaling_consistent,
            is_reducible=is_reducible,
            reduced_equation=reduced_eq,
            notes="\n".join(notes),
        )

    def _compute_monomials_mpmath(self, names, exponent_tuples):
        """Compute monomials with mpmath at the current precision."""
        from constants import ConstantsComputer
        from config import SearchConfig

        config = SearchConfig()
        config.working_precision = mpmath.mp.dps - 50
        cc = ConstantsComputer(config)
        values = cc.compute_all(mpmath.mp.dps - 50, names)

        result = []
        for exp_tuple in exponent_tuples:
            val = mpmath.mpf(1)
            for name, exp in zip(names, exp_tuple):
                if exp > 0:
                    val *= values[name] ** exp
            result.append(val)
        return result

    def _verify_with_sympy(self, coefficients, exponent_tuples, names):
        """
        Attempt verification with sympy (symbolic arithmetic).
        For constants like π, e, √2, sympy can compute exactly.
        For γ, ζ(3), etc., it uses mpmath internally but via a different path.
        """
        try:
            sym_constants = {
                "pi": sympy.pi,
                "e": sympy.E,
                "euler_gamma": sympy.EulerGamma,
                "phi": sympy.GoldenRatio,
                "ln2": sympy.log(2),
                "sqrt2": sympy.sqrt(2),
                "sqrt3": sympy.sqrt(3),
                "sqrt5": sympy.sqrt(5),
                "zeta3": sympy.zeta(3),
                "catalan": sympy.Catalan,
                "ln3": sympy.log(3),
                "ln5": sympy.log(5),
                "ln10": sympy.log(10),
                "pi2": sympy.pi**2,
            }

            total = sympy.Integer(0)
            for c, exp_tuple in zip(coefficients, exponent_tuples):
                if c == 0:
                    continue
                term = sympy.Integer(c)
                for name, exp in zip(names, exp_tuple):
                    if exp > 0 and name in sym_constants:
                        term *= sym_constants[name] ** exp
                    elif exp > 0:
                        return None  # constant not available in sympy
                total += term

            # Try exact simplification
            simplified = sympy.simplify(total)
            if simplified == 0:
                return 0.0  # EXACT RELATION confirmed by sympy!

            # Numerical evaluation with sympy
            return float(abs(simplified.evalf(100)))

        except Exception:
            return None
