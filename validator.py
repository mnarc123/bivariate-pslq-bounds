"""
Modulo di validazione INDIPENDENTE dei risultati.

PRINCIPIO: la validazione deve usare un percorso di calcolo completamente
diverso dalla ricerca. Questo previene errori sistematici.

Strategie di validazione:
1. Ricalcolo a precisione maggiore (1000+ cifre)
2. Ricalcolo con libreria diversa (gmpy2/sympy invece di mpmath)
3. Verifica che il residuo scali correttamente con la precisione
4. Tentativo di riduzione: la relazione può essere semplificata?
"""

import mpmath
import sympy
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from math import gcd
from functools import reduce


@dataclass
class ValidationResult:
    """Risultato della validazione."""
    is_valid: bool
    residual_mpmath_1000: float
    residual_sympy: Optional[float]
    scaling_consistent: bool  # il residuo scala come atteso con la precisione?
    is_reducible: bool  # può essere semplificata (coefficienti con GCD > 1)?
    reduced_equation: Optional[str]
    notes: str


class ResultValidator:
    """Validatore indipendente dei risultati PSLQ."""

    def validate(
        self,
        coefficients: list,
        exponent_tuples: list,
        constant_names: list,
    ) -> ValidationResult:
        """
        Validazione completa di una relazione candidata.

        Esegue 4 livelli di verifica indipendente.
        """
        notes = []

        # === LIVELLO 1: Ricalcolo a 1000 cifre con mpmath ===
        mpmath.mp.dps = 1050
        values_1000 = self._compute_monomials_mpmath(
            constant_names, exponent_tuples
        )
        residual_1000 = abs(sum(
            c * v for c, v in zip(coefficients, values_1000)
        ))
        notes.append(f"Residuo mpmath@1000: {mpmath.nstr(residual_1000, 5)}")

        # === LIVELLO 2: Verifica con sympy (aritmetica esatta dove possibile) ===
        residual_sympy = self._verify_with_sympy(
            coefficients, exponent_tuples, constant_names
        )
        if residual_sympy is not None:
            notes.append(f"Residuo sympy: {residual_sympy}")

        # === LIVELLO 3: Scaling check ===
        # Se la relazione è vera, il residuo a N cifre dovrebbe essere ~10^(-N)
        # Se è spuria, il residuo non migliorerà con la precisione
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
            # Dovrebbe essere circa 500 (differenza di precisione)
            scaling_consistent = log_ratio > 400  # margine tollerante
            notes.append(f"Scaling: log10(r500/r1000) = {log_ratio:.1f} (atteso ~500)")
        elif residual_1000 == 0:
            scaling_consistent = True
            notes.append("Scaling: residuo esattamente zero a 1000 cifre!")

        # === LIVELLO 4: Riducibilità (GCD dei coefficienti) ===
        nonzero_coeffs = [abs(c) for c in coefficients if c != 0]
        g = reduce(gcd, nonzero_coeffs) if nonzero_coeffs else 1
        is_reducible = g > 1
        reduced_eq = None
        if is_reducible:
            reduced_coeffs = [c // g for c in coefficients]
            notes.append(f"GCD = {g}, coefficienti riducibili a {reduced_coeffs}")
        else:
            notes.append("Coefficienti irriducibili (GCD = 1)")

        # === VERDETTO ===
        if residual_1000 == 0:
            is_valid = True
        elif residual_1000 > 0:
            log_res = float(mpmath.log10(residual_1000))
            is_valid = (
                log_res < -900  # residuo molto piccolo
                and scaling_consistent  # il residuo scala correttamente
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
        """Calcola monomiali con mpmath alla precisione corrente."""
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
        Tenta verifica con sympy (aritmetica simbolica).
        Per costanti come π, e, √2, sympy può calcolare in modo esatto.
        Per γ, ζ(3), ecc., usa mpmath internamente ma con percorso diverso.
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
                        return None  # costante non disponibile in sympy
                total += term

            # Prova semplificazione esatta
            simplified = sympy.simplify(total)
            if simplified == 0:
                return 0.0  # RELAZIONE ESATTA confermata da sympy!

            # Valutazione numerica con sympy
            return float(abs(simplified.evalf(100)))

        except Exception:
            return None
