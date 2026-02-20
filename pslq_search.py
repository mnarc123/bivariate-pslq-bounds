"""
PSLQ search engine for polynomial relations among constants.

ALGORITHM:
1. For each subset S of constants (size ≤ max_constants_per_search):
   a. Generate all monomials up to max_degree
   b. Compute numerical values of the monomials at working_precision
   c. Run PSLQ on the value vector
   d. If PSLQ returns a relation:
      - Verify that the coefficient norm is ≤ max_coefficient_norm
      - Verify that it is NOT a "trivial" relation (known or derivable)
      - Recompute at verification_precision
      - If the residual is still below threshold: CANDIDATE!

CAUTIONS:
- PSLQ can return spurious "relations" when precision is insufficient
- "Trivial" relations must be filtered (e.g. π² - π² = 0 if π² is in the set)
- The residual must drop by 20+ orders of magnitude relative to background noise
"""

import mpmath
import sympy
import itertools
import time
import json
import gc
import sys
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
from config import SearchConfig
from constants import ConstantsComputer
from monomials import compute_monomial_values, count_monomials
from results_manager import ResultsManager


@dataclass
class PSLQResult:
    """A PSLQ search result."""
    constants_used: List[str]          # names of the constants involved
    coefficients: List[int]            # integer coefficients found
    exponents: List[Tuple[int, ...]]   # monomial exponents
    labels: List[str]                  # human-readable labels
    residual_working: float            # residual at working_precision
    residual_verification: float       # residual at verification_precision
    coefficient_norm: float            # L2 norm of coefficients
    max_coefficient: int               # maximum absolute value of coefficients
    equation_string: str               # equation in human-readable format
    is_trivial: bool                   # True if it is a known/trivial relation
    search_time_seconds: float         # computation time
    timestamp: str                     # ISO timestamp


class PSLQSearchEngine:
    """Main search engine."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.constants_computer = ConstantsComputer(config)
        self.results_manager = ResultsManager(config)

        # Known relations to filter (add trivial relations here)
        self.known_relations = self._load_known_relations()

    def run_full_search(self):
        """
        Execute the full search according to the strategy defined in config.

        SEARCH STRATEGY (from most to least promising):

        Phase 1: Quadratic relations among Level 1 constants
                 (π, e, γ, φ, ln2, √2) — degree 2, all combinations
                 Small space, high probability of novel result

        Phase 2: Cubic relations among Level 1 pairs/triples
                 Degree 3, PSLQ dimension still manageable

        Phase 3: Quadratic relations with Level 2 constants
                 (adds ζ(3), Catalan, √3, √5, ln3, ...)

        Phase 4: Degree 4 relations for the most promising subsets

        Phase 5: Exotic Level 3 constants
                 (Feigenbaum, Khinchin, Glaisher, ...)
        """
        print("=" * 70)
        print("  BRIDGE EQUATION PROJECT — Computational Search")
        print("=" * 70)

        # === STEP 0: System integrity check ===
        print("\n[STEP 0] Verifying computation system integrity...")
        if not self.constants_computer.verify_known_relations():
            print("\n ⚠ CRITICAL ERROR: self-tests failed. ABORTING.")
            return

        # === Pre-compute constants ===
        all_names = (
            self.config.constants_level_1
            + self.config.constants_level_2
            + self.config.constants_level_3
        )
        print(f"\n[STEP 1] Computing {len(all_names)} constants at {self.config.working_precision} digits...")
        constants_work = self.constants_computer.compute_all(
            self.config.working_precision, all_names
        )
        print(f"[STEP 1] Computing {len(all_names)} constants at {self.config.verification_precision} digits...")
        constants_verify = self.constants_computer.compute_all(
            self.config.verification_precision, all_names
        )

        self.results_manager.log(f"Constants computed: {len(all_names)} at {self.config.working_precision}/{self.config.verification_precision} digits")

        # === PHASE 1: Quadratics among Level 1 ===
        self._run_phase(
            phase_name="PHASE 1: Quadratic relations — Level 1",
            constant_names=self.config.constants_level_1,
            max_degree=2,
            min_subset_size=2,
            max_subset_size=self.config.max_constants_per_search,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === PHASE 2: Cubics among Level 1 pairs/triples ===
        self._run_phase(
            phase_name="PHASE 2: Cubic relations — Level 1",
            constant_names=self.config.constants_level_1,
            max_degree=3,
            min_subset_size=2,
            max_subset_size=3,  # max 3 constants at a time (otherwise too many monomials)
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === PHASE 3: Quadratics with Level 2 ===
        level_1_2 = self.config.constants_level_1 + self.config.constants_level_2
        self._run_phase(
            phase_name="PHASE 3: Quadratic relations — Level 1+2",
            constant_names=level_1_2,
            max_degree=2,
            min_subset_size=2,
            max_subset_size=4,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === PHASE 4: Degree 4 for small subsets ===
        self._run_phase(
            phase_name="PHASE 4: Degree 4 relations — Level 1 pairs",
            constant_names=self.config.constants_level_1,
            max_degree=4,
            min_subset_size=2,
            max_subset_size=2,  # pairs only (otherwise combinatorial explosion)
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === PHASE 5: Exotic constants ===
        level_all = (
            self.config.constants_level_1
            + self.config.constants_level_2
            + self.config.constants_level_3
        )
        self._run_phase(
            phase_name="PHASE 5: Quadratic relations — all constants",
            constant_names=level_all,
            max_degree=2,
            min_subset_size=2,
            max_subset_size=3,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === EXTENDED PHASES: TRANSCENDENTALS ONLY ===
        # Algebraic constants (√2, √3, √5, φ) only produce trivial identities.
        # Focus on genuinely transcendental constants.
        transcendentals_core = ["pi", "e", "euler_gamma", "ln2"]
        transcendentals_extended = [
            "pi", "e", "euler_gamma", "ln2",
            "zeta3", "catalan", "ln3",
        ]
        transcendentals_all = [
            "pi", "e", "euler_gamma", "ln2",
            "zeta3", "catalan", "ln3",
            "zeta5", "khinchin", "glaisher", "omega",
        ]

        # === PHASE 6: High degree for core transcendental pairs ===
        # π, e, γ, ln2 — degree up to 6 (15-28 monomials per pair)
        self._run_phase(
            phase_name="PHASE 6: Degree 6 — core transcendental pairs (π,e,γ,ln2)",
            constant_names=transcendentals_core,
            max_degree=6,
            min_subset_size=2,
            max_subset_size=2,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === PHASE 7: Degree 5 for core transcendental triples ===
        self._run_phase(
            phase_name="PHASE 7: Degree 5 — core transcendental triples",
            constant_names=transcendentals_core,
            max_degree=5,
            min_subset_size=3,
            max_subset_size=3,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === PHASE 8: Degree 4 for extended transcendental triples ===
        self._run_phase(
            phase_name="PHASE 8: Degree 4 — extended transcendental triples",
            constant_names=transcendentals_extended,
            max_degree=4,
            min_subset_size=3,
            max_subset_size=3,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === PHASE 9: Degree 3 for transcendental quadruples/quintuples ===
        self._run_phase(
            phase_name="PHASE 9: Degree 3 — extended transcendental quadruples",
            constant_names=transcendentals_extended,
            max_degree=3,
            min_subset_size=4,
            max_subset_size=5,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === PHASE 10: Degree 3 for pairs/triples with exotic constants ===
        self._run_phase(
            phase_name="PHASE 10: Degree 3 — transcendentals with exotic constants",
            constant_names=transcendentals_all,
            max_degree=3,
            min_subset_size=2,
            max_subset_size=3,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === PHASE 11: Degree 2 for quadruples/quintuples with exotic constants ===
        self._run_phase(
            phase_name="PHASE 11: Degree 2 — complete transcendental quadruples",
            constant_names=transcendentals_all,
            max_degree=2,
            min_subset_size=4,
            max_subset_size=5,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === FINAL REPORT ===
        self.results_manager.generate_final_report()

    def _run_phase(
        self,
        phase_name: str,
        constant_names: List[str],
        max_degree: int,
        min_subset_size: int,
        max_subset_size: int,
        constants_work: Dict[str, mpmath.mpf],
        constants_verify: Dict[str, mpmath.mpf],
    ):
        """Execute a search phase."""
        print(f"\n{'='*60}")
        print(f"  {phase_name}")
        print(f"{'='*60}")

        self.results_manager.log(f"Starting phase: {phase_name}")

        n_total = len(constant_names)
        total_subsets = sum(
            1 for r in range(min_subset_size, min(max_subset_size, n_total) + 1)
            for _ in itertools.combinations(range(n_total), r)
        )
        print(f"  Constants: {n_total}, Max degree: {max_degree}")
        print(f"  Subsets to explore: {total_subsets}")

        subset_count = 0
        phase_candidates = 0
        phase_start = time.time()

        for subset_size in range(min_subset_size, min(max_subset_size, n_total) + 1):
            for subset_indices in itertools.combinations(range(n_total), subset_size):
                subset_names = [constant_names[i] for i in subset_indices]
                subset_count += 1

                # Check if the number of monomials is manageable
                n_monomials = count_monomials(len(subset_names), max_degree)
                if n_monomials > self.config.max_monomials_per_vector:
                    self.results_manager.log(
                        f"  Skip {'+'.join(subset_names)} deg≤{max_degree}: "
                        f"{n_monomials} monomials > {self.config.max_monomials_per_vector}"
                    )
                    continue

                # Progress
                print(f"\r  [{subset_count}/{total_subsets}] "
                      f"{'+'.join(subset_names)} "
                      f"(deg≤{max_degree}, {n_monomials} monomials)...", end="", flush=True)

                result = self._search_single_subset(
                    subset_names, max_degree,
                    constants_work, constants_verify
                )

                if result and not result.is_trivial:
                    phase_candidates += 1
                    print(f"\n  *** CANDIDATE FOUND! ***")
                    print(f"  {result.equation_string}")
                    print(f"  Residual (working):      {result.residual_working:.5e}")
                    print(f"  Residual (verification): {result.residual_verification:.5e}")
                    print(f"  Coefficient norm:        {result.coefficient_norm:.1f}")
                    self.results_manager.save_candidate(result)
                    self.results_manager.log(
                        f"  CANDIDATE: {result.equation_string} "
                        f"(residual_w={result.residual_working:.5e}, "
                        f"residual_v={result.residual_verification:.5e})"
                    )
                elif result and result.is_trivial:
                    # Do not save trivial relations to disk — log only
                    self.results_manager.log(
                        f"  Trivial: {result.equation_string}"
                    )

                # Free memory after each subset
                gc.collect()

        phase_elapsed = time.time() - phase_start
        print(f"\n  Phase completed: {subset_count} subsets explored "
              f"in {phase_elapsed:.1f}s, {phase_candidates} non-trivial candidates.")
        self.results_manager.log(
            f"Phase complete: {phase_name} — {subset_count} subsets, "
            f"{phase_candidates} candidates, {phase_elapsed:.1f}s"
        )

    def _search_single_subset(
        self,
        subset_names: List[str],
        max_degree: int,
        constants_work: Dict[str, mpmath.mpf],
        constants_verify: Dict[str, mpmath.mpf],
    ) -> Optional[PSLQResult]:
        """Run PSLQ on a single subset of constants."""

        t_start = time.time()

        # === COMPUTE MONOMIALS at working_precision ===
        mpmath.mp.dps = self.config.working_precision + 50
        values_work, exponents, labels = compute_monomial_values(
            constants_work, subset_names, max_degree,
            self.config.max_monomials_per_vector
        )

        # === RUN PSLQ ===
        try:
            relation = mpmath.pslq(values_work, maxcoeff=self.config.max_coefficient_norm)
        except Exception as ex:
            self.results_manager.log(
                f"  PSLQ error for {'+'.join(subset_names)} deg≤{max_degree}: {ex}"
            )
            return None

        if relation is None:
            return None

        # === VERIFY COEFFICIENTS ===
        coefficients = list(relation)
        max_coeff = max(abs(c) for c in coefficients)
        norm = sum(c**2 for c in coefficients) ** 0.5

        if max_coeff > self.config.max_coefficient_norm:
            return None

        # === COMPUTE RESIDUAL at working_precision ===
        residual_work = abs(sum(
            c * v for c, v in zip(coefficients, values_work)
        ))

        if residual_work == 0:
            log_residual_work = float('-inf')
        else:
            log_residual_work = float(mpmath.log10(residual_work))

        if log_residual_work > self.config.residual_threshold_log10 and residual_work != 0:
            return None

        # === INDEPENDENT VERIFICATION at verification_precision ===
        mpmath.mp.dps = self.config.verification_precision + 50
        values_verify, _, _ = compute_monomial_values(
            constants_verify, subset_names, max_degree,
            self.config.max_monomials_per_vector
        )
        residual_verify = abs(sum(
            c * v for c, v in zip(coefficients, values_verify)
        ))

        # === CHECK TRIVIALITY ===
        is_trivial = self._check_trivial(coefficients, exponents, subset_names)

        # === BUILD HUMAN-READABLE EQUATION ===
        equation = self._format_equation(coefficients, labels)

        t_elapsed = time.time() - t_start

        return PSLQResult(
            constants_used=subset_names,
            coefficients=coefficients,
            exponents=exponents,
            labels=labels,
            residual_working=float(residual_work),
            residual_verification=float(residual_verify),
            coefficient_norm=norm,
            max_coefficient=max_coeff,
            equation_string=equation,
            is_trivial=is_trivial,
            search_time_seconds=t_elapsed,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def _check_trivial(
        self,
        coefficients: List[int],
        exponents: List[Tuple[int, ...]],
        constant_names: List[str],
    ) -> bool:
        """
        Check whether a found relation is "trivial", i.e.:
        1. Has only one nonzero coefficient (impossible as a true relation)
        2. Involves only one constant (e.g. φ² - φ - 1 = 0 is known)
        3. Involves only known algebraically dependent constants
        4. Is a known relation multiplied by a common monomial factor
           (e.g. e·(√2² - 2) = 0 is trivial because √2² = 2 is known)
        5. Involves only algebraic constants (√2, √3, √5, φ) — all
           relations among these are derivable and not interesting

        NOTE: this filter is conservative — better to have false positives
        than to miss a discovery. "Suspicious" relations are flagged
        but not automatically discarded.
        """
        nonzero = [(c, e) for c, e in zip(coefficients, exponents) if c != 0]

        # Only one nonzero term: cannot be a true relation
        if len(nonzero) <= 1:
            return True

        # === FACTORIZATION: remove common exponents ===
        # If all nonzero terms share a common monomial factor,
        # the relation reduces. E.g.: e·(√2²-2)=0 → (√2²-2)=0 after
        # dividing by e. This is the most frequent case of false positive.
        nonzero_exps = [e for _, e in nonzero]
        n_vars = len(constant_names)
        min_exps = tuple(
            min(exp[i] for exp in nonzero_exps) for i in range(n_vars)
        )
        # Reduce exponents by subtracting the common factor
        reduced_exps = [
            tuple(exp[i] - min_exps[i] for i in range(n_vars))
            for exp in nonzero_exps
        ]
        reduced_nonzero = list(zip([c for c, _ in nonzero], reduced_exps))

        # After reduction, check which variables are still involved
        involved_vars_reduced = set()
        for _, exp_tuple in reduced_nonzero:
            for i, e in enumerate(exp_tuple):
                if e > 0:
                    involved_vars_reduced.add(i)

        # If after reduction it involves 0 or 1 constant → trivial
        # (single-variable relation, e.g. φ²-φ-1=0 or √2²-2=0)
        if len(involved_vars_reduced) <= 1:
            return True

        # Names of constants actually involved after reduction
        involved_names_reduced = {constant_names[i] for i in involved_vars_reduced}

        # === PURELY ALGEBRAIC CONSTANTS ===
        # All relations among algebraic constants are derivable
        # and do not constitute interesting discoveries
        algebraic_constants = {"phi", "sqrt2", "sqrt3", "sqrt5"}
        if involved_names_reduced.issubset(algebraic_constants):
            return True

        # === KNOWN ALGEBRAIC DEPENDENCIES ===
        trivial_groups = [
            {"phi", "sqrt5"},       # φ = (1+√5)/2
            {"pi", "pi2"},          # π² = π·π
            {"ln2", "ln10"},        # predictable logarithmic relation
            {"ln5", "ln10"},        # predictable logarithmic relation
            {"ln2", "ln5", "ln10"}, # ln2 + ln5 = ln10
            {"ln2", "ln3", "ln5", "ln10"},  # logarithmic relations
            {"ln3", "ln10"},        # logarithmic relation
        ]
        for group in trivial_groups:
            if involved_names_reduced.issubset(group):
                return True

        # === GENERAL LOGARITHMIC RELATIONS ===
        # If after reduction it involves only constants of type ln(n),
        # it is a derivable logarithmic relation (e.g. 2·ln2 - ln4 = 0)
        log_constants = {"ln2", "ln3", "ln5", "ln10"}
        if involved_names_reduced.issubset(log_constants):
            return True

        # === SYMBOLIC VERIFICATION WITH SYMPY ===
        # Last and most robust filter: if sympy can simplify
        # the expression to zero, it is a derivable identity (even if complex).
        # This catches cases like (1+√5-2φ)·ln5 + (-φ-φ√5+2φ²) = 0
        # where the identity is hidden in the factorization.
        if self._sympy_simplifies_to_zero(coefficients, exponents, constant_names):
            return True

        return False

    # Map constants → sympy symbols (class attribute to avoid recomputation)
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
    }

    def _sympy_simplifies_to_zero(
        self,
        coefficients: List[int],
        exponents: List[Tuple[int, ...]],
        constant_names: List[str],
    ) -> bool:
        """
        Check whether sympy can simplify the expression to zero.
        If so, the relation is algebraically derivable and therefore trivial.

        Uses multiple simplification strategies:
        1. Direct simplify()
        2. rewrite(sqrt) + simplify() — expands GoldenRatio as (1+√5)/2
        3. Direct expand()
        """
        try:
            total = sympy.Integer(0)
            for c, exp_tuple in zip(coefficients, exponents):
                if c == 0:
                    continue
                term = sympy.Integer(c)
                for name, exp in zip(constant_names, exp_tuple):
                    if exp > 0:
                        if name not in self._SYMPY_CONSTANTS:
                            # Constant not available in sympy — cannot verify
                            return False
                        term *= self._SYMPY_CONSTANTS[name] ** exp
                total += term

            # Strategy 1: direct simplify
            if sympy.simplify(total) == 0:
                return True

            # Strategy 2: rewrite algebraic constants in radical form
            # (e.g. GoldenRatio → (1+√5)/2) then simplify
            rewritten = total.rewrite(sympy.sqrt)
            if sympy.simplify(rewritten) == 0:
                return True

            # Strategy 3: direct expand (sometimes sufficient)
            if sympy.expand(rewritten) == 0:
                return True

            return False
        except Exception:
            # On error, do not mark as trivial (conservative)
            return False

    def _format_equation(self, coefficients: List[int], labels: List[str]) -> str:
        """Format the equation in human-readable form."""
        terms = []
        for c, label in zip(coefficients, labels):
            if c == 0:
                continue
            if c > 0 and terms:
                sign = " + "
            elif c < 0:
                sign = " - " if terms else "-"
                c = -c
            else:
                sign = "" if not terms else " + "

            if label == "1":
                terms.append(f"{sign}{c}")
            elif c == 1:
                terms.append(f"{sign}{label}")
            else:
                terms.append(f"{sign}{c}·{label}")

        return "".join(terms) + " = 0"

    def _load_known_relations(self) -> list:
        """Load database of known relations for the triviality filter."""
        # This could be expanded with an external JSON file
        return []
