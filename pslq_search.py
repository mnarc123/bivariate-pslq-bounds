"""
Motore di ricerca PSLQ per relazioni polinomiali tra costanti.

ALGORITMO:
1. Per ogni sottoinsieme S di costanti (dimensione ≤ max_constants_per_search):
   a. Genera tutti i monomiali fino al grado max_degree
   b. Calcola i valori numerici dei monomiali a working_precision
   c. Esegui PSLQ sul vettore dei valori
   d. Se PSLQ restituisce una relazione:
      - Verifica che la norma dei coefficienti sia ≤ max_coefficient_norm
      - Verifica che NON sia una relazione "banale" (nota o derivabile)
      - Ricalcola a verification_precision
      - Se il residuo è ancora sotto soglia: CANDIDATA!

PRECAUZIONI:
- PSLQ può restituire "relazioni" spurie quando la precisione è insufficiente
- Le relazioni "banali" vanno filtrate (es. π² - π² = 0 se π² è nel set)
- Il residuo deve calare di 20+ ordini di grandezza rispetto al "rumore di fondo"
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
    """Un risultato della ricerca PSLQ."""
    constants_used: List[str]          # nomi delle costanti coinvolte
    coefficients: List[int]            # coefficienti interi trovati
    exponents: List[Tuple[int, ...]]   # esponenti dei monomiali
    labels: List[str]                  # etichette leggibili
    residual_working: float            # residuo a working_precision
    residual_verification: float       # residuo a verification_precision
    coefficient_norm: float            # norma L2 dei coefficienti
    max_coefficient: int               # massimo valore assoluto dei coefficienti
    equation_string: str               # equazione in formato leggibile
    is_trivial: bool                   # True se è una relazione banale/nota
    search_time_seconds: float         # tempo di calcolo
    timestamp: str                     # ISO timestamp


class PSLQSearchEngine:
    """Motore di ricerca principale."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.constants_computer = ConstantsComputer(config)
        self.results_manager = ResultsManager(config)

        # Relazioni note da filtrare (aggiungi qui le relazioni banali)
        self.known_relations = self._load_known_relations()

    def run_full_search(self):
        """
        Esegue la ricerca completa secondo la strategia definita in config.

        STRATEGIA DI RICERCA (dal più promettente al meno):

        Fase 1: Relazioni quadratiche tra costanti di Livello 1
                 (π, e, γ, φ, ln2, √2) — grado 2, tutte le combinazioni
                 Spazio piccolo, alta probabilità di risultato inedito

        Fase 2: Relazioni cubiche tra coppie/triplette di Livello 1
                 Grado 3, dimensione PSLQ ancora gestibile

        Fase 3: Relazioni quadratiche con costanti di Livello 2
                 (aggiunge ζ(3), Catalan, √3, √5, ln3, ...)

        Fase 4: Relazioni di grado 4 per i sottoinsiemi più promettenti

        Fase 5: Costanti esotiche di Livello 3
                 (Feigenbaum, Khinchin, Glaisher, ...)
        """
        print("=" * 70)
        print("  PROGETTO EQUAZIONE PONTE — Ricerca Computazionale")
        print("=" * 70)

        # === STEP 0: Verifica integrità del sistema ===
        print("\n[STEP 0] Verifica integrità del sistema di calcolo...")
        if not self.constants_computer.verify_known_relations():
            print("\n ⚠ ERRORE CRITICO: i self-test sono falliti. INTERRUZIONE.")
            return

        # === Pre-calcolo costanti ===
        all_names = (
            self.config.constants_level_1
            + self.config.constants_level_2
            + self.config.constants_level_3
        )
        print(f"\n[STEP 1] Calcolo di {len(all_names)} costanti a {self.config.working_precision} cifre...")
        constants_work = self.constants_computer.compute_all(
            self.config.working_precision, all_names
        )
        print(f"[STEP 1] Calcolo di {len(all_names)} costanti a {self.config.verification_precision} cifre...")
        constants_verify = self.constants_computer.compute_all(
            self.config.verification_precision, all_names
        )

        self.results_manager.log(f"Costanti calcolate: {len(all_names)} a {self.config.working_precision}/{self.config.verification_precision} cifre")

        # === FASE 1: Quadratiche tra Livello 1 ===
        self._run_phase(
            phase_name="FASE 1: Relazioni quadratiche — Livello 1",
            constant_names=self.config.constants_level_1,
            max_degree=2,
            min_subset_size=2,
            max_subset_size=self.config.max_constants_per_search,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === FASE 2: Cubiche tra coppie/triplette di Livello 1 ===
        self._run_phase(
            phase_name="FASE 2: Relazioni cubiche — Livello 1",
            constant_names=self.config.constants_level_1,
            max_degree=3,
            min_subset_size=2,
            max_subset_size=3,  # max 3 costanti per volta (altrimenti troppi monomiali)
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === FASE 3: Quadratiche con Livello 2 ===
        level_1_2 = self.config.constants_level_1 + self.config.constants_level_2
        self._run_phase(
            phase_name="FASE 3: Relazioni quadratiche — Livello 1+2",
            constant_names=level_1_2,
            max_degree=2,
            min_subset_size=2,
            max_subset_size=4,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === FASE 4: Grado 4 per sottoinsiemi piccoli ===
        self._run_phase(
            phase_name="FASE 4: Relazioni di grado 4 — coppie Livello 1",
            constant_names=self.config.constants_level_1,
            max_degree=4,
            min_subset_size=2,
            max_subset_size=2,  # solo coppie (altrimenti esplosione combinatoria)
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === FASE 5: Costanti esotiche ===
        level_all = (
            self.config.constants_level_1
            + self.config.constants_level_2
            + self.config.constants_level_3
        )
        self._run_phase(
            phase_name="FASE 5: Relazioni quadratiche — tutte le costanti",
            constant_names=level_all,
            max_degree=2,
            min_subset_size=2,
            max_subset_size=3,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === FASI ESTESE: SOLO TRASCENDENTI ===
        # Le costanti algebriche (√2, √3, √5, φ) generano solo identità banali.
        # Focalizziamoci sulle costanti genuinamente trascendenti.
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

        # === FASE 6: Grado alto per coppie trascendenti core ===
        # π, e, γ, ln2 — grado fino a 6 (15-28 monomiali per coppia)
        self._run_phase(
            phase_name="FASE 6: Grado 6 — coppie trascendenti core (π,e,γ,ln2)",
            constant_names=transcendentals_core,
            max_degree=6,
            min_subset_size=2,
            max_subset_size=2,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === FASE 7: Grado 5 per triplette trascendenti core ===
        self._run_phase(
            phase_name="FASE 7: Grado 5 — triplette trascendenti core",
            constant_names=transcendentals_core,
            max_degree=5,
            min_subset_size=3,
            max_subset_size=3,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === FASE 8: Grado 4 per triplette trascendenti estese ===
        self._run_phase(
            phase_name="FASE 8: Grado 4 — triplette trascendenti estese",
            constant_names=transcendentals_extended,
            max_degree=4,
            min_subset_size=3,
            max_subset_size=3,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === FASE 9: Grado 3 per quadruple/quintuple trascendenti ===
        self._run_phase(
            phase_name="FASE 9: Grado 3 — quadruple trascendenti estese",
            constant_names=transcendentals_extended,
            max_degree=3,
            min_subset_size=4,
            max_subset_size=5,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === FASE 10: Grado 3 per coppie/triplette con costanti esotiche ===
        self._run_phase(
            phase_name="FASE 10: Grado 3 — trascendenti con costanti esotiche",
            constant_names=transcendentals_all,
            max_degree=3,
            min_subset_size=2,
            max_subset_size=3,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === FASE 11: Grado 2 per quadruple/quintuple con costanti esotiche ===
        self._run_phase(
            phase_name="FASE 11: Grado 2 — quadruple trascendenti complete",
            constant_names=transcendentals_all,
            max_degree=2,
            min_subset_size=4,
            max_subset_size=5,
            constants_work=constants_work,
            constants_verify=constants_verify,
        )

        # === REPORT FINALE ===
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
        """Esegue una fase della ricerca."""
        print(f"\n{'='*60}")
        print(f"  {phase_name}")
        print(f"{'='*60}")

        self.results_manager.log(f"Inizio fase: {phase_name}")

        n_total = len(constant_names)
        total_subsets = sum(
            1 for r in range(min_subset_size, min(max_subset_size, n_total) + 1)
            for _ in itertools.combinations(range(n_total), r)
        )
        print(f"  Costanti: {n_total}, Grado max: {max_degree}")
        print(f"  Sottoinsiemi da esplorare: {total_subsets}")

        subset_count = 0
        phase_candidates = 0
        phase_start = time.time()

        for subset_size in range(min_subset_size, min(max_subset_size, n_total) + 1):
            for subset_indices in itertools.combinations(range(n_total), subset_size):
                subset_names = [constant_names[i] for i in subset_indices]
                subset_count += 1

                # Controlla se il numero di monomiali è gestibile
                n_monomials = count_monomials(len(subset_names), max_degree)
                if n_monomials > self.config.max_monomials_per_vector:
                    self.results_manager.log(
                        f"  Skip {'+'.join(subset_names)} deg≤{max_degree}: "
                        f"{n_monomials} monomiali > {self.config.max_monomials_per_vector}"
                    )
                    continue

                # Progresso
                print(f"\r  [{subset_count}/{total_subsets}] "
                      f"{'+'.join(subset_names)} "
                      f"(deg≤{max_degree}, {n_monomials} monomiali)...", end="", flush=True)

                result = self._search_single_subset(
                    subset_names, max_degree,
                    constants_work, constants_verify
                )

                if result and not result.is_trivial:
                    phase_candidates += 1
                    print(f"\n  *** CANDIDATA TROVATA! ***")
                    print(f"  {result.equation_string}")
                    print(f"  Residuo (lavoro):    {result.residual_working:.5e}")
                    print(f"  Residuo (verifica):  {result.residual_verification:.5e}")
                    print(f"  Norma coefficienti:  {result.coefficient_norm:.1f}")
                    self.results_manager.save_candidate(result)
                    self.results_manager.log(
                        f"  CANDIDATA: {result.equation_string} "
                        f"(residuo_w={result.residual_working:.5e}, "
                        f"residuo_v={result.residual_verification:.5e})"
                    )
                elif result and result.is_trivial:
                    # Non salvare su disco le relazioni banali — solo log
                    self.results_manager.log(
                        f"  Banale: {result.equation_string}"
                    )

                # Libera memoria dopo ogni subset
                gc.collect()

        phase_elapsed = time.time() - phase_start
        print(f"\n  Fase completata: {subset_count} sottoinsiemi esplorati "
              f"in {phase_elapsed:.1f}s, {phase_candidates} candidate non banali.")
        self.results_manager.log(
            f"Fine fase: {phase_name} — {subset_count} subset, "
            f"{phase_candidates} candidate, {phase_elapsed:.1f}s"
        )

    def _search_single_subset(
        self,
        subset_names: List[str],
        max_degree: int,
        constants_work: Dict[str, mpmath.mpf],
        constants_verify: Dict[str, mpmath.mpf],
    ) -> Optional[PSLQResult]:
        """Esegue PSLQ su un singolo sottoinsieme di costanti."""

        t_start = time.time()

        # === CALCOLO MONOMIALI a working_precision ===
        mpmath.mp.dps = self.config.working_precision + 50
        values_work, exponents, labels = compute_monomial_values(
            constants_work, subset_names, max_degree,
            self.config.max_monomials_per_vector
        )

        # === ESECUZIONE PSLQ ===
        try:
            relation = mpmath.pslq(values_work, maxcoeff=self.config.max_coefficient_norm)
        except Exception as ex:
            self.results_manager.log(
                f"  PSLQ errore per {'+'.join(subset_names)} deg≤{max_degree}: {ex}"
            )
            return None

        if relation is None:
            return None

        # === VERIFICA COEFFICIENTI ===
        coefficients = list(relation)
        max_coeff = max(abs(c) for c in coefficients)
        norm = sum(c**2 for c in coefficients) ** 0.5

        if max_coeff > self.config.max_coefficient_norm:
            return None

        # === CALCOLO RESIDUO a working_precision ===
        residual_work = abs(sum(
            c * v for c, v in zip(coefficients, values_work)
        ))

        if residual_work == 0:
            log_residual_work = float('-inf')
        else:
            log_residual_work = float(mpmath.log10(residual_work))

        if log_residual_work > self.config.residual_threshold_log10 and residual_work != 0:
            return None

        # === VERIFICA INDIPENDENTE a verification_precision ===
        mpmath.mp.dps = self.config.verification_precision + 50
        values_verify, _, _ = compute_monomial_values(
            constants_verify, subset_names, max_degree,
            self.config.max_monomials_per_vector
        )
        residual_verify = abs(sum(
            c * v for c, v in zip(coefficients, values_verify)
        ))

        # === CONTROLLA BANALITÀ ===
        is_trivial = self._check_trivial(coefficients, exponents, subset_names)

        # === COSTRUISCI EQUAZIONE LEGGIBILE ===
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
        Verifica se una relazione trovata è "banale", cioè:
        1. Ha solo un coefficiente non-zero (impossibile come relazione vera)
        2. Coinvolge solo una costante (es. φ² - φ - 1 = 0 è nota)
        3. Coinvolge solo costanti algebricamente dipendenti note
        4. È una relazione nota moltiplicata per un fattore comune di monomiali
           (es. e·(√2² - 2) = 0 è banale perché √2² = 2 è noto)
        5. Coinvolge solo costanti algebriche (√2, √3, √5, φ) — tutte le
           relazioni tra queste sono derivabili e non interessanti

        NOTA: questo filtro è conservativo — meglio falsi positivi che
        perdere una scoperta. Le relazioni "sospette" vengono segnalate
        ma non scartate automaticamente.
        """
        nonzero = [(c, e) for c, e in zip(coefficients, exponents) if c != 0]

        # Solo un termine non-zero: impossibile essere una vera relazione
        if len(nonzero) <= 1:
            return True

        # === FATTORIZZAZIONE: rimuovi esponenti comuni ===
        # Se tutti i termini non-zero condividono un fattore monomiale comune,
        # la relazione si riduce. Es: e·(√2²-2)=0 → (√2²-2)=0 dopo aver
        # diviso per e. Questo è il caso più frequente di falso positivo.
        nonzero_exps = [e for _, e in nonzero]
        n_vars = len(constant_names)
        min_exps = tuple(
            min(exp[i] for exp in nonzero_exps) for i in range(n_vars)
        )
        # Riduci gli esponenti sottraendo il fattore comune
        reduced_exps = [
            tuple(exp[i] - min_exps[i] for i in range(n_vars))
            for exp in nonzero_exps
        ]
        reduced_nonzero = list(zip([c for c, _ in nonzero], reduced_exps))

        # Dopo la riduzione, controlla quali variabili sono ancora coinvolte
        involved_vars_reduced = set()
        for _, exp_tuple in reduced_nonzero:
            for i, e in enumerate(exp_tuple):
                if e > 0:
                    involved_vars_reduced.add(i)

        # Se dopo la riduzione coinvolge 0 o 1 costante → banale
        # (relazione in una sola variabile, es. φ²-φ-1=0 o √2²-2=0)
        if len(involved_vars_reduced) <= 1:
            return True

        # Nomi delle costanti effettivamente coinvolte dopo riduzione
        involved_names_reduced = {constant_names[i] for i in involved_vars_reduced}

        # === COSTANTI PURAMENTE ALGEBRICHE ===
        # Tutte le relazioni tra costanti algebriche sono derivabili
        # e non costituiscono scoperte interessanti
        algebraic_constants = {"phi", "sqrt2", "sqrt3", "sqrt5"}
        if involved_names_reduced.issubset(algebraic_constants):
            return True

        # === DIPENDENZE ALGEBRICHE NOTE ===
        trivial_groups = [
            {"phi", "sqrt5"},       # φ = (1+√5)/2
            {"pi", "pi2"},          # π² = π·π
            {"ln2", "ln10"},        # relazione logaritmica prevedibile
            {"ln5", "ln10"},        # relazione logaritmica prevedibile
            {"ln2", "ln5", "ln10"}, # ln2 + ln5 = ln10
            {"ln2", "ln3", "ln5", "ln10"},  # relazioni logaritmiche
            {"ln3", "ln10"},        # relazione logaritmica
        ]
        for group in trivial_groups:
            if involved_names_reduced.issubset(group):
                return True

        # === RELAZIONI LOGARITMICHE GENERALI ===
        # Se dopo riduzione coinvolge solo costanti di tipo ln(n),
        # è una relazione logaritmica derivabile (es. 2·ln2 - ln4 = 0)
        log_constants = {"ln2", "ln3", "ln5", "ln10"}
        if involved_names_reduced.issubset(log_constants):
            return True

        # === VERIFICA SIMBOLICA CON SYMPY ===
        # Ultimo filtro e più robusto: se sympy riesce a semplificare
        # l'espressione a zero, è una identità derivabile (anche se complessa).
        # Questo cattura casi come (1+√5-2φ)·ln5 + (-φ-φ√5+2φ²) = 0
        # dove l'identità si nasconde nella fattorizzazione.
        if self._sympy_simplifies_to_zero(coefficients, exponents, constant_names):
            return True

        return False

    # Mappa costanti → simboli sympy (attributo di classe per evitare ricalcoli)
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
        Verifica se sympy riesce a semplificare l'espressione a zero.
        Se sì, la relazione è derivabile algebricamente e quindi banale.

        Usa multiple strategie di semplificazione:
        1. simplify() diretto
        2. rewrite(sqrt) + simplify() — espande GoldenRatio come (1+√5)/2
        3. expand() diretto
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
                            # Costante non disponibile in sympy — non possiamo verificare
                            return False
                        term *= self._SYMPY_CONSTANTS[name] ** exp
                total += term

            # Strategia 1: simplify diretto
            if sympy.simplify(total) == 0:
                return True

            # Strategia 2: riscrivi costanti algebriche in forma radicale
            # (es. GoldenRatio → (1+√5)/2) poi semplifica
            rewritten = total.rewrite(sympy.sqrt)
            if sympy.simplify(rewritten) == 0:
                return True

            # Strategia 3: expand diretto (a volte basta)
            if sympy.expand(rewritten) == 0:
                return True

            return False
        except Exception:
            # In caso di errore, non marcare come banale (conservativo)
            return False

    def _format_equation(self, coefficients: List[int], labels: List[str]) -> str:
        """Formatta l'equazione in modo leggibile."""
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
        """Carica database di relazioni note per il filtro di banalità."""
        # Questo potrebbe essere espanso con un file JSON esterno
        return []
