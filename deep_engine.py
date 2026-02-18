"""
Motore di ricerca profonda per relazioni polinomiali bivariate.

Questo modulo esegue PSLQ sulle coppie di costanti trascendenti
spingendo fino al grado massimo consentito dal budget di tempo.

ALGORITMO PER OGNI COPPIA (α, β):
1. Calcola grado massimo fattibile dato il budget di tempo
2. Per ogni grado d da 9 fino al massimo (incrementale):
   a. Genera il vettore dei monomiali [1, α, β, α², αβ, β², ..., β^d]
   b. Calcola i valori a precisione adeguata (N×D×safety cifre)
   c. Esegui PSLQ con maxcoeff=100 (prima relazioni "eleganti")
   d. Se nessun risultato: esegui PSLQ con maxcoeff=10000
   e. Se nessun risultato: esegui PSLQ con maxcoeff=10^6
   f. Log del bound stabilito a questo grado
   g. Checkpoint
3. Verifica eventuale relazione trovata a precisione doppia

STRATEGIA INCREMENTALE:
- Partiamo dal grado 9 (dove la Fase 1 si è fermata)
- Incrementiamo di 1 in 1 fino al grado massimo
- Questo permette di interrompere in qualsiasi momento con risultati parziali
- I checkpoint garantiscono che non si ripeta lavoro già fatto
"""

import mpmath
import sympy
import time
import json
import sys
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from config import SearchConfig
from constants import ConstantsComputer
from precision_manager import compute_precision_plan, PrecisionPlan
from checkpoint import CheckpointManager
from bound_calculator import format_bound_statement, compute_search_space_size


# === DEFINIZIONE DELLE COPPIE ===

TOP_PAIRS = [
    ("pi", "e"),
    ("pi", "euler_gamma"),
    ("e", "euler_gamma"),
]

HIGH_PAIRS = [
    ("pi", "zeta3"),
    ("e", "zeta3"),
    ("euler_gamma", "zeta3"),
    ("pi", "catalan"),
    ("e", "catalan"),
    ("euler_gamma", "catalan"),
]

MEDIUM_PAIRS = [
    ("pi", "omega"),
    ("e", "omega"),
    ("euler_gamma", "omega"),
    ("zeta3", "catalan"),
    ("pi", "zeta5"),
    ("euler_gamma", "glaisher"),
    ("zeta3", "zeta5"),
    ("pi", "ln2"),
    ("e", "ln2"),
    ("euler_gamma", "ln2"),
]


@dataclass
class DeepSearchResult:
    """Risultato di una singola ricerca profonda."""
    pair: Tuple[str, str]
    degree: int
    max_coeff: int
    n_monomials: int
    precision_digits: int
    found_relation: bool
    relation: Optional[List[int]]
    residual_working: Optional[float]
    residual_verification: Optional[float]
    bound_statement: str
    elapsed_seconds: float
    timestamp: str


# Mappa costanti → simboli sympy per verifica banalità
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
    "glaisher": sympy.Symbol("A_Glaisher"),  # no closed form in sympy
}


def generate_bivariate_monomials(
    alpha: mpmath.mpf,
    beta: mpmath.mpf,
    degree: int,
) -> Tuple[List[mpmath.mpf], List[Tuple[int, int]], List[str]]:
    """
    Genera tutti i monomiali α^i · β^j con i+j ≤ degree.

    Ordine: grado totale crescente, poi i crescente dentro ogni grado.
    Il primo elemento è sempre 1 (i=0, j=0).

    Returns:
        values: valori numerici dei monomiali
        exponents: lista di (i, j)
        labels: stringhe leggibili
    """
    values = []
    exponents = []
    labels = []

    # Precomputa le potenze
    alpha_powers = [mpmath.mpf(1)]
    beta_powers = [mpmath.mpf(1)]
    for k in range(1, degree + 1):
        alpha_powers.append(alpha_powers[-1] * alpha)
        beta_powers.append(beta_powers[-1] * beta)

    for total_deg in range(degree + 1):
        for i in range(total_deg + 1):
            j = total_deg - i
            val = alpha_powers[i] * beta_powers[j]
            values.append(val)
            exponents.append((i, j))

            # Etichetta leggibile
            parts = []
            if i > 0:
                if i == 1:
                    parts.append("α")
                else:
                    parts.append(f"α^{i}")
            if j > 0:
                if j == 1:
                    parts.append("β")
                else:
                    parts.append(f"β^{j}")
            if not parts:
                parts = ["1"]
            labels.append("·".join(parts))

    return values, exponents, labels


def check_relation_trivial_sympy(
    relation: List[int],
    exponents: List[Tuple[int, int]],
    const1: str,
    const2: str,
) -> bool:
    """
    Verifica se una relazione trovata è banale usando sympy.
    Restituisce True se sympy riesce a semplificare a zero.
    """
    if const1 not in _SYMPY_CONSTANTS or const2 not in _SYMPY_CONSTANTS:
        return False

    try:
        sym1 = _SYMPY_CONSTANTS[const1]
        sym2 = _SYMPY_CONSTANTS[const2]

        total = sympy.Integer(0)
        for c, (i, j) in zip(relation, exponents):
            if c == 0:
                continue
            term = sympy.Integer(c) * sym1**i * sym2**j
            total += term

        if sympy.simplify(total) == 0:
            return True
        rewritten = total.rewrite(sympy.sqrt)
        if sympy.simplify(rewritten) == 0:
            return True
        if sympy.expand(rewritten) == 0:
            return True
        return False
    except Exception:
        return False


class DeepSearchEngine:
    """Motore di ricerca profonda."""

    def __init__(self, config: SearchConfig, max_hours_per_pair: float = 24.0):
        self.config = config
        self.max_hours_per_pair = max_hours_per_pair

        deep_dir = config.results_dir / "deep_v2"
        deep_dir.mkdir(parents=True, exist_ok=True)
        (deep_dir / "bounds").mkdir(exist_ok=True)
        (deep_dir / "logs").mkdir(exist_ok=True)

        self.checkpoint = CheckpointManager(deep_dir / "checkpoints")
        self.results: List[DeepSearchResult] = []
        self.constants_computer = ConstantsComputer(config)

        self.log_file = deep_dir / "logs" / f"deep_search_{time.strftime('%Y%m%d_%H%M%S')}.log"
        self.bounds_dir = deep_dir / "bounds"
        self.deep_dir = deep_dir

    def log(self, msg: str):
        """Log a file e console."""
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def get_constant_value(self, name: str, digits: int) -> mpmath.mpf:
        """Recupera o calcola una costante alla precisione data."""
        all_vals = self.constants_computer.compute_all(digits, [name])
        return all_vals[name]

    def run(self, pair_filter: Optional[str] = None):
        """Esecuzione principale della ricerca profonda."""
        self.log("=" * 70)
        self.log("  EQUAZIONE PONTE — Fase 2: Ricerca Profonda ad Alto Grado")
        self.log("=" * 70)

        # === SELF-TEST ===
        self.log("\n[SELF-TEST] Verifica PSLQ su relazione nota φ²-φ-1=0...")
        mpmath.mp.dps = 200
        phi = (1 + mpmath.sqrt(5)) / 2
        rel = mpmath.pslq([phi**2, phi, mpmath.mpf(1)])
        if rel is None or (rel != [1, -1, -1] and rel != [-1, 1, 1]):
            self.log(f"  ✗ ERRORE: PSLQ restituisce {rel}! Interruzione.")
            return
        self.log("  ✓ PSLQ funzionante.")

        # === SELF-TEST 2: ζ(2) = π²/6 ===
        self.log("[SELF-TEST] Verifica PSLQ su ζ(2) - π²/6 = 0...")
        mpmath.mp.dps = 200
        rel2 = mpmath.pslq([mpmath.zeta(2), mpmath.pi**2])
        if rel2 is None or (rel2 != [6, -1] and rel2 != [-6, 1]):
            self.log(f"  ✗ ERRORE: PSLQ restituisce {rel2}! Interruzione.")
            return
        self.log("  ✓ PSLQ funzionante.")

        # === PIANIFICAZIONE ===
        self.log(f"\n[PIANIFICAZIONE] Budget: {self.max_hours_per_pair:.1f}h per coppia")
        self.log(f"  Checkpoint completati: {self.checkpoint.get_completed_count()}")

        all_pairs = (
            [(p, "top") for p in TOP_PAIRS]
            + [(p, "high") for p in HIGH_PAIRS]
            + [(p, "medium") for p in MEDIUM_PAIRS]
        )

        # Filtra se richiesto
        if pair_filter:
            all_pairs = [
                (p, pri) for p, pri in all_pairs
                if f"{p[0]}+{p[1]}" == pair_filter
            ]
            if not all_pairs:
                self.log(f"  ⚠ Coppia '{pair_filter}' non trovata!")
                return

        for (c1, c2), priority in all_pairs:
            pair_name = f"{c1}+{c2}"

            # Budget di tempo basato sulla priorità
            if priority == "top":
                time_budget = self.max_hours_per_pair * 2
            elif priority == "high":
                time_budget = self.max_hours_per_pair
            else:
                time_budget = self.max_hours_per_pair * 0.5

            self.log(f"\n{'='*60}")
            self.log(f"  COPPIA: ({c1}, {c2})  [priorità: {priority}]")
            self.log(f"  Budget tempo: {time_budget:.1f}h")
            self.log(f"{'='*60}")

            self._search_pair(c1, c2, pair_name, time_budget)

        # === REPORT FINALE ===
        self._generate_report()

    def _search_pair(
        self,
        const1: str,
        const2: str,
        pair_name: str,
        time_budget_hours: float,
    ):
        """Ricerca profonda incrementale per una coppia."""

        pair_start_time = time.time()
        start_degree = 9  # da dove continua la Fase 1

        for degree in range(start_degree, 200):
            # Controlla tempo rimanente
            elapsed_hours = (time.time() - pair_start_time) / 3600
            remaining_hours = time_budget_hours - elapsed_hours
            if remaining_hours <= 0:
                self.log(f"  ⏱ Budget tempo esaurito a grado {degree-1}.")
                break

            # === LIVELLO A: Coefficienti piccoli (≤100) ===
            plan_small = compute_precision_plan(
                degree, max_coeff=100, max_hours=remaining_hours
            )
            if not plan_small.is_feasible:
                self.log(f"  📊 Grado {degree} (|c|≤100) non fattibile "
                         f"(stima {plan_small.estimated_hours:.1f}h > {remaining_hours:.1f}h rimanenti). "
                         f"Grado max raggiunto: {degree-1}.")
                break

            if not self.checkpoint.is_completed(pair_name, degree, 100):
                result_a = self._run_single_pslq(
                    const1, const2, pair_name, degree,
                    max_coeff=100, plan=plan_small
                )
                if result_a and result_a.found_relation:
                    self.log(f"  ★★★ RELAZIONE TROVATA a grado {degree}, |c|≤100! ★★★")
                    return
            else:
                self.log(f"    Grado {degree}, |c|≤100: già completato (checkpoint)")

            # Aggiorna tempo rimanente
            elapsed_hours = (time.time() - pair_start_time) / 3600
            remaining_hours = time_budget_hours - elapsed_hours

            # === LIVELLO B: Coefficienti medi (≤10000) ===
            plan_medium = compute_precision_plan(
                degree, max_coeff=10000, max_hours=remaining_hours
            )
            if plan_medium.is_feasible and not self.checkpoint.is_completed(pair_name, degree, 10000):
                result_b = self._run_single_pslq(
                    const1, const2, pair_name, degree,
                    max_coeff=10000, plan=plan_medium
                )
                if result_b and result_b.found_relation:
                    self.log(f"  ★★★ RELAZIONE TROVATA a grado {degree}, |c|≤10000! ★★★")
                    return
            elif self.checkpoint.is_completed(pair_name, degree, 10000):
                self.log(f"    Grado {degree}, |c|≤10000: già completato (checkpoint)")

            # Aggiorna tempo rimanente
            elapsed_hours = (time.time() - pair_start_time) / 3600
            remaining_hours = time_budget_hours - elapsed_hours

            # === LIVELLO C: Coefficienti grandi (≤10⁶) — solo se veloce ===
            plan_large = compute_precision_plan(
                degree, max_coeff=10**6,
                max_hours=min(1.0, remaining_hours)
            )
            if plan_large.is_feasible and not self.checkpoint.is_completed(pair_name, degree, 10**6):
                result_c = self._run_single_pslq(
                    const1, const2, pair_name, degree,
                    max_coeff=10**6, plan=plan_large
                )
                if result_c and result_c.found_relation:
                    self.log(f"  ★★★ RELAZIONE TROVATA a grado {degree}, |c|≤10⁶! ★★★")
                    return

        # Salva bound per questa coppia
        self._save_pair_bounds(const1, const2, pair_name)

    def _run_single_pslq(
        self,
        const1: str,
        const2: str,
        pair_name: str,
        degree: int,
        max_coeff: int,
        plan: PrecisionPlan,
    ) -> Optional[DeepSearchResult]:
        """Esegue una singola ricerca PSLQ."""

        self.checkpoint.mark_started(pair_name, degree, max_coeff)
        t_start = time.time()

        # Imposta precisione
        mpmath.mp.dps = plan.working_digits + 50

        # Calcola costanti
        alpha = self.get_constant_value(const1, plan.working_digits + 50)
        beta = self.get_constant_value(const2, plan.working_digits + 50)

        # Genera monomiali
        values, exponents, labels = generate_bivariate_monomials(alpha, beta, degree)

        self.log(f"    Grado {degree}, |c|≤{max_coeff}: "
                 f"{plan.n_monomials} mono, {plan.working_digits} cifre "
                 f"(stima {plan.estimated_hours*60:.1f}min)... ")

        # Esegui PSLQ
        try:
            relation = mpmath.pslq(values, maxcoeff=max_coeff, maxsteps=0)
        except Exception as exc:
            self.log(f"    ⚠ Errore PSLQ: {exc}")
            relation = None

        elapsed = time.time() - t_start
        found = relation is not None

        # Costruisci il bound statement
        bound_stmt = format_bound_statement(const1, const2, degree, max_coeff, found)

        residual_w = None
        residual_v = None

        if found:
            # Calcola residuo a working precision
            residual_val = abs(sum(c * v for c, v in zip(relation, values)))
            residual_w = float(residual_val) if residual_val > 0 else 0.0

            # Controlla se degenere
            nonzero = [c for c in relation if c != 0]
            if len(nonzero) <= 1:
                self.log(f"    ⚠ Relazione degenere (un solo termine non-zero). Scartata.")
                found = False
            else:
                # Controlla banalità con sympy
                is_trivial = check_relation_trivial_sympy(
                    relation, exponents, const1, const2
                )
                if is_trivial:
                    self.log(f"    ⚠ Relazione banale (sympy simplifica a 0). Scartata.")
                    found = False

            if found:
                # Verifica a precisione doppia
                mpmath.mp.dps = plan.verification_digits + 50
                alpha_v = self.get_constant_value(const1, plan.verification_digits + 50)
                beta_v = self.get_constant_value(const2, plan.verification_digits + 50)
                values_v, _, _ = generate_bivariate_monomials(alpha_v, beta_v, degree)
                residual_v_val = abs(sum(c * v for c, v in zip(relation, values_v)))
                residual_v = float(residual_v_val) if residual_v_val > 0 else 0.0

                # Formatta la relazione
                eq_parts = []
                for c, (i, j) in zip(relation, exponents):
                    if c == 0:
                        continue
                    eq_parts.append(f"{c}·{const1}^{i}·{const2}^{j}")
                eq_str = " + ".join(eq_parts)

                self.log(f"    ★ TROVATA! {eq_str} = 0")
                self.log(f"    ★ Coefficienti: {relation}")
                self.log(f"    ★ Residuo lavoro:    {residual_w:.5e}")
                self.log(f"    ★ Residuo verifica:  {residual_v:.5e}")

                # Salva candidata su disco
                candidate = {
                    "pair": [const1, const2],
                    "degree": degree,
                    "max_coeff": max_coeff,
                    "coefficients": relation,
                    "exponents": exponents,
                    "labels": labels,
                    "equation": eq_str,
                    "residual_working": residual_w,
                    "residual_verification": residual_v,
                    "precision_digits": plan.working_digits,
                    "verification_digits": plan.verification_digits,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                cand_path = (
                    self.deep_dir / "bounds"
                    / f"CANDIDATE_{const1}_{const2}_d{degree}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                )
                cand_path.write_text(json.dumps(candidate, indent=2))
                self.log(f"    → Salvata in {cand_path}")
        else:
            elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"
            self.log(f"    ✓ Nessuna relazione. {elapsed_str}. {bound_stmt}")

        result = DeepSearchResult(
            pair=(const1, const2),
            degree=degree,
            max_coeff=max_coeff,
            n_monomials=plan.n_monomials,
            precision_digits=plan.working_digits,
            found_relation=found,
            relation=relation if found else None,
            residual_working=residual_w,
            residual_verification=residual_v,
            bound_statement=bound_stmt,
            elapsed_seconds=elapsed,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        self.results.append(result)

        # Checkpoint
        self.checkpoint.mark_completed(pair_name, degree, max_coeff, {
            "found": found,
            "elapsed_s": round(elapsed, 1),
            "bound": bound_stmt,
        })

        return result

    def _save_pair_bounds(self, const1: str, const2: str, pair_name: str):
        """Salva un riepilogo dei bound per questa coppia."""
        pair_results = [r for r in self.results if r.pair == (const1, const2)]
        if not pair_results:
            return

        max_degree_100 = max(
            (r.degree for r in pair_results if r.max_coeff == 100 and not r.found_relation),
            default=0
        )
        max_degree_10k = max(
            (r.degree for r in pair_results if r.max_coeff == 10000 and not r.found_relation),
            default=0
        )
        max_degree_1M = max(
            (r.degree for r in pair_results if r.max_coeff == 10**6 and not r.found_relation),
            default=0
        )
        total_time = sum(r.elapsed_seconds for r in pair_results)

        summary = {
            "pair": [const1, const2],
            "bounds": {
                "max_coeff_100": {"max_degree": max_degree_100},
                "max_coeff_10000": {"max_degree": max_degree_10k},
                "max_coeff_1000000": {"max_degree": max_degree_1M},
            },
            "total_time_seconds": round(total_time, 1),
            "n_searches": len(pair_results),
            "found_any": any(r.found_relation for r in pair_results),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        bound_path = self.bounds_dir / f"bound_{pair_name}.json"
        bound_path.write_text(json.dumps(summary, indent=2))

    def _generate_report(self):
        """Genera il report finale della ricerca profonda."""
        report_path = self.deep_dir / "REPORT_DEEP_V2.md"

        lines = [
            "# Report — Equazione Ponte, Fase 2: Ricerca Profonda ad Alto Grado",
            f"**Data:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Hardware:** Intel i7-12700F, 64 GB DDR5, Debian 13",
            f"**Software:** Python 3, mpmath, sympy",
            f"**Budget:** {self.max_hours_per_pair:.1f}h per coppia",
            "",
            "---",
            "",
            "## Risultato principale",
            "",
        ]

        found_any = any(r.found_relation for r in self.results)
        if found_any:
            lines.append("**⭐ RELAZIONE NON-BANALE TROVATA!** Vedere dettagli sotto.")
        else:
            lines.append("**Nessuna relazione polinomiale non-banale trovata.**")
            lines.append("")
            lines.append("Questo è un risultato negativo computazionale significativo.")

        # Tabella dei bound per coppia
        lines.extend([
            "",
            "---",
            "",
            "## Bound stabiliti per coppia",
            "",
            "| Coppia | Grado max (|c|≤100) | Grado max (|c|≤10⁴) | Grado max (|c|≤10⁶) | Tempo |",
            "|--------|----------------------|----------------------|----------------------|-------|",
        ])

        # Raggruppa per coppia
        pairs_seen = {}
        for r in self.results:
            key = f"({r.pair[0]}, {r.pair[1]})"
            if key not in pairs_seen:
                pairs_seen[key] = {"100": 0, "10000": 0, "1000000": 0, "time": 0.0, "found": False}
            mc = str(r.max_coeff)
            if mc in pairs_seen[key] and not r.found_relation:
                pairs_seen[key][mc] = max(pairs_seen[key][mc], r.degree)
            pairs_seen[key]["time"] += r.elapsed_seconds
            if r.found_relation:
                pairs_seen[key]["found"] = True

        for key, data in pairs_seen.items():
            time_str = f"{data['time']:.0f}s" if data['time'] < 60 else f"{data['time']/60:.1f}min"
            status = " ⭐" if data["found"] else ""
            lines.append(
                f"| {key}{status} | {data['100']} | {data['10000']} | {data['1000000']} | {time_str} |"
            )

        # Dettaglio relazioni trovate
        found_results = [r for r in self.results if r.found_relation]
        if found_results:
            lines.extend([
                "",
                "---",
                "",
                "## ⭐ Relazioni trovate",
                "",
            ])
            for r in found_results:
                lines.extend([
                    f"### ({r.pair[0]}, {r.pair[1]}) — grado {r.degree}",
                    f"- **Coefficienti:** `{r.relation}`",
                    f"- **Residuo lavoro:** {r.residual_working:.5e}",
                    f"- **Residuo verifica:** {r.residual_verification:.5e}",
                    f"- **Precisione:** {r.precision_digits} cifre (lavoro), "
                    f"{r.precision_digits * 3 // 2} cifre (verifica)",
                    f"- **Tempo:** {r.elapsed_seconds:.1f}s",
                    "",
                ])

        # Statistiche
        total_time = sum(r.elapsed_seconds for r in self.results)
        total_searches = len(self.results)
        lines.extend([
            "",
            "---",
            "",
            "## Statistiche",
            "",
            f"- **Ricerche PSLQ totali:** {total_searches}",
            f"- **Tempo totale:** {total_time/3600:.2f} ore ({total_time:.0f}s)",
            f"- **Relazioni trovate:** {len(found_results)}",
            f"- **Coppie esplorate:** {len(pairs_seen)}",
            "",
            "---",
            "",
            "## Riproducibilità",
            "",
            "```bash",
            "source ~/bridge_eq_env/bin/activate",
            "cd ~/bridge_equation",
            f"python3 run_deep_search_v2.py --max-hours-per-pair {self.max_hours_per_pair}",
            "```",
            "",
            "## Riferimenti",
            "",
            "- Bailey & Ferguson, 'Numerical results on relations between fundamental constants' (1989)",
            "- Ferguson, Bailey & Arno, 'Analysis of PSLQ' (1999)",
            "- Bailey & Borwein, 'PSLQ: An Algorithm to Discover Integer Relations' (2009)",
        ])

        report_path.write_text("\n".join(lines))
        self.log(f"\n  Report salvato in: {report_path}")
