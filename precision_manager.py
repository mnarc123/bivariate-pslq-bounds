"""
Gestione della precisione multi-livello.

STRATEGIA: Per ogni coppia e grado, calcola la precisione minima necessaria,
poi aggiungi un margine di sicurezza. Usa una strategia "a scaletta":
prima testa a precisione moderata, poi se serve aumenta.
"""

import math
import mpmath
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PrecisionPlan:
    """Piano di precisione per una ricerca PSLQ."""
    n_monomials: int
    max_coeff_digits: int       # D: log10 della norma max dei coefficienti
    working_digits: int         # cifre di lavoro per PSLQ
    verification_digits: int    # cifre per verifica indipendente
    is_feasible: bool           # True se il calcolo è fattibile in tempo ragionevole
    estimated_hours: float      # stima del tempo in ore


# Parametro di calibrazione globale — calibrato da benchmark reale
# su i7-12700F con mpmath PSLQ, 2026-02-18.
# Modello: T(ore) ≈ k · N^alpha · P^beta
# dove N = n_monomials, P = working_digits
# Benchmark (π,e): grado 10-20, errore <1% su tutti i punti.
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
    Calcola il piano di precisione ottimale per una coppia a un dato grado.

    Args:
        degree: grado massimo del polinomio
        max_coeff: norma massima dei coefficienti
        safety_factor: moltiplicatore di sicurezza sulla precisione
        max_hours: tempo massimo consentito in ore

    Returns:
        PrecisionPlan con tutti i parametri calcolati
    """
    n_monomials = (degree + 1) * (degree + 2) // 2
    max_coeff_digits = max(1, math.ceil(math.log10(max_coeff + 1)))

    # Precisione minima: N × D (regola di Bailey)
    min_digits = n_monomials * max_coeff_digits

    # Precisione di lavoro con margine di sicurezza
    working_digits = int(min_digits * safety_factor)

    # Precisione di verifica: almeno 1.5x la working
    verification_digits = int(working_digits * 1.5)

    # Stima del tempo
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
    Trova il grado massimo fattibile per una coppia bivariata.
    Fa una ricerca binaria sul grado.
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
    Calibra i parametri k, alpha, beta dai dati di benchmark.

    benchmark_data: lista di (n_monomials, working_digits, elapsed_seconds)
    """
    global CALIBRATION_K, CALIBRATION_ALPHA, CALIBRATION_BETA

    if len(benchmark_data) < 2:
        # Con un solo punto, aggiusta solo k mantenendo alpha e beta
        if benchmark_data:
            n, p, t = benchmark_data[0]
            t_hours = t / 3600.0
            CALIBRATION_K = t_hours / (n ** CALIBRATION_ALPHA * p ** CALIBRATION_BETA)
        return

    # Con più punti, fit log-lineare: log(T) = log(k) + alpha*log(N) + beta*log(P)
    # Semplificazione: se P è costante tra i benchmark, fit solo su N
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
        # Fallback: usa solo il punto più grande
        n, p, t = max(benchmark_data, key=lambda x: x[0])
        t_hours = t / 3600.0
        CALIBRATION_K = t_hours / (n ** CALIBRATION_ALPHA * p ** CALIBRATION_BETA)
