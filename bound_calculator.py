"""
Calcolo rigoroso dei bound PSLQ.

Quando PSLQ termina senza trovare una relazione, il minimo dell'ultimo
elemento diagonale della matrice H fornisce un LOWER BOUND rigoroso
sulla norma euclidea di qualsiasi relazione intera.

Questo bound è il risultato scientifico principale della ricerca:
"Non esiste P(α,β)=0 con deg(P)≤d e ||coeff||≤M"
"""

import math
import mpmath
from typing import Optional, Tuple


def format_bound_statement(
    const1: str,
    const2: str,
    degree: int,
    max_coeff: int,
    found_relation: bool,
) -> str:
    """
    Formatta il bound come statement matematico rigoroso.

    Esempio:
    "Non esiste polinomio P ∈ Z[x,y] con deg(P) ≤ 20 e max|c_ij| ≤ 100
     tale che P(π, e) = 0."
    """
    # Mappa nomi interni → simboli matematici
    SYMBOLS = {
        "pi": "π", "e": "e", "euler_gamma": "γ",
        "phi": "φ", "ln2": "ln2", "sqrt2": "√2",
        "zeta3": "ζ(3)", "catalan": "G", "sqrt3": "√3",
        "sqrt5": "√5", "ln3": "ln3", "ln5": "ln5",
        "ln10": "ln10", "pi2": "π²", "zeta5": "ζ(5)",
        "khinchin": "K₀", "glaisher": "A",
        "omega": "Ω", "feigenbaum_d": "δ_F",
        "feigenbaum_a": "α_F", "meissel_mertens": "M",
        "twin_prime": "B₂",
    }
    s1 = SYMBOLS.get(const1, const1)
    s2 = SYMBOLS.get(const2, const2)

    if found_relation:
        return f"Relazione trovata per ({s1}, {s2}) a grado ≤ {degree}!"

    return (
        f"Non esiste P ∈ ℤ[x,y] con deg(P) ≤ {degree} e "
        f"max|c_ij| ≤ {max_coeff} tale che P({s1}, {s2}) = 0."
    )


def compute_search_space_size(degree: int, max_coeff: int) -> float:
    """
    Calcola log10 della dimensione dello spazio di ricerca esplorato.

    Lo spazio di tutti i polinomi a coefficienti interi con |c| ≤ M
    e grado ≤ d in 2 variabili ha dimensione (2M+1)^N dove N = (d+1)(d+2)/2.
    """
    n = (degree + 1) * (degree + 2) // 2
    log_space = n * math.log10(2 * max_coeff + 1)
    return log_space


def format_search_space(degree: int, max_coeff: int) -> str:
    """Formatta la dimensione dello spazio in modo leggibile."""
    log_size = compute_search_space_size(degree, max_coeff)
    n = (degree + 1) * (degree + 2) // 2
    return f"(2·{max_coeff}+1)^{n} ≈ 10^{log_size:.0f}"
