"""
Generatore di monomiali multivariati per la ricerca PSLQ.

Data una lista di costanti [c₁, c₂, ..., cₘ] e un grado massimo d,
genera tutti i monomiali c₁^a₁ · c₂^a₂ · ... · cₘ^aₘ con Σaᵢ ≤ d.

NOTA: l'inclusione del termine costante "1" (tutti gli esponenti zero)
è ESSENZIALE — senza di esso non si trovano relazioni affini (con termine noto).
"""

from itertools import combinations_with_replacement
from typing import List, Tuple, Dict
import mpmath


def generate_exponent_tuples(n_vars: int, max_degree: int) -> List[Tuple[int, ...]]:
    """
    Genera tutte le tuple di esponenti (a₁, ..., aₙ) con Σaᵢ ≤ max_degree.

    Usa il metodo "stars and bars" per enumerare efficacemente.

    Returns:
        Lista di tuple, ciascuna di lunghezza n_vars.
        La prima tupla è sempre (0, 0, ..., 0) — il termine costante "1".
    """
    tuples = []
    for total_degree in range(max_degree + 1):
        # Genera tutte le partizioni di total_degree in n_vars parti non-negative
        for combo in _partitions(total_degree, n_vars):
            tuples.append(combo)
    return tuples


def _partitions(total: int, n_parts: int) -> List[Tuple[int, ...]]:
    """Genera tutte le composizioni deboli di 'total' in 'n_parts' parti."""
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
    Calcola i valori numerici di tutti i monomiali.

    Args:
        constant_values: dizionario nome -> valore mpf
        constant_names: nomi delle costanti da usare (ordine fisso)
        max_degree: grado massimo
        max_monomials: numero massimo di monomiali (per controllare i tempi PSLQ)

    Returns:
        - values: lista di valori mpf dei monomiali
        - exponents: lista delle tuple di esponenti corrispondenti
        - labels: lista di stringhe leggibili (es. "π²·e·γ")

    NOTA: Il primo valore è sempre 1 (termine costante).
    """
    n_vars = len(constant_names)
    all_exponents = generate_exponent_tuples(n_vars, max_degree)

    if len(all_exponents) > max_monomials:
        # Strategia: dare priorità ai gradi bassi e ai monomiali "misti"
        # (prodotti di costanti diverse, che sono più probabilmente inesplorati)
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
    Prioritizza monomiali: prima grado basso, poi monomiali "misti"
    (con più variabili coinvolte), poi grado alto.
    """
    def sort_key(exp_tuple):
        total_deg = sum(exp_tuple)
        n_nonzero = sum(1 for e in exp_tuple if e > 0)
        max_single = max(exp_tuple) if exp_tuple else 0
        # Priorità: grado basso, molte variabili coinvolte, esponenti bassi
        return (total_deg, -n_nonzero, max_single)

    sorted_exp = sorted(exponents, key=sort_key)
    return sorted_exp[:max_count]


def count_monomials(n_vars: int, max_degree: int) -> int:
    """
    Calcola il numero totale di monomiali C(n_vars + max_degree, max_degree).
    Utile per stimare la dimensione dello spazio di ricerca.
    """
    from math import comb
    return comb(n_vars + max_degree, max_degree)
