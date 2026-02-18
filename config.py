"""
Configurazione del progetto Equazione Ponte.
Tutti i parametri di ricerca sono centralizzati qui.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

@dataclass
class SearchConfig:
    """Parametri della ricerca PSLQ."""

    # === PRECISIONE ===
    # La precisione in cifre decimali per il calcolo delle costanti.
    # REGOLA: Per trovare una relazione con coefficienti di norma massima D
    # in un vettore di dimensione N, servono almeno N*log10(D) cifre.
    # Con N=50 e D=1000, servono ~150 cifre. Usiamo un margine di sicurezza 3x.
    working_precision: int = 500  # cifre decimali di lavoro
    verification_precision: int = 1000  # cifre per la verifica indipendente

    # === COSTANTI DA INCLUDERE ===
    # Livello 1: le costanti più fondamentali (ricerca prioritaria)
    constants_level_1: List[str] = field(default_factory=lambda: [
        "pi", "e", "euler_gamma", "phi",  # golden ratio
        "ln2", "sqrt2",
    ])

    # Livello 2: costanti analitiche importanti
    constants_level_2: List[str] = field(default_factory=lambda: [
        "zeta3",          # Costante di Apéry ζ(3)
        "catalan",        # Costante di Catalan G
        "sqrt3", "sqrt5",
        "ln3", "ln5", "ln10",
        "pi2",            # π² (trattato come costante indipendente per efficienza)
    ])

    # Livello 3: costanti più esotiche (ricerca secondaria)
    constants_level_3: List[str] = field(default_factory=lambda: [
        "zeta5",          # ζ(5)
        "khinchin",       # Costante di Khinchin K₀
        "glaisher",       # Costante di Glaisher-Kinkelin A
        "omega",          # Costante Omega Ω (soluzione di x*e^x = 1)
        "feigenbaum_d",   # Prima costante di Feigenbaum δ
        "feigenbaum_a",   # Seconda costante di Feigenbaum α
        "meissel_mertens", # Costante di Meissel-Mertens M
        "twin_prime",     # Costante dei primi gemelli C₂
    ])

    # === GRADI DI RICERCA ===
    # Grado massimo del polinomio nelle costanti.
    # Grado 1 = relazioni lineari (molto esplorate, improbabile trovare nuove)
    # Grado 2 = relazioni quadratiche (poco esplorate per combinazioni miste)
    # Grado 3 = relazioni cubiche (territorio quasi inesplorato)
    # Grado 4+ = computazionalmente costoso, solo per sottoinsiemi piccoli
    max_degree: int = 4

    # === COEFFICIENTI ===
    # Soglia della norma dei coefficienti interi.
    # Relazioni con coefficienti grandi sono meno "interessanti" matematicamente.
    # Bailey usa tipicamente soglia ~10^6 per risultati significativi.
    max_coefficient_norm: int = 10**6

    # === CRITERI DI VALIDITÀ ===
    # Una relazione è considerata "candidata" se il residuo è < 10^(-threshold)
    # quando calcolata a precision=working_precision.
    # REGOLA EMPIRICA (Bailey): un "drop" di 20+ ordini di grandezza nel residuo
    # rispetto al rumore di fondo indica quasi certamente una relazione reale.
    residual_threshold_log10: int = -50  # |residuo| < 10^(-50) a 500 cifre

    # Per la verifica: ricalcolo a precision doppia
    verification_residual_threshold_log10: int = -100  # |residuo| < 10^(-100) a 1000 cifre

    # === STRATEGIA DI RICERCA ===
    # Numero massimo di costanti per singola ricerca PSLQ
    # (la complessità è polinomiale in N ma il tempo pratico cresce rapidamente)
    max_constants_per_search: int = 5

    # Numero massimo di monomiali nel vettore PSLQ
    # (oltre ~100 diventa molto lento anche con 500 cifre)
    max_monomials_per_vector: int = 80

    # === PERCORSI ===
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
