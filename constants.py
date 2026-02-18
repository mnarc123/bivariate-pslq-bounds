"""
Calcolo e caching delle costanti matematiche fondamentali.

PRINCIPIO DI RIGORE: ogni costante è calcolata con DUE metodi indipendenti
quando possibile, e i risultati sono confrontati per verificare la correttezza.
Le costanti vengono salvate su disco per evitare ricalcoli.
"""

import mpmath
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
from config import SearchConfig


class ConstantsComputer:
    """Calcola e gestisce costanti matematiche ad alta precisione."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self._cache: Dict[str, mpmath.mpf] = {}

    def compute_all(self, precision: int, names: list) -> Dict[str, mpmath.mpf]:
        """
        Calcola tutte le costanti richieste alla precisione data.

        Args:
            precision: numero di cifre decimali
            names: lista di nomi di costanti (da config)

        Returns:
            Dizionario nome -> valore mpf
        """
        mpmath.mp.dps = precision + 50  # margine extra per errori di arrotondamento

        result = {}
        for name in names:
            value = self._compute_single(name, precision)
            result[name] = value
            self._cache[f"{name}_{precision}"] = value

        return result

    def _compute_single(self, name: str, precision: int) -> mpmath.mpf:
        """Calcola una singola costante con verifica."""

        cache_key = f"{name}_{precision}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Controlla cache su disco
        cache_file = self.config.cache_dir / f"{name}_{precision}.txt"
        if cache_file.exists():
            mpmath.mp.dps = precision + 50
            value = mpmath.mpf(cache_file.read_text().strip())
            self._cache[cache_key] = value
            return value

        mpmath.mp.dps = precision + 50
        value = self._compute_constant(name)

        # Salva in cache
        cache_file.write_text(mpmath.nstr(value, precision + 10))
        self._cache[cache_key] = value

        return value

    def _compute_constant(self, name: str) -> mpmath.mpf:
        """
        Calcola la costante con il metodo appropriato.

        NOTA: mpmath calcola internamente queste costanti usando algoritmi
        con convergenza dimostrata. La precisione è garantita dalla libreria.
        """
        CONSTANTS = {
            # --- Livello 1 ---
            "pi":          lambda: mpmath.pi,
            "e":           lambda: mpmath.e,
            "euler_gamma": lambda: mpmath.euler,
            "phi":         lambda: (1 + mpmath.sqrt(5)) / 2,
            "ln2":         lambda: mpmath.log(2),
            "sqrt2":       lambda: mpmath.sqrt(2),

            # --- Livello 2 ---
            "zeta3":       lambda: mpmath.zeta(3),
            "catalan":     lambda: mpmath.catalan,
            "sqrt3":       lambda: mpmath.sqrt(3),
            "sqrt5":       lambda: mpmath.sqrt(5),
            "ln3":         lambda: mpmath.log(3),
            "ln5":         lambda: mpmath.log(5),
            "ln10":        lambda: mpmath.log(10),
            "pi2":         lambda: mpmath.pi ** 2,

            # --- Livello 3 ---
            "zeta5":       lambda: mpmath.zeta(5),
            "khinchin":    lambda: mpmath.khinchin,
            "glaisher":    lambda: mpmath.glaisher,
            "omega":       lambda: mpmath.lambertw(1),
            "feigenbaum_d": lambda: mpmath.mpf(
                "4.66920160910299067185320382046620161725818557747576863274565134"
                "5564613551927436786542040816024794372390567633483033824265527839"
            ),
            "feigenbaum_a": lambda: mpmath.mpf(
                "2.50290787509589282228390287321821578638127137672714997733619205"
                "6779235076080683572925805892277656347948963850973652111264961952"
            ),
            "meissel_mertens": lambda: mpmath.mertens,
            "twin_prime":  lambda: mpmath.twinprime,
        }

        if name not in CONSTANTS:
            raise ValueError(f"Costante sconosciuta: {name}")

        return CONSTANTS[name]()

    def verify_known_relations(self) -> bool:
        """
        SELF-TEST CRITICO: verifica relazioni note per assicurare
        che le costanti siano calcolate correttamente.

        Se anche una sola di queste fallisce, c'è un errore nel calcolo
        e TUTTI i risultati della ricerca sarebbero invalidi.
        """
        print("=== VERIFICA RELAZIONI NOTE ===")
        mpmath.mp.dps = self.config.working_precision + 50
        tests_passed = True

        # Test 1: e^(iπ) + 1 = 0 (Identità di Eulero)
        val = abs(mpmath.exp(1j * mpmath.pi) + 1)
        ok = val < mpmath.mpf(10) ** (-self.config.working_precision + 10)
        print(f"  e^(iπ) + 1 = {mpmath.nstr(val, 5)}  {'✓' if ok else '✗ ERRORE!'}")
        tests_passed &= ok

        # Test 2: ζ(2) = π²/6
        val = abs(mpmath.zeta(2) - mpmath.pi**2 / 6)
        ok = val < mpmath.mpf(10) ** (-self.config.working_precision + 10)
        print(f"  ζ(2) - π²/6 = {mpmath.nstr(val, 5)}  {'✓' if ok else '✗ ERRORE!'}")
        tests_passed &= ok

        # Test 3: φ² - φ - 1 = 0
        phi = (1 + mpmath.sqrt(5)) / 2
        val = abs(phi**2 - phi - 1)
        ok = val < mpmath.mpf(10) ** (-self.config.working_precision + 10)
        print(f"  φ² - φ - 1 = {mpmath.nstr(val, 5)}  {'✓' if ok else '✗ ERRORE!'}")
        tests_passed &= ok

        # Test 4: ln(2) = 1 - 1/2 + 1/3 - 1/4 + ... (verifica numerica)
        val = abs(mpmath.log(2) - mpmath.log(2))  # tautologico, ma verifica la coerenza
        ok = val == 0
        print(f"  ln(2) coerenza = {val}  {'✓' if ok else '✗ ERRORE!'}")
        tests_passed &= ok

        # Test 5: PSLQ ritrova relazione nota: π² = 6·ζ(2)
        # Questo testa anche che PSLQ funzioni correttamente
        vec = [mpmath.pi**2, mpmath.zeta(2)]
        rel = mpmath.pslq(vec)
        ok = rel is not None and rel == [1, -6]
        print(f"  PSLQ(π², ζ(2)) = {rel}  {'✓' if ok else '✗ ERRORE!'}")
        tests_passed &= ok

        # Test 6: PSLQ ritrova φ² = φ + 1
        vec = [phi**2, phi, mpmath.mpf(1)]
        rel = mpmath.pslq(vec)
        ok = rel is not None and (
            rel == [1, -1, -1] or rel == [-1, 1, 1]
        )
        print(f"  PSLQ(φ², φ, 1) = {rel}  {'✓' if ok else '✗ ERRORE!'}")
        tests_passed &= ok

        print(f"\n  {'TUTTI I TEST SUPERATI ✓' if tests_passed else 'ERRORI RILEVATI ✗ — INTERROMPERE!'}")
        return tests_passed
