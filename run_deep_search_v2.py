#!/usr/bin/env python3
"""
EQUAZIONE PONTE — Fase 2: Ricerca Profonda ad Alto Grado

Esegue PSLQ sulle coppie di costanti trascendenti fino al grado
massimo consentito dal budget di tempo, con checkpoint/resume.

Uso:
    python3 run_deep_search_v2.py                          # default 24h per coppia
    python3 run_deep_search_v2.py --max-hours-per-pair 48  # 48h per coppia
    python3 run_deep_search_v2.py --pair pi+e              # solo una coppia
    python3 run_deep_search_v2.py --estimate               # solo stima tempi
    python3 run_deep_search_v2.py --benchmark              # calibra stime di tempo
"""

import argparse
import sys
import time
import mpmath
from config import SearchConfig
from deep_engine import DeepSearchEngine
from precision_manager import (
    compute_precision_plan, find_max_feasible_degree,
    calibrate_from_benchmarks, CALIBRATION_K, CALIBRATION_ALPHA, CALIBRATION_BETA,
)


def run_benchmark():
    """
    Esegue un benchmark PSLQ su (π,e) a gradi crescenti per calibrare
    il modello di stima dei tempi.
    """
    print("\n=== BENCHMARK PSLQ su (π, e) ===\n")
    print(f"{'Grado':>8} {'N mono':>8} {'Cifre':>8} {'Tempo':>12} {'Risultato':>12}")
    print("-" * 54)

    benchmark_data = []

    for d in [10, 12, 15, 18, 20]:
        n = (d + 1) * (d + 2) // 2
        # Precisione: N * 2 * safety=2 per |c|≤100
        digits = n * 2 * 2
        mpmath.mp.dps = digits + 50

        alpha = mpmath.pi
        beta = mpmath.e

        # Genera monomiali
        ap = [mpmath.mpf(1)]
        bp = [mpmath.mpf(1)]
        for k in range(1, d + 1):
            ap.append(ap[-1] * alpha)
            bp.append(bp[-1] * beta)

        vals = []
        for td in range(d + 1):
            for i in range(td + 1):
                j = td - i
                vals.append(ap[i] * bp[j])

        t0 = time.time()
        r = mpmath.pslq(vals, maxcoeff=100)
        t1 = time.time()
        elapsed = t1 - t0

        result_str = "None" if r is None else str(r)[:30]
        time_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"
        print(f"{d:>8} {n:>8} {digits:>8} {time_str:>12} {result_str:>12}")

        benchmark_data.append((n, digits, elapsed))

    # Calibra
    print("\n=== CALIBRAZIONE ===")
    print(f"  Prima: k={CALIBRATION_K:.2e}, α={CALIBRATION_ALPHA:.2f}, β={CALIBRATION_BETA:.2f}")
    calibrate_from_benchmarks(benchmark_data)
    from precision_manager import CALIBRATION_K as K, CALIBRATION_ALPHA as A, CALIBRATION_BETA as B
    print(f"  Dopo:  k={K:.2e}, α={A:.2f}, β={B:.2f}")

    # Mostra stime calibrate
    print("\n=== STIME CALIBRATE ===\n")
    print(f"{'Grado':>8} {'N mono':>10} {'Cifre':>8} {'Tempo stimato':>16}")
    print("-" * 46)
    for d in [10, 15, 20, 25, 30, 40, 50]:
        plan = compute_precision_plan(d, max_coeff=100)
        if plan.estimated_hours < 1/60:
            time_str = f"{plan.estimated_hours*3600:.1f}s"
        elif plan.estimated_hours < 1:
            time_str = f"{plan.estimated_hours*60:.1f} min"
        elif plan.estimated_hours < 24:
            time_str = f"{plan.estimated_hours:.1f} ore"
        else:
            time_str = f"{plan.estimated_hours/24:.1f} giorni"
        print(f"{d:>8} {plan.n_monomials:>10} {plan.working_digits:>8} {time_str:>16}")

    return benchmark_data


def estimate_only():
    """Stampa una stima dei gradi raggiungibili e dei tempi."""
    print("\n=== STIMA GRADI RAGGIUNGIBILI ===\n")
    print(f"{'Budget (ore)':>14} {'|c|≤100':>12} {'|c|≤10⁴':>12} {'|c|≤10⁶':>12}")
    print("-" * 54)
    for hours in [1, 4, 12, 24, 48, 96, 168]:
        d100 = find_max_feasible_degree(100, hours)
        d10k = find_max_feasible_degree(10000, hours)
        d1M = find_max_feasible_degree(10**6, hours)
        print(f"{hours:>12}h  grado {d100:>4}  grado {d10k:>4}  grado {d1M:>4}")

    print("\n=== DETTAGLIO PER (π, e), |c|≤100 ===\n")
    print(f"{'Grado':>8} {'Monomiali':>10} {'Cifre':>8} {'Tempo stimato':>16}")
    print("-" * 46)
    for d in [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
        plan = compute_precision_plan(d, max_coeff=100)
        if plan.estimated_hours < 1/60:
            time_str = f"{plan.estimated_hours*3600:.1f}s"
        elif plan.estimated_hours < 1:
            time_str = f"{plan.estimated_hours*60:.1f} min"
        elif plan.estimated_hours < 24:
            time_str = f"{plan.estimated_hours:.1f} ore"
        else:
            time_str = f"{plan.estimated_hours/24:.1f} giorni"
        feas = "✓" if plan.is_feasible else "✗"
        print(f"{d:>8} {plan.n_monomials:>10} {plan.working_digits:>8} {time_str:>16}  {feas}")


def main():
    parser = argparse.ArgumentParser(
        description="Equazione Ponte — Fase 2: Ricerca Profonda"
    )
    parser.add_argument(
        "--max-hours-per-pair", type=float, default=24.0,
        help="Ore massime per ogni coppia di costanti (default: 24)"
    )
    parser.add_argument(
        "--pair", type=str, default=None,
        help="Cerca solo questa coppia (es. 'pi+e', 'pi+euler_gamma')"
    )
    parser.add_argument(
        "--estimate", action="store_true",
        help="Stampa solo la stima dei tempi, non eseguire"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Esegui benchmark per calibrare le stime di tempo"
    )
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        return

    if args.estimate:
        estimate_only()
        return

    config = SearchConfig()
    engine = DeepSearchEngine(config, max_hours_per_pair=args.max_hours_per_pair)

    t_start = time.time()
    engine.run(pair_filter=args.pair)
    t_total = time.time() - t_start

    hours = t_total / 3600
    print(f"\nRicerca completata in {hours:.2f} ore ({t_total:.0f}s).")


if __name__ == "__main__":
    main()
