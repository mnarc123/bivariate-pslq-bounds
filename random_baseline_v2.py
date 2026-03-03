#!/usr/bin/env python3
"""
Expanded Random Baseline Generator — 200 pseudo-generic pairs.

Generates 200 pairs of pseudo-generic transcendental constants and runs PSLQ
at degrees 3–12 with maxcoeff=100, extracting the precision fraction λ for
each run. The first 30 pairs match the original random_baseline.py for
backwards compatibility.

This provides a high-power null-model baseline for statistical comparison
with the 138 real-constant λ values from near_miss_report.json.

Usage:
    python3 random_baseline_v2.py

Output:
    results/statistics/baseline_v2_checkpoint.json  (incremental)
    results/statistics/baseline_v2_pairs.json       (pair definitions)
    results/statistics/baseline_v2_results.json     (final results)
"""

import mpmath
import math
import json
import time
import os
import sys
from pathlib import Path
from collections import namedtuple

# ---------------------------------------------------------------------------
# Pair definitions
# ---------------------------------------------------------------------------
# Each pair is (func1_name, prime1, func2_name, prime2, label).
# The first 30 match the original random_baseline.py exactly.
# func1 and func2 are ALWAYS from different function families to avoid
# algebraic relations.

FUNCTIONS = {
    'sin':    lambda p: mpmath.sin(mpmath.sqrt(p)),
    'cos':    lambda p: mpmath.cos(mpmath.sqrt(p)),
    'log':    lambda p: mpmath.log(mpmath.sqrt(p) + 1),
    'log2':   lambda p: mpmath.log(mpmath.sqrt(p) + 2),
    'log3':   lambda p: mpmath.log(mpmath.sqrt(p) + 3),
    'tanh':   lambda p: mpmath.tanh(mpmath.sqrt(p)),
    'erf':    lambda p: mpmath.erf(mpmath.sqrt(p)),
    'J0':     lambda p: mpmath.besselj(0, mpmath.sqrt(p)),
    'J1':     lambda p: mpmath.besselj(1, mpmath.sqrt(p)),
    'exp1':   lambda p: mpmath.exp(1 / mpmath.sqrt(p)),
    'sinh':   lambda p: mpmath.sinh(1 / mpmath.sqrt(p)),
    'cosh':   lambda p: mpmath.cosh(1 / mpmath.sqrt(p)),
    'arctan': lambda p: mpmath.atan(mpmath.sqrt(p)),
    'Ai':     lambda p: mpmath.airyai(mpmath.sqrt(p)),
    'Li2':    lambda p: mpmath.polylog(2, 1 / mpmath.sqrt(p)),
    'zetax':  lambda p: mpmath.zeta(1 + 1 / mpmath.sqrt(p)),
    'Gamma':  lambda p: mpmath.gamma(mpmath.sqrt(p)),
}

# Function families — pairs must use functions from DIFFERENT families
FAMILIES = {
    'trig':       ['sin', 'cos', 'arctan'],
    'hyp':        ['tanh', 'sinh', 'cosh'],
    'exp':        ['exp1'],
    'log':        ['log', 'log2', 'log3'],
    'error':      ['erf'],
    'bessel':     ['J0', 'J1'],
    'special':    ['Ai', 'Li2', 'zetax', 'Gamma'],
}

def _family_of(fname):
    for fam, members in FAMILIES.items():
        if fname in members:
            return fam
    return fname  # unknown = unique family


# First 50 primes (enough for 200 pairs needing distinct primes per pair)
PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271,
    277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
    367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443,
    449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
    547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619,
    631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719,
    727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821,
    823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911,
    919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
    1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
    1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181,
    1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277,
    1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361,
    1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451,
    1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531,
]

# --- Original 30 pairs (exact match with random_baseline.py) ---
ORIGINAL_30 = [
    ('sin', 2,   'cos', 3),
    ('log2', 5,  'tanh', 7),
    ('J0', 11,   'erf', 13),
    ('sin', 17,  'exp1', 19),
    ('log2', 23, 'cos', 29),
    ('tanh', 31, 'sin', 37),
    ('J1', 41,   'log', 43),
    ('erf', 47,  'cos', 53),
    ('sin', 59,  'tanh', 61),
    ('log3', 67, 'J0', 71),
    ('exp1', 73, 'sin', 79),
    ('cos', 83,  'erf', 89),
    ('tanh', 97, 'log', 101),
    ('J0', 103,  'sin', 107),
    ('erf', 109, 'cos', 113),
    ('sin', 127, 'exp1', 131),
    ('log2', 137, 'tanh', 139),
    ('cos', 149, 'J1', 151),
    ('sin', 157, 'erf', 163),
    ('tanh', 167, 'log', 173),
    ('exp1', 179, 'cos', 181),
    ('J0', 191,  'sin', 193),
    ('erf', 197, 'tanh', 199),
    ('sin', 211, 'log2', 223),
    ('cos', 227, 'exp1', 229),
    ('tanh', 233, 'J0', 239),
    ('log', 241, 'sin', 251),
    ('erf', 257, 'cos', 263),
    ('sin', 269, 'tanh', 271),
    ('J1', 277,  'log3', 281),
]


def _generate_new_170_pairs():
    """Generate 170 new pairs using expanded function set, all cross-family.

    Primes may be reused across different pairs but are always distinct
    within each pair. Each (f1, p1, f2, p2) tuple is unique.
    """
    # Build cross-family function combos
    func_names = list(FUNCTIONS.keys())
    cross_family_combos = []
    for f1 in func_names:
        for f2 in func_names:
            if _family_of(f1) != _family_of(f2):
                cross_family_combos.append((f1, f2))

    # Track existing pair signatures to avoid duplicates
    existing = set()
    for f1, p1, f2, p2 in ORIGINAL_30:
        existing.add((f1, p1, f2, p2))

    pairs = []
    combo_idx = 0
    # Use a broad selection of primes for variety
    prime_pool = PRIMES[:80]  # 80 primes gives plenty of combinations

    for combo_idx, (f1, f2) in enumerate(cross_family_combos):
        if len(pairs) >= 170:
            break
        for pi_idx in range(len(prime_pool)):
            if len(pairs) >= 170:
                break
            p1 = prime_pool[pi_idx]
            # Pick a different prime for second constant
            p2 = prime_pool[(pi_idx + combo_idx + 1) % len(prime_pool)]
            if p1 == p2:
                p2 = prime_pool[(pi_idx + combo_idx + 2) % len(prime_pool)]
            if p1 == p2:
                continue
            sig = (f1, p1, f2, p2)
            if sig not in existing:
                existing.add(sig)
                pairs.append(sig)
                break  # One pair per function combo, move to next combo

    # If we still need more pairs, cycle through combos with different primes
    round_num = 1
    while len(pairs) < 170:
        for combo_idx, (f1, f2) in enumerate(cross_family_combos):
            if len(pairs) >= 170:
                break
            pi_idx = round_num * 7 + combo_idx * 3  # spread across primes
            p1 = prime_pool[pi_idx % len(prime_pool)]
            p2 = prime_pool[(pi_idx + round_num + 1) % len(prime_pool)]
            if p1 == p2:
                p2 = prime_pool[(pi_idx + round_num + 2) % len(prime_pool)]
            if p1 == p2:
                continue
            sig = (f1, p1, f2, p2)
            if sig not in existing:
                existing.add(sig)
                pairs.append(sig)
        round_num += 1
        if round_num > 20:  # safety valve
            break

    return pairs


def get_all_200_pairs():
    """Return list of 200 pair definitions: (func1_name, prime1, func2_name, prime2)."""
    new_170 = _generate_new_170_pairs()
    return ORIGINAL_30 + new_170


def pair_label(f1, p1, f2, p2):
    return f"{f1}(√{p1}), {f2}(√{p2})"


# ---------------------------------------------------------------------------
# Worker function — runs in a separate process
# ---------------------------------------------------------------------------

def _generate_monomials(alpha, beta, degree):
    """Generate bivariate monomial vector (1, α, β, α², αβ, β², ...)."""
    values = []
    exponents = []
    for total_deg in range(degree + 1):
        for j in range(total_deg + 1):
            i = total_deg - j
            val = alpha**i * beta**j
            values.append(val)
            exponents.append((i, j))
    return values, exponents


def run_single_task(args):
    """
    Execute a single PSLQ baseline run in a worker process.

    Args is a tuple: (pair_index, func1_name, prime1, func2_name, prime2, degree, max_coeff)
    Returns a dict with the result.
    """
    pair_index, f1_name, p1, f2_name, p2, degree, max_coeff = args

    N = (degree + 1) * (degree + 2) // 2
    D = math.ceil(math.log10(max_coeff + 1))
    working_digits = 2 * N * D

    mpmath.mp.dps = working_digits + 50

    result = {
        'pair_index': pair_index,
        'pair_label': pair_label(f1_name, p1, f2_name, p2),
        'degree': degree,
        'max_coeff': max_coeff,
        'n_monomials': N,
        'working_digits': working_digits,
    }

    t0 = time.time()

    try:
        f1 = FUNCTIONS[f1_name]
        f2 = FUNCTIONS[f2_name]
        alpha = f1(p1)
        beta = f2(p2)
    except Exception as exc:
        result['status'] = 'error_constants'
        result['error'] = str(exc)
        result['elapsed'] = round(time.time() - t0, 2)
        return result

    # Check for problematic values
    if abs(alpha) < mpmath.mpf(10)**(-working_digits // 2) or \
       abs(beta) < mpmath.mpf(10)**(-working_digits // 2):
        result['status'] = 'error_small'
        result['error'] = 'constant too small'
        result['elapsed'] = round(time.time() - t0, 2)
        return result

    values, exponents = _generate_monomials(alpha, beta, degree)

    # Run patched PSLQ
    try:
        _proj = str(Path(__file__).parent)
        if _proj not in sys.path:
            sys.path.insert(0, _proj)
        # Import inside worker to avoid pickling issues
        import pslq_bounds_fast as _pbf
        pslq_result = _pbf.pslq_with_bound(values, maxcoeff=max_coeff, maxsteps=0)
    except Exception as exc:
        result['status'] = 'error_pslq'
        result['error'] = str(exc)
        result['elapsed'] = round(time.time() - t0, 2)
        return result

    if pslq_result is None:
        result['status'] = 'error_pslq_none'
        result['elapsed'] = round(time.time() - t0, 2)
        return result

    found_relation = pslq_result.relation is not None
    result['found_relation'] = found_relation

    if found_relation:
        # Spurious relation found — record but mark as excluded
        result['status'] = 'spurious'
        result['lambda_value'] = None
        result['elapsed'] = round(time.time() - t0, 2)
        return result

    # Extract best candidate and compute residual at double precision
    candidate = pslq_result.best_candidate
    if candidate is None:
        result['status'] = 'no_candidate'
        result['elapsed'] = round(time.time() - t0, 2)
        return result

    # Verify at double precision
    double_digits = working_digits * 2
    mpmath.mp.dps = double_digits + 50
    try:
        alpha_2x = FUNCTIONS[f1_name](p1)
        beta_2x = FUNCTIONS[f2_name](p2)
    except Exception:
        alpha_2x = alpha
        beta_2x = beta

    vals_2x, _ = _generate_monomials(alpha_2x, beta_2x, degree)
    residual = abs(sum(c * v for c, v in zip(candidate, vals_2x)))

    if residual > 0:
        log10_res = float(mpmath.log10(residual))
        lambda_val = -log10_res / working_digits if working_digits > 0 else 0.0
    else:
        log10_res = None
        lambda_val = None

    norm_inf = max(abs(c) for c in candidate)
    norm_l2 = math.sqrt(sum(c**2 for c in candidate))

    result['status'] = 'ok'
    result['found_relation'] = False
    result['residual'] = float(residual) if residual > 0 else 0.0
    result['log10_residual'] = round(log10_res, 2) if log10_res is not None else None
    result['lambda_value'] = round(lambda_val, 4) if lambda_val is not None else None
    result['norm_inf'] = int(norm_inf)
    result['norm_l2'] = round(norm_l2, 2)
    result['elapsed'] = round(time.time() - t0, 2)

    return result


# ---------------------------------------------------------------------------
# Main — uses multiprocessing.Pool for robust parallelism
# ---------------------------------------------------------------------------

def main():
    import multiprocessing as mp

    project_root = Path(__file__).parent
    output_dir = project_root / "results" / "statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    N_WORKERS = min(20, os.cpu_count() or 12)
    MAX_COEFF = 100
    DEGREES = list(range(3, 13))  # 3, 4, ..., 12
    CHECKPOINT_INTERVAL = 100
    PROGRESS_INTERVAL = 20

    all_pairs = get_all_200_pairs()
    n_pairs = len(all_pairs)

    print("=" * 65, flush=True)
    print(f"  Expanded Baseline v2: {n_pairs} pairs x {len(DEGREES)} degrees = "
          f"{n_pairs * len(DEGREES)} PSLQ runs", flush=True)
    print(f"  Workers: {N_WORKERS} | Degrees: {DEGREES[0]}-{DEGREES[-1]} | "
          f"maxcoeff: {MAX_COEFF}", flush=True)
    print("=" * 65, flush=True)

    # Save pair definitions
    pair_defs = []
    for i, (f1, p1, f2, p2) in enumerate(all_pairs):
        pair_defs.append({
            'index': i,
            'func1': f1, 'prime1': p1,
            'func2': f2, 'prime2': p2,
            'label': pair_label(f1, p1, f2, p2),
            'family1': _family_of(f1),
            'family2': _family_of(f2),
        })
    (output_dir / "baseline_v2_pairs.json").write_text(
        json.dumps({'pairs': pair_defs, 'n_pairs': n_pairs,
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}, indent=2))
    print(f"  Saved {n_pairs} pair definitions", flush=True)

    # Load checkpoint if exists
    checkpoint_path = output_dir / "baseline_v2_checkpoint.json"
    completed = {}
    if checkpoint_path.exists():
        cp = json.loads(checkpoint_path.read_text())
        for r in cp.get('results', []):
            key = (r['pair_index'], r['degree'])
            completed[key] = r
        print(f"  Loaded checkpoint: {len(completed)} tasks already done", flush=True)

    # Build task list — sorted by degree (fast tasks first for early progress)
    tasks = []
    for pair_idx, (f1, p1, f2, p2) in enumerate(all_pairs):
        for degree in DEGREES:
            if (pair_idx, degree) in completed:
                continue
            tasks.append((pair_idx, f1, p1, f2, p2, degree, MAX_COEFF))
    tasks.sort(key=lambda t: t[5])  # sort by degree ascending

    total_tasks = n_pairs * len(DEGREES)
    done_count = len(completed)
    print(f"  Tasks remaining: {len(tasks)} / {total_tasks}", flush=True)

    if not tasks:
        print("  All tasks already completed!", flush=True)
        all_results = list(completed.values())
    else:
        all_results = list(completed.values())
        t_start = time.time()
        n_spurious = sum(1 for r in all_results if r.get('status') == 'spurious')
        lambda_sum = sum(r.get('lambda_value', 0) or 0
                         for r in all_results if r.get('lambda_value'))
        lambda_count = sum(1 for r in all_results
                           if r.get('lambda_value') and r['lambda_value'] > 0)

        # Use fork-based Pool for better performance with mpmath
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=N_WORKERS) as pool:
            for result in pool.imap_unordered(run_single_task, tasks, chunksize=1):
                all_results.append(result)
                done_count += 1

                if result.get('status') == 'spurious':
                    n_spurious += 1
                if result.get('lambda_value') and result['lambda_value'] > 0:
                    lambda_sum += result['lambda_value']
                    lambda_count += 1

                # Progress
                new_done = done_count - len(completed)
                if new_done % PROGRESS_INTERVAL == 0 or done_count == total_tasks:
                    elapsed = time.time() - t_start
                    remaining = len(tasks) - new_done
                    rate = new_done / elapsed if elapsed > 0 else 0
                    eta = remaining / rate if rate > 0 else 0
                    lm = lambda_sum / lambda_count if lambda_count > 0 else 0
                    print(f"  [{done_count:5d}/{total_tasks}] "
                          f"{100*done_count/total_tasks:5.1f}% | "
                          f"elapsed {elapsed:.0f}s | "
                          f"ETA {eta:.0f}s | "
                          f"lam_mean={lm:.3f} | "
                          f"spurious={n_spurious} | "
                          f"d={result.get('degree','?')} "
                          f"t={result.get('elapsed',0):.1f}s",
                          flush=True)

                # Checkpoint
                if new_done % CHECKPOINT_INTERVAL == 0:
                    _save_checkpoint(all_results, checkpoint_path)
                    print(f"    >> checkpoint saved ({len(all_results)} results)",
                          flush=True)

        total_elapsed = time.time() - t_start
        print(f"\n  Completed {len(tasks)} new tasks in {total_elapsed:.0f}s "
              f"({total_elapsed/60:.1f} min)", flush=True)

    # Save final results
    _save_checkpoint(all_results, checkpoint_path)

    # Compute summary stats
    usable = [r for r in all_results
              if r.get('lambda_value') is not None and r['lambda_value'] > 0]
    spurious = [r for r in all_results if r.get('status') == 'spurious']
    errors = [r for r in all_results if r.get('status', '').startswith('error')]

    final = {
        'baseline_results': all_results,
        'n_pairs': n_pairs,
        'n_results': len(all_results),
        'n_usable': len(usable),
        'n_spurious': len(spurious),
        'n_errors': len(errors),
        'degrees': DEGREES,
        'max_coeff': MAX_COEFF,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    (output_dir / "baseline_v2_results.json").write_text(
        json.dumps(final, indent=2))

    print(f"\n  === Summary ===", flush=True)
    print(f"  Total runs: {len(all_results)}", flush=True)
    print(f"  Usable (lam > 0): {len(usable)}", flush=True)
    print(f"  Spurious relations: {len(spurious)}", flush=True)
    print(f"  Errors: {len(errors)}", flush=True)
    if usable:
        lambdas = [r['lambda_value'] for r in usable]
        print(f"  Mean lam: {sum(lambdas)/len(lambdas):.4f}", flush=True)
    print(f"  Results saved to baseline_v2_results.json", flush=True)


def _save_checkpoint(results, path):
    cp = {
        'results': results,
        'n_results': len(results),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    path.write_text(json.dumps(cp, indent=2))


if __name__ == "__main__":
    main()
