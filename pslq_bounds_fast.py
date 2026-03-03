"""
Fast H-matrix bound extraction by patching mpmath's PSLQ at the source level.

Strategy: read mpmath's identification.py source, apply minimal patches to
capture the H matrix at termination, then exec the patched source. This
gives native-speed PSLQ (with gmpy2 backend) plus the H matrix for
extracting the rigorous diagonal bound from Ferguson-Bailey-Arno (1999).

Patches applied to mpmath's pslq() source:
  1. maxsteps=0 is converted to 10**7 (effectively unlimited).
  2. REP is initialized to 0 before the main loop.
  3. Before the final 'return None', H/n/prec/REP are saved to _captured.
  4. Before 'return vec' (relation found), same capture.

The bound from the diagonal (Theorem 1 of FBA99) is:
    ||m||_2 >= 1 / max_j |H_{jj}|

Usage:
    from pslq_bounds_fast import pslq_with_bound, PSLQResult
    result = pslq_with_bound(x, maxcoeff=100, maxsteps=0)
    if result.relation is None:
        print(f"No relation found. ||m||_2 >= {result.norm_bound}")

Reference:
    H.R.P. Ferguson, D.H. Bailey, S. Arno, "Analysis of PSLQ, an
    integer relation finding algorithm", Math. Comp. 68 (1999),
    no. 225, 351-369. Theorem 1.
"""

import mpmath
import mpmath.identification as _ident
import inspect
import textwrap
from collections import namedtuple

PSLQResult = namedtuple('PSLQResult', [
    'relation',          # list[int] or None
    'norm_bound',        # int: 1/max_j|H_{jj}| (Theorem 1, FBA99)
    'norm_bound_allH',   # int: 1/max_{ij}|H_{ij}| (mpmath-style)
    'max_coeff_tested',  # int: maxcoeff parameter
    'iterations',        # int: number of iterations performed
    'best_candidate',    # list[int] or None: B-column with smallest |y_i|
    'best_candidate_err', # float or None: |y_i| for the best candidate
])

# Module-level storage for captured H matrix from patched PSLQ
_captured = {}


def _build_patched_pslq():
    """
    Build a patched version of mpmath's pslq that captures the H matrix
    and iteration count at termination.
    """
    src = inspect.getsource(_ident.pslq)
    src = textwrap.dedent(src)

    # --- Patch 1: maxsteps=0 -> 10**7 (effectively unlimited) ---
    # Insert after the line '    assert tol' (which is after tol setup)
    # We add it right before the main loop marker
    src = src.replace(
        '    # Main algorithm\n    for REP in range(maxsteps):',
        '    # Main algorithm\n'
        '    if maxsteps == 0:\n'
        '        maxsteps = 10**7\n'
        '    REP = 0\n'
        '    for REP in range(maxsteps):',
    )

    # --- Patch 2: capture H before final 'return None' ---
    # The final 'return None' is the last one in the source
    idx = src.rfind('    return None')
    if idx == -1:
        raise RuntimeError("Could not find final 'return None' in mpmath pslq source")
    capture_code = (
        '    _captured["H"] = H\n'
        '    _captured["B"] = B\n'
        '    _captured["y"] = y\n'
        '    _captured["n"] = n\n'
        '    _captured["prec"] = prec\n'
        '    _captured["REP"] = REP + 1\n'
        '    return None'
    )
    src = src[:idx] + capture_code + src[idx + len('    return None'):]

    # --- Patch 3: capture H before 'return vec' (relation found) ---
    src = src.replace(
        '                    return vec',
        '                    _captured["H"] = H\n'
        '                    _captured["B"] = B\n'
        '                    _captured["y"] = y\n'
        '                    _captured["n"] = n\n'
        '                    _captured["prec"] = prec\n'
        '                    _captured["REP"] = REP + 1\n'
        '                    return vec',
    )

    # Compile and extract the function
    globs = {
        'xrange': getattr(_ident, 'xrange', range),
        'round_fixed': _ident.round_fixed,
        'sqrt_fixed': __import__('mpmath.libmp', fromlist=['sqrt_fixed']).sqrt_fixed,
        '_captured': _captured,
    }
    exec(compile(src, '<patched_pslq>', 'exec'), globs)
    return globs['pslq']


# Build the patched function once at import time
_patched_pslq = _build_patched_pslq()


def pslq_with_bound(x, tol=None, maxcoeff=1000, maxsteps=100, verbose=False):
    """
    Run mpmath's native PSLQ (with gmpy2 if available) and extract the
    rigorous H-matrix norm bound from Theorem 1 of FBA99.

    Returns a PSLQResult namedtuple.
    """
    ctx = mpmath.mp
    n = len(x)
    _captured.clear()

    relation = _patched_pslq(ctx, x, tol=tol, maxcoeff=maxcoeff,
                             maxsteps=maxsteps, verbose=verbose)

    H = _captured.get('H')
    n_cap = _captured.get('n', n)
    prec = _captured.get('prec', ctx.prec)
    iterations = _captured.get('REP', 0)

    B = _captured.get('B')
    y = _captured.get('y')

    if relation is not None:
        return PSLQResult(
            relation=relation,
            norm_bound=0,
            norm_bound_allH=0,
            max_coeff_tested=maxcoeff,
            iterations=iterations,
            best_candidate=relation,
            best_candidate_err=0.0,
        )

    if H is None:
        return PSLQResult(
            relation=None,
            norm_bound=None,
            norm_bound_allH=None,
            max_coeff_tested=maxcoeff,
            iterations=iterations,
            best_candidate=None,
            best_candidate_err=None,
        )

    # Compute diagonal bound: 1/max_j|H_{jj}| (Theorem 1, FBA99)
    recnorm_diag = max(abs(H[j, j]) for j in range(1, n_cap))
    if recnorm_diag:
        norm_diag = ((1 << (2 * prec)) // recnorm_diag) >> prec
    else:
        norm_diag = None

    # Compute allH bound (same as mpmath's internal calculation)
    recnorm_allH = max(abs(h) for h in H.values())
    if recnorm_allH:
        norm_allH = ((1 << (2 * prec)) // recnorm_allH) >> prec
        norm_allH //= 100
    else:
        norm_allH = None

    if verbose:
        print("Norm bound (diagonal, Thm 1): %s" % norm_diag)
        print("Norm bound (all H entries):    %s" % norm_allH)

    # Extract best candidate: B-column with smallest |y_i|
    best_vec = None
    best_err_val = None
    if B is not None and y is not None:
        try:
            from mpmath.libmp.backend import xrange
        except ImportError:
            xrange = range
        def _round_fixed_local(x, p):
            return ((x + (1 << (p - 1))) >> p) << p
        best_i = min(range(1, n_cap + 1), key=lambda i: abs(y[i]))
        best_vec = [int(_round_fixed_local(B[j, best_i], prec) >> prec)
                    for j in range(1, n_cap + 1)]
        best_err_val = float(abs(y[best_i]) / ctx.mpf(2)**prec)

    return PSLQResult(
        relation=None,
        norm_bound=int(norm_diag) if norm_diag is not None else None,
        norm_bound_allH=int(norm_allH) if norm_allH is not None else None,
        max_coeff_tested=maxcoeff,
        iterations=iterations,
        best_candidate=best_vec,
        best_candidate_err=best_err_val,
    )
