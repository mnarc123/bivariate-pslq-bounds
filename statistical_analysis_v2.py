#!/usr/bin/env python3
"""
Statistical Analysis v2 — Expanded baseline (200 pairs).

Compares 138 real-constant λ values with the expanded baseline.
Produces:
  - KS, AD, MW tests
  - Bootstrap CI for mean difference
  - Cohen's d effect size
  - Per-pair z-scores
  - Subgroup analysis (low/high degree, top/medium priority)
  - 4 publication-quality PDF plots
  - JSON results + markdown report

Usage:
    python3 statistical_analysis_v2.py
"""

import json
import math
import time
import numpy as np
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_real_lambdas(stats_dir: Path):
    """Load 138 real-constant λ values from real_residuals.json."""
    path = stats_dir / "real_residuals.json"
    data = json.loads(path.read_text())
    items = data['real_residuals']
    lambdas = []
    meta = []  # (pair_symbol, degree, lambda)
    for item in items:
        lv = item.get('lambda_value', 0)
        if lv is not None and lv > 0:
            lambdas.append(lv)
            meta.append({
                'pair_symbol': item.get('pair_symbol', '?'),
                'degree': item.get('degree', 0),
                'lambda': lv,
            })
    return np.array(lambdas), meta


def load_baseline_lambdas(stats_dir: Path):
    """Load baseline λ values from baseline_v2_results.json."""
    path = stats_dir / "baseline_v2_results.json"
    data = json.loads(path.read_text())
    items = data['baseline_results']
    lambdas = []
    meta = []
    for item in items:
        lv = item.get('lambda_value')
        if lv is not None and lv > 0:
            lambdas.append(lv)
            meta.append({
                'pair_label': item.get('pair_label', '?'),
                'pair_index': item.get('pair_index', -1),
                'degree': item.get('degree', 0),
                'lambda': lv,
            })
    return np.array(lambdas), meta, data


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_tests(real_lam, base_lam):
    """Run KS, AD, MW tests + bootstrap + Cohen's d."""
    results = {}

    # Descriptive stats
    results['n_real'] = len(real_lam)
    results['n_baseline'] = len(base_lam)
    results['real_mean'] = float(np.mean(real_lam))
    results['real_std'] = float(np.std(real_lam, ddof=1))
    results['real_median'] = float(np.median(real_lam))
    results['baseline_mean'] = float(np.mean(base_lam))
    results['baseline_std'] = float(np.std(base_lam, ddof=1))
    results['baseline_median'] = float(np.median(base_lam))

    # 1. Kolmogorov-Smirnov
    ks_stat, ks_p = stats.ks_2samp(real_lam, base_lam)
    results['ks_statistic'] = float(ks_stat)
    results['ks_pvalue'] = float(ks_p)

    # 2. Anderson-Darling
    try:
        ad = stats.anderson_ksamp([real_lam, base_lam])
        results['ad_statistic'] = float(ad.statistic)
        results['ad_pvalue'] = float(ad.pvalue)
    except Exception as e:
        results['ad_error'] = str(e)

    # 3. Mann-Whitney U
    try:
        u_stat, u_p = stats.mannwhitneyu(real_lam, base_lam, alternative='two-sided')
        results['mw_statistic'] = float(u_stat)
        results['mw_pvalue'] = float(u_p)
    except Exception as e:
        results['mw_error'] = str(e)

    # 4. Bootstrap CI for mean difference
    rng = np.random.default_rng(42)
    n_boot = 10000
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        r_boot = rng.choice(real_lam, size=len(real_lam), replace=True)
        b_boot = rng.choice(base_lam, size=len(base_lam), replace=True)
        diffs[i] = r_boot.mean() - b_boot.mean()
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    results['bootstrap_mean_diff'] = float(np.mean(diffs))
    results['bootstrap_ci_lo'] = ci_lo
    results['bootstrap_ci_hi'] = ci_hi
    results['bootstrap_diffs'] = diffs.tolist()  # for plotting

    # 5. Cohen's d
    n1, n2 = len(real_lam), len(base_lam)
    s1, s2 = np.std(real_lam, ddof=1), np.std(base_lam, ddof=1)
    s_pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = float((np.mean(real_lam) - np.mean(base_lam)) / s_pooled) if s_pooled > 0 else 0.0
    results['cohens_d'] = cohens_d

    return results


# ---------------------------------------------------------------------------
# Per-pair z-scores
# ---------------------------------------------------------------------------

def compute_zscores(real_meta, base_lam):
    """Compute z-score for each real pair's mean λ vs baseline distribution."""
    base_mean = float(np.mean(base_lam))
    base_std = float(np.std(base_lam, ddof=1))
    if base_std == 0:
        base_std = 1e-10

    # Group real data by pair
    pairs = {}
    for item in real_meta:
        ps = item['pair_symbol']
        pairs.setdefault(ps, []).append(item['lambda'])

    z_scores = []
    for pair_sym, lambdas in pairs.items():
        pair_mean = np.mean(lambdas)
        z = float((pair_mean - base_mean) / base_std)
        z_scores.append({
            'pair': pair_sym,
            'n_samples': len(lambdas),
            'mean_lambda': round(float(pair_mean), 4),
            'z_score': round(z, 3),
            'flag_2sigma': abs(z) > 2,
            'flag_3sigma': abs(z) > 3,
        })

    z_scores.sort(key=lambda x: abs(x['z_score']), reverse=True)
    return z_scores


# ---------------------------------------------------------------------------
# Subgroup analysis
# ---------------------------------------------------------------------------

def subgroup_analysis(real_lam, real_meta, base_lam, base_meta):
    """Compare by degree subgroups and pair priority."""
    results = {}

    # --- Degree subgroups ---
    low_degs = {3, 4, 5, 6}
    high_degs = {7, 8, 9, 10, 11, 12}

    real_low = np.array([m['lambda'] for m in real_meta if m['degree'] in low_degs])
    real_high = np.array([m['lambda'] for m in real_meta if m['degree'] in high_degs])
    base_low = np.array([m['lambda'] for m in base_meta if m['degree'] in low_degs])
    base_high = np.array([m['lambda'] for m in base_meta if m['degree'] in high_degs])

    for label, rl, bl in [('low_degree_3_6', real_low, base_low),
                           ('high_degree_7_12', real_high, base_high)]:
        if len(rl) >= 5 and len(bl) >= 5:
            ks_s, ks_p = stats.ks_2samp(rl, bl)
            mw_s, mw_p = stats.mannwhitneyu(rl, bl, alternative='two-sided')
            results[label] = {
                'n_real': int(len(rl)), 'n_baseline': int(len(bl)),
                'real_mean': round(float(np.mean(rl)), 4),
                'base_mean': round(float(np.mean(bl)), 4),
                'ks_pvalue': round(float(ks_p), 4),
                'mw_pvalue': round(float(mw_p), 4),
            }

    # --- Priority subgroups ---
    # Top pairs (pi,e), (pi,gamma), (e,gamma)
    top_symbols = {'(π, e)', '(π, γ)', '(e, γ)'}
    real_top = np.array([m['lambda'] for m in real_meta if m['pair_symbol'] in top_symbols])
    real_other = np.array([m['lambda'] for m in real_meta if m['pair_symbol'] not in top_symbols])

    if len(real_top) >= 5 and len(base_lam) >= 5:
        ks_s, ks_p = stats.ks_2samp(real_top, base_lam)
        results['top_priority_vs_baseline'] = {
            'n_real': int(len(real_top)),
            'real_mean': round(float(np.mean(real_top)), 4),
            'ks_pvalue': round(float(ks_p), 4),
        }
    if len(real_other) >= 5 and len(base_lam) >= 5:
        ks_s, ks_p = stats.ks_2samp(real_other, base_lam)
        results['other_priority_vs_baseline'] = {
            'n_real': int(len(real_other)),
            'real_mean': round(float(np.mean(real_other)), 4),
            'ks_pvalue': round(float(ks_p), 4),
        }

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def setup_rcparams():
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_histogram(real_lam, base_lam, output_dir):
    """Overlapping histogram of λ distributions."""
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(
        min(real_lam.min(), base_lam.min()) - 0.05,
        max(real_lam.max(), base_lam.max()) + 0.05,
        25
    )
    ax.hist(base_lam, bins=bins, alpha=0.5, density=True,
            color='#ff7f0e', label=f'Baseline ($n={len(base_lam)}$)',
            edgecolor='white', linewidth=0.5)
    ax.hist(real_lam, bins=bins, alpha=0.5, density=True,
            color='#1f77b4', label=f'Real constants ($n={len(real_lam)}$)',
            edgecolor='white', linewidth=0.5)

    # Mean lines
    ax.axvline(np.mean(real_lam), color='#1f77b4', linestyle='--', linewidth=1.5,
               label=f'Real mean = {np.mean(real_lam):.3f}')
    ax.axvline(np.mean(base_lam), color='#ff7f0e', linestyle='--', linewidth=1.5,
               label=f'Baseline mean = {np.mean(base_lam):.3f}')

    ax.set_xlabel(r'$\lambda = -\log_{10}|R| \;/\; p$')
    ax.set_ylabel('Density')
    ax.set_title(r'Distribution of precision fraction $\lambda$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "residual_histogram.pdf")
    plt.close(fig)


def plot_qq(real_lam, base_lam, output_dir):
    """Q-Q plot with KS confidence bands."""
    fig, ax = plt.subplots(figsize=(6, 6))

    n_quantiles = min(len(real_lam), len(base_lam), 200)
    probs = np.linspace(0, 1, n_quantiles + 2)[1:-1]
    rq = np.quantile(real_lam, probs)
    bq = np.quantile(base_lam, probs)

    ax.scatter(bq, rq, s=15, alpha=0.7, color='#2166ac', zorder=3)

    # Reference line
    all_vals = np.concatenate([rq, bq])
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            'r--', linewidth=1, alpha=0.7, label='$y = x$')

    # KS 95% confidence band
    n_eff = min(len(real_lam), len(base_lam))
    ks_crit = 1.36 / math.sqrt(n_eff)  # 95% critical value
    ax.fill_between([lo - margin, hi + margin],
                    [lo - margin - ks_crit, hi + margin - ks_crit],
                    [lo - margin + ks_crit, hi + margin + ks_crit],
                    alpha=0.15, color='red', label='95% KS band')

    ax.set_xlabel(r'Baseline quantiles ($\lambda$)')
    ax.set_ylabel(r'Real constants quantiles ($\lambda$)')
    ax.set_title(r'Q-Q plot: Real vs.\ Baseline $\lambda$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_dir / "qq_plot.pdf")
    plt.close(fig)


def plot_lambda_vs_degree(real_meta, base_meta, output_dir):
    """Mean λ ± SE vs degree for both groups."""
    fig, ax = plt.subplots(figsize=(7, 5))

    def _by_degree(meta):
        d = {}
        for m in meta:
            d.setdefault(m['degree'], []).append(m['lambda'])
        degs = sorted(d.keys())
        means = [np.mean(d[deg]) for deg in degs]
        ses = [np.std(d[deg], ddof=1) / math.sqrt(len(d[deg])) if len(d[deg]) > 1 else 0
               for deg in degs]
        return degs, means, ses

    bd, bm, bse = _by_degree(base_meta)
    rd, rm, rse = _by_degree(real_meta)

    ax.errorbar(bd, bm, yerr=bse, fmt='o-', color='#ff7f0e',
                label='Baseline', capsize=3, markersize=5, linewidth=1.5)
    ax.errorbar(rd, rm, yerr=rse, fmt='s-', color='#1f77b4',
                label='Real constants', capsize=3, markersize=5, linewidth=1.5)

    ax.set_xlabel('Polynomial degree $d$')
    ax.set_ylabel(r'Mean $\lambda \pm$ SE')
    ax.set_title(r'Precision fraction $\lambda$ vs.\ degree')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "lambda_vs_degree.pdf")
    plt.close(fig)


def plot_bootstrap(diffs, ci_lo, ci_hi, output_dir):
    """Bootstrap distribution of mean difference."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(diffs, bins=50, density=True, color='#7570b3', alpha=0.7,
            edgecolor='white', linewidth=0.5)

    ax.axvline(0, color='black', linestyle='--', linewidth=1.5,
               label='Zero difference')
    ax.axvspan(ci_lo, ci_hi, alpha=0.2, color='#d95f02',
               label=f'95% CI [{ci_lo:.4f}, {ci_hi:.4f}]')

    ax.set_xlabel(r'$\bar\lambda_{\rm real} - \bar\lambda_{\rm baseline}$')
    ax.set_ylabel('Density')
    ax.set_title('Bootstrap distribution of mean difference (10,000 resamples)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "bootstrap_diff.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def generate_markdown_report(test_results, z_scores, subgroups, output_dir):
    """Generate statistical_report_v2.md."""
    tr = test_results
    lines = [
        "# Statistical Analysis v2 — Expanded Baseline (K=200)",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Sample Sizes",
        f"- Real constants: n = {tr['n_real']}",
        f"- Baseline (K=200 pairs): n = {tr['n_baseline']}",
        "",
        "## Descriptive Statistics",
        "",
        f"| | Real | Baseline |",
        f"|--|------|----------|",
        f"| Mean | {tr['real_mean']:.4f} | {tr['baseline_mean']:.4f} |",
        f"| Std  | {tr['real_std']:.4f} | {tr['baseline_std']:.4f} |",
        f"| Median | {tr['real_median']:.4f} | {tr['baseline_median']:.4f} |",
        "",
        "## Hypothesis Tests",
        "",
        f"| Test | Statistic | p-value | Reject H₀ (α=0.05)? |",
        f"|------|-----------|---------|---------------------|",
    ]

    ks_reject = "Yes" if tr['ks_pvalue'] < 0.05 else "No"
    lines.append(f"| Kolmogorov-Smirnov | {tr['ks_statistic']:.4f} | "
                 f"{tr['ks_pvalue']:.4f} | {ks_reject} |")

    if 'ad_pvalue' in tr:
        ad_reject = "Yes" if tr['ad_pvalue'] < 0.05 else "No"
        lines.append(f"| Anderson-Darling | {tr['ad_statistic']:.4f} | "
                     f"{tr['ad_pvalue']:.4f} | {ad_reject} |")

    if 'mw_pvalue' in tr:
        mw_reject = "Yes" if tr['mw_pvalue'] < 0.05 else "No"
        lines.append(f"| Mann-Whitney U | {tr['mw_statistic']:.1f} | "
                     f"{tr['mw_pvalue']:.4f} | {mw_reject} |")

    lines.extend([
        "",
        "## Effect Size & Bootstrap",
        "",
        f"- **Cohen's d:** {tr['cohens_d']:.4f}",
        f"- **Bootstrap mean diff (95% CI):** [{tr['bootstrap_ci_lo']:.4f}, "
        f"{tr['bootstrap_ci_hi']:.4f}]",
        "",
        "## Per-Pair Z-Scores",
        "",
        "| Pair | n | Mean λ | z-score | |z|>2? | |z|>3? |",
        "|------|---|--------|---------|--------|--------|",
    ])

    for zs in z_scores:
        f2 = "**YES**" if zs['flag_2sigma'] else "No"
        f3 = "**YES**" if zs['flag_3sigma'] else "No"
        lines.append(f"| {zs['pair']} | {zs['n_samples']} | "
                     f"{zs['mean_lambda']:.4f} | {zs['z_score']:.3f} | "
                     f"{f2} | {f3} |")

    lines.extend(["", "## Subgroup Analysis", ""])
    for sg_name, sg_data in subgroups.items():
        lines.append(f"### {sg_name}")
        for k, v in sg_data.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    (output_dir / "statistical_report_v2.md").write_text("\n".join(lines))
    print(f"  Report: statistical_report_v2.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    stats_dir = Path(__file__).parent / "results" / "statistics"
    figures_dir = Path(__file__).parent / "results" / "deep_v2" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Statistical Analysis v2 — Expanded Baseline")
    print("=" * 60)

    # Load data
    real_lam_all, real_meta_all = load_real_lambdas(stats_dir)
    base_lam_all, base_meta_all, base_data = load_baseline_lambdas(stats_dir)

    n_total = base_data.get('n_results', len(base_data.get('baseline_results', [])))
    n_usable_all = len(base_lam_all)
    n_spurious = base_data.get('n_spurious', n_total - n_usable_all)

    # Determine common degree range for fair comparison
    base_degrees = set(m['degree'] for m in base_meta_all)
    real_degrees = set(m['degree'] for m in real_meta_all)
    common_degrees = base_degrees & real_degrees
    print(f"  Baseline degrees available: {sorted(base_degrees)}")
    print(f"  Real degrees available: {sorted(real_degrees)}")
    print(f"  Common degrees for comparison: {sorted(common_degrees)}")

    # Filter to common degrees
    real_meta = [m for m in real_meta_all if m['degree'] in common_degrees]
    real_lam = np.array([m['lambda'] for m in real_meta])
    base_meta = [m for m in base_meta_all if m['degree'] in common_degrees]
    base_lam = np.array([m['lambda'] for m in base_meta])

    print(f"\n  Real λ samples (matched): {len(real_lam)} (of {len(real_lam_all)} total)")
    print(f"  Baseline total runs: {n_total}")
    print(f"  Baseline usable λ (matched): {len(base_lam)} (of {n_usable_all} total)")
    print(f"  Baseline spurious: {n_spurious}")
    print(f"  Real mean λ: {np.mean(real_lam):.4f}")
    print(f"  Baseline mean λ: {np.mean(base_lam):.4f}")

    # Run tests on degree-matched data
    print("\n  Running statistical tests (degree-matched)...")
    test_results = run_tests(real_lam, base_lam)

    print(f"    KS: stat={test_results['ks_statistic']:.4f}, "
          f"p={test_results['ks_pvalue']:.4f}")
    if 'ad_pvalue' in test_results:
        print(f"    AD: stat={test_results['ad_statistic']:.4f}, "
              f"p={test_results['ad_pvalue']:.4f}")
    if 'mw_pvalue' in test_results:
        print(f"    MW: stat={test_results['mw_statistic']:.1f}, "
              f"p={test_results['mw_pvalue']:.4f}")
    print(f"    Cohen's d: {test_results['cohens_d']:.4f}")
    print(f"    Bootstrap 95% CI: [{test_results['bootstrap_ci_lo']:.4f}, "
          f"{test_results['bootstrap_ci_hi']:.4f}]")

    # Z-scores
    print("\n  Computing per-pair z-scores...")
    z_scores = compute_zscores(real_meta, base_lam)
    n_2sig = sum(1 for z in z_scores if z['flag_2sigma'])
    n_3sig = sum(1 for z in z_scores if z['flag_3sigma'])
    print(f"    Pairs with |z| > 2: {n_2sig}")
    print(f"    Pairs with |z| > 3: {n_3sig}")

    # Subgroup analysis
    print("\n  Subgroup analysis...")
    subgroups = subgroup_analysis(real_lam, real_meta, base_lam, base_meta)
    for sg, data in subgroups.items():
        ks_p = data.get('ks_pvalue', '?')
        print(f"    {sg}: KS p={ks_p}")

    # Plots
    print("\n  Generating plots...")
    setup_rcparams()
    plot_histogram(real_lam, base_lam, figures_dir)
    plot_qq(real_lam, base_lam, figures_dir)
    plot_lambda_vs_degree(real_meta, base_meta, figures_dir)

    # Bootstrap plot (without full diffs array in saved JSON)
    diffs = np.array(test_results['bootstrap_diffs'])
    plot_bootstrap(diffs, test_results['bootstrap_ci_lo'],
                   test_results['bootstrap_ci_hi'], figures_dir)
    print(f"  4 plots saved to {figures_dir}")

    # Also copy plots to stats dir for reference
    import shutil
    for fname in ['residual_histogram.pdf', 'qq_plot.pdf',
                   'lambda_vs_degree.pdf', 'bootstrap_diff.pdf']:
        src = figures_dir / fname
        dst = stats_dir / fname
        if src.exists():
            shutil.copy2(src, dst)

    # Save JSON (without bootstrap_diffs to keep file small)
    json_results = {k: v for k, v in test_results.items() if k != 'bootstrap_diffs'}
    json_results['z_scores'] = z_scores
    json_results['subgroups'] = subgroups
    json_results['n_total_baseline_runs'] = n_total
    json_results['n_usable_baseline'] = len(base_lam)
    json_results['n_spurious_baseline'] = n_spurious
    json_results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    (stats_dir / "residual_analysis_v2.json").write_text(
        json.dumps(json_results, indent=2, cls=NpEncoder))
    print(f"  JSON: residual_analysis_v2.json")

    # Markdown report
    generate_markdown_report(test_results, z_scores, subgroups, stats_dir)

    # Print summary
    print("\n  " + "=" * 50)
    print("  SUMMARY")
    print("  " + "=" * 50)
    ks_p = test_results['ks_pvalue']
    ad_p = test_results.get('ad_pvalue', '?')
    mw_p = test_results.get('mw_pvalue', '?')
    print(f"  KS p-value:  {ks_p:.4f}")
    print(f"  AD p-value:  {ad_p:.4f}" if isinstance(ad_p, float) else f"  AD p-value:  {ad_p}")
    print(f"  MW p-value:  {mw_p:.4f}" if isinstance(mw_p, float) else f"  MW p-value:  {mw_p}")
    print(f"  Cohen's d:   {test_results['cohens_d']:.4f}")
    print(f"  Bootstrap CI: [{test_results['bootstrap_ci_lo']:.4f}, "
          f"{test_results['bootstrap_ci_hi']:.4f}]")

    any_reject = (ks_p < 0.05 or
                  (isinstance(ad_p, float) and ad_p < 0.05) or
                  (isinstance(mw_p, float) and mw_p < 0.05))
    if any_reject:
        print("\n  *** At least one test REJECTS H₀ at α=0.05 ***")
    else:
        print("\n  *** No test rejects H₀ at α=0.05 ***")

    print("  Done.")


if __name__ == "__main__":
    main()
