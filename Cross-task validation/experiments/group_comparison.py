"""
Statistical Validation: Group-Level Comparisons

Tests whether AD patients show significantly higher cross-task variability
than healthy controls, independent of modeling.

Tests:
    - Mann-Whitney U test (non-parametric, robust to non-normality)
    - Effect size: rank-biserial correlation
    - Multiple comparison correction: Benjamini-Hochberg FDR
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from config import METRICS_DIR

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    CLEANED_FEATURES_CSV, ENGINEERED_FEATURES_CSV, METRICS_DIR,
    LABEL_COL, LABEL_MAP, ensure_directories
)


def rank_biserial_correlation(U, n1, n2):
    """
    Compute rank-biserial correlation from Mann-Whitney U statistic.
    Effect size interpretation:
        |r| < 0.1: negligible
        |r| < 0.3: small
        |r| < 0.5: medium
        |r| >= 0.5: large
    """
    r = 1 - (2 * U) / (n1 * n2)
    return r


def benjamini_hochberg(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    
    # BH critical values
    bh_critical = (np.arange(1, n + 1) / n) * alpha
    
    # Find largest k where p_k <= (k/n)*alpha
    below_threshold = sorted_p <= bh_critical
    if not any(below_threshold):
        return np.zeros(n, dtype=bool)
    
    k = np.max(np.where(below_threshold)[0]) + 1
    
    # All p-values with rank <= k are significant
    significant = np.zeros(n, dtype=bool)
    significant[sorted_idx[:k]] = True
    
    return significant


def run_group_comparison(features_path=None, output_dir=None):
    """
    Run Mann-Whitney U tests comparing AD vs Healthy on all variability features.
    """
    ensure_directories()
    
    if features_path is None:
        # Try cleaned first, fall back to engineered
        if os.path.exists(CLEANED_FEATURES_CSV):
            features_path = CLEANED_FEATURES_CSV
        else:
            features_path = ENGINEERED_FEATURES_CSV
    
    if output_dir is None:
        output_dir = METRICS_DIR
    
    print("=" * 70)
    print("GROUP-LEVEL STATISTICAL COMPARISON: AD vs HEALTHY")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    print(f"Loading features from: {features_path}")
    df = pd.read_csv(features_path)
    
    y = df[LABEL_COL]
    if y.dtype == 'object':
        y = y.map(LABEL_MAP)
    
    X = df.drop(columns=[LABEL_COL])
    
    # Split by group
    ad_mask = y == 1
    healthy_mask = y == 0
    n_ad = ad_mask.sum()
    n_healthy = healthy_mask.sum()
    
    print(f"Samples: AD = {n_ad}, Healthy = {n_healthy}")
    
    # Identify variability features
    variability_suffixes = ['_std', '_cv', '_iqr', '_range']
    var_cols = [c for c in X.columns if any(c.endswith(s) for s in variability_suffixes)]
    mean_cols = [c for c in X.columns if c.endswith('_mean')]
    
    print(f"Testing {len(var_cols)} variability features")
    print()
    
    # Run tests on variability features
    results = []
    
    for col in var_cols:
        ad_values = X.loc[ad_mask, col].dropna()
        healthy_values = X.loc[healthy_mask, col].dropna()
        
        # Mann-Whitney U test
        # Alternative 'greater': test if AD has larger values (more variability)
        U_stat, p_value = stats.mannwhitneyu(
            ad_values, healthy_values, 
            alternative='greater'
        )
        
        # Effect size
        r = rank_biserial_correlation(U_stat, len(ad_values), len(healthy_values))
        
        # Descriptive stats
        ad_median = ad_values.median()
        healthy_median = healthy_values.median()
        median_diff = ad_median - healthy_median
        
        results.append({
            'feature': col,
            'ad_median': ad_median,
            'healthy_median': healthy_median,
            'median_diff': median_diff,
            'U_statistic': U_stat,
            'p_value': p_value,
            'effect_size_r': r
        })
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    significant_fdr = benjamini_hochberg(results_df['p_value'].values, alpha=0.05)
    results_df['significant_fdr'] = significant_fdr
    results_df['significant_uncorrected'] = results_df['p_value'] < 0.05
    
    # Sort by effect size
    results_df = results_df.sort_values('effect_size_r', ascending=False)
    
    # Summary
    n_sig_uncorrected = results_df['significant_uncorrected'].sum()
    n_sig_fdr = results_df['significant_fdr'].sum()
    
    print("-" * 50)
    print("SUMMARY")
    print("-" * 50)
    print(f"Features tested: {len(var_cols)}")
    print(f"Significant (uncorrected p < 0.05): {n_sig_uncorrected}")
    print(f"Significant (FDR-corrected): {n_sig_fdr}")
    print()
    
    # Top features by effect size
    print("TOP 10 FEATURES BY EFFECT SIZE (AD > Healthy)")
    print("-" * 50)
    top_10 = results_df.head(10)
    for _, row in top_10.iterrows():
        sig_marker = "*" if row['significant_fdr'] else ""
        print(f"  {row['feature']:30s}  r = {row['effect_size_r']:+.3f}  p = {row['p_value']:.4f} {sig_marker}")
    
    # Also test mean features for comparison
    print("\n" + "-" * 50)
    print("COMPARISON: MEAN FEATURES (for reference)")
    print("-" * 50)
    
    mean_results = []
    for col in mean_cols[:10]:  # Just show first 10
        ad_values = X.loc[ad_mask, col].dropna()
        healthy_values = X.loc[healthy_mask, col].dropna()
        U_stat, p_value = stats.mannwhitneyu(ad_values, healthy_values, alternative='two-sided')
        r = rank_biserial_correlation(U_stat, len(ad_values), len(healthy_values))
        mean_results.append({'feature': col, 'effect_size_r': r, 'p_value': p_value})
    
    mean_df = pd.DataFrame(mean_results).sort_values('p_value')
    for _, row in mean_df.head(5).iterrows():
        print(f"  {row['feature']:30s}  r = {row['effect_size_r']:+.3f}  p = {row['p_value']:.4f}")
    
    # Save results
    output_path = os.path.join(output_dir, 'group_comparison_results.csv')
    results_df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"\nResults saved to: {output_path}")
    
    # Generate report
    report_path = os.path.join(output_dir, 'group_comparison_report.txt')
    report_file = os.path.join(METRICS_DIR, 'group_comparison_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("GROUP-LEVEL STATISTICAL COMPARISON: AD vs HEALTHY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: DARWIN Handwriting Dataset\n")
        f.write(f"Samples: AD = {n_ad}, Healthy = {n_healthy}\n\n")
        
        f.write("HYPOTHESIS\n")
        f.write("-" * 50 + "\n")
        f.write("AD patients exhibit higher cross-task variability (CV, IQR, etc.)\n")
        f.write("compared to healthy controls, reflecting cognitive instability.\n\n")
        
        f.write("STATISTICAL METHOD\n")
        f.write("-" * 50 + "\n")
        f.write("Test: Mann-Whitney U (one-tailed, AD > Healthy)\n")
        f.write("Effect size: Rank-biserial correlation\n")
        f.write("Multiple testing correction: Benjamini-Hochberg FDR (α = 0.05)\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Variability features tested: {len(var_cols)}\n")
        f.write(f"Significant (uncorrected p < 0.05): {n_sig_uncorrected}\n")
        f.write(f"Significant (FDR-corrected): {n_sig_fdr}\n\n")
        
        f.write("TOP VARIABILITY FEATURES (by effect size, AD > Healthy)\n")
        f.write("-" * 50 + "\n")
        for i, (_, row) in enumerate(results_df.head(15).iterrows(), 1):
            sig = "**" if row['significant_fdr'] else ("*" if row['significant_uncorrected'] else "")
            f.write(f"{i:2d}. {row['feature']:35s} r = {row['effect_size_r']:+.4f}, p = {row['p_value']:.5f} {sig}\n")
        
        f.write("\n* = p < 0.05 (uncorrected), ** = p < 0.05 (FDR-corrected)\n\n")
        
        f.write("EFFECT SIZE INTERPRETATION\n")
        f.write("-" * 50 + "\n")
        f.write("|r| < 0.1: negligible\n")
        f.write("|r| < 0.3: small\n")
        f.write("|r| < 0.5: medium\n")
        f.write("|r| >= 0.5: large\n")
    
    print(f"Report saved to: {report_path}")
    
    return results_df


if __name__ == "__main__":
    run_group_comparison()