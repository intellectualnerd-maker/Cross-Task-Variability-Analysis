"""
Baseline vs. Variability Comparison Experiment

This is the CRITICAL validation for the cross-task variability hypothesis.
It answers: Does variability add discriminative signal beyond mean performance?

Experiment Design:
    Model A (Baseline): Trained on mean features only (*_mean)
    Model B (Full):     Trained on mean + variability features (*_mean, *_std, *_cv, *_iqr, *_range)
    
Comparison:
    - 5-fold stratified cross-validation
    - Metrics: Accuracy, AUC-ROC, F1
    - Statistical test: Paired t-test on fold-level AUCs
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    CLEANED_FEATURES_CSV, METRICS_DIR, FIGURES_DIR,
    LABEL_COL, LABEL_MAP, RANDOM_STATE, N_SPLITS_CV, ensure_directories
)


def load_engineered_features(filepath):
    """Load engineered features and encode labels."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]
    if y.dtype == 'object':
        y = y.map(LABEL_MAP)
    return X, y


def split_features_by_type(X):
    """
    Split feature matrix into baseline (mean only) vs full (all variability metrics).
    
    Returns:
        X_baseline: Only *_mean features (what static screening captures)
        X_full: All features including variability (*_mean, *_std, *_cv, *_iqr, *_range)
    """
    mean_cols = [c for c in X.columns if c.endswith('_mean')]
    variability_cols = [c for c in X.columns if any(c.endswith(s) for s in ['_std', '_cv', '_iqr', '_range'])]
    
    X_baseline = X[mean_cols].copy()
    X_full = X[mean_cols + variability_cols].copy()
    
    return X_baseline, X_full, mean_cols, variability_cols


def run_cv_experiment(X, y, model_name="Model", n_splits=N_SPLITS_CV):
    """
    Run stratified k-fold CV and return per-fold metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        fold_results.append({
            'fold': fold,
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob),
            'f1': f1_score(y_test, y_pred),
            'n_test': len(y_test)
        })
    
    return pd.DataFrame(fold_results)


def compute_effect_size(baseline_aucs, full_aucs):
    """Compute Cohen's d for paired samples."""
    diff = np.array(full_aucs) - np.array(baseline_aucs)
    d = np.mean(diff) / np.std(diff, ddof=1)
    return d


def run_baseline_vs_variability_experiment(features_path=None):
    """
    Main experiment: Compare baseline (mean-only) vs full (mean + variability) models.
    """
    ensure_directories()
    
    if features_path is None:
        features_path = CLEANED_FEATURES_CSV
    
    print("=" * 70)
    print("BASELINE VS. VARIABILITY COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    print(f"Loading features from: {features_path}")
    X, y = load_engineered_features(features_path)
    print(f"Samples: {len(y)} | AD: {sum(y == 1)} | Healthy: {sum(y == 0)}")
    
    # Split features
    X_baseline, X_full, mean_cols, var_cols = split_features_by_type(X)
    print(f"\nFeature Split:")
    print(f"  Baseline (mean only): {len(mean_cols)} features")
    print(f"  Variability features: {len(var_cols)} features")
    print(f"  Full model: {X_full.shape[1]} features")
    
    # Run experiments
    print("\n" + "-" * 50)
    print("Running Baseline Model (mean features only)...")
    baseline_results = run_cv_experiment(X_baseline, y, "Baseline")
    
    print("Running Full Model (mean + variability features)...")
    full_results = run_cv_experiment(X_full, y, "Full")
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    baseline_summary = {
        'model': 'Baseline (mean only)',
        'accuracy_mean': baseline_results['accuracy'].mean(),
        'accuracy_std': baseline_results['accuracy'].std(),
        'auc_mean': baseline_results['auc'].mean(),
        'auc_std': baseline_results['auc'].std(),
        'f1_mean': baseline_results['f1'].mean(),
        'f1_std': baseline_results['f1'].std(),
        'n_features': len(mean_cols)
    }
    
    full_summary = {
        'model': 'Full (mean + variability)',
        'accuracy_mean': full_results['accuracy'].mean(),
        'accuracy_std': full_results['accuracy'].std(),
        'auc_mean': full_results['auc'].mean(),
        'auc_std': full_results['auc'].std(),
        'f1_mean': full_results['f1'].mean(),
        'f1_std': full_results['f1'].std(),
        'n_features': X_full.shape[1]
    }
    
    print(f"\nBaseline Model (mean features only):")
    print(f"  Accuracy: {baseline_summary['accuracy_mean']:.3f} ± {baseline_summary['accuracy_std']:.3f}")
    print(f"  AUC-ROC:  {baseline_summary['auc_mean']:.3f} ± {baseline_summary['auc_std']:.3f}")
    print(f"  F1 Score: {baseline_summary['f1_mean']:.3f} ± {baseline_summary['f1_std']:.3f}")
    
    print(f"\nFull Model (mean + variability features):")
    print(f"  Accuracy: {full_summary['accuracy_mean']:.3f} ± {full_summary['accuracy_std']:.3f}")
    print(f"  AUC-ROC:  {full_summary['auc_mean']:.3f} ± {full_summary['auc_std']:.3f}")
    print(f"  F1 Score: {full_summary['f1_mean']:.3f} ± {full_summary['f1_std']:.3f}")
    
    # Statistical comparison
    print("\n" + "-" * 50)
    print("STATISTICAL COMPARISON")
    print("-" * 50)
    
    # Paired t-test on AUCs
    t_stat, p_value = stats.ttest_rel(full_results['auc'], baseline_results['auc'])
    effect_size = compute_effect_size(baseline_results['auc'].values, full_results['auc'].values)
    
    auc_improvement = full_summary['auc_mean'] - baseline_summary['auc_mean']
    auc_pct_improvement = (auc_improvement / baseline_summary['auc_mean']) * 100
    
    print(f"\nAUC Improvement: {auc_improvement:+.4f} ({auc_pct_improvement:+.2f}%)")
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_value:.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.3f}")
    
    if p_value < 0.05:
        print(f"\n✓ SIGNIFICANT: Variability features provide statistically significant improvement (p < 0.05)")
    else:
        print(f"\n○ NOT SIGNIFICANT: Variability features do not significantly improve performance (p >= 0.05)")
    
    # Effect size interpretation
    if abs(effect_size) < 0.2:
        effect_interp = "negligible"
    elif abs(effect_size) < 0.5:
        effect_interp = "small"
    elif abs(effect_size) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    print(f"Effect size interpretation: {effect_interp}")
    
    # Save results
    results_df = pd.DataFrame([baseline_summary, full_summary])
    results_path = os.path.join(METRICS_DIR, 'baseline_vs_variability_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Save detailed fold-level results
    baseline_results['model'] = 'baseline'
    full_results['model'] = 'full'
    detailed_df = pd.concat([baseline_results, full_results], ignore_index=True)
    detailed_path = os.path.join(METRICS_DIR, 'baseline_vs_variability_folds.csv')
    detailed_df.to_csv(detailed_path, index=False)
    
    # Generate report
    report_path = os.path.join(METRICS_DIR, 'baseline_vs_variability_report.txt')
    with open(report_path, 'w') as f:
        f.write("BASELINE VS. VARIABILITY COMPARISON EXPERIMENT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: DARWIN Handwriting Dataset\n")
        f.write(f"Samples: {len(y)} (AD: {sum(y == 1)}, Healthy: {sum(y == 0)})\n\n")
        
        f.write("HYPOTHESIS\n")
        f.write("-" * 50 + "\n")
        f.write("Cross-task variability captures early cognitive instability\n")
        f.write("that is not reflected in mean task performance.\n\n")
        
        f.write("EXPERIMENTAL DESIGN\n")
        f.write("-" * 50 + "\n")
        f.write(f"Baseline Model: {len(mean_cols)} mean features only (*_mean)\n")
        f.write(f"Full Model: {X_full.shape[1]} features (mean + std + cv + iqr + range)\n")
        f.write(f"Classifier: Random Forest (300 trees, max_depth=10)\n")
        f.write(f"Validation: {N_SPLITS_CV}-fold stratified cross-validation\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Baseline AUC: {baseline_summary['auc_mean']:.4f} ± {baseline_summary['auc_std']:.4f}\n")
        f.write(f"Full AUC:     {full_summary['auc_mean']:.4f} ± {full_summary['auc_std']:.4f}\n")
        f.write(f"AUC Improvement: {auc_improvement:+.4f} ({auc_pct_improvement:+.2f}%)\n\n")
        
        f.write("STATISTICAL SIGNIFICANCE\n")
        f.write("-" * 50 + "\n")
        f.write(f"Paired t-test: t = {t_stat:.4f}, p = {p_value:.4f}\n")
        f.write(f"Cohen's d: {effect_size:.4f} ({effect_interp} effect)\n")
        if p_value < 0.05:
            f.write("\nCONCLUSION: Variability features provide statistically significant\n")
            f.write("improvement in AD detection, supporting the cognitive instability hypothesis.\n")
        else:
            f.write("\nCONCLUSION: No statistically significant improvement detected.\n")
            f.write("This may indicate insufficient sample size or that variability\n")
            f.write("information is partially captured by mean features.\n")
    
    print(f"Report saved to: {report_path}")
    
    return {
        'baseline': baseline_summary,
        'full': full_summary,
        'improvement': auc_improvement,
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': p_value < 0.05
    }


if __name__ == "__main__":
 run_baseline_vs_variability_experiment()