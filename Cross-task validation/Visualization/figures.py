"""
Visualization Module for Cross-Task Variability Analysis

Generates publication-quality figures:
    1. Box plots: CV/IQR distributions by group
    2. Feature importance bar chart (highlighting variability features)
    3. Model comparison bar chart (baseline vs full)
    4. Effect size forest plot
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    CLEANED_FEATURES_CSV, ENGINEERED_FEATURES_CSV, METRICS_DIR, FIGURES_DIR,
    LABEL_COL, LABEL_MAP, ensure_directories
)

# Style configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Colors
COLOR_AD = '#D55E00'      # Vermillion (colorblind-safe)
COLOR_HEALTHY = '#0072B2'  # Blue (colorblind-safe)
COLOR_MEAN = '#999999'     # Gray for mean features
COLOR_VAR = '#009E73'      # Teal for variability features


def plot_variability_boxplots(features_path=None, output_dir=None, top_n=8):
    """
    Box plots comparing variability feature distributions between AD and Healthy.
    """
    ensure_directories()
    
    if features_path is None:
        features_path = ENGINEERED_FEATURES_CSV if os.path.exists(ENGINEERED_FEATURES_CSV) else CLEANED_FEATURES_CSV
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    df = pd.read_csv(features_path)
    y = df[LABEL_COL]
    if y.dtype == 'object':
        y = y.map(LABEL_MAP)
    X = df.drop(columns=[LABEL_COL])
    
    # Get CV features (most interpretable variability measure)
    cv_cols = [c for c in X.columns if c.endswith('_cv')]
    
    # Select top features by AD-Healthy difference
    diffs = []
    for col in cv_cols:
        ad_med = X.loc[y == 1, col].median()
        healthy_med = X.loc[y == 0, col].median()
        diffs.append((col, ad_med - healthy_med))
    
    diffs.sort(key=lambda x: x[1], reverse=True)
    top_features = [d[0] for d in diffs[:top_n]]
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, col in enumerate(top_features):
        ax = axes[i]
        
        ad_data = X.loc[y == 1, col].dropna()
        healthy_data = X.loc[y == 0, col].dropna()
        
        bp = ax.boxplot(
            [healthy_data, ad_data],
            labels=['Healthy', 'AD'],
            patch_artist=True,
            widths=0.6
        )
        
        bp['boxes'][0].set_facecolor(COLOR_HEALTHY)
        bp['boxes'][1].set_facecolor(COLOR_AD)
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        # Clean feature name for title
        feature_name = col.replace('_cv', '').replace('_', ' ').title()
        ax.set_title(f'{feature_name}\n(CV)', fontsize=9)
        ax.set_ylabel('Coefficient of Variation')
        
    plt.suptitle('Cross-Task Variability: AD vs Healthy Controls', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'variability_boxplots.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def plot_feature_importance(importance_path=None, output_dir=None, top_n=20):
    """
    Bar chart of feature importances, color-coded by feature type (mean vs variability).
    """
    ensure_directories()
    
    if importance_path is None:
        importance_path = os.path.join(METRICS_DIR, 'feature_importance.csv')
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    if not os.path.exists(importance_path):
        print(f"Feature importance file not found: {importance_path}")
        print("Run feature selection first.")
        return None
    
    imp_df = pd.read_csv(importance_path)
    
    # Handle different column name formats
    if 'feature' in imp_df.columns:
        imp_df = imp_df.set_index('feature')
    
    # Get importance column
    imp_col = [c for c in imp_df.columns if 'importance' in c.lower()][0]
    imp_df = imp_df.sort_values(imp_col, ascending=True).tail(top_n)
    
    # Classify features
    colors = []
    for feat in imp_df.index:
        if feat.endswith('_mean'):
            colors.append(COLOR_MEAN)
        else:  # _std, _cv, _iqr, _range
            colors.append(COLOR_VAR)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(imp_df))
    ax.barh(y_pos, imp_df[imp_col], color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ') for f in imp_df.index])
    ax.set_xlabel('Feature Importance (Random Forest)')
    ax.set_title('Top Features for AD Detection\n(Variability features in teal, Mean features in gray)')
    
    # Legend
    mean_patch = mpatches.Patch(color=COLOR_MEAN, label='Mean features')
    var_patch = mpatches.Patch(color=COLOR_VAR, label='Variability features')
    ax.legend(handles=[var_patch, mean_patch], loc='lower right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def plot_model_comparison(results_path=None, output_dir=None):
    """
    Bar chart comparing baseline (mean-only) vs full (mean + variability) model performance.
    """
    ensure_directories()
    
    if results_path is None:
        results_path = os.path.join(METRICS_DIR, 'baseline_vs_variability_results.csv')
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Run baseline_vs_variability experiment first.")
        return None
    
    results_df = pd.read_csv(results_path)
    
    metrics = ['accuracy', 'auc', 'f1']
    x = np.arange(len(metrics))
    width = 0.35
    
    baseline = results_df[results_df['model'].str.contains('Baseline')].iloc[0]
    full = results_df[results_df['model'].str.contains('Full')].iloc[0]
    
    baseline_vals = [baseline[f'{m}_mean'] for m in metrics]
    baseline_errs = [baseline[f'{m}_std'] for m in metrics]
    full_vals = [full[f'{m}_mean'] for m in metrics]
    full_errs = [full[f'{m}_std'] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, yerr=baseline_errs,
                   label='Baseline (mean only)', color=COLOR_MEAN, capsize=3)
    bars2 = ax.bar(x + width/2, full_vals, width, yerr=full_errs,
                   label='Full (mean + variability)', color=COLOR_VAR, capsize=3)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance: Baseline vs. Full Model')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'AUC-ROC', 'F1 Score'])
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def plot_effect_sizes(comparison_path=None, output_dir=None, top_n=15):
    """
    Forest plot of effect sizes for top variability features.
    """
    ensure_directories()
    
    if comparison_path is None:
        comparison_path = os.path.join(METRICS_DIR, 'group_comparison_results.csv')
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    if not os.path.exists(comparison_path):
        print(f"Comparison file not found: {comparison_path}")
        print("Run group_comparison experiment first.")
        return None
    
    df = pd.read_csv(comparison_path)
    df = df.sort_values('effect_size_r', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    y_pos = np.arange(len(df))
    colors = [COLOR_AD if r > 0 else COLOR_HEALTHY for r in df['effect_size_r']]
    
    ax.barh(y_pos, df['effect_size_r'], color=colors, edgecolor='white', linewidth=0.5)
    
    # Add significance markers
    for i, (_, row) in enumerate(df.iterrows()):
        if row.get('significant_fdr', False):
            ax.annotate('**', xy=(row['effect_size_r'], i), 
                       xytext=(3, 0), textcoords='offset points',
                       fontsize=10, va='center')
        elif row['p_value'] < 0.05:
            ax.annotate('*', xy=(row['effect_size_r'], i),
                       xytext=(3, 0), textcoords='offset points',
                       fontsize=10, va='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ') for f in df['feature']])
    ax.set_xlabel('Effect Size (Rank-Biserial Correlation)')
    ax.set_title('Effect Sizes: AD vs Healthy on Variability Features\n(positive = higher in AD)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Effect size reference lines
    for thresh, label in [(0.1, ''), (0.3, 'small'), (0.5, 'medium')]:
        ax.axvline(x=thresh, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'effect_sizes.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def generate_all_figures(features_path=None):
    """Generate all figures for the project."""
    ensure_directories()
    
    print("=" * 50)
    print("GENERATING ALL FIGURES")
    print("=" * 50)
    
    figures = []
    
    print("\n1. Variability box plots...")
    try:
        fig = plot_variability_boxplots(features_path)
        if fig:
            figures.append(fig)
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Feature importance chart...")
    try:
        fig = plot_feature_importance()
        if fig:
            figures.append(fig)
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Model comparison chart...")
    try:
        fig = plot_model_comparison()
        if fig:
            figures.append(fig)
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n4. Effect size forest plot...")
    try:
        fig = plot_effect_sizes()
        if fig:
            figures.append(fig)
    except Exception as e:
        print(f"   Error: {e}")
    
    print(f"\nGenerated {len(figures)} figures in {FIGURES_DIR}")
    return figures


if __name__ == "__main__":
    generate_all_figures()