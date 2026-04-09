#!/usr/bin/env python3
"""
Cross-Task Variability Analysis Pipeline
=========================================

Master script that runs the full analysis pipeline:
    1. Feature Engineering: Extract variability metrics from DARWIN dataset
    2. Feature Cleaning: Handle missing values, outliers, scaling
    3. Feature Selection: Correlation filtering + importance ranking
    4. Baseline vs. Variability Experiment: The critical validation
    5. Group Comparison: Statistical tests (Mann-Whitney U)
    6. Visualization: Generate publication figures

Usage:
    python run_pipeline.py                  # Run full pipeline
    python run_pipeline.py --stage=1        # Run only stage 1
    python run_pipeline.py --from-stage=3   # Run from stage 3 onwards

Requirements:
    - DARWIN dataset at: data/raw/DARWIN_DATASET/data.csv
    - Python packages: numpy, pandas, scikit-learn, scipy, matplotlib, xgboost
"""
import os
import sys
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


sys.path.insert(0, PROJECT_ROOT)          
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))  

from config import (
    DARWIN_DATASET, ENGINEERED_FEATURES_CSV, CLEANED_FEATURES_CSV,
    RESULTS_DIR, RESULTS_BASE_DIR, METRICS_DIR, FIGURES_DIR, 
    ensure_directories, check_dataset_exists, get_previous_runs, RUN_ID
)


def print_banner(stage_num, stage_name):
    print("\n" + "=" * 70)
    print(f"STAGE {stage_num}: {stage_name}")
    print("=" * 70)


def stage_1_feature_engineering():
    """Extract variability features from raw DARWIN data."""
    print_banner(1, "FEATURE ENGINEERING")
    
    from preprocessing.feature_engineering import main as engineer_features
    engineer_features()
    
    if os.path.exists(ENGINEERED_FEATURES_CSV):
        print(f"✓ Engineered features saved to: {ENGINEERED_FEATURES_CSV}")
        return True
    return False


def stage_2_feature_cleaning():
    """Clean engineered features (missing values, outliers, scaling)."""
    print_banner(2, "FEATURE CLEANING")
    
    from preprocessing.feature_cleaning import main as clean_features
    clean_features()
    
    if os.path.exists(CLEANED_FEATURES_CSV):
        print(f"✓ Cleaned features saved to: {CLEANED_FEATURES_CSV}")
        return True
    return False


def stage_3_feature_selection():
    """Select top features using RF importance and XGBoost-RFE."""
    print_banner(3, "FEATURE SELECTION")
    
    from preprocessing.feature_selection import main as select_features
    select_features()
    
    print(f"✓ Feature selection complete")
    return True


def stage_4_baseline_vs_variability():
    """Run the critical baseline vs. variability comparison experiment."""
    print_banner(4, "BASELINE VS. VARIABILITY EXPERIMENT")
    
    from experiments.baseline_vs_variability import run_baseline_vs_variability_experiment
    results = run_baseline_vs_variability_experiment()
    
    if results['significant']:
        print("\n✓ HYPOTHESIS SUPPORTED: Variability features significantly improve detection")
    else:
        print("\n○ HYPOTHESIS NOT SUPPORTED: No significant improvement from variability features")
    
    return results


def stage_5_group_comparison():
    """Run statistical group comparison (Mann-Whitney U tests)."""
    print_banner(5, "GROUP COMPARISON (STATISTICAL TESTS)")
    
    from experiments.group_comparison import run_group_comparison
    results_df = run_group_comparison()
    
    n_sig = results_df['significant_fdr'].sum()
    print(f"\n✓ {n_sig} variability features significantly higher in AD group (FDR-corrected)")
    
    return results_df


def stage_6_visualization():
    """Generate publication-quality figures."""
    print_banner(6, "VISUALIZATION")
    
    from visualization.figures import generate_all_figures
    figures = generate_all_figures()
    
    print(f"\n✓ Generated {len(figures)} figures")
    return figures


def run_pipeline(from_stage=1, to_stage=6, single_stage=None):
    """Run the full or partial pipeline."""
    
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "CROSS-TASK VARIABILITY ANALYSIS" + " " * 17 + "#")
    print("#" + " " * 20 + "DARWIN Dataset Pipeline" + " " * 25 + "#")
    print("#" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")
    
    # Ensure directories exist
    ensure_directories()
    
    # Show previous runs
    previous_runs = get_previous_runs()
    if previous_runs:
        print(f"\nPrevious runs found: {len(previous_runs)}")
        print(f"  Latest: {previous_runs[-1]}")
        print(f"  Current: {RUN_ID}")
    else:
        print(f"\nThis is the first run: {RUN_ID}")
    
    print(f"Results directory: {RESULTS_DIR}")
    
    # Check dataset
    if from_stage == 1 or single_stage == 1:
        if not check_dataset_exists():
            print("\nPipeline aborted: Dataset not found.")
            return False
    
    stages = {
        1: ("Feature Engineering", stage_1_feature_engineering),
        2: ("Feature Cleaning", stage_2_feature_cleaning),
        3: ("Feature Selection", stage_3_feature_selection),
        4: ("Baseline vs. Variability", stage_4_baseline_vs_variability),
        5: ("Group Comparison", stage_5_group_comparison),
        6: ("Visualization", stage_6_visualization),
    }
    
    if single_stage:
        stages_to_run = [single_stage]
    else:
        stages_to_run = range(from_stage, to_stage + 1)
    
    results = {}
    for stage_num in stages_to_run:
        if stage_num not in stages:
            print(f"Unknown stage: {stage_num}")
            continue
        
        stage_name, stage_func = stages[stage_num]
        try:
            result = stage_func()
            results[stage_num] = result
        except Exception as e:
            print(f"\n✗ Stage {stage_num} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {RESULTS_DIR}")
    print(f"Metrics directory: {METRICS_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    
    # Key result
    if 4 in results:
        exp_results = results[4]
        print(f"\n--- KEY RESULT ---")
        print(f"Baseline AUC: {exp_results['baseline']['auc_mean']:.4f}")
        print(f"Full AUC:     {exp_results['full']['auc_mean']:.4f}")
        print(f"Improvement:  {exp_results['improvement']:+.4f} (p = {exp_results['p_value']:.4f})")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Task Variability Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py                    # Run full pipeline (stages 1-6)
    python run_pipeline.py --stage 4          # Run only stage 4 (baseline vs variability)
    python run_pipeline.py --from-stage 3     # Run stages 3-6
    python run_pipeline.py --to-stage 4       # Run stages 1-4
        """
    )
    
    parser.add_argument('--stage', type=int, help='Run only this stage')
    parser.add_argument('--from-stage', type=int, default=1, help='Start from this stage')
    parser.add_argument('--to-stage', type=int, default=6, help='Stop at this stage')
    
    args = parser.parse_args()
    
    success = run_pipeline(
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        single_stage=args.stage
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()