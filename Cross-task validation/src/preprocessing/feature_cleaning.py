import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Add src to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    ENGINEERED_FEATURES_CSV, CLEANED_FEATURES_CSV, METRICS_DIR, RESULTS_DIR,
    LABEL_COL, ensure_directories
)

# Ensure directories exist
ensure_directories()

def remove_missing_values(df):
    """Impute missing values using median"""
    na_before = df.isna().sum().sum()
    df = df.fillna(df.median(numeric_only=True ))
    na_after = df.isna().sum().sum()
    print(f"Missing values before: {na_before}, after: {na_after}")
    return df

def remove_outliers(df, z_thresh=3, min_fraction=0.8):
    """Remove outliers using Z-score method (tolerates slight feature deviations)."""
    print(f"Applying Z-threshold={z_thresh} with minimum inlier fraction={min_fraction}")
    numeric_cols = df.select_dtypes(include=[np.number])
    # Compute standard deviations and handle zero-variance columns
    stds = numeric_cols.std(ddof=0).replace(0, np.nan)
    z_scores = np.abs((numeric_cols - numeric_cols.mean()) / stds)
    mask = (z_scores < z_thresh).mean(axis=1) > min_fraction
    outliers_removed = len(df) - mask.sum()
    if outliers_removed > 0:
        removed_indices = df.index[~mask]
        outlier_file = os.path.join(METRICS_DIR, 'detected_outliers.csv')
        df.loc[removed_indices].to_csv(outlier_file, index=False)
        print(f"Saved details of {outliers_removed} outliers to {outlier_file}")
    print(f"Outliers removed: {outliers_removed}")
    return df[mask].reset_index(drop=True)


def remove_outliers_with_labels(df, z_thresh=3, min_fraction=0.8):
    """Remove outliers while keeping label column intact."""
    print(f"Applying Z-threshold={z_thresh} with minimum inlier fraction={min_fraction}")
    numeric_cols = df.select_dtypes(include=[np.number])
    # Compute standard deviations and handle zero-variance columns
    stds = numeric_cols.std(ddof=0).replace(0, np.nan)
    z_scores = np.abs((numeric_cols - numeric_cols.mean()) / stds)
    mask = (z_scores < z_thresh).mean(axis=1) > min_fraction
    outliers_removed = len(df) - mask.sum()
    if outliers_removed > 0:
        removed_indices = df.index[~mask]
        outlier_file = os.path.join(METRICS_DIR, 'detected_outliers.csv')
        df.loc[removed_indices].to_csv(outlier_file, index=False)
        print(f"Saved details of {outliers_removed} outliers to {outlier_file}")
    print(f"Outliers removed: {outliers_removed}")
    return df[mask].reset_index(drop=True)


def scale_features(df):
    """Scale features to zero mean and unit variance"""
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"Scaled features: {len(numeric_cols)} numeric features.")
    return df

def main():
    ensure_directories()
    df = pd.read_csv(ENGINEERED_FEATURES_CSV)
    labels = df[LABEL_COL]
    df = df.drop(columns=[LABEL_COL])
    df = remove_missing_values(df)
    
    # Remove outliers - but keep labels aligned
    df_with_labels = pd.concat([df, labels], axis=1)
    df_with_labels = remove_outliers_with_labels(df_with_labels)
    
    labels = df_with_labels[LABEL_COL]
    df = df_with_labels.drop(columns=[LABEL_COL])
    
    df = scale_features(df)
    cleaned = pd.concat([df.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    cleaned.to_csv(CLEANED_FEATURES_CSV, index=False)
    print("Cleaned data saved to:", CLEANED_FEATURES_CSV)
    
    report_path = os.path.join(METRICS_DIR, 'cleaned_features_report.txt')
    with open(report_path, 'w') as f:
        f.write("Feature Cleaning Report\n")
        f.write("="*50 + "\n")
        f.write(f"Report generated: {datetime.now()}\n")
        f.write(f"Samples after cleaning: {cleaned.shape[0]}\n")
        f.write(f"Features after cleaning: {cleaned.shape[1] - 1}\n")
        f.write(f"Output_file: {CLEANED_FEATURES_CSV}\n")
        print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()