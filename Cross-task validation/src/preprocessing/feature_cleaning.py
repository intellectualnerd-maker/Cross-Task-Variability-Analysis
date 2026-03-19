import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

Base_dir = 'D:/project-root/Cross-task validation'
Engineered_dir = os.path.join(Base_dir,'results/engineered_features.csv')
Cleaned_dir = os.path.join(Base_dir,'data/cleaned/cleaned_features.csv')
Report_dir = os.path.join(Base_dir,'results/metrics/cleaned_features_report.txt')
os.makedirs(os.path.dirname(Cleaned_dir), exist_ok=True)
os.makedirs(os.path.dirname(Report_dir), exist_ok=True)

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
        outlier_file = os.path.join(Base_dir, 'results', 'metrics', 'detected_outliers.csv')
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
    df = pd.read_csv(Engineered_dir)
    label_col = 'class'
    labels = df[label_col]
    df = df.drop(columns=[label_col])
    df = remove_missing_values(df)
    df = remove_outliers(df)
    df = scale_features(df)
    cleaned = pd.concat([df, labels], axis=1)
    cleaned.to_csv(Cleaned_dir, index=False)
    print("Cleaned data saved to:", Cleaned_dir)
    
    with open(Report_dir, 'w') as f:
        f.write("Feature Cleaning Report\n")
        f.write("="*50 + "\n")
        f.write(f"Report generated: {datetime.now()}\n")
        f.write(f"Samples after cleaning: {cleaned.shape[0]}\n")
        f.write(f"Features after cleaning: {cleaned.shape[1] - 1}\n")
        f.write(f"Output_file: {Cleaned_dir}\n")
        print(f"Report saved to: {Report_dir}")

if __name__ == "__main__":
    main()