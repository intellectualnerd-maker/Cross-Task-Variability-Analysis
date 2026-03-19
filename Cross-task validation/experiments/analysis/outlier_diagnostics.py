import os
import pandas as pd
import matplotlib.pyplot as plt

# === Base paths ===
Base_dir = 'D:/project-root/Cross-task validation'
Engineered_file = os.path.join(Base_dir, 'results', 'engineered_features.csv')
Cleaned_file = os.path.join(Base_dir, 'data', 'cleaned', 'cleaned_features.csv')
Outlier_file = os.path.join(Base_dir, 'results', 'metrics', 'outlier_samples.csv')

# Load dataset
print("Loading engineered and cleaned datasets...")
engineered = pd.read_csv(Engineered_file)
cleaned = pd.read_csv(Cleaned_file)

print(f"Engineered dataset shape: {engineered.shape}")
print(f"Cleaned dataset shape: {cleaned.shape}")

# Identify dropped (outlier) samples
# Align by index if index was not reset
engineered = engineered.reset_index(drop=True)
cleaned = cleaned.reset_index(drop=True)

# Check which rows are missing from cleaned
removed_count = len(engineered) - len(cleaned)
if removed_count > 0:
    dropped_indices = list(set(engineered.index) - set(cleaned.index))
    outliers = engineered.iloc[dropped_indices]
    outliers.to_csv(Outlier_file, index=False)
    print(f"\n {removed_count} outlier samples detected and saved to:")
    print(f"   {Outlier_file}")
else:
    print("\nNo outliers detected — cleaned dataset matches engineered dataset.")
    outliers = pd.DataFrame()

#  Summary statistics
if not outliers.empty:
    print("\nOutlier Feature Summary (first 10 features):")
    print(outliers.describe().T.head(10))
    
    # Quick correlation insight
    corr = outliers.select_dtypes(include=['number']).corr().abs().mean().mean()
    print(f"\nAverage absolute correlation among outlier features: {corr:.3f}")
else:
    print("No statistical summary generated (no outliers found).")

# Optional: Visualization
# Plot some key features if they exist
example_features = ['air_time1', 'pressure_var1', 'disp_index1']

for feat in example_features:
    if feat in engineered.columns:
        plt.figure(figsize=(6,4))
        plt.hist(engineered[feat], bins=30, alpha=0.6, label='All data')
        if not outliers.empty:
            plt.hist(outliers[feat], bins=30, alpha=0.6, label='Outliers')
        plt.title(f"Distribution of {feat}")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

print("\n Outlier diagnostics completed successfully.")
