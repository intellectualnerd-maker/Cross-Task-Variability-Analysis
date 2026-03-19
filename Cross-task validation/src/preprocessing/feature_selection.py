import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

# Base directories
Base_dir = 'D:/project-root/Cross-task validation'
Cleaned_dir = os.path.join(Base_dir, 'data/cleaned/cleaned_features.csv')
Selected_dir = os.path.join(Base_dir, 'results/selected_features.csv')
Importance_dir = os.path.join(Base_dir, 'results/metrics/feature_importance.csv')
Report_dir = os.path.join(Base_dir, 'results/metrics/feature_selection_report.txt')
Dropped_dir = os.path.join(Base_dir, 'results/metrics/dropped_features.csv')
Before_filter_dir = os.path.join(Base_dir, 'results/full_features_before_filter.csv')
After_filter_dir = os.path.join(Base_dir, 'results/features_after_correlation_filter.csv')
RFE_Comparison_dir = os.path.join(Base_dir, 'results/metrics/xgboost_rfe_comparison.csv')

# Ensure required directories exist
os.makedirs(os.path.dirname(Selected_dir), exist_ok=True)
os.makedirs(os.path.dirname(Report_dir), exist_ok=True)
os.makedirs(os.path.dirname(Importance_dir), exist_ok=True)

def load_data():
    """Load the cleaned dataset and split into features (X) and labels (y)."""
    df = pd.read_csv(Cleaned_dir)
    X = df.drop(columns=['class'])
    y = df['class']

    # Encode string labels into numeric values if needed
    if y.dtype == 'object':
        y = y.map({'H': 0, 'P': 1})  
        print("Encoded class labels: {'H': 0, 'P': 1}")

    return X, y


def remove_correlated_features(X, threshold=0.9):
    """Remove features that are highly correlated with each other."""
    print(f"Removing features with correlation higher than {threshold}...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if len(to_drop) > X.shape[1] * 0.5:
        print(f"Warning: {len(to_drop)} of {X.shape[1]} features are highly correlated. "
              f"Automatically relaxing threshold from {threshold} → 0.95.")
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    X_reduced = X.drop(columns=to_drop)
    print(f"Removed {len(to_drop)} features due to high correlation.")
    return X_reduced, to_drop

def compute_feature_importance(X, y, top_n=25):
    """Compute feature importance using Random Forest importance and select top N."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train_scaled, y_train)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    importances.to_frame(name='importance').to_csv(
        Importance_dir,
        index_label='feature',
        float_format='%.5f'
    )
    print(f"Feature importances saved to {Importance_dir}")

    top_features = importances.head(top_n).index.tolist()
    print(f"Selected top {top_n} features based on importance.")
    return top_features, importances

def compute_xgboost_rfe(X, y, top_n=25):
    """Perform feature selection using XGBoost with Recursive Feature Elimination (RFE)."""
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_estimators=200
    )
    selector = RFE(model, n_features_to_select=top_n, step=1)
    selector = selector.fit(X, y)
    selected_features = X.columns[selector.support_].tolist()

    rankings = pd.DataFrame({
        'feature': X.columns,
        'RFE_rank': selector.ranking_
    }).sort_values(by='RFE_rank')

    rankings.to_csv(RFE_Comparison_dir, index=False)
    print(f"XGBoost-RFE feature rankings saved to {RFE_Comparison_dir}")

    return selected_features, rankings

def main():
    X, y = load_data()
    X.to_csv(Before_filter_dir, index=False)
    print(f"Full feature dataset saved to {Before_filter_dir}")

    X_filtered, dropped_features = remove_correlated_features(X.copy(), threshold=0.9)
    X_filtered.to_csv(After_filter_dir, index=False)
    pd.DataFrame({'dropped_features': dropped_features}).to_csv(Dropped_dir, index=False)
    print(f"Filtered dataset saved to {After_filter_dir}")
    print(f"Dropped feature names saved to {Dropped_dir}")

    top_features_rf, importances = compute_feature_importance(X_filtered, y, top_n=25)
    top_features_rfe, rfe_rankings = compute_xgboost_rfe(X_filtered, y, top_n=25)

    overlap_features = set(top_features_rf).intersection(set(top_features_rfe))
    overlap_ratio = len(overlap_features) / 25

    selected = pd.concat([X_filtered[top_features_rf], y], axis=1)
    selected.to_csv(Selected_dir, index=False)
    print(f"Selected features dataset saved to {Selected_dir}")

    # Report Generation
    with open(Report_dir, 'w') as f:
        f.write("Feature Selection Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("1. Dataset Overview\n")
        f.write(f"   - Samples: {X.shape[0]}\n")
        f.write(f"   - Initial features: {X.shape[1]}\n\n")

        f.write("2. Correlation Filtering\n")
        f.write(f"   - Threshold used: 0.9 (adaptive to 0.95 if needed)\n")
        f.write(f"   - Features removed due to correlation: {len(dropped_features)}\n")
        f.write(f"   - Remaining features after filtering: {X_filtered.shape[1]}\n\n")

        f.write("3. Random Forest Feature Importance (Top 25)\n")
        for i, feature in enumerate(top_features_rf, start=1):
            f.write(f"   {i}. {feature} — importance: {importances[feature]:.5f}\n")

        f.write("\n4. XGBoost + Recursive Feature Elimination (Top 25)\n")
        for i, feature in enumerate(top_features_rfe, start=1):
            f.write(f"   {i}. {feature}\n")

        f.write("\n5. Cross-Model Comparison\n")
        f.write(f"   - Overlap count between RF and XGBoost-RFE: {len(overlap_features)} / 25\n")
        f.write(f"   - Overlap ratio: {overlap_ratio:.2f}\n")
        f.write(f"   - Common selected features:\n")
        for feature in sorted(overlap_features):
            f.write(f"      • {feature}\n")

        f.write("\n6. Output Files\n")
        f.write(f"   - Selected features CSV: {Selected_dir}\n")
        f.write(f"   - Dropped features CSV: {Dropped_dir}\n")
        f.write(f"   - Feature importances CSV (Random Forest): {Importance_dir}\n")
        f.write(f"   - XGBoost-RFE rankings CSV: {RFE_Comparison_dir}\n")
        f.write(f"   - Full dataset before filtering: {Before_filter_dir}\n")
        f.write(f"   - Filtered dataset after correlation removal: {After_filter_dir}\n")

    print(f"Report saved to {Report_dir}")

if __name__ == "__main__":
    main()
