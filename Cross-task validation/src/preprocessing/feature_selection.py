import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

# Add src to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    CLEANED_FEATURES_CSV, SELECTED_FEATURES_CSV, RESULTS_DIR, METRICS_DIR,
    LABEL_COL, LABEL_MAP, TOP_N_FEATURES, CORRELATION_THRESHOLD, RANDOM_STATE,
    ensure_directories
)

# Output file paths
Importance_path = os.path.join(METRICS_DIR, 'feature_importance.csv')
Report_path = os.path.join(METRICS_DIR, 'feature_selection_report.txt')
Dropped_path = os.path.join(METRICS_DIR, 'dropped_features.csv')
Before_filter_path = os.path.join(RESULTS_DIR, 'full_features_before_filter.csv')
After_filter_path = os.path.join(RESULTS_DIR, 'features_after_correlation_filter.csv')
RFE_Comparison_path = os.path.join(METRICS_DIR, 'xgboost_rfe_comparison.csv')

def load_data():
    """Load the cleaned dataset and split into features (X) and labels (y)."""
    df = pd.read_csv(CLEANED_FEATURES_CSV)
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]

    # Encode string labels into numeric values if needed
    if y.dtype == 'object':
        y = y.map(LABEL_MAP)
        print(f"Encoded class labels: {LABEL_MAP}")

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

def compute_feature_importance(X, y, top_n=TOP_N_FEATURES):
    """Compute feature importance using Random Forest importance and select top N."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    rf.fit(X_train_scaled, y_train)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    importances.to_frame(name='importance').to_csv(
        Importance_path,
        index_label='feature',
        float_format='%.5f'
    )
    print(f"Feature importances saved to {Importance_path}")

    top_features = importances.head(top_n).index.tolist()
    print(f"Selected top {top_n} features based on importance.")
    return top_features, importances

def compute_xgboost_rfe(X, y, top_n=TOP_N_FEATURES):
    """Perform feature selection using XGBoost with Recursive Feature Elimination (RFE)."""
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_estimators=200
    )
    selector = RFE(model, n_features_to_select=top_n, step=1)
    selector = selector.fit(X, y)
    selected_features = X.columns[selector.support_].tolist()

    rankings = pd.DataFrame({
        'feature': X.columns,
        'RFE_rank': selector.ranking_
    }).sort_values(by='RFE_rank')

    rankings.to_csv(RFE_Comparison_path, index=False)
    print(f"XGBoost-RFE feature rankings saved to {RFE_Comparison_path}")

    return selected_features, rankings

def main():
    ensure_directories()
    X, y = load_data()
    X.to_csv(Before_filter_path, index=False)
    print(f"Full feature dataset saved to {Before_filter_path}")

    X_filtered, dropped_features = remove_correlated_features(X.copy(), threshold=CORRELATION_THRESHOLD)
    X_filtered.to_csv(After_filter_path, index=False)
    pd.DataFrame({'dropped_features': dropped_features}).to_csv(Dropped_path, index=False)
    print(f"Filtered dataset saved to {After_filter_path}")
    print(f"Dropped feature names saved to {Dropped_path}")

    top_features_rf, importances = compute_feature_importance(X_filtered, y, top_n=TOP_N_FEATURES)
    top_features_rfe, rfe_rankings = compute_xgboost_rfe(X_filtered, y, top_n=TOP_N_FEATURES)

    overlap_features = set(top_features_rf).intersection(set(top_features_rfe))
    overlap_ratio = len(overlap_features) / TOP_N_FEATURES

    selected = pd.concat([X_filtered[top_features_rf], y], axis=1)
    selected.to_csv(SELECTED_FEATURES_CSV, index=False)
    print(f"Selected features dataset saved to {SELECTED_FEATURES_CSV}")

    # Report Generation
    with open(Report_path, 'w') as f:
        f.write("Feature Selection Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("1. Dataset Overview\n")
        f.write(f"   - Samples: {X.shape[0]}\n")
        f.write(f"   - Initial features: {X.shape[1]}\n\n")

        f.write("2. Correlation Filtering\n")
        f.write(f"   - Threshold used: {CORRELATION_THRESHOLD} (adaptive to 0.95 if needed)\n")
        f.write(f"   - Features removed due to correlation: {len(dropped_features)}\n")
        f.write(f"   - Remaining features after filtering: {X_filtered.shape[1]}\n\n")

        f.write(f"3. Random Forest Feature Importance (Top {TOP_N_FEATURES})\n")
        for i, feature in enumerate(top_features_rf, start=1):
            f.write(f"   {i}. {feature} — importance: {importances[feature]:.5f}\n")

        f.write(f"\n4. XGBoost + Recursive Feature Elimination (Top {TOP_N_FEATURES})\n")
        for i, feature in enumerate(top_features_rfe, start=1):
            f.write(f"   {i}. {feature}\n")

        f.write("\n5. Cross-Model Comparison\n")
        f.write(f"   - Overlap count between RF and XGBoost-RFE: {len(overlap_features)} / {TOP_N_FEATURES}\n")
        f.write(f"   - Overlap ratio: {overlap_ratio:.2f}\n")
        f.write(f"   - Common selected features:\n")
        for feature in sorted(overlap_features):
            f.write(f"      • {feature}\n")

        f.write("\n6. Output Files\n")
        f.write(f"   - Selected features CSV: {SELECTED_FEATURES_CSV}\n")
        f.write(f"   - Dropped features CSV: {Dropped_path}\n")
        f.write(f"   - Feature importances CSV (Random Forest): {Importance_path}\n")
        f.write(f"   - XGBoost-RFE rankings CSV: {RFE_Comparison_path}\n")
        f.write(f"   - Full dataset before filtering: {Before_filter_path}\n")
        f.write(f"   - Filtered dataset after correlation removal: {After_filter_path}\n")

    print(f"Report saved to {Report_path}")

if __name__ == "__main__":
    main()