import os
import re
import sys
import numpy as np
import pandas as pd

# Add src to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    DARWIN_DATASET, ENGINEERED_FEATURES_CSV, RESULTS_DIR, METRICS_DIR,
    LABEL_COL, EPS, ensure_directories
)

def extract_base_features(columns):
    """From columns extract base features"""
    base_features = {}
    for col in columns:
        if col == LABEL_COL:
            continue                    
        m = re.match(r"(.+?)(\d+)$", col)
        if not m:
            continue
        base = m.group(1)
        if base not in base_features:
            base_features[base] = []
        base_features[base].append(col)
    return base_features
            
def compute_stats(df, cols, prefix):
    """Compute variability stats for one base feature across all the tasks"""
    values                  = df[cols].apply(pd.to_numeric, errors='coerce').values
    stats                   = pd.DataFrame(index = df.index)
    stats[f'{prefix}_mean'] = np.nanmean(values, axis=1)
    stats[f'{prefix}_std']  = np.nanstd(values, axis=1)
    stats[f'{prefix}_cv']   = stats[f'{prefix}_std'] / (stats[f'{prefix}_mean'] + EPS)
    stats[f'{prefix}_range']= np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    stats[f'{prefix}_iqr']  = np.nanpercentile(values, 75, axis=1) - np.nanpercentile(values, 25, axis=1)
    return stats

def main():
    ensure_directories()
    print(f'Loading DARWIN dataset from {DARWIN_DATASET}')
    df = pd.read_csv(DARWIN_DATASET)
    print('Extracting base features')
    base_features = extract_base_features(df.columns)
    print('Computing engineered features')
    engineered_list = []
    for base, cols in base_features.items():
        stats = compute_stats(df, cols, base)
        engineered_list.append(stats)
    
    # Concatenate all engineered features at once for efficiency
    if engineered_list:
        engineered = pd.concat(engineered_list, axis=1)
    else:
        engineered = pd.DataFrame(index=df.index)
        
    engineered[LABEL_COL] = df[LABEL_COL]
    engineered.to_csv(ENGINEERED_FEATURES_CSV, index=False)
    print(f'Saving engineered features to {ENGINEERED_FEATURES_CSV}')

    report_file = os.path.join(METRICS_DIR, 'feature_engineering.txt')
    with open(report_file, 'w') as f:
        f.write(f'Feature Engineering Report (DARWIN Dataset)\n')
        f.write(f'='*60 + '\n')
        f.write(f"subjects:{df.shape[0]}\n")
        f.write(f'original features:{(df.shape)[1]-1}(excluding label)\n')
        f.write(f'base features:{len(base_features)}\n')
        f.write(f'engineered features per base: 5 (mean, std, cv, range, iqr)\n')
        f.write(f'Final feature count:{engineered.shape[1]-1}(excluding label)\n')
        f.write(f'output file: {ENGINEERED_FEATURES_CSV}\n')
        print(f'report saved to {report_file}')        

if __name__ == "__main__":
 main()