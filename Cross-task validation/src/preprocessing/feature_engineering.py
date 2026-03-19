import os
import re
import numpy as np
import pandas as pd

Label_Col = 'class'
DARWIN_DATASET = 'D:/project-root/Cross-task validation/data/raw/DARWIN_DATASET/data.csv'
Results_dir = 'D:/project-root/Cross-task validation/results'
Metrics_dir = 'D:/project-root/Cross-task validation/results/metrics'
os.makedirs(Results_dir,exist_ok = True)
os.makedirs(Metrics_dir,exist_ok = True)
EPS = 1e-9

def extract_base_features(columns):
    """From columns extract base features"""
    base_features = {}
    for col in columns:
        if col == Label_Col:
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
    stats[f'{prefix}_mean'] = np.nanmean(values, axis= 1)
    stats[f'{prefix}_std']  = np.nanstd(values, axis= 1)
    stats[f'{prefix}_cv']   = stats[f'{prefix}_std'] / (stats[f'{prefix}_mean'] + EPS)
    stats[f'{prefix}_range']= np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    stats[f'{prefix}_iqr']  = np.nanpercentile(values, 75, axis=1) - np.nanpercentile(values, 25, axis=1)
    return stats

def main():
    print('Loading DARWIN dataset')
    df=pd.read_csv('D:/project-root/Cross-task validation/data/raw/DARWIN_DATASET/data.csv')
    print('Extracting base features')
    base_features = extract_base_features(df.columns)
    engineered   = pd.DataFrame(index=df.index)
    print('Computing engineered features')
    for base, cols in base_features.items():
        stats = compute_stats(df, cols, base)
        engineered = pd.concat([engineered, stats], axis=1)
    engineered[Label_Col] = df[Label_Col]
    out_csv = os.path.join(Results_dir, 'engineered_features.csv')
    engineered.to_csv(out_csv, index=False)
    print(f'Saving engineered features to {out_csv}')

    report_file = os.path.join(Metrics_dir, 'feature_engineering.txt')
    with open(report_file, 'w') as f:
        f.write(f'Feature Engineering Report (DARWIN Dataset)\n')
        f.write(f'='*60 + '\n')
        f.write(f"subjects:{df.shape[0]}\n")
        f.write(f'original features:{(df.shape)[1]-1}(excluding label)\n')
        f.write(f'base features:{len(base_features)}\n')
        f.write(f'engineered features per base: 5 (mean, std, cv, range, iqr)\n')
        f.write(f'Final feature count:{engineered.shape[1]-1}(excluding label)\n')
        f.write(f'output file: {out_csv}\n')
        print(f'report saved to {report_file}')        

if __name__ == "__main__":
 main()