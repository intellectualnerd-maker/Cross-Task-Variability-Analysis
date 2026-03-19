import pandas as pd
import os
import numpy as np
from scipy.spatial import ConvexHull

input_path = "D:/project-root/Cross-task validation/data/raw/DARWIN_DATASET/data.csv"

def time_cols_conversion(input_path,output_path):
    """
    Converts ms into sec where as per requirement.
    """
df = pd.read_csv(input_path)
time_cols = [col for col in df.columns if "time" in col.lower()]
for col in time_cols:
    stats = df[col].describe()
    mean_val = stats['mean']
    min_val  = stats["min"]
if mean_val > 1000 and min_val > 100:
    df[col]  = df[col]/1000
else:
    print('Resepective column is already in seconds.')
os.makedirs(outut_path,exist_ok = True)
df.to_csv(os.path.join(output_path,"features.csv"),index=False)

print("Features.csv saved to {output_path}")
if __name__=="__main__":
    input_path = "D:/project-root/Cross-task validation/data/raw/DARWIN_DATASET/data.csv"
    output_path= "D:/project-root/Cross-task validation/data/processed"
