import re
import os
import json
from pathlib import Path
import pandas as pd
DARWIN_file_path = "D:/project-root/Cross-task validation/data/raw/DARWIN_DATASET/data.csv"
df=pd.read_csv(DARWIN_file_path)
def standardize_column_names(df,json_path='configs/column_changes.json'):
    """
    Standardizes column names by converting them to lowercase, replacing spaces with underscores,
    and removing special characters and finally saving changes into JSON file.
    """
    revamped_columns = {}
    cleaned=[]
    for col in df.columns:
        new_col = col.lower()
        
        new_col = col.strip()
        new_col = re.sub(r'[^a-z0-9_]', '',new_col)
        new_col = re.sub(r'__+', '_', new_col)  # Replace multiple underscores with a single underscore
        new_col = new_col.replace(' ', '_') # Replace spaces with underscores
        new_col = new_col.replace('-', '_') # Replace hyphens with underscores
        new_col = new_col.replace('(', '')  # Remove opening parentheses
        new_col = new_col.replace(')', '')  # Remove closing parentheses
        new_col = new_col.strip()  # Remove leading/trailing underscores
        cleaned.append(new_col)
    if col != new_col:
        revamped_columns[col] = new_col
        df.columns = cleaned # apply cleaned names to dataframe
        os.makedirs(os.path.dirname(json_path),exist_ok = True)
        #save changes to JSON if any 
    if revamped_columns:
        with open (json_path,'w') as f:
            json.dump(revamped_columns, f, indent=4)
    return df, revamped_columns