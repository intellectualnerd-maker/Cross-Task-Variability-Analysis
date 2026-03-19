import pandas as pd
file_path="D:/DARWIN_DATASET/data.csv"
df=pd.read_csv(file_path)
print('Columns_1-50:', df.columns[:50])
print('Columns_51-100:', df.columns[50:100])










