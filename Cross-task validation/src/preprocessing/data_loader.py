import pandas as pd
DARWIN_file_path = "D:/project-root/Cross-task validation/data/raw/DARWIN_DATASET/data.csv"

def data_loader(DARWIN_file_path='D:/project-root/Cross-task validation/data/raw/DARWIN_DATASET/data.csv'):
    data = pd.read_csv(DARWIN_file_path)
    print('Data loaded successfully.')
    return data

def basic_data_audit(DARWIN):
    print("First 5 rows of the dataset:"
          , DARWIN.head())
    print("The shape of the DARWIN dataset is:"
          , DARWIN.shape)
    print('The columns of the DARWIN dataset are:'
          , DARWIN.columns)
    print('The basic information of the dataset is:'
          , DARWIN.info())
    print('The "missing values" status of the dataset is:'
          , DARWIN.isnull().sum()
          , DARWIN.isna().sum()*100)
    return DARWIN
    
def basic_stats_audit(DARWIN):
    print('The basic descriptive stats audit of the dataset (all types):'
      , DARWIN.describe(include='all'))
    return DARWIN

