import pandas as pd

def load_raw_data(path="data/raw/Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    print(df.head())
    return df

load_raw_data()