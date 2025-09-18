import pandas as pd

def load_raw_data(path="data/raw/Telco-Customer-Churn.csv"):
    pd.set_option('future.no_silent_downcasting', True)
    df = pd.read_csv(path)

    df = df.drop(columns=["customerID"])

    # transform money in float
    money_columns = ['TotalCharges', 'MonthlyCharges']
    df[money_columns] = df[money_columns].apply(pd.to_numeric, errors='coerce')
    
    df = df.dropna(subset=money_columns)
    return df