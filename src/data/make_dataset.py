import pandas as pd

def load_raw_data(path="data/raw/Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)

    df = df.drop(columns=["customerID"])

    ##Binary Enconding - YES AND NO
    colums_to_change = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    maping = {'Yes': 1, 'No': 0}
    for column in colums_to_change:
        df[column] = df[column].replace(maping)

    ##Binary Enconding - Gender
    maping_gender = {'Female': 0, 'Male': 1}
    df['Gender'] = df['Gender'].replace(maping_gender)

    print(df.head())
    return df

load_raw_data()