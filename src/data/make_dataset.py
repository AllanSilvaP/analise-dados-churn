import pandas as pd

def load_raw_data(path="data/raw/Telco-Customer-Churn.csv"):
    pd.set_option('future.no_silent_downcasting', True)
    df = pd.read_csv(path)

    df = df.drop(columns=["customerID"])

    # transform money in float
    money_columns = ['TotalCharges', 'MonthlyCharges']
    for column in money_columns:
        df[column] = pd.to_numeric(df[column])

    ##Binary Enconding - YES AND NO
    columns_to_change = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    maping = {'Yes': 1, 'No': 0}
    for column in columns_to_change:
        df[column] = df[column].replace(maping)

    ##Binary Enconding - Gender
    maping_gender = {'Female': 0, 'Male': 1}
    df['gender'] = df['gender'].replace(maping_gender)

    # One Hot encondig - 3 or more options
    colums_to_change_onehot = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    for column in colums_to_change_onehot:
        df = pd.get_dummies(df, columns=[column], dtype=int)
    print(df.head())
    return df

load_raw_data()