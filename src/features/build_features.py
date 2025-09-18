import pandas as pd

def build_features(df):
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