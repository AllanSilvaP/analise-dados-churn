#INTERNAL
from src.data.make_dataset import load_raw_data
from src.features.build_features import process_data
from src.models.train_model import  train_model
from src.models.predict_model import load_model, predict, predict_probality
from src.models.evaluate_model import evaluate_model

#EXTERNAL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def main():
    df = load_raw_data()
    df = process_data(df)
    
    X = df.drop("Churn", axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    model_path = "models/log_reg.pkl"
    
    model = train_model(X_train, y_train, save_path=model_path)
    
    model = load_model(model_path)
    evaluate_model(model, X_test, y_test, threshold=0.7)
    
    

if __name__ == '__main__':
    main()