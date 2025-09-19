#INTERNAL
from src.data.make_dataset import load_raw_data
from src.features.build_features import process_data
from src.models.train_model import  train_model
from src.models.predict_model import load_model, predict, predict_probality
from src.models.evaluate_model import evaluate_model

#EXTERNAL
from sklearn.model_selection import train_test_split

def main():
    df = load_raw_data()
    df = process_data(df)
    
    X = df.drop("Churn", axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model_path = "models/log_reg.pkl"
    
    model = train_model(X_train, y_train, save_path=model_path)
    
    model = load_model(model_path)
    evaluate_model(model, X_test, y_test)
    
    

if __name__ == '__main__':
    main()