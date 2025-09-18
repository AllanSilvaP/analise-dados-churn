from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X_train, y_train, save_path="models/log_reg.pkl"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model