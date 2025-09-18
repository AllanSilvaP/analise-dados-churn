import pickle
import pandas as pd

def load_model(path="models/log_reg.pkl"):
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model

def predict(model, X):
    return model.predict(X)

def predict_probality(model, X):
    return model.predict_proba(X)[:, 1]