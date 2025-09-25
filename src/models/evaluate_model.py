from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from scipy import stats
import numpy as np

def evaluate_model(model, X_test, y_test, threshold=0.5):
    # Adjust predictions based on the given threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > threshold).astype(int)
    
    print (f"Threshold: {threshold}")
    print("Score:", accuracy_score(y_test, y_pred))
    print ("ROC AUC:", roc_auc_score(y_test, y_proba))
    print('\nClassify', classification_report(y_test, y_pred))
    print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
    
    ks_stat, p_value = stats.ks_2samp(y_proba[y_test==1], y_proba[y_test==0])
    print(f"KS Test: {ks_stat:.4f} (p={p_value:.4f})")