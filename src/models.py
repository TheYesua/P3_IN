"""
Definición de modelos para la Práctica 3 - Kaggle
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np


def get_baseline_models():
    """
    Retorna un diccionario con modelos baseline para comparación inicial.
    """
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
    }
    return models


def evaluate_model_cv(model, X, y, cv=5, scoring='f1'):
    """
    Evalúa un modelo usando validación cruzada estratificada.
    
    Returns:
        mean_score, std_score, scores
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    
    return scores.mean(), scores.std(), scores


def compare_models(models, X, y, cv=5):
    """
    Compara múltiples modelos usando validación cruzada.
    
    Returns:
        DataFrame con resultados
    """
    results = []
    
    for name, model in models.items():
        mean_score, std_score, _ = evaluate_model_cv(model, X, y, cv=cv)
        results.append({
            'Model': name,
            'F1_Mean': mean_score,
            'F1_Std': std_score
        })
        print(f"{name}: F1 = {mean_score:.4f} (+/- {std_score:.4f})")
    
    return results


def train_and_predict(model, X_train, y_train, X_test):
    """
    Entrena un modelo y genera predicciones.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def get_classification_report(y_true, y_pred):
    """
    Genera un reporte de clasificación completo.
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['control', 'botrytis']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    f1 = f1_score(y_true, y_pred)
    print(f"\nF1-Score: {f1:.4f}")
    
    return f1
