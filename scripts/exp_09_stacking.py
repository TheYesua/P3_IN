"""
Experimento 09: StackingClassifier
Práctica 3 - Competición Kaggle

Objetivo: Usar stacking con múltiples modelos base y Logistic Regression
como meta-learner para combinar las predicciones de forma óptima.
"""

import sys
from pathlib import Path

# Añadir src al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from src.preprocessing import (
    load_data, prepare_features, scale_features,
    FLUORESCENCE_COLS, get_spectral_columns
)
from src.models import evaluate_model_cv, train_and_predict
from src.utils import create_submission_from_test, print_class_distribution

# Configuración
DATA_PATH = PROJECT_ROOT / 'data'
SUBMISSIONS_PATH = PROJECT_ROOT / 'submissions'
RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("EXPERIMENTO 09: STACKING CLASSIFIER")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # 1. Cargar datos
    print("\n[1] Cargando datos...")
    train_df, test_df = load_data(
        DATA_PATH / 'train.csv',
        DATA_PATH / 'test.csv'
    )
    print(f"    Train: {train_df.shape[0]} muestras")
    print(f"    Test: {test_df.shape[0]} muestras")
    
    # 2. Preparar features
    print("\n[2] Preparando features...")
    X_train, X_test, y_train, feature_cols = prepare_features(train_df, test_df)
    print(f"    Features: {len(feature_cols)}")
    
    # Distribución de clases
    print_class_distribution(y_train)
    
    # 3. Escalar features
    print("\n[3] Escalando features (StandardScaler)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, method='standard')
    
    # 4. Definir modelos base
    print("\n[4] Configurando Stacking Classifier...")
    
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('nb', GaussianNB()),
    ]
    
    # Meta-learner: Logistic Regression
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        stack_method='auto',
        n_jobs=-1
    )
    
    # 5. Evaluar con CV
    print("\n[5] Evaluando StackingClassifier (5-fold CV)...")
    mean_score, std_score, scores = evaluate_model_cv(stacking, X_train_scaled, y_train, cv=5)
    print(f"    F1-Score: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"    Scores por fold: {[f'{s:.4f}' for s in scores]}")
    
    # 6. Entrenar modelo final y predecir
    print("\n[6] Entrenando modelo final...")
    stacking.fit(X_train_scaled, y_train)
    predictions = stacking.predict(X_test_scaled)
    
    # Estadísticas de predicciones
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"    Predicciones: {dict(zip(unique, counts))}")
    
    # 7. Crear submission
    print("\n[7] Generando submission...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    submission_filename = SUBMISSIONS_PATH / f'submission_09_stacking_{timestamp}.csv'
    
    submission = create_submission_from_test(predictions, submission_filename)
    
    # 8. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL EXPERIMENTO")
    print("=" * 60)
    print(f"Preprocesado: StandardScaler")
    print(f"Modelo: StackingClassifier")
    print(f"Modelos base: LR, SVM, RF, KNN, NB")
    print(f"Meta-learner: Logistic Regression")
    print(f"Score CV (F1): {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"Submission: {submission_filename.name}")
    print("=" * 60)
    
    return {
        'exp_num': 9,
        'score_cv': mean_score,
        'submission_file': submission_filename.name
    }

if __name__ == '__main__':
    results = main()
