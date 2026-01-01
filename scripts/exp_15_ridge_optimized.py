"""
Experimento 15: Ridge Classifier Optimizado
Práctica 3 - Competición Kaggle

Objetivo: Optimizar el parámetro alpha de Ridge Classifier
que mostró el mejor rendimiento en exp_14.
"""

import sys
from pathlib import Path

# Añadir src al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
    print("EXPERIMENTO 15: RIDGE CLASSIFIER OPTIMIZADO")
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
    
    # 4. Grid Search para optimizar alpha
    print("\n[4] Optimizando alpha con GridSearchCV...")
    
    param_grid = {
        'alpha': [0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
        'class_weight': [None, 'balanced'],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        RidgeClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n    Mejores parámetros: {grid_search.best_params_}")
    print(f"    Mejor F1-Score CV: {grid_search.best_score_:.4f}")
    
    # 5. Evaluar mejor modelo
    print("\n[5] Evaluando mejor modelo...")
    best_model = grid_search.best_estimator_
    mean_score, std_score, scores = evaluate_model_cv(best_model, X_train_scaled, y_train, cv=5)
    print(f"    F1-Score: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"    Scores por fold: {[f'{s:.4f}' for s in scores]}")
    
    # 6. Entrenar modelo final y predecir
    print("\n[6] Entrenando modelo final con todos los datos...")
    best_model.fit(X_train_scaled, y_train)
    predictions = best_model.predict(X_test_scaled)
    
    # Estadísticas de predicciones
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"    Predicciones: {dict(zip(unique, counts))}")
    
    # 7. Crear submission
    print("\n[7] Generando submission...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    submission_filename = SUBMISSIONS_PATH / f'submission_15_ridge_optimized_{timestamp}.csv'
    
    submission = create_submission_from_test(predictions, submission_filename)
    
    # 8. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL EXPERIMENTO")
    print("=" * 60)
    print(f"Preprocesado: StandardScaler")
    print(f"Modelo: Ridge Classifier (optimizado)")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Score CV (F1): {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"Submission: {submission_filename.name}")
    print("=" * 60)
    
    return {
        'exp_num': 15,
        'best_params': grid_search.best_params_,
        'score_cv': mean_score,
        'submission_file': submission_filename.name
    }

if __name__ == '__main__':
    results = main()
