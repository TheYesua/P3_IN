"""
Experimento 12: Logistic Regression con Calibración de Probabilidades
Práctica 3 - Competición Kaggle

Objetivo: Usar CalibratedClassifierCV para mejorar las probabilidades
y potencialmente el umbral de decisión óptimo.
"""

import sys
from pathlib import Path

# Añadir src al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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

def find_optimal_threshold(y_true, y_proba):
    """Encuentra el umbral óptimo para maximizar F1."""
    from sklearn.metrics import f1_score
    
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.3, 0.7, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

def main():
    print("=" * 60)
    print("EXPERIMENTO 12: LR CON CALIBRACIÓN Y UMBRAL ÓPTIMO")
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
    
    # 4. Modelo base
    print("\n[4] Configurando modelo...")
    base_model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    
    # 5. Obtener probabilidades con CV para encontrar umbral óptimo
    print("\n[5] Buscando umbral óptimo...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_proba_cv = cross_val_predict(base_model, X_train_scaled, y_train, cv=cv, method='predict_proba')[:, 1]
    
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_train, y_proba_cv)
    print(f"    Umbral óptimo: {optimal_threshold:.2f}")
    print(f"    F1 con umbral óptimo: {optimal_f1:.4f}")
    
    # 6. Calibrar modelo
    print("\n[6] Calibrando modelo...")
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    
    # Evaluar modelo calibrado
    mean_score, std_score, scores = evaluate_model_cv(calibrated_model, X_train_scaled, y_train, cv=5)
    print(f"    F1-Score calibrado (umbral 0.5): {mean_score:.4f} (+/- {std_score:.4f})")
    
    # 7. Entrenar modelo final
    print("\n[7] Entrenando modelo final...")
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Predecir con umbral óptimo
    y_proba_test = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    predictions_optimal = (y_proba_test >= optimal_threshold).astype(int)
    predictions_default = calibrated_model.predict(X_test_scaled)
    
    # Estadísticas
    unique_opt, counts_opt = np.unique(predictions_optimal, return_counts=True)
    unique_def, counts_def = np.unique(predictions_default, return_counts=True)
    print(f"    Predicciones (umbral óptimo {optimal_threshold:.2f}): {dict(zip(unique_opt, counts_opt))}")
    print(f"    Predicciones (umbral 0.5): {dict(zip(unique_def, counts_def))}")
    
    # 8. Crear submissions
    print("\n[8] Generando submissions...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Submission con umbral óptimo
    submission_filename_opt = SUBMISSIONS_PATH / f'submission_12a_calibrated_optimal_{timestamp}.csv'
    create_submission_from_test(predictions_optimal, submission_filename_opt)
    
    # Submission con umbral default
    submission_filename_def = SUBMISSIONS_PATH / f'submission_12b_calibrated_default_{timestamp}.csv'
    create_submission_from_test(predictions_default, submission_filename_def)
    
    # 9. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL EXPERIMENTO")
    print("=" * 60)
    print(f"Preprocesado: StandardScaler")
    print(f"Modelo: Logistic Regression + Calibración Isotónica")
    print(f"Umbral óptimo encontrado: {optimal_threshold:.2f}")
    print(f"F1 CV (umbral óptimo): {optimal_f1:.4f}")
    print(f"F1 CV (umbral 0.5): {mean_score:.4f}")
    print(f"Submissions generadas:")
    print(f"  - {submission_filename_opt.name} (umbral {optimal_threshold:.2f})")
    print(f"  - {submission_filename_def.name} (umbral 0.5)")
    print("=" * 60)
    
    return {
        'exp_num': 12,
        'optimal_threshold': optimal_threshold,
        'score_cv_optimal': optimal_f1,
        'score_cv_default': mean_score
    }

if __name__ == '__main__':
    results = main()
