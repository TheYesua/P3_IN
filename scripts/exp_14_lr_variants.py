"""
Experimento 14: Variantes de Logistic Regression
Práctica 3 - Competición Kaggle

Objetivo: Probar diferentes configuraciones de LR incluyendo
regularización L1, ElasticNet, y diferentes valores de C.
"""

import sys
from pathlib import Path

# Añadir src al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

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
    print("EXPERIMENTO 14: VARIANTES DE LOGISTIC REGRESSION")
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
    
    # 4. Probar múltiples variantes
    print("\n[4] Evaluando variantes de LR...")
    
    models = {
        # LR con L2 (baseline optimizado)
        'LR L2 C=5': LogisticRegression(C=5.0, penalty='l2', solver='liblinear', max_iter=2000, random_state=RANDOM_STATE),
        'LR L2 C=10': LogisticRegression(C=10.0, penalty='l2', solver='liblinear', max_iter=2000, random_state=RANDOM_STATE),
        'LR L2 C=2': LogisticRegression(C=2.0, penalty='l2', solver='liblinear', max_iter=2000, random_state=RANDOM_STATE),
        
        # LR con L1 (sparsity)
        'LR L1 C=1': LogisticRegression(C=1.0, penalty='l1', solver='liblinear', max_iter=2000, random_state=RANDOM_STATE),
        'LR L1 C=5': LogisticRegression(C=5.0, penalty='l1', solver='liblinear', max_iter=2000, random_state=RANDOM_STATE),
        'LR L1 C=10': LogisticRegression(C=10.0, penalty='l1', solver='liblinear', max_iter=2000, random_state=RANDOM_STATE),
        
        # ElasticNet
        'LR ElasticNet C=1': LogisticRegression(C=1.0, penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=3000, random_state=RANDOM_STATE),
        'LR ElasticNet C=5': LogisticRegression(C=5.0, penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=3000, random_state=RANDOM_STATE),
        
        # SGD con diferentes loss
        'SGD log_loss': SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=2000, random_state=RANDOM_STATE),
        'SGD modified_huber': SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.0001, max_iter=2000, random_state=RANDOM_STATE),
        
        # Ridge Classifier
        'Ridge alpha=1': RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE),
        'Ridge alpha=0.1': RidgeClassifier(alpha=0.1, random_state=RANDOM_STATE),
    }
    
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
            mean_score = scores.mean()
            std_score = scores.std()
            results.append((name, mean_score, std_score, model))
            print(f"    {name}: F1 = {mean_score:.4f} (+/- {std_score:.4f})")
        except Exception as e:
            print(f"    {name}: ERROR - {str(e)[:50]}")
    
    # 5. Seleccionar mejor modelo
    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score, best_std, best_model = results[0]
    
    print(f"\n    MEJOR: {best_name} (F1 = {best_score:.4f})")
    
    # 6. Entrenar y predecir
    print("\n[5] Entrenando modelo final...")
    best_model.fit(X_train_scaled, y_train)
    predictions = best_model.predict(X_test_scaled)
    
    # Estadísticas
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"    Predicciones: {dict(zip(unique, counts))}")
    
    # 7. Crear submission
    print("\n[6] Generando submission...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    submission_filename = SUBMISSIONS_PATH / f'submission_14_lr_variants_{timestamp}.csv'
    
    submission = create_submission_from_test(predictions, submission_filename)
    
    # 8. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL EXPERIMENTO")
    print("=" * 60)
    print(f"Preprocesado: StandardScaler")
    print(f"Mejor modelo: {best_name}")
    print(f"Score CV (F1): {best_score:.4f} (+/- {best_std:.4f})")
    print(f"Submission: {submission_filename.name}")
    print("\nTop 5 modelos:")
    for name, score, std, _ in results[:5]:
        print(f"  - {name}: {score:.4f} (+/- {std:.4f})")
    print("=" * 60)
    
    return {
        'exp_num': 14,
        'best_model': best_name,
        'score_cv': best_score,
        'submission_file': submission_filename.name
    }

if __name__ == '__main__':
    results = main()
