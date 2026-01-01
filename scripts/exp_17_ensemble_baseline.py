"""
Experimento 17: Ensemble basado en el baseline
Práctica 3 - Competición Kaggle

Objetivo: El baseline LR simple (0.84782) es el mejor.
Crear un ensemble que combine múltiples LR con diferentes seeds
para reducir varianza y potencialmente mejorar.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score

from src.preprocessing import load_data, prepare_features, scale_features
from src.models import evaluate_model_cv
from src.utils import create_submission_from_test, print_class_distribution

# Configuración
DATA_PATH = PROJECT_ROOT / 'data'
SUBMISSIONS_PATH = PROJECT_ROOT / 'submissions'
RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("EXPERIMENTO 17: ENSEMBLE DE MÚLTIPLES LR")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # 1. Cargar datos
    print("\n[1] Cargando datos...")
    train_df, test_df = load_data(DATA_PATH / 'train.csv', DATA_PATH / 'test.csv')
    X_train, X_test, y_train, feature_cols = prepare_features(train_df, test_df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print(f"    Train: {X_train.shape[0]} muestras, Test: {X_test.shape[0]} muestras")
    
    # 2. Crear ensemble de LR con diferentes configuraciones
    print("\n[2] Creando ensemble de modelos LR...")
    
    models = [
        # Variaciones del baseline
        LogisticRegression(max_iter=1000, random_state=42),
        LogisticRegression(max_iter=1000, random_state=123),
        LogisticRegression(max_iter=1000, random_state=456),
        LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        LogisticRegression(max_iter=1000, C=2.0, random_state=42),
        LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        LogisticRegression(max_iter=1000, solver='saga', random_state=42),
    ]
    
    # 3. Entrenar todos los modelos y obtener probabilidades
    print("\n[3] Entrenando modelos y obteniendo probabilidades...")
    all_probas = []
    
    for i, model in enumerate(models):
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        all_probas.append(proba)
        pred = (proba >= 0.5).astype(int)
        print(f"    Modelo {i+1}: {pred.sum()} clase 1")
    
    # 4. Combinar probabilidades (promedio)
    print("\n[4] Combinando predicciones...")
    avg_proba = np.mean(all_probas, axis=0)
    
    # Predicciones con diferentes umbrales
    pred_050 = (avg_proba >= 0.50).astype(int)
    pred_048 = (avg_proba >= 0.48).astype(int)
    pred_045 = (avg_proba >= 0.45).astype(int)
    
    print(f"    Ensemble (umbral 0.50): {pred_050.sum()} clase 1")
    print(f"    Ensemble (umbral 0.48): {pred_048.sum()} clase 1")
    print(f"    Ensemble (umbral 0.45): {pred_045.sum()} clase 1")
    
    # 5. Voting por mayoría
    print("\n[5] Voting por mayoría...")
    all_preds = np.array([(p >= 0.5).astype(int) for p in all_probas])
    majority_vote = (all_preds.sum(axis=0) >= len(models) / 2).astype(int)
    print(f"    Majority vote: {majority_vote.sum()} clase 1")
    
    # 6. Comparar con baseline
    print("\n[6] Comparación con baseline...")
    baseline = pd.read_csv(SUBMISSIONS_PATH / 'submission_01_baseline_logisticregression_20251223_1856.csv')
    baseline_pred = baseline['class'].values
    
    diff_ensemble = (pred_050 != baseline_pred).sum()
    diff_majority = (majority_vote != baseline_pred).sum()
    
    print(f"    Baseline: {baseline_pred.sum()} clase 1")
    print(f"    Diferencias ensemble vs baseline: {diff_ensemble}")
    print(f"    Diferencias majority vs baseline: {diff_majority}")
    
    # 7. Generar submissions
    print("\n[7] Generando submissions...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Ensemble promedio
    filename_ensemble = SUBMISSIONS_PATH / f'submission_17a_ensemble_avg_{timestamp}.csv'
    create_submission_from_test(pred_050, filename_ensemble)
    print(f"    -> {filename_ensemble.name}")
    
    # Majority vote
    filename_majority = SUBMISSIONS_PATH / f'submission_17b_majority_vote_{timestamp}.csv'
    create_submission_from_test(majority_vote, filename_majority)
    print(f"    -> {filename_majority.name}")
    
    # Ensemble con umbral más bajo (más agresivo)
    filename_low = SUBMISSIONS_PATH / f'submission_17c_ensemble_thresh048_{timestamp}.csv'
    create_submission_from_test(pred_048, filename_low)
    print(f"    -> {filename_low.name}")
    
    # 8. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Baseline original: {baseline_pred.sum()} clase 1 (F1 Kaggle = 0.84782)")
    print(f"Ensemble avg (0.50): {pred_050.sum()} clase 1")
    print(f"Majority vote: {majority_vote.sum()} clase 1")
    print(f"\nSi el ensemble tiene predicciones similares al baseline,")
    print(f"debería obtener un F1 similar (~0.84782)")
    print("=" * 60)

if __name__ == '__main__':
    main()
