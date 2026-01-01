"""
Experimento 16: Logistic Regression con ajuste de umbral
Práctica 3 - Competición Kaggle

Objetivo: El baseline LR (0.84782) funciona mejor que modelos más complejos.
Probar diferentes umbrales para maximizar F1 en test.

Análisis previo mostró:
- Baseline predice más clase 1 (botrytis) que Ridge
- Ridge es más conservador pero obtiene peor F1
- Esto sugiere que ser más agresivo prediciendo clase 1 puede ayudar
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

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
    print("EXPERIMENTO 16: LR CON AJUSTE DE UMBRAL")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # 1. Cargar datos
    print("\n[1] Cargando datos...")
    train_df, test_df = load_data(DATA_PATH / 'train.csv', DATA_PATH / 'test.csv')
    X_train, X_test, y_train, feature_cols = prepare_features(train_df, test_df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print(f"    Train: {X_train.shape[0]} muestras")
    print(f"    Test: {X_test.shape[0]} muestras")
    print_class_distribution(y_train)
    
    # 2. Modelo base (igual que baseline)
    print("\n[2] Entrenando modelo base (LR default)...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    
    # 3. Obtener probabilidades con CV
    print("\n[3] Evaluando diferentes umbrales con CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_proba_cv = cross_val_predict(model, X_train_scaled, y_train, cv=cv, method='predict_proba')[:, 1]
    
    # Probar diferentes umbrales
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba_cv >= thresh).astype(int)
        f1 = f1_score(y_train, y_pred)
        n_class1 = y_pred.sum()
        results.append({
            'threshold': thresh,
            'f1_cv': f1,
            'n_class1': n_class1,
            'pct_class1': 100 * n_class1 / len(y_pred)
        })
        print(f"    Umbral {thresh:.2f}: F1 = {f1:.4f}, Clase 1 = {n_class1} ({100*n_class1/len(y_pred):.1f}%)")
    
    # Mejor umbral
    best_result = max(results, key=lambda x: x['f1_cv'])
    print(f"\n    MEJOR UMBRAL: {best_result['threshold']:.2f} (F1 = {best_result['f1_cv']:.4f})")
    
    # 4. Entrenar modelo final
    print("\n[4] Entrenando modelo final...")
    model.fit(X_train_scaled, y_train)
    
    # 5. Generar predicciones con diferentes umbrales
    print("\n[5] Generando submissions con diferentes umbrales...")
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Submission con umbral default (0.5) - debería ser igual al baseline
    predictions_default = (y_proba_test >= 0.5).astype(int)
    filename_default = SUBMISSIONS_PATH / f'submission_16a_lr_thresh050_{timestamp}.csv'
    create_submission_from_test(predictions_default, filename_default)
    print(f"    Umbral 0.50: {predictions_default.sum()} clase 1 -> {filename_default.name}")
    
    # Submission con umbral más bajo (más agresivo en clase 1)
    predictions_low = (y_proba_test >= 0.45).astype(int)
    filename_low = SUBMISSIONS_PATH / f'submission_16b_lr_thresh045_{timestamp}.csv'
    create_submission_from_test(predictions_low, filename_low)
    print(f"    Umbral 0.45: {predictions_low.sum()} clase 1 -> {filename_low.name}")
    
    # Submission con umbral aún más bajo
    predictions_lower = (y_proba_test >= 0.40).astype(int)
    filename_lower = SUBMISSIONS_PATH / f'submission_16c_lr_thresh040_{timestamp}.csv'
    create_submission_from_test(predictions_lower, filename_lower)
    print(f"    Umbral 0.40: {predictions_lower.sum()} clase 1 -> {filename_lower.name}")
    
    # Submission con el mejor umbral de CV
    best_thresh = best_result['threshold']
    predictions_best = (y_proba_test >= best_thresh).astype(int)
    filename_best = SUBMISSIONS_PATH / f'submission_16d_lr_thresh{int(best_thresh*100):03d}_{timestamp}.csv'
    create_submission_from_test(predictions_best, filename_best)
    print(f"    Umbral {best_thresh:.2f}: {predictions_best.sum()} clase 1 -> {filename_best.name}")
    
    # 6. Comparar con baseline original
    print("\n[6] Comparación con baseline original...")
    baseline = pd.read_csv(SUBMISSIONS_PATH / 'submission_01_baseline_logisticregression_20251223_1856.csv')
    print(f"    Baseline original: {baseline['class'].sum()} clase 1")
    print(f"    Nuevo (0.50): {predictions_default.sum()} clase 1")
    print(f"    Diferencia: {predictions_default.sum() - baseline['class'].sum()}")
    
    # 7. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Mejor umbral CV: {best_thresh:.2f} (F1 = {best_result['f1_cv']:.4f})")
    print(f"\nSubmissions generadas:")
    print(f"  - {filename_default.name} (umbral 0.50)")
    print(f"  - {filename_low.name} (umbral 0.45)")
    print(f"  - {filename_lower.name} (umbral 0.40)")
    print(f"  - {filename_best.name} (umbral {best_thresh:.2f})")
    print("=" * 60)

if __name__ == '__main__':
    main()
