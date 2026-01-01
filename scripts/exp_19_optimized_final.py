"""
Experimento 19: Modelo Final Optimizado
Práctica 3 - Competición Kaggle

Combinando los mejores hallazgos:
- RFE con 100-150 features mejora F1-CV
- Solo features espectrales dan buen resultado
- LR simple generaliza mejor que modelos complejos
- El baseline predice 99 clase 1 y obtiene 0.84782
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.preprocessing import load_data, prepare_features, scale_features, get_spectral_columns
from src.utils import create_submission_from_test

# Configuración
DATA_PATH = PROJECT_ROOT / 'data'
SUBMISSIONS_PATH = PROJECT_ROOT / 'submissions'
RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("EXPERIMENTO 19: MODELO FINAL OPTIMIZADO")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # 1. Cargar datos
    print("\n[1] Cargando datos...")
    train_df, test_df = load_data(DATA_PATH / 'train.csv', DATA_PATH / 'test.csv')
    X_train, X_test, y_train, feature_cols = prepare_features(train_df, test_df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    
    # 2. Solo features espectrales
    print("\n[2] Probando solo features espectrales...")
    spectral_cols = get_spectral_columns(train_df)
    spectral_idx = [i for i, col in enumerate(feature_cols) if col in spectral_cols]
    X_train_spec = X_train_scaled[:, spectral_idx]
    X_test_spec = X_test_scaled[:, spectral_idx]
    
    scores = cross_val_score(lr, X_train_spec, y_train, cv=cv, scoring='f1')
    print(f"    Espectrales (300): F1 = {scores.mean():.4f}")
    
    # 3. RFE sobre espectrales
    print("\n[3] RFE sobre features espectrales...")
    best_f1 = 0
    best_config = None
    
    for n_features in [50, 75, 100, 125, 150]:
        rfe = RFE(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), 
                  n_features_to_select=n_features, step=25)
        rfe.fit(X_train_spec, y_train)
        X_rfe = rfe.transform(X_train_spec)
        scores = cross_val_score(lr, X_rfe, y_train, cv=cv, scoring='f1')
        f1 = scores.mean()
        print(f"    RFE({n_features}) espectrales: F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_config = ('spectral_rfe', n_features, rfe)
    
    # 4. RFE sobre todas las features (comparación)
    print("\n[4] RFE sobre todas las features...")
    for n_features in [100, 150]:
        rfe = RFE(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), 
                  n_features_to_select=n_features, step=50)
        rfe.fit(X_train_scaled, y_train)
        X_rfe = rfe.transform(X_train_scaled)
        scores = cross_val_score(lr, X_rfe, y_train, cv=cv, scoring='f1')
        f1 = scores.mean()
        print(f"    RFE({n_features}) todas: F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_config = ('all_rfe', n_features, rfe)
    
    print(f"\n    MEJOR: {best_config[0]} con {best_config[1]} features, F1 = {best_f1:.4f}")
    
    # 5. Generar submissions
    print("\n[5] Generando submissions...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Submission con mejor configuración
    if best_config[0] == 'spectral_rfe':
        X_train_best = best_config[2].transform(X_train_spec)
        X_test_best = best_config[2].transform(X_test_spec)
    else:
        X_train_best = best_config[2].transform(X_train_scaled)
        X_test_best = best_config[2].transform(X_test_scaled)
    
    lr.fit(X_train_best, y_train)
    pred_best = lr.predict(X_test_best)
    filename_best = SUBMISSIONS_PATH / f'submission_19a_best_rfe_{timestamp}.csv'
    create_submission_from_test(pred_best, filename_best)
    print(f"    Mejor config: {pred_best.sum()} clase 1 -> {filename_best.name}")
    
    # Submission con solo espectrales (sin RFE)
    lr.fit(X_train_spec, y_train)
    pred_spec = lr.predict(X_test_spec)
    filename_spec = SUBMISSIONS_PATH / f'submission_19b_spectral_only_{timestamp}.csv'
    create_submission_from_test(pred_spec, filename_spec)
    print(f"    Solo espectrales: {pred_spec.sum()} clase 1 -> {filename_spec.name}")
    
    # Submission con RFE(100) espectrales
    rfe_100 = RFE(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), 
                  n_features_to_select=100, step=25)
    rfe_100.fit(X_train_spec, y_train)
    X_train_100 = rfe_100.transform(X_train_spec)
    X_test_100 = rfe_100.transform(X_test_spec)
    lr.fit(X_train_100, y_train)
    pred_100 = lr.predict(X_test_100)
    filename_100 = SUBMISSIONS_PATH / f'submission_19c_spectral_rfe100_{timestamp}.csv'
    create_submission_from_test(pred_100, filename_100)
    print(f"    Spectral RFE(100): {pred_100.sum()} clase 1 -> {filename_100.name}")
    
    # 6. Comparar con baseline
    print("\n[6] Comparación con baseline...")
    baseline = pd.read_csv(SUBMISSIONS_PATH / 'submission_01_baseline_logisticregression_20251223_1856.csv')
    print(f"    Baseline: {baseline['class'].sum()} clase 1 (Kaggle F1 = 0.84782)")
    print(f"    Mejor nuevo: {pred_best.sum()} clase 1")
    print(f"    Diferencias: {(pred_best != baseline['class'].values).sum()}")
    
    # 7. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"Mejor configuración: {best_config[0]} con {best_config[1]} features")
    print(f"F1-CV: {best_f1:.4f} (baseline: 0.8278)")
    print(f"\nSubmissions a probar en Kaggle:")
    print(f"  1. {filename_best.name} - Mejor F1-CV")
    print(f"  2. {filename_spec.name} - Solo espectrales")
    print(f"  3. {filename_100.name} - Spectral RFE(100)")
    print("=" * 60)

if __name__ == '__main__':
    main()
