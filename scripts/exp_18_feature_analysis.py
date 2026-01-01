"""
Experimento 18: Análisis de features y selección inteligente
Práctica 3 - Competición Kaggle

Objetivo: Analizar si hay features que añaden ruido y perjudican
la generalización. Probar LR con subconjuntos de features.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.preprocessing import load_data, prepare_features, scale_features, FLUORESCENCE_COLS, get_spectral_columns
from src.utils import create_submission_from_test

# Configuración
DATA_PATH = PROJECT_ROOT / 'data'
SUBMISSIONS_PATH = PROJECT_ROOT / 'submissions'
RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("EXPERIMENTO 18: ANÁLISIS DE FEATURES")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # 1. Cargar datos
    print("\n[1] Cargando datos...")
    train_df, test_df = load_data(DATA_PATH / 'train.csv', DATA_PATH / 'test.csv')
    X_train, X_test, y_train, feature_cols = prepare_features(train_df, test_df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print(f"    Total features: {len(feature_cols)}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # 2. Baseline con todas las features
    print("\n[2] Baseline (todas las features)...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    scores = cross_val_score(lr, X_train_scaled, y_train, cv=cv, scoring='f1')
    print(f"    F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
    baseline_f1 = scores.mean()
    
    # 3. Solo features de fluorescencia
    print("\n[3] Solo fluorescencia (4 features)...")
    fluor_idx = [i for i, col in enumerate(feature_cols) if col in FLUORESCENCE_COLS]
    if fluor_idx:
        X_train_fluor = X_train_scaled[:, fluor_idx]
        X_test_fluor = X_test_scaled[:, fluor_idx]
        scores = cross_val_score(lr, X_train_fluor, y_train, cv=cv, scoring='f1')
        print(f"    F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # 4. Solo features espectrales
    print("\n[4] Solo espectrales (300 features)...")
    spectral_cols = get_spectral_columns(train_df)
    spectral_idx = [i for i, col in enumerate(feature_cols) if col in spectral_cols]
    if spectral_idx:
        X_train_spec = X_train_scaled[:, spectral_idx]
        X_test_spec = X_test_scaled[:, spectral_idx]
        scores = cross_val_score(lr, X_train_spec, y_train, cv=cv, scoring='f1')
        print(f"    F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # 5. Feature selection con L1
    print("\n[5] Feature selection con L1 (Lasso)...")
    lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr_l1.fit(X_train_scaled, y_train)
    n_nonzero = np.sum(lr_l1.coef_ != 0)
    print(f"    Features con coef != 0: {n_nonzero}")
    
    # Usar SelectFromModel
    selector = SelectFromModel(lr_l1, prefit=True)
    X_train_l1 = selector.transform(X_train_scaled)
    X_test_l1 = selector.transform(X_test_scaled)
    print(f"    Features seleccionadas: {X_train_l1.shape[1]}")
    
    scores = cross_val_score(lr, X_train_l1, y_train, cv=cv, scoring='f1')
    print(f"    F1 con features L1: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # 6. RFE (Recursive Feature Elimination)
    print("\n[6] RFE con 100 features...")
    rfe = RFE(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), n_features_to_select=100, step=50)
    rfe.fit(X_train_scaled, y_train)
    X_train_rfe = rfe.transform(X_train_scaled)
    X_test_rfe = rfe.transform(X_test_scaled)
    
    scores = cross_val_score(lr, X_train_rfe, y_train, cv=cv, scoring='f1')
    print(f"    F1 con RFE(100): {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # 7. Probar diferentes cantidades de features con RFE
    print("\n[7] Probando diferentes cantidades de features...")
    best_n = 0
    best_f1 = 0
    
    for n_features in [50, 100, 150, 200, 250]:
        rfe = RFE(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), 
                  n_features_to_select=n_features, step=50)
        rfe.fit(X_train_scaled, y_train)
        X_rfe = rfe.transform(X_train_scaled)
        scores = cross_val_score(lr, X_rfe, y_train, cv=cv, scoring='f1')
        f1 = scores.mean()
        print(f"    RFE({n_features}): F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_n = n_features
    
    print(f"\n    Mejor: RFE({best_n}) con F1 = {best_f1:.4f}")
    
    # 8. Generar submission con mejor configuración
    print("\n[8] Generando submissions...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Con todas las features (baseline)
    lr.fit(X_train_scaled, y_train)
    pred_all = lr.predict(X_test_scaled)
    filename_all = SUBMISSIONS_PATH / f'submission_18a_lr_all_features_{timestamp}.csv'
    create_submission_from_test(pred_all, filename_all)
    print(f"    Todas features: {pred_all.sum()} clase 1 -> {filename_all.name}")
    
    # Con features L1
    lr.fit(X_train_l1, y_train)
    pred_l1 = lr.predict(X_test_l1)
    filename_l1 = SUBMISSIONS_PATH / f'submission_18b_lr_l1_features_{timestamp}.csv'
    create_submission_from_test(pred_l1, filename_l1)
    print(f"    Features L1: {pred_l1.sum()} clase 1 -> {filename_l1.name}")
    
    # Con mejor RFE
    rfe_best = RFE(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), 
                   n_features_to_select=best_n, step=50)
    rfe_best.fit(X_train_scaled, y_train)
    X_train_best = rfe_best.transform(X_train_scaled)
    X_test_best = rfe_best.transform(X_test_scaled)
    lr.fit(X_train_best, y_train)
    pred_rfe = lr.predict(X_test_best)
    filename_rfe = SUBMISSIONS_PATH / f'submission_18c_lr_rfe{best_n}_{timestamp}.csv'
    create_submission_from_test(pred_rfe, filename_rfe)
    print(f"    RFE({best_n}): {pred_rfe.sum()} clase 1 -> {filename_rfe.name}")
    
    # 9. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Baseline (todas): F1 CV = {baseline_f1:.4f}")
    print(f"Mejor RFE({best_n}): F1 CV = {best_f1:.4f}")
    if best_f1 > baseline_f1:
        print(f"MEJORA: +{(best_f1-baseline_f1)*100:.2f}%")
    else:
        print(f"Sin mejora significativa")
    print("=" * 60)

if __name__ == '__main__':
    main()
