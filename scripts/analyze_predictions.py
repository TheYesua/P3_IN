"""
Análisis de diferencias entre predicciones del baseline y Ridge.
Objetivo: Entender por qué el baseline (F1=0.84782) supera a Ridge (F1=0.82485) en Kaggle.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from collections import Counter

# Paths
SUBMISSIONS_PATH = PROJECT_ROOT / 'submissions'

# Cargar submissions
baseline = pd.read_csv(SUBMISSIONS_PATH / 'submission_01_baseline_logisticregression_20251223_1856.csv')
ridge = pd.read_csv(SUBMISSIONS_PATH / 'submission_15_ridge_optimized_20260101_1651.csv')

print("=" * 60)
print("ANÁLISIS DE PREDICCIONES: BASELINE vs RIDGE")
print("=" * 60)

print("\n[1] Distribución de predicciones:")
print(f"    Baseline: {dict(Counter(baseline['class']))}")
print(f"    Ridge:    {dict(Counter(ridge['class']))}")

# Diferencias
diff_mask = baseline['class'] != ridge['class']
n_diff = diff_mask.sum()
print(f"\n[2] Predicciones diferentes: {n_diff} de {len(baseline)} ({100*n_diff/len(baseline):.1f}%)")

if n_diff > 0:
    print("\n[3] IDs con predicciones diferentes:")
    diff_df = pd.DataFrame({
        'Id': baseline.loc[diff_mask, 'Id'],
        'baseline': baseline.loc[diff_mask, 'class'],
        'ridge': ridge.loc[diff_mask, 'class']
    })
    print(diff_df.to_string(index=False))
    
    # Analizar el patrón
    baseline_0_ridge_1 = ((baseline['class'] == 0) & (ridge['class'] == 1)).sum()
    baseline_1_ridge_0 = ((baseline['class'] == 1) & (ridge['class'] == 0)).sum()
    print(f"\n[4] Patrón de cambios:")
    print(f"    Baseline=0, Ridge=1 (Ridge predice más botrytis): {baseline_0_ridge_1}")
    print(f"    Baseline=1, Ridge=0 (Ridge predice más control): {baseline_1_ridge_0}")

# Cargar datos de entrenamiento para análisis
from src.preprocessing import load_data, prepare_features, scale_features

DATA_PATH = PROJECT_ROOT / 'data'
train_df, test_df = load_data(DATA_PATH / 'train.csv', DATA_PATH / 'test.csv')
X_train, X_test, y_train, feature_cols = prepare_features(train_df, test_df)

print(f"\n[5] Distribución en entrenamiento:")
train_dist = dict(Counter(y_train))
print(f"    Clase 0 (control): {train_dist[0]} ({100*train_dist[0]/len(y_train):.1f}%)")
print(f"    Clase 1 (botrytis): {train_dist[1]} ({100*train_dist[1]/len(y_train):.1f}%)")

print(f"\n[6] Distribución predicha en test:")
baseline_dist = dict(Counter(baseline['class']))
ridge_dist = dict(Counter(ridge['class']))
print(f"    Baseline - Clase 0: {baseline_dist.get(0, 0)} ({100*baseline_dist.get(0, 0)/len(baseline):.1f}%)")
print(f"    Baseline - Clase 1: {baseline_dist.get(1, 0)} ({100*baseline_dist.get(1, 0)/len(baseline):.1f}%)")
print(f"    Ridge    - Clase 0: {ridge_dist.get(0, 0)} ({100*ridge_dist.get(0, 0)/len(ridge):.1f}%)")
print(f"    Ridge    - Clase 1: {ridge_dist.get(1, 0)} ({100*ridge_dist.get(1, 0)/len(ridge):.1f}%)")

# Análisis de probabilidades
print("\n[7] Análisis de confianza del modelo...")
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Baseline LR con probabilidades
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_proba = lr.predict_proba(X_test_scaled)

print(f"    LR - Probabilidad media clase 1: {lr_proba[:, 1].mean():.4f}")
print(f"    LR - Probabilidad min/max clase 1: {lr_proba[:, 1].min():.4f} / {lr_proba[:, 1].max():.4f}")

# Casos cercanos al umbral
near_threshold = np.abs(lr_proba[:, 1] - 0.5) < 0.1
print(f"    LR - Casos cercanos al umbral (0.4-0.6): {near_threshold.sum()}")

# Ridge no tiene predict_proba directamente, usar decision_function
ridge_model = RidgeClassifier(alpha=1.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)
ridge_decision = ridge_model.decision_function(X_test_scaled)

print(f"\n    Ridge - Decision function media: {ridge_decision.mean():.4f}")
print(f"    Ridge - Decision function min/max: {ridge_decision.min():.4f} / {ridge_decision.max():.4f}")

# Casos cercanos al umbral (0)
near_zero = np.abs(ridge_decision) < 0.5
print(f"    Ridge - Casos cercanos al umbral: {near_zero.sum()}")

print("\n" + "=" * 60)
print("CONCLUSIONES")
print("=" * 60)
print("""
El baseline LR y Ridge tienen predicciones muy similares pero difieren
en algunos casos límite. El F1-score en Kaggle depende de:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)  
- F1 = 2 * (Precision * Recall) / (Precision + Recall)

Si Ridge predice más clase 1 (botrytis) pero algunos son FP,
esto reduce la Precision y por tanto el F1.

RECOMENDACIÓN: Probar el baseline original con pequeñas variaciones
o ajustar el umbral de decisión para ser más conservador.
""")
