"""
Script para generar las gráficas de la Fase 2 de experimentos.
Genera las mismas gráficas que el notebook 03_Experimentos_Fase2.ipynb
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score

from src.preprocessing import load_data, prepare_features, scale_features
from src.models import evaluate_model_cv

# Paths
DATA_PATH = PROJECT_ROOT / 'data'
GRAFICAS_PATH = PROJECT_ROOT / 'docs' / 'graficas'
RANDOM_STATE = 42

print("=" * 60)
print("GENERANDO GRÁFICAS FASE 2")
print("=" * 60)

# 1. Cargar datos
print("\n[1] Cargando datos...")
train_df, test_df = load_data(DATA_PATH / 'train.csv', DATA_PATH / 'test.csv')
X_train, X_test, y_train, feature_cols = prepare_features(train_df, test_df)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
print(f"    Muestras: {X_train.shape[0]}, Features: {X_train.shape[1]}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 2. Evaluar modelos Fase 2
print("\n[2] Evaluando modelos Fase 2...")
results_fase2 = []

# Exp07: LR optimizado
lr_opt = LogisticRegression(C=5.0, solver='liblinear', penalty='l2', max_iter=2000, random_state=RANDOM_STATE)
m, s, _ = evaluate_model_cv(lr_opt, X_train_scaled, y_train)
results_fase2.append({'Experimento': 'Exp07: LR Optimizado', 'F1_Mean': m, 'F1_Std': s})
print(f"    Exp07 - LR Optimizado: F1 = {m:.4f}")

# Exp08: Voting
estimators_voting = [
    ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
    ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)),
    ('nb', GaussianNB()),
]
voting = VotingClassifier(estimators=estimators_voting, voting='hard')
m, s, _ = evaluate_model_cv(voting, X_train_scaled, y_train)
results_fase2.append({'Experimento': 'Exp08: Voting Hard', 'F1_Mean': m, 'F1_Std': s})
print(f"    Exp08 - Voting Hard: F1 = {m:.4f}")

# Exp09: Stacking
estimators_stack = [
    ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
    ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('nb', GaussianNB()),
]
stacking = StackingClassifier(estimators=estimators_stack, final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), cv=5)
m, s, _ = evaluate_model_cv(stacking, X_train_scaled, y_train)
results_fase2.append({'Experimento': 'Exp09: Stacking', 'F1_Mean': m, 'F1_Std': s})
print(f"    Exp09 - Stacking: F1 = {m:.4f}")

# Exp10: SVM linear
svm_opt = SVC(C=1.0, kernel='linear', random_state=RANDOM_STATE)
m, s, _ = evaluate_model_cv(svm_opt, X_train_scaled, y_train)
results_fase2.append({'Experimento': 'Exp10: SVM Linear', 'F1_Mean': m, 'F1_Std': s})
print(f"    Exp10 - SVM Linear: F1 = {m:.4f}")

# Exp15: Ridge
ridge = RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE)
m, s, _ = evaluate_model_cv(ridge, X_train_scaled, y_train)
results_fase2.append({'Experimento': 'Exp15: Ridge (α=1)', 'F1_Mean': m, 'F1_Std': s})
print(f"    Exp15 - Ridge: F1 = {m:.4f}")

results_fase2_df = pd.DataFrame(results_fase2)

# 3. Gráfica: Comparación Fase 2
print("\n[3] Generando gráfica: comparacion_modelos_fase2.png...")
fig, ax = plt.subplots(figsize=(12, 6))

results_sorted = results_fase2_df.sort_values('F1_Mean', ascending=True)
colors = ['#e74c3c' if x < 0.78 else '#f39c12' if x < 0.84 else '#27ae60' for x in results_sorted['F1_Mean']]

bars = ax.barh(results_sorted['Experimento'], results_sorted['F1_Mean'], color=colors, edgecolor='black', linewidth=0.5)
ax.errorbar(results_sorted['F1_Mean'], results_sorted['Experimento'], xerr=results_sorted['F1_Std'], fmt='none', color='black', capsize=3)

for bar, score in zip(bars, results_sorted['F1_Mean']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.4f}', va='center', fontsize=10)

ax.axvline(x=0.8278, color='blue', linestyle='--', linewidth=2, label='Baseline Fase 1: 0.8278')
ax.set_xlabel('F1-Score (Validación Cruzada 5-fold)')
ax.set_title('Comparación de Modelos - Fase 2 (Optimización)')
ax.set_xlim(0.65, 0.95)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(GRAFICAS_PATH / 'comparacion_modelos_fase2.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Guardada")

# 4. Gráfica: Comparación completa
print("\n[4] Generando gráfica: comparacion_modelos_completa.png...")
results_fase1 = [
    {'Experimento': 'Exp01: Baseline LR', 'F1_Mean': 0.8278, 'F1_Std': 0.0232, 'Fase': 'Fase 1'},
    {'Experimento': 'Exp02: PCA(3) + LR', 'F1_Mean': 0.7368, 'F1_Std': 0.0020, 'Fase': 'Fase 1'},
    {'Experimento': 'Exp03: PCA(7) + XGB', 'F1_Mean': 0.7268, 'F1_Std': 0.0508, 'Fase': 'Fase 1'},
    {'Experimento': 'Exp04: SKB(50) + RF', 'F1_Mean': 0.6802, 'F1_Std': 0.0544, 'Fase': 'Fase 1'},
    {'Experimento': 'Exp05: XGBoost', 'F1_Mean': 0.7472, 'F1_Std': 0.0520, 'Fase': 'Fase 1'},
    {'Experimento': 'Exp06: LightGBM', 'F1_Mean': 0.7232, 'F1_Std': 0.0556, 'Fase': 'Fase 1'},
]

for r in results_fase2:
    r['Fase'] = 'Fase 2'

all_results = pd.DataFrame(results_fase1 + results_fase2)
all_results_sorted = all_results.sort_values('F1_Mean', ascending=True)

fig, ax = plt.subplots(figsize=(14, 8))

colors = ['#3498db' if fase == 'Fase 1' else '#9b59b6' for fase in all_results_sorted['Fase']]

bars = ax.barh(all_results_sorted['Experimento'], all_results_sorted['F1_Mean'], color=colors, edgecolor='black', linewidth=0.5)
ax.errorbar(all_results_sorted['F1_Mean'], all_results_sorted['Experimento'], xerr=all_results_sorted['F1_Std'], fmt='none', color='black', capsize=3)

for bar, score in zip(bars, all_results_sorted['F1_Mean']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{score:.4f}', va='center', fontsize=9)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='Fase 1 (Exploración)'),
    Patch(facecolor='#9b59b6', edgecolor='black', label='Fase 2 (Optimización)')
]
ax.legend(handles=legend_elements, loc='lower right')

ax.set_xlabel('F1-Score (Validación Cruzada 5-fold)')
ax.set_title('Comparación Completa de Todos los Experimentos')
ax.set_xlim(0.6, 0.95)

plt.tight_layout()
plt.savefig(GRAFICAS_PATH / 'comparacion_modelos_completa.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Guardada")

# 5. Matriz de confusión Ridge
print("\n[5] Generando gráfica: matriz_confusion_ridge.png...")
best_model = RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

y_pred = cross_val_predict(best_model, X_train_scaled, y_train, cv=skf)
cm = confusion_matrix(y_train, y_pred)
f1 = f1_score(y_train, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=['Control (sana)', 'Botrytis (infectada)'],
            yticklabels=['Control (sana)', 'Botrytis (infectada)'],
            annot_kws={'size': 16})

ax.set_title(f'Matriz de Confusión - Ridge Classifier\nF1-Score = {f1:.4f}', fontsize=14)
ax.set_xlabel('Predicción', fontsize=12)
ax.set_ylabel('Valor Real', fontsize=12)

plt.tight_layout()
plt.savefig(GRAFICAS_PATH / 'matriz_confusion_ridge.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Guardada")

# 6. Análisis de regularización
print("\n[6] Generando gráfica: analisis_regularizacion.png...")
alphas_ridge = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
Cs_lr = [100, 10, 2, 1, 0.5, 0.2, 0.1]

ridge_scores = []
lr_scores = []

for alpha, C in zip(alphas_ridge, Cs_lr):
    ridge = RidgeClassifier(alpha=alpha, random_state=RANDOM_STATE)
    scores = cross_val_score(ridge, X_train_scaled, y_train, cv=cv, scoring='f1')
    ridge_scores.append({'alpha': alpha, 'F1_Mean': scores.mean(), 'F1_Std': scores.std()})
    
    lr = LogisticRegression(C=C, solver='liblinear', max_iter=2000, random_state=RANDOM_STATE)
    scores = cross_val_score(lr, X_train_scaled, y_train, cv=cv, scoring='f1')
    lr_scores.append({'C': C, 'F1_Mean': scores.mean(), 'F1_Std': scores.std()})

ridge_df = pd.DataFrame(ridge_scores)
lr_df = pd.DataFrame(lr_scores)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.errorbar(ridge_df['alpha'], ridge_df['F1_Mean'], yerr=ridge_df['F1_Std'], 
             fmt='o-', color='#27ae60', linewidth=2, markersize=8, capsize=4)
ax1.set_xlabel('Alpha (regularización)', fontsize=12)
ax1.set_ylabel('F1-Score', fontsize=12)
ax1.set_title('Ridge Classifier: Efecto de Alpha', fontsize=14)
ax1.set_xscale('log')
ax1.axhline(y=ridge_df['F1_Mean'].max(), color='red', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3)

ax2.errorbar(lr_df['C'], lr_df['F1_Mean'], yerr=lr_df['F1_Std'],
             fmt='s-', color='#3498db', linewidth=2, markersize=8, capsize=4)
ax2.set_xlabel('C (inverso de regularización)', fontsize=12)
ax2.set_ylabel('F1-Score', fontsize=12)
ax2.set_title('Logistic Regression: Efecto de C', fontsize=14)
ax2.set_xscale('log')
ax2.axhline(y=lr_df['F1_Mean'].max(), color='red', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(GRAFICAS_PATH / 'analisis_regularizacion.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Guardada")

# Resumen
print("\n" + "=" * 60)
print("GRÁFICAS GENERADAS")
print("=" * 60)
for f in ['comparacion_modelos_fase2.png', 'comparacion_modelos_completa.png', 
          'matriz_confusion_ridge.png', 'analisis_regularizacion.png']:
    path = GRAFICAS_PATH / f
    if path.exists():
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f}")
print("=" * 60)
