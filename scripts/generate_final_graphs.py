"""
Script para generar gráficas finales para la documentación.
Incluye todos los experimentos realizados (Fases 1, 2 y 3).
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

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

GRAFICAS_PATH = PROJECT_ROOT / 'docs' / 'graficas'

print("=" * 60)
print("GENERANDO GRÁFICAS FINALES")
print("=" * 60)

# Datos de todos los experimentos
experimentos = [
    # Fase 1: Baseline y Reducción Dimensional
    {'Exp': 'Exp01', 'Nombre': 'Baseline LR', 'F1_CV': 0.8278, 'F1_Kaggle': 0.84782, 'Fase': 'Fase 1'},
    {'Exp': 'Exp02', 'Nombre': 'PCA(3) + LR', 'F1_CV': 0.7368, 'F1_Kaggle': 0.74561, 'Fase': 'Fase 1'},
    {'Exp': 'Exp03', 'Nombre': 'PCA(7) + XGBoost', 'F1_CV': 0.7268, 'F1_Kaggle': 0.69364, 'Fase': 'Fase 1'},
    {'Exp': 'Exp04', 'Nombre': 'SKB(50) + RF', 'F1_CV': 0.6802, 'F1_Kaggle': 0.59459, 'Fase': 'Fase 1'},
    
    # Fase 2: Modelos Avanzados
    {'Exp': 'Exp05', 'Nombre': 'XGBoost', 'F1_CV': 0.7472, 'F1_Kaggle': 0.70454, 'Fase': 'Fase 2'},
    {'Exp': 'Exp06', 'Nombre': 'LightGBM', 'F1_CV': 0.7232, 'F1_Kaggle': 0.73033, 'Fase': 'Fase 2'},
    
    # Fase 3: Optimización (solo los más relevantes)
    {'Exp': 'Exp07', 'Nombre': 'LR Optimizado (C=5)', 'F1_CV': 0.8484, 'F1_Kaggle': None, 'Fase': 'Fase 3'},
    {'Exp': 'Exp10', 'Nombre': 'SVM Linear', 'F1_CV': 0.8388, 'F1_Kaggle': None, 'Fase': 'Fase 3'},
    {'Exp': 'Exp15', 'Nombre': 'Ridge Classifier', 'F1_CV': 0.8518, 'F1_Kaggle': 0.82485, 'Fase': 'Fase 3'},
    {'Exp': 'Exp18', 'Nombre': 'LR + RFE(150)', 'F1_CV': 0.8479, 'F1_Kaggle': None, 'Fase': 'Fase 3'},
    {'Exp': 'Exp19', 'Nombre': 'Spectral RFE(100)', 'F1_CV': 0.8517, 'F1_Kaggle': 0.84491, 'Fase': 'Fase 3'},
]

df = pd.DataFrame(experimentos)

# 1. Gráfica: Comparación F1-CV de todos los experimentos
print("\n[1] Generando: comparacion_todos_experimentos.png...")
fig, ax = plt.subplots(figsize=(14, 8))

df_sorted = df.sort_values('F1_CV', ascending=True)
colors_fase = {'Fase 1': '#3498db', 'Fase 2': '#e74c3c', 'Fase 3': '#27ae60'}
colors = [colors_fase[f] for f in df_sorted['Fase']]

bars = ax.barh(df_sorted['Nombre'], df_sorted['F1_CV'], color=colors, edgecolor='black', linewidth=0.5)

for bar, score in zip(bars, df_sorted['F1_CV']):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{score:.4f}', va='center', fontsize=9)

# Línea del baseline
ax.axvline(x=0.8278, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (0.8278)')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='Fase 1: Reducción Dim.'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='Fase 2: Modelos Avanzados'),
    Patch(facecolor='#27ae60', edgecolor='black', label='Fase 3: Optimización'),
    plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='Baseline CV')
]
ax.legend(handles=legend_elements, loc='lower right')

ax.set_xlabel('F1-Score (Validación Cruzada 5-fold)')
ax.set_title('Comparación de Todos los Experimentos por Fase')
ax.set_xlim(0.55, 0.92)

plt.tight_layout()
plt.savefig(GRAFICAS_PATH / 'comparacion_todos_experimentos.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Guardada")

# 2. Gráfica: CV vs Kaggle
print("\n[2] Generando: cv_vs_kaggle.png...")
df_kaggle = df[df['F1_Kaggle'].notna()].copy()

fig, ax = plt.subplots(figsize=(10, 8))

colors = [colors_fase[f] for f in df_kaggle['Fase']]
ax.scatter(df_kaggle['F1_CV'], df_kaggle['F1_Kaggle'], c=colors, s=150, edgecolors='black', linewidth=1.5, zorder=5)

# Línea diagonal (perfecta correlación)
lims = [0.55, 0.90]
ax.plot(lims, lims, 'k--', alpha=0.5, label='CV = Kaggle')

# Etiquetas
for _, row in df_kaggle.iterrows():
    ax.annotate(row['Nombre'], (row['F1_CV'], row['F1_Kaggle']), 
                textcoords="offset points", xytext=(5, 5), fontsize=8)

ax.set_xlabel('F1-Score (Validación Cruzada)')
ax.set_ylabel('F1-Score (Kaggle)')
ax.set_title('Correlación entre F1-CV y F1-Kaggle')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(GRAFICAS_PATH / 'cv_vs_kaggle.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Guardada")

# 3. Gráfica: Resumen por fases
print("\n[3] Generando: resumen_fases.png...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (fase, color) in enumerate([('Fase 1', '#3498db'), ('Fase 2', '#e74c3c'), ('Fase 3', '#27ae60')]):
    ax = axes[i]
    df_fase = df[df['Fase'] == fase].sort_values('F1_CV', ascending=True)
    
    bars = ax.barh(df_fase['Nombre'], df_fase['F1_CV'], color=color, edgecolor='black')
    for bar, score in zip(bars, df_fase['F1_CV']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{score:.4f}', va='center', fontsize=9)
    
    ax.axvline(x=0.8278, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('F1-Score CV')
    ax.set_title(fase)
    ax.set_xlim(0.55, 0.92)

plt.tight_layout()
plt.savefig(GRAFICAS_PATH / 'resumen_fases.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ Guardada")

# 4. Tabla resumen en formato texto
print("\n[4] Tabla resumen:")
print("-" * 80)
print(f"{'Exp':<6} {'Nombre':<25} {'F1-CV':<10} {'F1-Kaggle':<12} {'Fase':<10}")
print("-" * 80)
for _, row in df.sort_values('F1_Kaggle', ascending=False, na_position='last').iterrows():
    kaggle = f"{row['F1_Kaggle']:.5f}" if pd.notna(row['F1_Kaggle']) else "N/A"
    print(f"{row['Exp']:<6} {row['Nombre']:<25} {row['F1_CV']:.4f}     {kaggle:<12} {row['Fase']:<10}")
print("-" * 80)

print("\n" + "=" * 60)
print("GRÁFICAS GENERADAS")
print("=" * 60)
for f in ['comparacion_todos_experimentos.png', 'cv_vs_kaggle.png', 'resumen_fases.png']:
    print(f"  ✓ {f}")
print("=" * 60)
