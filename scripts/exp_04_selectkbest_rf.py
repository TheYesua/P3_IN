"""
Experimento 04: SelectKBest (k=50) + Random Forest
Práctica 3 - Competición Kaggle

Hipótesis: Seleccionar las 50 mejores features usando ANOVA F-value
puede mejorar el rendimiento al eliminar features ruidosas.
"""

import sys
from pathlib import Path

# Añadir src al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import (
    load_data, prepare_features, scale_features, select_k_best,
    FLUORESCENCE_COLS, get_spectral_columns
)
from src.models import evaluate_model_cv, train_and_predict
from src.utils import create_submission_from_test, print_class_distribution

# Configuración
DATA_PATH = PROJECT_ROOT / 'data'
SUBMISSIONS_PATH = PROJECT_ROOT / 'submissions'
RANDOM_STATE = 42
K_BEST = 50  # Número de features a seleccionar

def main():
    print("=" * 60)
    print("EXPERIMENTO 04: SELECTKBEST (k=50) + RANDOM FOREST")
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
    print(f"    Features originales: {len(feature_cols)}")
    
    # Distribución de clases
    print_class_distribution(y_train)
    
    # 3. Escalar features
    print("\n[3] Escalando features (StandardScaler)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, method='standard')
    
    # 4. SelectKBest
    print(f"\n[4] Seleccionando {K_BEST} mejores features...")
    X_train_selected, X_test_selected, selector = select_k_best(
        X_train_scaled, X_test_scaled, y_train, k=K_BEST
    )
    print(f"    Features seleccionadas: {X_train_selected.shape[1]}")
    
    # Mostrar top 10 features seleccionadas
    feature_scores = pd.DataFrame({
        'feature': feature_cols,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    print(f"    Top 10 features:")
    for i, row in feature_scores.head(10).iterrows():
        print(f"      - {row['feature']}: {row['score']:.2f}")
    
    # 5. Evaluar Random Forest con CV
    print("\n[5] Evaluando Random Forest (5-fold CV)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    mean_score, std_score, scores = evaluate_model_cv(model, X_train_selected, y_train, cv=5)
    print(f"    F1-Score: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"    Scores por fold: {[f'{s:.4f}' for s in scores]}")
    
    # 6. Entrenar modelo final y predecir
    print("\n[6] Entrenando modelo final...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    predictions = train_and_predict(model, X_train_selected, y_train, X_test_selected)
    
    # Estadísticas de predicciones
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"    Predicciones: {dict(zip(unique, counts))}")
    
    # 7. Crear submission
    print("\n[7] Generando submission...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    submission_filename = SUBMISSIONS_PATH / f'submission_04_selectkbest{K_BEST}_rf_{timestamp}.csv'
    
    submission = create_submission_from_test(predictions, submission_filename)
    
    # 8. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL EXPERIMENTO")
    print("=" * 60)
    print(f"Preprocesado: StandardScaler + SelectKBest(k={K_BEST})")
    print(f"Modelo: Random Forest (n_estimators=100, class_weight='balanced')")
    print(f"Score CV (F1): {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"Submission: {submission_filename.name}")
    print("\n¡Recuerda subir el archivo a Kaggle y actualizar REGISTRO_EXPERIMENTOS.md!")
    print("=" * 60)
    
    return {
        'exp_num': 4,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'preprocessing': f'StandardScaler + SelectKBest(k={K_BEST})',
        'model': 'RandomForest',
        'model_params': "n_estimators=100, class_weight='balanced'",
        'score_cv': mean_score,
        'score_std': std_score,
        'submission_file': submission_filename.name,
        'top_features': feature_scores.head(10)['feature'].tolist()
    }

if __name__ == '__main__':
    results = main()
