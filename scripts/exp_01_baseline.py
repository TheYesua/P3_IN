"""
Experimento 01: Baseline con múltiples modelos
Práctica 3 - Competición Kaggle

Este script entrena varios modelos baseline y genera la primera submission.
"""

import sys
from pathlib import Path

# Añadir src al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from src.preprocessing import (
    load_data, prepare_features, scale_features, 
    FLUORESCENCE_COLS, get_spectral_columns
)
from src.models import get_baseline_models, evaluate_model_cv, compare_models, train_and_predict
from src.utils import create_submission_from_test, print_class_distribution

# Configuración
DATA_PATH = PROJECT_ROOT / 'data'
SUBMISSIONS_PATH = PROJECT_ROOT / 'submissions'
RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("EXPERIMENTO 01: BASELINE")
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
    print(f"    - Fluorescencia: {len(FLUORESCENCE_COLS)}")
    print(f"    - Espectrales: {len(get_spectral_columns(train_df))}")
    
    # Distribución de clases
    print_class_distribution(y_train)
    
    # 3. Escalar features
    print("\n[3] Escalando features (StandardScaler)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, method='standard')
    
    # 4. Comparar modelos baseline
    print("\n[4] Evaluando modelos baseline (5-fold CV)...")
    print("-" * 50)
    models = get_baseline_models()
    results = compare_models(models, X_train_scaled, y_train, cv=5)
    print("-" * 50)
    
    # 5. Seleccionar mejor modelo
    results_df = pd.DataFrame(results)
    best_model_name = results_df.loc[results_df['F1_Mean'].idxmax(), 'Model']
    best_score = results_df['F1_Mean'].max()
    
    print(f"\n[5] Mejor modelo: {best_model_name} (F1 = {best_score:.4f})")
    
    # 6. Entrenar mejor modelo con todos los datos y predecir
    print(f"\n[6] Entrenando {best_model_name} con todos los datos...")
    best_model = models[best_model_name]
    predictions = train_and_predict(best_model, X_train_scaled, y_train, X_test_scaled)
    
    # Estadísticas de predicciones
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"    Predicciones: {dict(zip(unique, counts))}")
    
    # 7. Crear submission
    print("\n[7] Generando submission...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    submission_filename = SUBMISSIONS_PATH / f'submission_01_baseline_{best_model_name.lower()}_{timestamp}.csv'
    
    submission = create_submission_from_test(predictions, submission_filename)
    
    # 8. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL EXPERIMENTO")
    print("=" * 60)
    print(f"Preprocesado: StandardScaler")
    print(f"Modelo: {best_model_name}")
    print(f"Score CV (F1): {best_score:.4f}")
    print(f"Submission: {submission_filename.name}")
    print("\n¡Recuerda subir el archivo a Kaggle y actualizar REGISTRO_EXPERIMENTOS.md!")
    print("=" * 60)
    
    # Guardar resultados para registro
    return {
        'exp_num': 1,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'preprocessing': 'StandardScaler',
        'model': best_model_name,
        'score_cv': best_score,
        'submission_file': submission_filename.name,
        'all_results': results_df
    }

if __name__ == '__main__':
    results = main()
