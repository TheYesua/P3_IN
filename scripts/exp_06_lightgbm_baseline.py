"""
Experimento 06: LightGBM Baseline (sin reducción de dimensionalidad)
Práctica 3 - Competición Kaggle

Hipótesis: LightGBM puede ser más rápido que XGBoost con
rendimiento similar o superior en datasets pequeños-medianos.
"""

import sys
from pathlib import Path

# Añadir src al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    print("ERROR: LightGBM no está instalado. Ejecuta: pip install lightgbm")
    sys.exit(1)

from src.preprocessing import (
    load_data, prepare_features, scale_features,
    FLUORESCENCE_COLS, get_spectral_columns
)
from src.models import evaluate_model_cv, train_and_predict
from src.utils import create_submission_from_test, print_class_distribution

# Suprimir warnings de LightGBM
warnings.filterwarnings('ignore', category=UserWarning)

# Configuración
DATA_PATH = PROJECT_ROOT / 'data'
SUBMISSIONS_PATH = PROJECT_ROOT / 'submissions'
RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("EXPERIMENTO 06: LIGHTGBM BASELINE (304 features)")
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
    
    # 4. Configurar y evaluar LightGBM
    print("\n[4] Evaluando LightGBM (5-fold CV)...")
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
        force_col_wise=True
    )
    
    mean_score, std_score, scores = evaluate_model_cv(model, X_train_scaled, y_train, cv=5)
    print(f"    F1-Score: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"    Scores por fold: {[f'{s:.4f}' for s in scores]}")
    
    # 5. Entrenar modelo final y predecir
    print("\n[5] Entrenando modelo final...")
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
        force_col_wise=True
    )
    predictions = train_and_predict(model, X_train_scaled, y_train, X_test_scaled)
    
    # Estadísticas de predicciones
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"    Predicciones: {dict(zip(unique, counts))}")
    
    # Feature importance (top 20)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n[6] Top 20 features más importantes:")
    for i, row in feature_importance.head(20).iterrows():
        print(f"      - {row['feature']}: {row['importance']:.0f}")
    
    # 7. Crear submission
    print("\n[7] Generando submission...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    submission_filename = SUBMISSIONS_PATH / f'submission_06_lightgbm_baseline_{timestamp}.csv'
    
    submission = create_submission_from_test(predictions, submission_filename)
    
    # 8. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL EXPERIMENTO")
    print("=" * 60)
    print(f"Preprocesado: StandardScaler (sin reducción dimensionalidad)")
    print(f"Modelo: LightGBM (n_estimators=100, max_depth=6, lr=0.1)")
    print(f"Score CV (F1): {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"Submission: {submission_filename.name}")
    print("\n¡Recuerda subir el archivo a Kaggle y actualizar REGISTRO_EXPERIMENTOS.md!")
    print("=" * 60)
    
    return {
        'exp_num': 6,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'preprocessing': 'StandardScaler',
        'model': 'LightGBM',
        'model_params': 'n_estimators=100, max_depth=6, lr=0.1',
        'score_cv': mean_score,
        'score_std': std_score,
        'submission_file': submission_filename.name,
        'top_features': feature_importance.head(20)['feature'].tolist()
    }

if __name__ == '__main__':
    results = main()
