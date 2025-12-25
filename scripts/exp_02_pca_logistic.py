"""
Experimento 02: PCA (3 componentes) + Logistic Regression
Práctica 3 - Competición Kaggle

Hipótesis: PCA con 3 componentes (95% varianza) puede mejorar
la generalización al reducir el ruido y el overfitting.
"""

import sys
from pathlib import Path

# Añadir src al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    load_data, prepare_features, scale_features, apply_pca,
    FLUORESCENCE_COLS, get_spectral_columns
)
from src.models import evaluate_model_cv, train_and_predict
from src.utils import create_submission_from_test, print_class_distribution

# Configuración
DATA_PATH = PROJECT_ROOT / 'data'
SUBMISSIONS_PATH = PROJECT_ROOT / 'submissions'
RANDOM_STATE = 42
N_COMPONENTS = 3  # 95% de varianza según EDA

def main():
    print("=" * 60)
    print("EXPERIMENTO 02: PCA (3 componentes) + LOGISTIC REGRESSION")
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
    
    # 4. Aplicar PCA
    print(f"\n[4] Aplicando PCA con {N_COMPONENTS} componentes...")
    X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled, n_components=N_COMPONENTS)
    print(f"    Dimensiones reducidas: {X_train_pca.shape[1]}")
    print(f"    Varianza explicada total: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"    Varianza por componente: {[f'{v*100:.1f}%' for v in pca.explained_variance_ratio_]}")
    
    # 5. Evaluar modelo con CV
    print("\n[5] Evaluando Logistic Regression (5-fold CV)...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    mean_score, std_score, scores = evaluate_model_cv(model, X_train_pca, y_train, cv=5)
    print(f"    F1-Score: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"    Scores por fold: {[f'{s:.4f}' for s in scores]}")
    
    # 6. Entrenar modelo final y predecir
    print("\n[6] Entrenando modelo final...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    predictions = train_and_predict(model, X_train_pca, y_train, X_test_pca)
    
    # Estadísticas de predicciones
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"    Predicciones: {dict(zip(unique, counts))}")
    
    # 7. Crear submission
    print("\n[7] Generando submission...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    submission_filename = SUBMISSIONS_PATH / f'submission_02_pca{N_COMPONENTS}_logistic_{timestamp}.csv'
    
    submission = create_submission_from_test(predictions, submission_filename)
    
    # 8. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL EXPERIMENTO")
    print("=" * 60)
    print(f"Preprocesado: StandardScaler + PCA({N_COMPONENTS})")
    print(f"Modelo: Logistic Regression")
    print(f"Score CV (F1): {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"Varianza explicada: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"Submission: {submission_filename.name}")
    print("\n¡Recuerda subir el archivo a Kaggle y actualizar REGISTRO_EXPERIMENTOS.md!")
    print("=" * 60)
    
    return {
        'exp_num': 2,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'preprocessing': f'StandardScaler + PCA({N_COMPONENTS})',
        'model': 'LogisticRegression',
        'score_cv': mean_score,
        'score_std': std_score,
        'variance_explained': pca.explained_variance_ratio_.sum(),
        'submission_file': submission_filename.name
    }

if __name__ == '__main__':
    results = main()
