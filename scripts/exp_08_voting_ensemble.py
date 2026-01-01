"""
Experimento 08: VotingClassifier Ensemble
Práctica 3 - Competición Kaggle

Objetivo: Combinar múltiples modelos que generalizan bien para mejorar
el rendimiento mediante votación.
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from src.preprocessing import (
    load_data, prepare_features, scale_features,
    FLUORESCENCE_COLS, get_spectral_columns
)
from src.models import evaluate_model_cv, train_and_predict
from src.utils import create_submission_from_test, print_class_distribution

# Configuración
DATA_PATH = PROJECT_ROOT / 'data'
SUBMISSIONS_PATH = PROJECT_ROOT / 'submissions'
RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("EXPERIMENTO 08: VOTING CLASSIFIER ENSEMBLE")
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
    
    # Distribución de clases
    print_class_distribution(y_train)
    
    # 3. Escalar features
    print("\n[3] Escalando features (StandardScaler)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, method='standard')
    
    # 4. Definir modelos base
    print("\n[4] Configurando modelos base...")
    
    # Modelos que mostraron buena generalización
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)),
        ('nb', GaussianNB()),
    ]
    
    # 5. Voting Classifier - Hard voting
    print("\n[5] Evaluando VotingClassifier (hard voting)...")
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    mean_hard, std_hard, scores_hard = evaluate_model_cv(voting_hard, X_train_scaled, y_train, cv=5)
    print(f"    Hard Voting F1: {mean_hard:.4f} (+/- {std_hard:.4f})")
    
    # 6. Voting Classifier - Soft voting
    print("\n[6] Evaluando VotingClassifier (soft voting)...")
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    mean_soft, std_soft, scores_soft = evaluate_model_cv(voting_soft, X_train_scaled, y_train, cv=5)
    print(f"    Soft Voting F1: {mean_soft:.4f} (+/- {std_soft:.4f})")
    
    # 7. Seleccionar mejor voting
    if mean_soft > mean_hard:
        best_voting = voting_soft
        best_score = mean_soft
        best_std = std_soft
        voting_type = 'soft'
    else:
        best_voting = voting_hard
        best_score = mean_hard
        best_std = std_hard
        voting_type = 'hard'
    
    print(f"\n    Mejor voting: {voting_type} (F1 = {best_score:.4f})")
    
    # 8. Entrenar modelo final y predecir
    print("\n[8] Entrenando modelo final...")
    best_voting.fit(X_train_scaled, y_train)
    predictions = best_voting.predict(X_test_scaled)
    
    # Estadísticas de predicciones
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"    Predicciones: {dict(zip(unique, counts))}")
    
    # 9. Crear submission
    print("\n[9] Generando submission...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    submission_filename = SUBMISSIONS_PATH / f'submission_08_voting_{voting_type}_{timestamp}.csv'
    
    submission = create_submission_from_test(predictions, submission_filename)
    
    # 10. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL EXPERIMENTO")
    print("=" * 60)
    print(f"Preprocesado: StandardScaler")
    print(f"Modelo: VotingClassifier ({voting_type})")
    print(f"Modelos base: LR, SVM, RF, NB")
    print(f"Score CV (F1): {best_score:.4f} (+/- {best_std:.4f})")
    print(f"Submission: {submission_filename.name}")
    print("=" * 60)
    
    return {
        'exp_num': 8,
        'voting_type': voting_type,
        'score_cv': best_score,
        'submission_file': submission_filename.name
    }

if __name__ == '__main__':
    results = main()
