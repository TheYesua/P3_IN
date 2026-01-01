"""
Experimento 13: Voting de LR optimizado + SVM linear
Práctica 3 - Competición Kaggle

Objetivo: Combinar los dos mejores modelos lineales (LR y SVM linear)
que mostraron mejor rendimiento individual.
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
from sklearn.ensemble import VotingClassifier

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
    print("EXPERIMENTO 13: VOTING LR + SVM LINEAR")
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
    
    # 4. Definir modelos optimizados
    print("\n[4] Configurando modelos...")
    
    # LR optimizado (del exp 07)
    lr = LogisticRegression(
        C=5.0, 
        solver='liblinear', 
        penalty='l2',
        max_iter=2000, 
        random_state=RANDOM_STATE
    )
    
    # SVM linear (del exp 10)
    svm = SVC(
        C=1.0, 
        kernel='linear', 
        probability=True,
        random_state=RANDOM_STATE
    )
    
    # 5. Evaluar modelos individuales
    print("\n[5] Evaluando modelos individuales...")
    mean_lr, std_lr, _ = evaluate_model_cv(lr, X_train_scaled, y_train, cv=5)
    print(f"    LR optimizado: F1 = {mean_lr:.4f} (+/- {std_lr:.4f})")
    
    mean_svm, std_svm, _ = evaluate_model_cv(svm, X_train_scaled, y_train, cv=5)
    print(f"    SVM linear: F1 = {mean_svm:.4f} (+/- {std_svm:.4f})")
    
    # 6. Voting Classifier
    print("\n[6] Evaluando VotingClassifier...")
    
    # Soft voting (usa probabilidades)
    voting_soft = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm)],
        voting='soft',
        weights=[1, 1]  # Igual peso
    )
    mean_soft, std_soft, scores_soft = evaluate_model_cv(voting_soft, X_train_scaled, y_train, cv=5)
    print(f"    Soft Voting (1:1): F1 = {mean_soft:.4f} (+/- {std_soft:.4f})")
    
    # Probar con diferentes pesos
    voting_lr_heavy = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm)],
        voting='soft',
        weights=[2, 1]  # Más peso a LR
    )
    mean_lr_heavy, std_lr_heavy, _ = evaluate_model_cv(voting_lr_heavy, X_train_scaled, y_train, cv=5)
    print(f"    Soft Voting (2:1 LR): F1 = {mean_lr_heavy:.4f} (+/- {std_lr_heavy:.4f})")
    
    # Hard voting
    voting_hard = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm)],
        voting='hard'
    )
    mean_hard, std_hard, _ = evaluate_model_cv(voting_hard, X_train_scaled, y_train, cv=5)
    print(f"    Hard Voting: F1 = {mean_hard:.4f} (+/- {std_hard:.4f})")
    
    # 7. Seleccionar mejor configuración
    results = [
        ('LR solo', mean_lr, std_lr, lr),
        ('SVM solo', mean_svm, std_svm, svm),
        ('Soft 1:1', mean_soft, std_soft, voting_soft),
        ('Soft 2:1', mean_lr_heavy, std_lr_heavy, voting_lr_heavy),
        ('Hard', mean_hard, std_hard, voting_hard),
    ]
    
    best_name, best_score, best_std, best_model = max(results, key=lambda x: x[1])
    print(f"\n    Mejor configuración: {best_name} (F1 = {best_score:.4f})")
    
    # 8. Entrenar y predecir
    print("\n[8] Entrenando modelo final...")
    best_model.fit(X_train_scaled, y_train)
    predictions = best_model.predict(X_test_scaled)
    
    # Estadísticas
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"    Predicciones: {dict(zip(unique, counts))}")
    
    # 9. Crear submission
    print("\n[9] Generando submission...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    submission_filename = SUBMISSIONS_PATH / f'submission_13_lr_svm_voting_{timestamp}.csv'
    
    submission = create_submission_from_test(predictions, submission_filename)
    
    # 10. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL EXPERIMENTO")
    print("=" * 60)
    print(f"Preprocesado: StandardScaler")
    print(f"Mejor modelo: {best_name}")
    print(f"Score CV (F1): {best_score:.4f} (+/- {best_std:.4f})")
    print(f"Submission: {submission_filename.name}")
    print("=" * 60)
    
    return {
        'exp_num': 13,
        'best_config': best_name,
        'score_cv': best_score,
        'submission_file': submission_filename.name
    }

if __name__ == '__main__':
    results = main()
