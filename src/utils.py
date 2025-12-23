"""
Utilidades generales para la Práctica 3 - Kaggle
"""

import pandas as pd
import numpy as np
import csv
from datetime import datetime


def create_submission(test_ids, predictions, filename):
    """
    Crea un archivo de submission para Kaggle.
    
    Args:
        test_ids: IDs de las muestras de test (1 a N)
        predictions: Predicciones (0 o 1)
        filename: Nombre del archivo de salida
    
    Note:
        El formato replica el sample_submission.csv de Kaggle:
        - Cabecera con comillas: "Id","class"
        - Datos sin comillas
    """
    submission = pd.DataFrame({
        'Id': test_ids,
        'class': predictions
    })
    # quoting=csv.QUOTE_NONNUMERIC pone comillas solo en strings (cabecera)
    submission.to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Submission guardada en: {filename}")
    return submission


def create_submission_from_test(predictions, filename, n_samples=None):
    """
    Crea submission asumiendo IDs de 1 a n_samples.
    Si n_samples es None, usa la longitud de predictions.
    """
    if n_samples is None:
        n_samples = len(predictions)
    test_ids = list(range(1, n_samples + 1))
    return create_submission(test_ids, predictions, filename)


def log_experiment(exp_num, score_train, score_kaggle, preprocessing, 
                   algorithm, params, observations=""):
    """
    Genera una entrada formateada para el registro de experimentos.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    entry = f"""
### Experimento {exp_num:02d}
- **Fecha y hora de subida**: {timestamp}
- **Posición en Kaggle**: [COMPLETAR]
- **Score en entrenamiento (CV)**: {score_train:.4f}
- **Score en Kaggle (test)**: {score_kaggle if score_kaggle else '[COMPLETAR]'}
- **Preprocesado**: {preprocessing}
- **Algoritmo**: {algorithm}
- **Parámetros**: {params}
- **Archivos asociados**:
  - Script: `scripts/exp_{exp_num:02d}_xxx.py`
  - Submission: `submissions/submission_{exp_num:02d}_xxx.csv`
- **Observaciones**: {observations}
"""
    print(entry)
    return entry


def print_class_distribution(y, title="Distribución de clases"):
    """
    Imprime la distribución de clases.
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    print(f"\n{title}")
    print("-" * 40)
    for val, count in zip(unique, counts):
        pct = count / total * 100
        label = "control" if val == 0 else "botrytis"
        print(f"  Clase {val} ({label}): {count} ({pct:.1f}%)")
    print(f"  Total: {total}")
    print("-" * 40)
