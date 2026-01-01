"""
Script para verificar los valores correctos de F1-CV para todos los modelos baseline.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.preprocessing import load_data, prepare_features, scale_features

DATA_PATH = PROJECT_ROOT / 'data'
RANDOM_STATE = 42

print("=" * 60)
print("VERIFICACIÓN DE VALORES BASELINE")
print("=" * 60)

# Cargar datos
train_df, test_df = load_data(DATA_PATH / 'train.csv', DATA_PATH / 'test.csv')
X_train, X_test, y_train, feature_cols = prepare_features(train_df, test_df)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ('SVM (RBF)', SVC(kernel='rbf', random_state=RANDOM_STATE)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ('KNN (k=5)', KNeighborsClassifier(n_neighbors=5)),
]

print("\nModelos Baseline con StandardScaler:")
print("-" * 60)
print(f"{'Modelo':<25} {'F1-CV Mean':<12} {'F1-CV Std':<12}")
print("-" * 60)

for name, model in models:
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
    print(f"{name:<25} {scores.mean():.4f}       {scores.std():.4f}")

print("-" * 60)
