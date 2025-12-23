# Módulo src - Código Fuente Reutilizable

Este directorio contiene el código Python modular y reutilizable para la práctica.

---

## Arquitectura

```
src/
├── __init__.py         # Inicialización del módulo
├── preprocessing.py    # Funciones de carga y preprocesado de datos
├── models.py          # Definición y evaluación de modelos
├── utils.py           # Utilidades generales (submissions, logging)
└── README.md          # Este archivo
```

---

## preprocessing.py

### Constantes
| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `METADATA_COLS` | `['exp', 'dpi', 'leaf', 'spot']` | Columnas que NO deben usarse como features |
| `TARGET_COL` | `'class'` | Nombre de la columna objetivo |
| `FLUORESCENCE_COLS` | `['F440', 'F520', 'F680', 'F740']` | Variables de fluorescencia |

### Funciones Principales

#### `load_data(train_path, test_path)`
Carga los datasets de entrenamiento y test, aplicando limpieza automática de valores numéricos con formato incorrecto (ej: '232 .25' → 232.25).

```python
train_df, test_df = load_data('data/train.csv', 'data/test.csv')
```

#### `clean_numeric_columns(df)`
Limpia columnas numéricas que contienen espacios u otros caracteres no válidos.

#### `get_spectral_columns(df)`
Retorna lista de columnas espectrales (las que empiezan con 'w').

#### `get_feature_columns(df)`
Retorna lista de columnas válidas para usar como features (excluye metadatos y target).

#### `encode_target(y)` / `decode_target(y)`
Codifica/decodifica la variable objetivo: control → 0, botrytis → 1.

#### `prepare_features(train_df, test_df)`
Prepara X_train, X_test, y_train y lista de feature_cols.

#### `scale_features(X_train, X_test, method='standard')`
Escala features usando StandardScaler o MinMaxScaler.

#### `apply_pca(X_train, X_test, n_components=0.95)`
Aplica PCA para reducción de dimensionalidad.

#### `select_k_best(X_train, X_test, y_train, k=50)`
Selecciona las k mejores features usando ANOVA F-value.

---

## models.py

### Funciones Principales

#### `get_baseline_models()`
Retorna diccionario con 5 modelos baseline preconfigurados:
- LogisticRegression
- RandomForest
- SVM
- KNN
- GradientBoosting

#### `evaluate_model_cv(model, X, y, cv=5, scoring='f1')`
Evalúa modelo con validación cruzada estratificada. Retorna (mean, std, scores).

#### `compare_models(models, X, y, cv=5)`
Compara múltiples modelos e imprime resultados ordenados por F1-score.

#### `train_and_predict(model, X_train, y_train, X_test)`
Entrena modelo y genera predicciones.

#### `get_classification_report(y_true, y_pred)`
Genera reporte de clasificación completo con matriz de confusión.

---

## utils.py

### Funciones Principales

#### `create_submission(test_ids, predictions, filename)`
Crea archivo CSV con formato de submission de Kaggle (columnas: Id, class).

#### `create_submission_from_test(predictions, filename, n_samples=None)`
Crea submission asumiendo IDs de 1 a n_samples. Si n_samples es None, usa len(predictions).

#### `log_experiment(exp_num, score_train, score_kaggle, preprocessing, algorithm, params, observations="")`
Genera entrada formateada para el registro de experimentos.

#### `print_class_distribution(y, title="Distribución de clases")`
Imprime distribución de clases con porcentajes.

---

## Uso Típico

```python
import sys
sys.path.append('.')  # Si se ejecuta desde la raíz del proyecto

from src.preprocessing import load_data, prepare_features, scale_features
from src.models import get_baseline_models, compare_models, train_and_predict
from src.utils import create_submission_from_test

# 1. Cargar datos
train_df, test_df = load_data('data/train.csv', 'data/test.csv')

# 2. Preparar features
X_train, X_test, y_train, feature_cols = prepare_features(train_df, test_df)

# 3. Escalar
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# 4. Comparar modelos
models = get_baseline_models()
results = compare_models(models, X_train_scaled, y_train)

# 5. Entrenar mejor modelo y predecir
best_model = models['LogisticRegression']
predictions = train_and_predict(best_model, X_train_scaled, y_train, X_test_scaled)

# 6. Crear submission
create_submission_from_test(predictions, 'submissions/my_submission.csv')
```

---

*Última actualización: 23/12/2024*
