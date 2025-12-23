# Registro de Experimentos - P3 Kaggle

## Tabla de Submissions (OBLIGATORIO)

| Nº | Fecha/Hora | Posición | Score Train | Score Kaggle | Preprocesado | Algoritmo | Parámetros | Observaciones |
|----|------------|----------|-------------|--------------|--------------|-----------|------------|---------------|
| 01 | 23/12/2024 18:56 | [PENDIENTE] | 0.9388 | [PENDIENTE] | StandardScaler | LogisticRegression | max_iter=1000 | Mejor modelo en CV |
| 02 | | | | | | | | |
| 03 | | | | | | | | |
| 04 | | | | | | | | |
| 05 | | | | | | | | |
| 06 | | | | | | | | |
| 07 | | | | | | | | |
| 08 | | | | | | | | |
| 09 | | | | | | | | |
| 10 | | | | | | | | |

---

## Detalle de Experimentos

### Experimento 01 - Baseline con Logistic Regression
- **Fecha y hora de subida**: 23/12/2024 18:56
- **Posición en Kaggle**: [PENDIENTE - Subir a Kaggle]
- **Score en entrenamiento (CV)**: 0.9388 (± 0.0254)
- **Score en Kaggle (test)**: [PENDIENTE - Subir a Kaggle]
- **Preprocesado**:
  - Normalización: StandardScaler
  - Reducción dimensionalidad: Ninguna (todas las features)
  - Balanceo de clases: No aplicado
  - Limpieza de datos: Eliminación de espacios en valores numéricos
- **Algoritmo**: Logistic Regression
- **Parámetros**:
```python
LogisticRegression(max_iter=1000, random_state=42)
```
- **Archivos asociados**:
  - Script: `scripts/exp_01_baseline.py`
  - Submission: `submissions/submission_01_baseline_logisticregression_20251223_1856.csv`
- **Observaciones**:
  - Mejor modelo en validación cruzada (5-fold estratificado)
  - Comparado con: RandomForest (0.9266), SVM (0.9326), KNN (0.8943), GradientBoosting (0.9203)
  - Dataset: 336 muestras train, 143 muestras test
  - Features: 304 (4 fluorescencia + 300 espectrales)

---

### Experimento 02
- **Fecha y hora de subida**: 
- **Posición en Kaggle**: 
- **Score en entrenamiento (CV)**: 
- **Score en Kaggle (test)**: 
- **Preprocesado**:
  - Normalización: 
  - Reducción dimensionalidad: 
  - Balanceo de clases: 
  - Otros: 
- **Algoritmo**: 
- **Parámetros**:
```python
# Configuración del modelo
```
- **Archivos asociados**:
  - Script: `scripts/exp_02_xxx.py`
  - Submission: `submissions/submission_02_xxx.csv`
- **Observaciones**:

---

### Experimento 03
- **Fecha y hora de subida**: 
- **Posición en Kaggle**: 
- **Score en entrenamiento (CV)**: 
- **Score en Kaggle (test)**: 
- **Preprocesado**:
  - Normalización: 
  - Reducción dimensionalidad: 
  - Balanceo de clases: 
  - Otros: 
- **Algoritmo**: 
- **Parámetros**:
```python
# Configuración del modelo
```
- **Archivos asociados**:
  - Script: `scripts/exp_03_xxx.py`
  - Submission: `submissions/submission_03_xxx.csv`
- **Observaciones**:

---

## Notas Adicionales

### Mejores Configuraciones Encontradas
- **Baseline**: LogisticRegression con StandardScaler alcanza F1=0.9388 en CV
- El escalado es importante dado el rango diferente entre fluorescencia (~200-2000) y espectros (~700-8000)

### Ideas por Probar
- [ ] PCA con 3-7 componentes (95-99% varianza)
- [ ] SelectKBest para selección de features
- [ ] XGBoost / LightGBM
- [ ] Optimización de hiperparámetros con Optuna
- [ ] Ensemble (Voting/Stacking)
- [ ] class_weight='balanced' en modelos que lo soporten

### Problemas Encontrados y Soluciones
| Problema | Solución |
|----------|----------|
| Valores con espacios ('232 .25') | Función `clean_numeric_columns()` en preprocessing.py |
| Error en correlaciones del EDA | Usar `load_data()` que aplica limpieza automática |
| Tamaño de submission incorrecto | `create_submission_from_test()` usa len(predictions) dinámicamente |


---

*Añadir más secciones de experimento según sea necesario copiando la plantilla anterior*
