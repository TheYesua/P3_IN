# Práctica 3 - Competición Kaggle: Clasificación de Hojas de Tomate
## Inteligencia de Negocio - Curso 2025-2026

---

## 📋 Información General

| Campo | Valor |
|-------|-------|
| **Asignatura** | Inteligencia de Negocio |
| **Práctica** | P3 - Competición Kaggle |
| **Fecha límite** | 7 de enero de 2026, 23:30 |
| **Puntuación máxima** | 2.5 puntos |
| **Métrica de evaluación** | F1-score |
| **Nombre en Kaggle** | [TuNombre][TuApellido]_UGR_IN |

---

## 🎯 Objetivo del Problema

Clasificación binaria de hojas de tomate:
- **Clase 0 (control)**: Hojas sanas
- **Clase 1 (botrytis)**: Hojas infectadas

### Datos Disponibles

| Archivo | Descripción | Tamaño |
|---------|-------------|--------|
| `train.csv` | Conjunto de entrenamiento con etiquetas | 337 muestras |
| `test.csv` | Conjunto de test sin etiquetas | 144 muestras |
| `sample_submission.csv` | Formato de envío | 144 filas |

### Variables del Dataset

| Tipo | Columnas | Descripción |
|------|----------|-------------|
| **Metadatos (NO USAR)** | `exp`, `dpi`, `leaf`, `spot` | Información experimental |
| **Fluorescencia** | `F440`, `F520`, `F680`, `F740` | 4 valores de fluorescencia multicolor |
| **Hiperespectral** | `w388.13` a `w1028.28` | ~300 variables espectrales (longitudes de onda) |
| **Target** | `class` | `control` (0) o `botrytis` (1) |

---

## 📁 Estructura del Proyecto

```
P3/
├── data/                      # Datos originales (NO MODIFICAR)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/                 # Jupyter notebooks de experimentación
│   └── 01_EDA.ipynb          # Análisis exploratorio inicial
├── src/                       # Código fuente reutilizable
│   ├── preprocessing.py       # Funciones de preprocesado
│   ├── models.py             # Definición de modelos
│   └── utils.py              # Utilidades generales
├── submissions/               # Archivos CSV enviados a Kaggle
│   └── README.md             # Índice de submissions
├── scripts/                   # Scripts de experimentos
│   └── README.md             # Descripción de cada script
├── docs/                      # Documentación
│   └── capturas/             # Capturas de pantalla de Kaggle
├── PLAN_TRABAJO.md           # Este archivo
├── REGISTRO_EXPERIMENTOS.md  # Tabla de experimentos (OBLIGATORIO)
└── requirements.txt          # Dependencias del proyecto
```

---

## 🗺️ Ruta de Trabajo

### Fase 1: Configuración y EDA (Análisis Exploratorio)
- [x] **1.1** Configurar entorno de trabajo y dependencias ✅
- [x] **1.2** Cargar y explorar los datos ✅
- [x] **1.3** Análisis de distribución de clases (balance/desbalance) ✅
- [x] **1.4** Visualización de variables de fluorescencia ✅
- [x] **1.5** Visualización de espectros hiperespectrales ✅
- [x] **1.6** Análisis de correlaciones entre variables ✅
- [x] **1.7** Detección de valores atípicos (outliers) ✅
- [x] **1.8** Documentar hallazgos del EDA ✅

### Fase 2: Preprocesamiento de Datos
- [x] **2.1** Separar variables válidas de metadatos ✅
- [x] **2.2** Análisis de valores faltantes ✅ (No hay valores faltantes)
- [x] **2.3** Normalización/Estandarización de datos ✅ (StandardScaler implementado)
- [x] **2.4** Reducción de dimensionalidad (PCA, selección de características) ✅ (PCA y SelectKBest implementados)
- [ ] **2.5** Técnicas de balanceo de clases (si aplica: SMOTE, undersampling) - Pendiente evaluar necesidad
- [x] **2.6** Crear pipeline de preprocesamiento reutilizable ✅ (src/preprocessing.py)

### Fase 3: Modelado Baseline
- [x] **3.1** Implementar validación cruzada estratificada ✅ (StratifiedKFold 5-fold)
- [x] **3.2** Entrenar modelo baseline simple (Logistic Regression / Decision Tree) ✅
- [x] **3.3** Evaluar con F1-score en validación ✅ (LogReg: 0.9388, RF: 0.9266, SVM: 0.9326)
- [x] **3.4** Primera submission a Kaggle ✅ (submission_01_baseline_logisticregression)
- [ ] **3.5** Documentar resultados en tabla de experimentos - Pendiente score Kaggle

### Fase 4: Experimentación con Modelos
- [ ] **4.1** Random Forest
- [ ] **4.2** Gradient Boosting (XGBoost, LightGBM, CatBoost)
- [ ] **4.3** Support Vector Machine (SVM)
- [ ] **4.4** K-Nearest Neighbors (KNN)
- [ ] **4.5** Redes Neuronales (MLP)
- [ ] **4.6** Comparativa de modelos

### Fase 5: Optimización
- [ ] **5.1** Búsqueda de hiperparámetros (GridSearch / RandomSearch / Optuna)
- [ ] **5.2** Feature Engineering avanzado
- [ ] **5.3** Ensemble methods (Voting, Stacking)
- [ ] **5.4** Análisis de importancia de características
- [ ] **5.5** Validación cruzada anidada para estimación robusta

### Fase 6: Submissions Finales
- [ ] **6.1** Seleccionar mejores modelos
- [ ] **6.2** Entrenar con todos los datos de entrenamiento
- [ ] **6.3** Generar predicciones finales
- [ ] **6.4** Submissions estratégicas a Kaggle

### Fase 7: Documentación Final
- [ ] **7.1** Completar tabla de experimentos
- [ ] **7.2** Captura de pantalla del Leaderboard
- [ ] **7.3** Redactar documentación PDF
- [ ] **7.4** Organizar scripts y CSVs con nomenclatura clara
- [ ] **7.5** Revisión final y entrega

---

## 📊 Registro de Experimentos

> **IMPORTANTE**: Mantener actualizado el archivo `REGISTRO_EXPERIMENTOS.md` con cada submission.

Ver archivo: [REGISTRO_EXPERIMENTOS.md](./REGISTRO_EXPERIMENTOS.md)

---

## 📝 Registro de Progreso

### [Fecha: 23/12/2024]
**Actividad realizada:**
- Configuración inicial del proyecto y estructura de directorios
- Implementación de módulos en `src/`: preprocessing.py, models.py, utils.py
- Análisis Exploratorio de Datos completo (notebook 01_EDA.ipynb)
- Entrenamiento y comparación de 5 modelos baseline
- Generación de primera submission para Kaggle

**Problemas encontrados:**
- Datos con formato incorrecto: valores como '232 .25' (espacio antes del punto decimal)
- Error al calcular correlaciones por tipos de datos incorrectos

**Soluciones aplicadas:**
- Implementación de función `clean_numeric_columns()` en preprocessing.py
- Integración de limpieza automática en `load_data()`

**Próximos pasos:**
- Subir submission a Kaggle y registrar score
- Experimentar con PCA + diferentes modelos
- Probar XGBoost/LightGBM
- Optimización de hiperparámetros

---

## 🔧 Notas Técnicas

### Librerías Recomendadas
```python
# Data manipulation
pandas, numpy

# Visualization
matplotlib, seaborn, plotly

# Machine Learning
scikit-learn, xgboost, lightgbm, catboost

# Imbalanced data
imbalanced-learn (SMOTE, etc.)

# Hyperparameter tuning
optuna, scikit-optimize

# Deep Learning (opcional)
tensorflow, pytorch
```

### Consideraciones Especiales
1. **NO usar** columnas `exp`, `dpi`, `leaf`, `spot` como features
2. **Codificación de clases**: control → 0, botrytis → 1
3. **Métrica objetivo**: F1-score
4. **Validación**: Usar validación cruzada estratificada por el desbalance potencial

---

## ⚠️ Recordatorios Importantes

- [ ] Registrar CADA submission en la tabla de experimentos
- [ ] Guardar CADA script usado con nomenclatura clara (ej: `exp_01_baseline_lr.py`)
- [ ] Guardar CADA CSV de submission (ej: `submission_01_baseline_lr.csv`)
- [ ] No usar datos de test para entrenar/configurar modelos
- [ ] Nombre en Kaggle: `[Nombre][Apellido]_UGR_IN`
- [ ] Captura del Leaderboard para la documentación

---

## 📚 Referencias y Recursos

- [Documentación scikit-learn](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Imbalanced-learn](https://imbalanced-learn.org/)
- [Kaggle Competition Tips](https://www.kaggle.com/docs/competitions)

---

*Última actualización: 23/12/2024*
