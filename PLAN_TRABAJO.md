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
- [ ] **1.1** Configurar entorno de trabajo y dependencias
- [ ] **1.2** Cargar y explorar los datos
- [ ] **1.3** Análisis de distribución de clases (balance/desbalance)
- [ ] **1.4** Visualización de variables de fluorescencia
- [ ] **1.5** Visualización de espectros hiperespectrales
- [ ] **1.6** Análisis de correlaciones entre variables
- [ ] **1.7** Detección de valores atípicos (outliers)
- [ ] **1.8** Documentar hallazgos del EDA

### Fase 2: Preprocesamiento de Datos
- [ ] **2.1** Separar variables válidas de metadatos
- [ ] **2.2** Análisis de valores faltantes
- [ ] **2.3** Normalización/Estandarización de datos
- [ ] **2.4** Reducción de dimensionalidad (PCA, selección de características)
- [ ] **2.5** Técnicas de balanceo de clases (si aplica: SMOTE, undersampling)
- [ ] **2.6** Crear pipeline de preprocesamiento reutilizable

### Fase 3: Modelado Baseline
- [ ] **3.1** Implementar validación cruzada estratificada
- [ ] **3.2** Entrenar modelo baseline simple (Logistic Regression / Decision Tree)
- [ ] **3.3** Evaluar con F1-score en validación
- [ ] **3.4** Primera submission a Kaggle
- [ ] **3.5** Documentar resultados en tabla de experimentos

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

### [Fecha: ____]
**Actividad realizada:**
- 

**Problemas encontrados:**
- 

**Soluciones aplicadas:**
- 

**Próximos pasos:**
- 

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

*Última actualización: [Fecha de inicio]*
