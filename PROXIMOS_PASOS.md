# Próximos Pasos - P3 Kaggle

## Estado Actual
- **Score actual**: 0.8478 (posición 1 de 2)
- **Modelo baseline**: Logistic Regression + StandardScaler
- **F1-score CV**: 0.8278, **F1-score Kaggle**: 0.8478

---

## Estrategias para Mejorar el Score

### 1. Reducción de Dimensionalidad (Prioridad: Alta)
**Razón**: El EDA mostró que 3 componentes PCA explican el 95% de la varianza.

**Experimentos propuestos**:
- `exp_02_pca_3components.py`: PCA con 3 componentes + Logistic Regression
- `exp_03_pca_7components.py`: PCA con 7 componentes + Random Forest
- `exp_04_selectkbest.py`: SelectKBest con k=50 + XGBoost

**Expected improvement**: +0.01-0.03 en F1-score

### 2. Modelos Avanzados (Prioridad: Alta)
**Algoritmos a probar**:
- **XGBoost**: Excelente para datos tabulares, robusto al overfitting
- **LightGBM**: Más rápido que XGBoost, similar rendimiento
- **CatBoost**: Manejo automático de variables categóricas (no aplica aquí pero bueno para comparar)

**Configuración base sugerida**:
```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### 3. Optimización de Hiperparámetros (Prioridad: Media)
**Herramienta**: Optuna
**Modelos a optimizar**:
- Logistic Regression (C, penalty, solver)
- XGBoost (n_estimators, max_depth, learning_rate, etc.)
- Random Forest (n_estimators, max_depth, min_samples_split)

### 4. Técnicas de Ensemble (Prioridad: Media)
**Métodos**:
- **Voting Classifier**: Combinar predicciones de múltiples modelos
- **Stacking**: Usar predicciones de modelos base como features para un meta-modelo
- **Bagging**: Multiple instancias del mismo modelo con diferentes subsets

### 5. Feature Engineering (Prioridad: Baja-Media)
**Ideas**:
- **Polynomial features**: Interacciones entre variables de fluorescencia
- **Spectral indices**: Ratios entre bandas espectrales específicas
- **Domain knowledge**: Índices vegetativos (NDVI-like) adaptados a fluorescencia

---

## Plan de Ejecución Sugerido

### Fase 1: Reducción Dimensionalidad (1-2 días)
1. Implementar experimentos con PCA
2. Evaluar impacto en diferentes modelos
3. Seleccionar mejor configuración de dimensionalidad

### Fase 2: Modelos Avanzados (2-3 días)
1. Implementar XGBoost baseline
2. Probar LightGBM y CatBoost
3. Comparar con Logistic Regression

### Fase 3: Optimización (2-3 días)
1. Configurar Optuna para búsqueda de hiperparámetros
2. Optimizar top 2 modelos
3. Validar con cross-validation estratificado

### Fase 4: Ensemble y Finalización (1-2 días)
1. Implementar Voting Classifier con mejores modelos
2. Probar Stacking si el tiempo lo permite
3. Generar submission final

---

## Scripts a Crear

```
scripts/
├── exp_02_pca_logistic.py
├── exp_03_pca_xgboost.py
├── exp_04_selectkbest_rf.py
├── exp_05_xgboost_baseline.py
├── exp_06_lightgbm_baseline.py
├── exp_07_optuna_xgboost.py
├── exp_08_optuna_rf.py
└── exp_09_ensemble_voting.py
```

---

## Métricas de Progreso

**Objetivos intermedios**:
- Superar 0.85: ✅ (actual: 0.8478)
- Alcanzar 0.86: 🎯 (próximo objetivo)
- Superar 0.87: 🚀 (optimista)
- Llegar a 0.88: 🏆 (muy ambicioso)

**Deadline sugerido**: 1-2 semanas para implementar y probar las mejoras principales.

---

## Notas Importantes

- **Semillas aleatorias**: Mantener `random_state=42` para reproducibilidad
- **Validación cruzada**: Usar siempre StratifiedKFold(n_splits=5)
- **Documentación**: Actualizar `REGISTRO_EXPERIMENTOS.md` después de cada submission
- **Backups**: Guardar modelos y predicciones de cada experimento
- **Tiempo**: Priorizar experimentos con mayor potencial de mejora

---

*Última actualización: 23/12/2025*
