# Scripts de Experimentos

Este directorio contiene los scripts Python utilizados para cada experimento.

## Nomenclatura

Los archivos deben seguir el formato:
```
exp_XX_descripcion.py
```

Donde:
- `XX`: Número de experimento (01, 02, 03...)
- `descripcion`: Breve descripción del modelo/técnica

## Ejemplo
- `exp_01_baseline_lr.py` - Logistic Regression baseline
- `exp_02_rf_default.py` - Random Forest con parámetros por defecto
- `exp_03_xgb_tuned.py` - XGBoost con hiperparámetros optimizados

## Lista de Scripts

| Script | Descripción | Submission Asociada |
|--------|-------------|---------------------|
| `exp_01_baseline.py` | Comparación de 5 modelos baseline con StandardScaler | `submission_01_baseline_logisticregression_20251223_1856.csv` |

---

## Detalle de Scripts

### exp_01_baseline.py

**Objetivo**: Establecer una línea base comparando múltiples algoritmos de clasificación.

**Metodología**:
1. Carga de datos con limpieza automática (`load_data()`)
2. Preparación de features (excluye metadatos: exp, dpi, leaf, spot)
3. Escalado con StandardScaler
4. Evaluación con validación cruzada estratificada (5-fold)
5. Comparación de 5 modelos: LogisticRegression, RandomForest, SVM, KNN, GradientBoosting
6. Selección del mejor modelo según F1-score
7. Entrenamiento final y generación de submission

**Resultados**:
| Modelo | F1-Score (CV) |
|--------|---------------|
| LogisticRegression | 0.9388 ± 0.0254 |
| SVM | 0.9326 |
| RandomForest | 0.9266 |
| GradientBoosting | 0.9203 |
| KNN | 0.8943 |

**Conclusión**: LogisticRegression fue el mejor modelo baseline, seleccionado para la primera submission.

---

*Actualizar esta tabla con cada nuevo script*
