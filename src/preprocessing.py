"""
Funciones de preprocesamiento para la Práctica 3 - Kaggle
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


# Columnas que NO se pueden usar como features
METADATA_COLS = ['exp', 'dpi', 'leaf', 'spot']
TARGET_COL = 'class'

# Columnas de fluorescencia
FLUORESCENCE_COLS = ['F440', 'F520', 'F680', 'F740']


def clean_numeric_columns(df):
    """
    Limpia columnas numéricas que pueden tener espacios u otros caracteres.
    Ej: '232 .25' -> 232.25
    """
    for col in df.columns:
        if df[col].dtype == 'object' and col not in METADATA_COLS + [TARGET_COL]:
            df[col] = df[col].astype(str).str.replace(' ', '').astype(float)
    return df


def load_data(train_path, test_path):
    """
    Carga los datos de entrenamiento y test.
    
    Returns:
        train_df, test_df: DataFrames con los datos
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Limpiar columnas numéricas con formato incorrecto
    train_df = clean_numeric_columns(train_df)
    test_df = clean_numeric_columns(test_df)
    
    return train_df, test_df


def get_feature_columns(df):
    """
    Obtiene las columnas válidas para usar como features.
    Excluye metadatos y la columna target.
    """
    exclude_cols = METADATA_COLS + [TARGET_COL]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def get_spectral_columns(df):
    """
    Obtiene las columnas espectrales (w388.13 a w1028.28).
    """
    return [col for col in df.columns if col.startswith('w')]


def encode_target(y):
    """
    Codifica la variable target: control -> 0, botrytis -> 1
    """
    mapping = {'control': 0, 'botrytis': 1}
    return y.map(mapping)


def decode_target(y):
    """
    Decodifica la variable target: 0 -> control, 1 -> botrytis
    """
    mapping = {0: 'control', 1: 'botrytis'}
    return pd.Series(y).map(mapping)


def prepare_features(train_df, test_df):
    """
    Prepara las features para entrenamiento.
    
    Returns:
        X_train, X_test, y_train, feature_cols
    """
    feature_cols = get_feature_columns(train_df)
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = encode_target(train_df[TARGET_COL]).values
    
    return X_train, X_test, y_train, feature_cols


def scale_features(X_train, X_test, method='standard'):
    """
    Escala las features.
    
    Args:
        method: 'standard' para StandardScaler, 'minmax' para MinMaxScaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Método no soportado: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def apply_pca(X_train, X_test, n_components=0.95):
    """
    Aplica PCA para reducción de dimensionalidad.
    
    Args:
        n_components: número de componentes o varianza a retener (0-1)
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    return X_train_pca, X_test_pca, pca


def select_k_best(X_train, X_test, y_train, k=50):
    """
    Selecciona las k mejores features usando ANOVA F-value.
    """
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_test_selected, selector
