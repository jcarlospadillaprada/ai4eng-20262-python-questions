import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize


def preparar_firmas_espectrales(df, target_col, umbral_minimo):
    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # Imputar valores faltantes con la media de cada columna
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Eliminar filas cuya suma de intensidades sea menor que umbral_minimo
    sumas = X_imputed.sum(axis=1)
    mask = sumas >= umbral_minimo
    X_filtrado = X_imputed[mask]
    y_filtrado = y[mask]

    # Normalizar cada fila con norma L2
    X_normalizado = normalize(X_filtrado, norm='l2')

    return X_normalizado, y_filtrado
