import numpy as np
from sklearn.ensemble import RandomForestClassifier


def seleccionar_features_importantes(X, y, k):
    # Entrenar RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Obtener importancias
    importancias = model.feature_importances_

    # Índices de las k más importantes, de mayor a menor
    indices_top_k = np.argsort(importancias)[::-1][:k]

    # Valores de importancia correspondientes
    importancias_top_k = importancias[indices_top_k]

    return indices_top_k, importancias_top_k
