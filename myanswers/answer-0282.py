import pandas as pd
import numpy as np


def detectar_discrepancias_inventario(df, umbral_error):
    resultado = df.copy()

    # Calcular stock final esperado
    resultado['stock_final_esperado'] = resultado['stock_inicial'] - resultado['ventas']

    # Calcular discrepancia absoluta
    resultado['discrepancia'] = (
        resultado['stock_final_real'] - resultado['stock_final_esperado']
    ).abs()

    # Filtrar productos con discrepancia estrictamente mayor al umbral
    filtrado = resultado[resultado['discrepancia'] > umbral_error].copy()

    # Ordenar de mayor a menor discrepancia
    filtrado = filtrado.sort_values('discrepancia', ascending=False).reset_index(drop=True)

    return filtrado
