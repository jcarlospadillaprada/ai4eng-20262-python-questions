import pandas as pd
import numpy as np


def detectar_discrepancias_inventario(df, umbral_error):
    resultado = df.copy()

    # Calcular discrepancia absoluta entre stock real y esperado
    stock_esperado = resultado['stock_inicial'] - resultado['ventas']
    resultado['discrepancia'] = (resultado['stock_final_real'] - stock_esperado).abs()

    # Filtrar productos con discrepancia estrictamente mayor al umbral
    filtrado = resultado[resultado['discrepancia'] > umbral_error].copy()

    # Ordenar de mayor a menor discrepancia (sin reset_index, igual que el generador)
    filtrado = filtrado.sort_values('discrepancia', ascending=False)

    return filtrado
