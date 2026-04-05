import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generar_caso_de_uso_descomponer_serie_ventas():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función descomponer_serie_ventas.
    """
    periodo = random.choice([4, 6, 7, 12])
    n_periodos = random.randint(3, 6)
    n_filas = periodo * n_periodos

    # Generar fechas diarias/semanales
    fecha_inicio = datetime(2022, 1, 1)
    fechas = [fecha_inicio + timedelta(days=i) for i in range(n_filas)]

    # Serie sintética: tendencia + estacionalidad + ruido
    tendencia_base = np.linspace(100, 200, n_filas)
    factores_estacionales = np.tile(
        np.sin(np.linspace(0, 2 * np.pi, periodo)) * 20,
        n_periodos
    )
    ruido = np.random.randn(n_filas) * 5

    ventas = tendencia_base + factores_estacionales + ruido

    df = pd.DataFrame({
        "fecha": fechas,
        "ventas": ventas
    })

    fecha_col = "fecha"
    valor_col = "ventas"

    input_data = {
        "df":        df.copy(),
        "fecha_col": fecha_col,
        "valor_col": valor_col,
        "periodo":   periodo
    }

    # --- Calcular output esperado ---
    df_work = df.copy()
    df_work[fecha_col] = pd.to_datetime(df_work[fecha_col])
    df_work = df_work.sort_values(fecha_col).reset_index(drop=True)

    # Tendencia
    df_work["tendencia"] = (
        df_work[valor_col]
        .rolling(window=periodo, center=True)
        .mean()
        .ffill()
        .bfill()
    )

    # Serie sin tendencia
    sin_tendencia = df_work[valor_col] - df_work["tendencia"]

    # Factores estacionales por posición dentro del periodo
    posiciones = np.arange(len(df_work)) % periodo
    df_temp = pd.DataFrame({"sin_tendencia": sin_tendencia, "posicion": posiciones})
    factores = df_temp.groupby("posicion")["sin_tendencia"].mean()

    df_work["estacionalidad"] = posiciones
    df_work["estacionalidad"] = df_work["estacionalidad"].map(factores)

    # Residuo
    df_work["residuo"] = (
        df_work[valor_col] - df_work["tendencia"] - df_work["estacionalidad"]
    )

    # Estadísticas del residuo
    media_residuo = float(round(np.mean(df_work["residuo"].values), 6))
    std_residuo   = float(round(np.std(df_work["residuo"].values), 6))
    es_aprox_normal = bool(abs(media_residuo) < 0.05 * std_residuo)

    dict_stats = {
        "media_residuo":   media_residuo,
        "std_residuo":     std_residuo,
        "es_aprox_normal": es_aprox_normal
    }

    output_data = (df_work, dict_stats)

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_descomponer_serie_ventas()
    df_desc, stats = salida_esperada
    print("=== INPUT ===")
    print(f"fecha_col: {entrada['fecha_col']}, valor_col: {entrada['valor_col']}, periodo: {entrada['periodo']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada["df"].head())
    print("\n=== OUTPUT ESPERADO ===")
    print("DataFrame descompuesto (primeras 5 filas):")
    print(df_desc[["fecha", "ventas", "tendencia", "estacionalidad", "residuo"]].head())
    print("\nEstadísticas del residuo:")
    print(stats)
