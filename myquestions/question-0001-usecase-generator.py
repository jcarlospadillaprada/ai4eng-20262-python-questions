import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_construir_red_ponderada():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función construir_red_ponderada.
    """
    n_usuarios = random.randint(4, 8)
    usuarios = [f"user_{i}" for i in range(n_usuarios)]

    n_mensajes = random.randint(20, 50)
    emisores   = [random.choice(usuarios) for _ in range(n_mensajes)]
    receptores = [random.choice([u for u in usuarios if u != e]) for e in emisores]
    sentimientos = [random.choice([-1, 0, 1]) for _ in range(n_mensajes)]

    df = pd.DataFrame({
        "emisor":      emisores,
        "receptor":    receptores,
        "sentimiento": sentimientos
    })

    emisor_col      = "emisor"
    receptor_col    = "receptor"
    sentimiento_col = "sentimiento"

    input_data = {
        "df":             df.copy(),
        "emisor_col":     emisor_col,
        "receptor_col":   receptor_col,
        "sentimiento_col": sentimiento_col
    }

    # --- Calcular output esperado ---
    grouped = df.groupby([emisor_col, receptor_col])[sentimiento_col].agg(
        frecuencia="count",
        sentimiento_promedio="mean"
    ).reset_index()

    grouped["sentimiento_promedio"] = grouped["sentimiento_promedio"].round(4)
    grouped["intensidad_relacion"]  = (
        grouped["frecuencia"] * abs(grouped["sentimiento_promedio"] + 0.5)
    ).round(4)

    resultado = grouped[grouped["frecuencia"] >= 2].copy()
    resultado = resultado.sort_values("intensidad_relacion", ascending=False)
    resultado = resultado.reset_index(drop=True)
    resultado = resultado[
        [emisor_col, receptor_col, "frecuencia",
         "sentimiento_promedio", "intensidad_relacion"]
    ]

    output_data = resultado

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_construir_red_ponderada()
    print("=== INPUT ===")
    print(f"emisor_col: {entrada['emisor_col']}")
    print(f"receptor_col: {entrada['receptor_col']}")
    print(f"sentimiento_col: {entrada['sentimiento_col']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada["df"].head())
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
