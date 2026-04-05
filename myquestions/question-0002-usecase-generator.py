import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_normalizar_expresion_cuantiles():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función normalizar_expresion_cuantiles.
    """
    n_genes   = random.randint(6, 15)
    n_muestras = random.randint(3, 6)

    sample_names = [f"muestra_{i}" for i in range(n_muestras)]
    gene_names   = [f"gen_{i}"     for i in range(n_genes)]

    data = np.random.randint(0, 500, size=(n_genes, n_muestras))
    df = pd.DataFrame(data, index=gene_names, columns=sample_names)

    input_data = {"df": df.copy()}

    # --- Calcular output esperado ---
    # Paso 1: log2(x + 1)
    df_log = np.log2(df + 1)

    # Paso 2: ranking de cada columna
    ranks = df_log.rank(method='average')

    # Paso 3: valor de referencia — media por fila del df ordenado por columna
    df_sorted = pd.DataFrame(
        np.sort(df_log.values, axis=0),
        index=df_log.index,
        columns=df_log.columns
    )
    row_means = df_sorted.mean(axis=1).values  # un valor por rango

    # Paso 4: reemplazar cada valor por su valor de referencia según su rango
    # rank va de 1..n_genes; el índice del row_means va 0..n_genes-1
    result = ranks.copy()
    for col in ranks.columns:
        result[col] = ranks[col].apply(
            lambda r: row_means[int(round(r)) - 1]
        )

    output_data = result.round(6)

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_normalizar_expresion_cuantiles()
    print("=== INPUT ===")
    print("DataFrame de expresión génica:")
    print(entrada["df"])
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
