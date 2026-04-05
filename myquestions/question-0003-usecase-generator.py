import numpy as np
import random
from sklearn.decomposition import TruncatedSVD

def generar_caso_de_uso_factorizar_matriz_usuarios():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función factorizar_matriz_usuarios.
    """
    n_usuarios = random.randint(8, 20)
    n_items    = random.randint(10, 30)

    # Matriz dispersa: ~60% ceros
    X = np.random.rand(n_usuarios, n_items)
    mask = np.random.rand(n_usuarios, n_items) < 0.6
    X[mask] = 0.0
    X = X.astype(float)

    max_componentes = min(n_usuarios, n_items) - 1
    n_componentes = random.randint(2, max(2, max_componentes // 2))
    umbral = round(random.uniform(0.1, 0.8), 2)

    input_data = {
        "X":                    X.copy(),
        "n_componentes":        n_componentes,
        "umbral_reconstruccion": umbral
    }

    # --- Calcular output esperado ---
    svd = TruncatedSVD(n_components=n_componentes, random_state=42)
    X_transformada   = svd.fit_transform(X)
    X_reconstruida   = svd.inverse_transform(X_transformada)

    # RMSE global
    diff = X - X_reconstruida
    rmse_global = float(round(np.sqrt(np.mean(diff ** 2)), 6))

    # RMSE por fila
    rmse_por_usuario = np.sqrt(np.mean(diff ** 2, axis=1))
    usuarios_bien_representados = rmse_por_usuario <= umbral

    output_data = (X_reconstruida, rmse_global, usuarios_bien_representados)

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_factorizar_matriz_usuarios()
    X_rec, rmse, bien = salida_esperada
    print("=== INPUT ===")
    print(f"Forma de X: {entrada['X'].shape}")
    print(f"n_componentes: {entrada['n_componentes']}")
    print(f"umbral_reconstruccion: {entrada['umbral_reconstruccion']}")
    print("\n=== OUTPUT ESPERADO ===")
    print(f"Forma X_reconstruida: {X_rec.shape}")
    print(f"RMSE global: {rmse}")
    print(f"Usuarios bien representados (primeros 5): {bien[:5]}")
    print(f"Total bien representados: {bien.sum()} de {len(bien)}")
