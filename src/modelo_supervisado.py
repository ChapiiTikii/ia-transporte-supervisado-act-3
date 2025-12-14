"""
Actividad 3 - Metodos de aprendizaje supervisado
Contexto: Transporte masivo (relacionado con actividades previas de busqueda/rutas)

Objetivo del componente:
- Construir un modelo supervisado (regresion) que aprenda a predecir el tiempo de viaje (tiempo_min)
  usando variables que normalmente influyen en un trayecto: distancia, estaciones, transbordos,
  hora pico, clima e incidentes.

Relacion con "problema de busqueda" (actividad anterior):
- En un problema de busqueda, el agente busca una ruta desde A hasta B (camino).
- En este trabajo (supervisado), el modelo no busca rutas directamente; mas bien predice un valor
  (tiempo/costo) que puede servir para evaluar rutas o condiciones del sistema.
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1) DATOS -> construccion o carga del dataset
# ---------------------------------------------------------------------
def crear_dataset_simulado(n: int = 800, seed: int = 123) -> pd.DataFrame:
    """
    Crea un dataset simulado de transporte masivo.

    Cada fila representa un "viaje" con condiciones observables (features) y un resultado medido:
    tiempo_min (label). Se simula porque no se cuenta con un dataset publico directo con estas
    variables para el sistema del proyecto.

    Parametros:
    - n: numero de registros (viajes simulados)
    - seed: semilla para reproducibilidad

    Retorna:
    - DataFrame con variables explicativas y la variable objetivo.
    """
    rng = np.random.default_rng(seed)

    # Variables relacionadas con el trayecto
    dist_km = rng.uniform(2, 35, size=n).round(2)
    num_estaciones = (dist_km * rng.uniform(1.8, 3.2, size=n)).round().astype(int)
    num_transbordos = rng.integers(0, 3, size=n)

    # Variables de contexto (condiciones del entorno)
    hora_pico = rng.integers(0, 2, size=n)        # 0 = no, 1 = si
    clima_lluvia = rng.integers(0, 2, size=n)     # 0 = no, 1 = si
    incidente = rng.binomial(1, 0.12, size=n)     # probabilidad ~12%
    dia_semana = rng.integers(0, 7, size=n)       # 0-6

    # "Regla generadora" del tiempo (simula como se comportaria el sistema)
    # Base: mas distancia y mas estaciones => mas tiempo
    tiempo_base = (
        dist_km * 2.2 +
        num_estaciones * 0.6 +
        num_transbordos * 6
    )

    # Penalizaciones: condiciones que aumentan el tiempo
    penal_pico = hora_pico * rng.uniform(4, 10, size=n)
    penal_lluvia = clima_lluvia * rng.uniform(2, 6, size=n)
    penal_incidente = incidente * rng.uniform(8, 18, size=n)

    # Ruido: variabilidad natural (retrasos aleatorios, tiempos de espera, etc.)
    ruido = rng.normal(0, 3.0, size=n)

    tiempo_min = (tiempo_base + penal_pico + penal_lluvia + penal_incidente + ruido).round(2)

    df = pd.DataFrame({
        "dist_km": dist_km,
        "num_estaciones": num_estaciones,
        "num_transbordos": num_transbordos,
        "hora_pico": hora_pico,
        "clima_lluvia": clima_lluvia,
        "incidente": incidente,
        "dia_semana": dia_semana,
        "tiempo_min": tiempo_min  # variable objetivo (supervisado)
    })

    return df


# ---------------------------------------------------------------------
# 2) PROBLEMA -> definicion del objetivo del modelo
# ---------------------------------------------------------------------
def definir_problema(df: pd.DataFrame):
    """
    Define X (variables explicativas) e y (objetivo) para el aprendizaje supervisado.

    En este caso:
    - X: condiciones del trayecto y contexto
    - y: tiempo_min (tiempo real/estimado del viaje)
    """
    X = df.drop(columns=["tiempo_min"])
    y = df["tiempo_min"]
    return X, y


# ---------------------------------------------------------------------
# 3) PREPROCESAMIENTO
# ---------------------------------------------------------------------
def preprocesar(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesamiento basico:
    - En este dataset no hay variables categoricas tipo texto, ni nulos.
    - Aun asi, dejamos esta funcion para mostrar la etapa del pipeline.

    Si en el futuro se usan fuentes reales:
    - Aqui se podrian imputar nulos, escalar variables, codificar categorias, etc.
    """
    X_proc = X.copy()

    # Ejemplo: asegurar tipos enteros donde corresponde
    cols_int = ["num_estaciones", "num_transbordos", "hora_pico", "clima_lluvia", "incidente", "dia_semana"]
    for c in cols_int:
        X_proc[c] = X_proc[c].astype(int)

    return X_proc


# ---------------------------------------------------------------------
# 4) SPLIT -> entrenamiento y prueba
# ---------------------------------------------------------------------
def dividir_datos(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, seed: int = 123):
    """
    Divide el dataset en entrenamiento y prueba.
    Esto permite evaluar el modelo en datos no vistos durante el entrenamiento.
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed)


# ---------------------------------------------------------------------
# 5) MODELO -> definicion del algoritmo supervisado
# ---------------------------------------------------------------------
def crear_modelo(seed: int = 123) -> DecisionTreeRegressor:
    """
    Crea un arbol de decision para regresion.

    Parametros para evitar un arbol demasiado grande:
    - max_depth: limita niveles, mejora interpretabilidad
    - min_samples_leaf / min_samples_split: evita hojas con muy pocos datos (reduce sobreajuste)
    """
    return DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=20,
        min_samples_split=40,
        random_state=seed
    )


# ---------------------------------------------------------------------
# 6) ENTRENAR MODELO
# ---------------------------------------------------------------------
def entrenar(modelo: DecisionTreeRegressor, X_train, y_train):
    """Entrena el modelo con los datos de entrenamiento."""
    modelo.fit(X_train, y_train)
    return modelo


# ---------------------------------------------------------------------
# 7) METRICAS -> evaluacion del modelo
# ---------------------------------------------------------------------
def evaluar(modelo: DecisionTreeRegressor, X_test, y_test) -> dict:
    """
    Calcula metricas tipicas de regresion:
    - MAE: error absoluto medio (en minutos)
    - RMSE: error cuadratico medio (penaliza mas errores grandes)
    - R2: que tanto explica el modelo la variabilidad del objetivo (1 es ideal)
    """
    preds = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    return {
        "MAE_min": float(mae),
        "RMSE_min": float(rmse),
        "R2": float(r2),
        "n_test": int(len(y_test))
    }


def guardar_arbol(modelo: DecisionTreeRegressor, feature_names, filepath: str):
    """Guarda el diagrama del arbol en un archivo PNG para incluirlo en el repo y el video."""
    plt.figure(figsize=(14, 7))
    plot_tree(
        modelo,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# 8) CONCLUSIONES -> interpretacion simple del resultado
# ---------------------------------------------------------------------
def conclusiones(resultados: dict) -> str:
    """
    Genera un texto corto de conclusiones basado en las metricas.
    Esto te sirve tal cual para el video y el documento de pruebas.
    """
    mae = resultados["MAE_min"]
    rmse = resultados["RMSE_min"]
    r2 = resultados["R2"]

    texto = []
    texto.append("Conclusiones del modelo supervisado (arbol de decision):")
    texto.append(f"- El MAE fue de {mae:.2f} minutos, lo que indica el error promedio en las predicciones.")
    texto.append(f"- El RMSE fue de {rmse:.2f} minutos, penalizando mas los errores grandes.")
    texto.append(f"- El R2 fue de {r2:.3f}, que muestra que el modelo explica una parte importante de la variabilidad del tiempo.")

    texto.append("- Se limitaron parametros del arbol (profundidad y minimo de muestras por hoja) para evitar sobreajuste y mejorar interpretabilidad.")
    texto.append("- Este tipo de modelo puede apoyar decisiones del sistema, por ejemplo estimar tiempos esperados y comparar condiciones de viaje.")

    return "\n".join(texto)


# ---------------------------------------------------------------------
# MAIN -> ejecucion del flujo completo del profesor
# ---------------------------------------------------------------------
def main():
    # Crear carpetas de salida
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # DATOS
    df = crear_dataset_simulado(n=400, seed=123)
    dataset_path = "data/raw/dataset_transporte_masivo.csv"
    df.to_csv(dataset_path, index=False)

    # PROBLEMA
    X, y = definir_problema(df)

    # PREPROCESAMIENTO
    X_proc = preprocesar(X)

    # SPLIT
    X_train, X_test, y_train, y_test = dividir_datos(X_proc, y, test_size=0.2, seed=123)

    # MODELO
    modelo = crear_modelo(seed=123)

    # ENTRENAR
    modelo = entrenar(modelo, X_train, y_train)

    # METRICAS
    resultados = evaluar(modelo, X_test, y_test)

    # Guardar arbol
    arbol_path = "outputs/arbol_decision.png"
    guardar_arbol(modelo, X_train.columns.tolist(), arbol_path)

    # Guardar metricas en JSON (evidencia facil)
    metricas_path = "outputs/metricas.json"
    with open(metricas_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)

    # Impresion final (para video)
    print("Dataset generado:", dataset_path)
    print("Arbol guardado:", arbol_path)
    print("Metricas guardadas:", metricas_path)
    print()
    print("Metricas:")
    for k, v in resultados.items():
        print(f"- {k}: {v}")

    print()
    print(conclusiones(resultados))


if __name__ == "__main__":
    main()
