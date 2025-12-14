import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


def crear_dataset_simulado(n=400, seed=123):
    rng = np.random.default_rng(seed)

    dist_km = rng.uniform(2, 35, size=n).round(2)
    num_estaciones = (dist_km * rng.uniform(1.8, 3.2, size=n)).round().astype(int)
    num_transbordos = rng.integers(0, 3, size=n)

    hora_pico = rng.integers(0, 2, size=n)          # 0 o 1
    clima_lluvia = rng.integers(0, 2, size=n)       # 0 o 1
    incidente = rng.binomial(1, 0.12, size=n)       # ~12% con incidentes
    dia_semana = rng.integers(0, 7, size=n)         # 0 a 6

    tiempo_base = (
        dist_km * 2.2 +
        num_estaciones * 0.6 +
        num_transbordos * 6
    )

    penal_pico = hora_pico * rng.uniform(4, 10, size=n)
    penal_lluvia = clima_lluvia * rng.uniform(2, 6, size=n)
    penal_incidente = incidente * rng.uniform(8, 18, size=n)

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
        "tiempo_min": tiempo_min
    })

    return df


def entrenar_modelo(df):
    X = df.drop(columns=["tiempo_min"])
    y = df["tiempo_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )

    modelo = DecisionTreeRegressor(
      max_depth=3,
      min_samples_leaf=20,
      min_samples_split=40,
      random_state=123
    )

    modelo.fit(X_train, y_train)

    preds = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    resultados = {
        "MAE_min": float(mae),
        "RMSE_min": float(rmse),
        "R2": float(r2),
        "n_total": int(len(df)),
        "test_size": 0.3
    }

    return modelo, X_train, resultados


def guardar_arbol(modelo, feature_names, filepath):
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


def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = crear_dataset_simulado(n=400, seed=123)
    dataset_path = "data/raw/dataset_transporte_masivo.csv"
    df.to_csv(dataset_path, index=False)

    modelo, X_train, resultados = entrenar_modelo(df)

    arbol_path = "outputs/arbol_decision.png"
    guardar_arbol(modelo, X_train.columns.tolist(), arbol_path)

    print("Dataset:", dataset_path)
    print("Arbol:", arbol_path)
    print("Resultados:")
    for k, v in resultados.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
