"""
==========================================================
FASE 6B - Entrenamiento del primer modelo de IA
==========================================================
Este script lee el dataset preparado en 06_prepare_dataset.py
y entrena un modelo de Gradient Boosting para predecir si
una señal será ganadora o perdedora.

Puntos clave del diseño:
  - Split temporal (NO aleatorio): entrenamos con datos
    antiguos y testeamos con datos recientes. Esto simula
    el uso real (predecir el futuro con datos del pasado).
  - Modelo: GradientBoostingClassifier de scikit-learn.
    Es un modelo robusto, interpretable y que funciona bien
    con datasets de tamaño medio como el nuestro (~4500 filas).
  - Métricas: accuracy, precision, recall, F1, y sobre todo
    una simulación de "¿qué pasa si solo operamos cuando el
    modelo dice que sí?".
==========================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import joblib


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
OUTPUT_DIR = "outputs"
DATASET_FILE = os.path.join(OUTPUT_DIR, "ml_dataset.csv")
MODEL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "model_results.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "model.joblib")
IMPORTANCE_CHART = os.path.join(OUTPUT_DIR, "feature_importance.png")

# Fecha de corte para split temporal.
# Entrenamos con datos hasta esta fecha (inclusive),
# testeamos con datos posteriores.
# Usamos finales de 2024 como corte: entrenamos con
# 2021-2024 (~3.5 años) y testeamos con 2025-2026 (~1.5 años).
TRAIN_CUTOFF = "2024-12-31"

# Features (deben coincidir con las de 06_prepare_dataset.py)
FEATURES = [
    # Indicadores base
    "rsi_14",
    "atr_14",

    # Features derivadas (Iteración 0)
    "dist_ema20_pct",
    "dist_ema50_pct",
    "ema_spread_pct",
    "ema_spread_4h_pct",
    "atr_pct",
    "price_position",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "hour",
    "day_of_week",
    "risk_pct",
    "reward_risk_ratio",
    "direction_num",
    "rule_num",

    # Nuevas features (Iteración 1)
    "volume_sma20_ratio",
    "rsi_14_4h",
    "atr_pct_4h",
    "roc_5",
    "roc_20",
    "atr_regime",
    "candle_range_ratio",
]

LABEL = "label"


# ==========================================================
# FUNCIÓN 1 - Cargar dataset y hacer split temporal
# ==========================================================
def cargar_y_split():
    """
    Carga el dataset y lo divide en train/test por fecha.

    IMPORTANTE: usamos split temporal, NO aleatorio.
    En series temporales, un split aleatorio "filtra" información
    del futuro al conjunto de entrenamiento, lo que produce
    métricas falsamente optimistas. El split temporal simula
    el uso real: entrenar con el pasado, predecir el futuro.
    """
    df = pd.read_csv(DATASET_FILE)

    df["open_time"] = pd.to_datetime(
        df["open_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")

    cutoff = pd.Timestamp(TRAIN_CUTOFF, tz="Europe/Madrid")

    df_train = df[df["open_time"] <= cutoff].copy()
    df_test = df[df["open_time"] > cutoff].copy()

    print(f"Dataset total: {len(df)} filas")
    print(f"Train (hasta {TRAIN_CUTOFF}): {len(df_train)} filas")
    print(f"Test  (después {TRAIN_CUTOFF}): {len(df_test)} filas")
    print(f"\nTrain label: {df_train[LABEL].value_counts().to_dict()}")
    print(f"Test  label: {df_test[LABEL].value_counts().to_dict()}")

    return df_train, df_test


# ==========================================================
# FUNCIÓN 2 - Entrenar modelo
# ==========================================================
def entrenar_modelo(df_train):
    """
    Entrena un GradientBoostingClassifier.

    Parámetros conservadores para evitar sobreajuste:
      - n_estimators=200 (no demasiados árboles)
      - max_depth=4 (árboles poco profundos)
      - learning_rate=0.05 (aprendizaje lento y gradual)
      - min_samples_leaf=20 (no crear hojas con pocos datos)
      - subsample=0.8 (usar 80% de los datos en cada árbol)
    """
    X_train = df_train[FEATURES]
    y_train = df_train[LABEL]

    modelo = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42
    )

    print("\nEntrenando modelo...")
    modelo.fit(X_train, y_train)
    print("Modelo entrenado.")

    # Guardar modelo para uso en vivo (08_live_signal.py)
    joblib.dump(modelo, MODEL_FILE)
    print(f"Modelo guardado: {MODEL_FILE}")

    return modelo


# ==========================================================
# FUNCIÓN 3 - Evaluar modelo con métricas de clasificación
# ==========================================================
def evaluar_clasificacion(modelo, df_test):
    """
    Métricas estándar de clasificación sobre el set de test.
    """
    X_test = df_test[FEATURES]
    y_test = df_test[LABEL]

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    print(f"\n{'='*60}")
    print("MÉTRICAS DE CLASIFICACIÓN (test set)")
    print(f"{'='*60}")

    print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")

    print(f"\nMatriz de confusión:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Pred→     Perd.  Ganad.")
    print(f"  Real↓")
    print(f"  Perd.  {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  Ganad. {cm[1][0]:>6}  {cm[1][1]:>6}")

    print(f"\nReporte completo:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Perdedor", "Ganador"],
        zero_division=0
    ))

    return y_pred, y_proba


# ==========================================================
# FUNCIÓN 4 - Simulación de trading con filtro del modelo
# ==========================================================
def simular_con_filtro(df_test, y_pred, y_proba, umbral=0.5):
    """
    La métrica más importante: ¿qué pasa si solo operamos
    cuando el modelo dice "ganador"?

    Comparamos:
      - "Sin filtro": operar todas las señales (lo que
        hace el sistema de reglas solo).
      - "Con filtro": operar solo las señales que el modelo
        predice como ganadoras.

    También probamos con diferentes umbrales de probabilidad
    para ver el tradeoff entre selectividad y rendimiento.
    """
    df = df_test.copy()
    df["y_pred"] = y_pred
    df["y_proba"] = y_proba

    print(f"\n{'='*60}")
    print("SIMULACIÓN DE TRADING: con y sin filtro IA")
    print(f"{'='*60}")

    # Sin filtro
    pnl_sin = df["pnl_abs"].sum()
    trades_sin = len(df)
    wr_sin = (df["pnl_abs"] > 0).mean() * 100

    print(f"\nSIN filtro IA:")
    print(f"  Trades: {trades_sin}")
    print(f"  Win rate: {wr_sin:.1f}%")
    print(f"  PnL total: {pnl_sin:,.2f} USD")

    # Con filtro (umbral por defecto = 0.5)
    print(f"\nCON filtro IA (umbrales de probabilidad):")
    print(f"{'Umbral':>8s} | {'Trades':>7s} | {'WR':>6s} | "
          f"{'PnL':>12s} | {'PnL/trade':>10s} | {'Filtradas':>9s}")
    print("-" * 70)

    resultados = []

    for umbral in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        filtrado = df[df["y_proba"] >= umbral]
        n = len(filtrado)
        if n == 0:
            continue
        wr = (filtrado["pnl_abs"] > 0).mean() * 100
        pnl = filtrado["pnl_abs"].sum()
        pnl_medio = filtrado["pnl_abs"].mean()
        pct_filtradas = (1 - n / trades_sin) * 100

        print(f"{umbral:>8.2f} | {n:>7d} | {wr:>5.1f}% | "
              f"{pnl:>11,.2f} | {pnl_medio:>9,.2f} | "
              f"{pct_filtradas:>8.1f}%")

        resultados.append({
            "umbral": umbral,
            "trades": n,
            "win_rate": wr,
            "pnl_total": pnl,
            "pnl_medio": pnl_medio,
            "pct_filtradas": pct_filtradas
        })

    return resultados


# ==========================================================
# FUNCIÓN 5 - Feature importance
# ==========================================================
def mostrar_feature_importance(modelo):
    """
    Muestra qué features son más importantes para el modelo.
    """
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n{'='*60}")
    print("IMPORTANCIA DE FEATURES")
    print(f"{'='*60}")

    for i, idx in enumerate(indices):
        print(f"  {i+1:>2}. {FEATURES[idx]:>25s}: "
              f"{importances[idx]:.4f}")

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_idx = np.argsort(importances)
    ax.barh(
        [FEATURES[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#2196F3"
    )
    ax.set_xlabel("Importancia")
    ax.set_title("Feature Importance — Gradient Boosting",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(IMPORTANCE_CHART, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nGráfico guardado: {IMPORTANCE_CHART}")


# ==========================================================
# FUNCIÓN 6 - Guardar resultados del test set
# ==========================================================
def guardar_resultados(df_test, y_pred, y_proba):
    """
    Guarda el test set con las predicciones del modelo,
    útil para análisis posterior.
    """
    df = df_test.copy()
    df["model_pred"] = y_pred
    df["model_proba"] = y_proba

    df.to_csv(MODEL_RESULTS_FILE, index=False)
    print(f"\nResultados guardados: {MODEL_RESULTS_FILE}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("Entrenando modelo de IA...\n")

    # 1. Cargar y split
    df_train, df_test = cargar_y_split()

    # 2. Entrenar
    modelo = entrenar_modelo(df_train)

    # 3. Evaluar clasificación
    y_pred, y_proba = evaluar_clasificacion(modelo, df_test)

    # 4. Simular trading con filtro
    simular_con_filtro(df_test, y_pred, y_proba)

    # 5. Feature importance
    mostrar_feature_importance(modelo)

    # 6. Guardar resultados
    guardar_resultados(df_test, y_pred, y_proba)

    print(f"\n{'='*60}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
