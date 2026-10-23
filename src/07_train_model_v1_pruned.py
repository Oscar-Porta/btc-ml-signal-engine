"""
==========================================================
FASE 9 (Iteración 1B.2) - Entrenamiento del modelo V1 PODADO
==========================================================
Copia de 07_train_model.py con UN ÚNICO cambio:
la lista FEATURES elimina 4 columnas que el análisis de
la Iteración 1B demostró inútiles o duplicadas:

  1. risk_pct           → duplicada con atr_pct (corr=1.000)
  2. direction_num      → importancia nativa Y permutation = 0
  3. reward_risk_ratio  → siempre ~2.0 por diseño, invariante
  4. rule_num           → importancia ≈ 0 en ambas métricas

De 24 features originales pasamos a 20.

Objetivo: comprobar si un modelo más simple mantiene (o
supera) el rendimiento del v1 original. Criterio de éxito:

    F1_pruned >= F1_v1 - 0.02  (tolerancia de 2 puntos abs)

Si se cumple: adoptamos el podado como nuevo v1 y
actualizamos 08_live_signal.py para que use 20 features.
Si no: nos quedamos con el v1 original de 24.

Salidas:
  - outputs/model_v1_pruned.joblib
  - outputs/model_results_v1_pruned.csv
  - outputs/feature_importance_v1_pruned.png

NO se pisa el model.joblib original del v1 — eso solo se
hace al final si el podado gana y decides promoverlo.
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
from sklearn.inspection import permutation_importance
import joblib


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
OUTPUT_DIR = "outputs"
DATASET_FILE = os.path.join(OUTPUT_DIR, "ml_dataset.csv")  # dataset v1
MODEL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "model_results_v1_pruned.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "model_v1_pruned.joblib")
IMPORTANCE_CHART = os.path.join(OUTPUT_DIR, "feature_importance_v1_pruned.png")

# Modelo v1 original (24 features) para comparación.
V1_MODEL_FILE = os.path.join(OUTPUT_DIR, "model.joblib")

# Mismo split temporal que v1 para comparación justa.
TRAIN_CUTOFF = "2024-12-31"

# Features PODADAS (20 = 24 originales − 4 descartadas).
FEATURES = [
    # Indicadores base
    "rsi_14", "atr_14",
    # Iteración 0 (sin risk_pct, direction_num, reward_risk_ratio, rule_num)
    "dist_ema20_pct", "dist_ema50_pct",
    "ema_spread_pct", "ema_spread_4h_pct",
    "atr_pct", "price_position",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "hour", "day_of_week",
    # Iteración 1
    "volume_sma20_ratio", "rsi_14_4h", "atr_pct_4h",
    "roc_5", "roc_20", "atr_regime", "candle_range_ratio",
]

# Features v1 ORIGINAL (24) — DEBE estar en el ORDEN EXACTO en
# que se entrenó el modelo v1 (ver 07_train_model.py). Si lo
# desordenamos, scikit-learn rechaza el predict con
# "feature names must be in the same order as fit".
FEATURES_V1 = [
    "rsi_14", "atr_14",
    "dist_ema20_pct", "dist_ema50_pct",
    "ema_spread_pct", "ema_spread_4h_pct",
    "atr_pct", "price_position",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "hour", "day_of_week",
    "risk_pct", "reward_risk_ratio",
    "direction_num", "rule_num",
    "volume_sma20_ratio", "rsi_14_4h", "atr_pct_4h",
    "roc_5", "roc_20", "atr_regime", "candle_range_ratio",
]

LABEL = "label"


# ==========================================================
# FUNCIÓN 1 - Cargar + split
# ==========================================================
def cargar_y_split():
    df = pd.read_csv(DATASET_FILE)
    df["open_time"] = pd.to_datetime(
        df["open_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")

    cutoff = pd.Timestamp(TRAIN_CUTOFF, tz="Europe/Madrid")
    df_train = df[df["open_time"] <= cutoff].copy()
    df_test = df[df["open_time"] > cutoff].copy()

    print(f"Dataset total: {len(df)} filas")
    print(f"Train: {len(df_train)}  |  Test: {len(df_test)}")
    return df_train, df_test


# ==========================================================
# FUNCIÓN 2 - Entrenar
# ==========================================================
def entrenar_modelo(df_train):
    X_train = df_train[FEATURES]
    y_train = df_train[LABEL]

    modelo = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )
    print(f"\nEntrenando modelo podado ({len(FEATURES)} features)...")
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, MODEL_FILE)
    print(f"Modelo guardado: {MODEL_FILE}")
    return modelo


# ==========================================================
# FUNCIÓN 3 - Evaluar clasificación
# ==========================================================
def evaluar_clasificacion(modelo, df_test, features, titulo):
    X_test = df_test[features]
    y_test = df_test[LABEL]
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    print(f"\n{'='*60}")
    print(f"MÉTRICAS — {titulo}")
    print(f"{'='*60}")

    metricas = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }
    for k, v in metricas.items():
        print(f"{k.capitalize():>10s}: {v:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusión:")
    print(f"  Pred→     Perd.  Ganad.")
    print(f"  Real↓")
    print(f"  Perd.  {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  Ganad. {cm[1][0]:>6}  {cm[1][1]:>6}")

    print("\nReporte:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Perdedor", "Ganador"],
        zero_division=0
    ))
    return y_pred, y_proba, metricas


# ==========================================================
# FUNCIÓN 4 - Simulación de trading con filtro
# ==========================================================
def simular_con_filtro(df_test, y_pred, y_proba):
    df = df_test.copy()
    df["y_pred"] = y_pred
    df["y_proba"] = y_proba

    print(f"\n{'='*60}")
    print("SIMULACIÓN DE TRADING (v1 podado)")
    print(f"{'='*60}")

    pnl_sin = df["pnl_abs"].sum()
    trades_sin = len(df)
    wr_sin = (df["pnl_abs"] > 0).mean() * 100
    print(f"\nSIN filtro:")
    print(f"  Trades: {trades_sin} | Win rate: {wr_sin:.1f}% | "
          f"PnL: {pnl_sin:,.2f} USD")

    print(f"\nCON filtro:")
    print(f"{'Umbral':>8s} | {'Trades':>7s} | {'WR':>6s} | "
          f"{'PnL':>12s} | {'PnL/trade':>10s} | {'Filt.':>6s}")
    print("-" * 65)
    for umbral in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        f = df[df["y_proba"] >= umbral]
        n = len(f)
        if n == 0:
            continue
        wr = (f["pnl_abs"] > 0).mean() * 100
        pnl = f["pnl_abs"].sum()
        pnl_m = f["pnl_abs"].mean()
        pct = (1 - n / trades_sin) * 100
        print(f"{umbral:>8.2f} | {n:>7d} | {wr:>5.1f}% | "
              f"{pnl:>11,.2f} | {pnl_m:>9,.2f} | {pct:>5.1f}%")


# ==========================================================
# FUNCIÓN 5 - Importancia nativa + permutation
# ==========================================================
def mostrar_importancias(modelo, df_test):
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n{'='*60}")
    print("IMPORTANCIA NATIVA (v1 podado)")
    print(f"{'='*60}")
    for i, idx in enumerate(indices):
        print(f"  {i+1:>2}. {FEATURES[idx]:>25s}: {importances[idx]:.4f}")

    # Permutation importance.
    print(f"\n{'='*60}")
    print("PERMUTATION IMPORTANCE (F1, n_repeats=10)")
    print(f"{'='*60}")
    result = permutation_importance(
        modelo, df_test[FEATURES], df_test[LABEL],
        n_repeats=10, random_state=42, scoring="f1", n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    print(f"{'Feature':>25s} | {'Δ F1':>8s} | {'std':>6s}")
    print("-" * 48)
    for idx in order:
        print(f"{FEATURES[idx]:>25s} | "
              f"{result.importances_mean[idx]:>+8.4f} | "
              f"{result.importances_std[idx]:>6.4f}")

    # Gráfico de importancia nativa.
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_idx = np.argsort(importances)
    ax.barh(
        [FEATURES[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#FF9800"
    )
    ax.set_xlabel("Importancia")
    ax.set_title("Feature Importance v1 PODADO",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGráfico: {IMPORTANCE_CHART}")


# ==========================================================
# FUNCIÓN 6 - Comparativa v1 original vs v1 podado
# ==========================================================
def comparar_con_v1_original(df_test, metricas_pruned):
    if not os.path.exists(V1_MODEL_FILE):
        print(f"\n[INFO] No se encontró {V1_MODEL_FILE}. "
              f"Saltando comparación.")
        return

    print(f"\n{'='*60}")
    print("COMPARATIVA V1 ORIGINAL (24 feat) vs V1 PODADO (20 feat)")
    print(f"{'='*60}")

    modelo_v1 = joblib.load(V1_MODEL_FILE)
    X_test_v1 = df_test[FEATURES_V1]
    y_test = df_test[LABEL]
    y_pred_v1 = modelo_v1.predict(X_test_v1)

    m_v1 = {
        "accuracy":  accuracy_score(y_test, y_pred_v1),
        "precision": precision_score(y_test, y_pred_v1, zero_division=0),
        "recall":    recall_score(y_test, y_pred_v1, zero_division=0),
        "f1":        f1_score(y_test, y_pred_v1, zero_division=0),
    }

    print(f"\n{'Métrica':>12s} | {'v1 orig':>9s} | {'v1 pod':>8s} | "
          f"{'Δ abs':>8s} | {'Δ %':>7s}")
    print("-" * 55)
    for k in ["accuracy", "precision", "recall", "f1"]:
        a, b = m_v1[k], metricas_pruned[k]
        d = b - a
        pct = (d / a * 100) if a > 0 else 0
        print(f"{k:>12s} | {a:>9.3f} | {b:>8.3f} | {d:>+8.3f} | "
              f"{pct:>+6.1f}%")

    # Veredicto: F1_pruned >= F1_v1 - 0.02 (2 puntos abs de tolerancia).
    delta_f1_abs = metricas_pruned["f1"] - m_v1["f1"]
    print(f"\n>>> VEREDICTO sobre criterio F1_pruned ≥ F1_v1 − 0.02:")
    if delta_f1_abs >= -0.02:
        print(f"    PODADO OK (ΔF1 abs = {delta_f1_abs:+.3f}). "
              f"Se puede adoptar como nuevo v1.")
        if delta_f1_abs > 0:
            print(f"    Además MEJORA el F1 en {delta_f1_abs:+.3f}. Bonus.")
    else:
        print(f"    PODADO EMPEORA demasiado (ΔF1 abs = {delta_f1_abs:+.3f}). "
              f"Mantener v1 original de 24 features.")


# ==========================================================
# FUNCIÓN 7 - Guardar resultados
# ==========================================================
def guardar_resultados(df_test, y_pred, y_proba):
    df = df_test.copy()
    df["model_pred"] = y_pred
    df["model_proba"] = y_proba
    df.to_csv(MODEL_RESULTS_FILE, index=False)
    print(f"\nResultados guardados: {MODEL_RESULTS_FILE}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print(f"Entrenando v1 PODADO ({len(FEATURES)} features)...\n")
    df_train, df_test = cargar_y_split()
    modelo = entrenar_modelo(df_train)

    y_pred, y_proba, metricas = evaluar_clasificacion(
        modelo, df_test, FEATURES, titulo="v1 PODADO (test)"
    )
    simular_con_filtro(df_test, y_pred, y_proba)
    mostrar_importancias(modelo, df_test)
    comparar_con_v1_original(df_test, metricas)
    guardar_resultados(df_test, y_pred, y_proba)

    print(f"\n{'='*60}")
    print("ENTRENAMIENTO V1 PODADO COMPLETADO")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
