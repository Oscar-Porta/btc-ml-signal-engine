"""
==========================================================
FASE 10 (Iteración 2) - Entrenamiento del modelo V3
==========================================================
Cambio principal respecto al v1 podado:
  - Label basado en R-multiple >= 0.5 (en lugar de pnl > 0).
  - Mismas 20 features del v1 podado (no se añaden nuevas).

La hipótesis es que al darle al modelo una definición más
limpia de "ganador" (excluir time_exits que ganan centavos
y breakeven), aprenderá patrones más claros.

Criterio de éxito:
  - Comparamos la SIMULACIÓN DE TRADING (PnL real), no solo
    métricas de clasificación, porque el label ha cambiado
    y el F1 no es directamente comparable entre labels.
  - El v3 gana si, para algún umbral razonable, genera más
    PnL que el v1 podado con el mismo rango de trades.

Salidas:
  - outputs/model_v3.joblib
  - outputs/model_results_v3.csv
  - outputs/feature_importance_v3.png
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
DATASET_FILE = os.path.join(OUTPUT_DIR, "ml_dataset.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "model_v3.joblib")
MODEL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "model_results_v3.csv")
IMPORTANCE_CHART = os.path.join(OUTPUT_DIR, "feature_importance_v3.png")

# Modelo v1 podado (producción actual) para comparación.
V1_MODEL_FILE = os.path.join(OUTPUT_DIR, "model.joblib")

# Mismo split temporal que todos los modelos anteriores.
TRAIN_CUTOFF = "2024-12-31"

# Mismas 20 features que el v1 podado.
FEATURES = [
    "rsi_14", "atr_14",
    "dist_ema20_pct", "dist_ema50_pct",
    "ema_spread_pct", "ema_spread_4h_pct",
    "atr_pct", "price_position",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "hour", "day_of_week",
    "volume_sma20_ratio", "rsi_14_4h", "atr_pct_4h",
    "roc_5", "roc_20", "atr_regime", "candle_range_ratio",
]

# Label v3 (R-multiple >= 0.5) y label v1 (pnl > 0).
LABEL = "label"        # v3: R >= 0.5
LABEL_V1 = "label_v1"  # v1 original: pnl > 0


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

    # Mostrar distribución del label v3 vs v1
    for name, label_col in [("v3 (R≥0.5)", LABEL),
                             ("v1 (pnl>0)", LABEL_V1)]:
        n_train = df_train[label_col].sum()
        n_test = df_test[label_col].sum()
        print(f"  {name}: train {n_train} gan. "
              f"({n_train/len(df_train)*100:.1f}%) | "
              f"test {n_test} gan. "
              f"({n_test/len(df_test)*100:.1f}%)")

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
    print(f"\nEntrenando modelo v3 ({len(FEATURES)} features, "
          f"label=R≥0.5)...")
    modelo.fit(X_train, y_train)
    joblib.dump(modelo, MODEL_FILE)
    print(f"Modelo guardado: {MODEL_FILE}")
    return modelo


# ==========================================================
# FUNCIÓN 3 - Evaluar clasificación
# ==========================================================
def evaluar_clasificacion(modelo, df_test, features, label_col,
                          titulo):
    X_test = df_test[features]
    y_test = df_test[label_col]
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

    return y_pred, y_proba, metricas


# ==========================================================
# FUNCIÓN 4 - Simulación de trading (métrica REAL)
# ==========================================================
def simular_con_filtro(df_test, y_proba, titulo):
    """
    Esta es la métrica que realmente importa: PnL real
    filtrando trades por probabilidad del modelo.

    Usa pnl_abs (USD reales), NO el label, porque el label
    es lo que cambia entre v1 y v3 y no sería comparable.
    """
    df = df_test.copy()
    df["y_proba"] = y_proba

    print(f"\n{'='*60}")
    print(f"SIMULACIÓN DE TRADING — {titulo}")
    print(f"{'='*60}")

    pnl_sin = df["pnl_abs"].sum()
    trades_sin = len(df)
    wr_sin = (df["pnl_abs"] > 0).mean() * 100
    print(f"\nSIN filtro:")
    print(f"  Trades: {trades_sin} | Win rate: {wr_sin:.1f}% | "
          f"PnL: {pnl_sin:,.2f} USD")

    print(f"\nCON filtro:")
    print(f"{'Umbral':>8s} | {'Trades':>7s} | {'WR':>6s} | "
          f"{'PnL':>12s} | {'PnL/trade':>10s} | "
          f"{'R medio':>8s}")
    print("-" * 70)

    resultados = []
    for umbral in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                    0.55, 0.60]:
        f = df[df["y_proba"] >= umbral]
        n = len(f)
        if n == 0:
            continue
        wr = (f["pnl_abs"] > 0).mean() * 100
        pnl = f["pnl_abs"].sum()
        pnl_m = f["pnl_abs"].mean()
        r_m = f["r_multiple"].mean()
        print(f"{umbral:>8.2f} | {n:>7d} | {wr:>5.1f}% | "
              f"{pnl:>11,.2f} | {pnl_m:>9,.2f} | "
              f"{r_m:>+7.3f}")
        resultados.append({
            "umbral": umbral, "trades": n, "wr": wr,
            "pnl": pnl, "pnl_trade": pnl_m, "r_medio": r_m
        })

    return resultados


# ==========================================================
# FUNCIÓN 5 - Importancia de features
# ==========================================================
def mostrar_importancias(modelo, df_test):
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n{'='*60}")
    print("IMPORTANCIA NATIVA (v3)")
    print(f"{'='*60}")
    for i, idx in enumerate(indices):
        print(f"  {i+1:>2}. {FEATURES[idx]:>25s}: "
              f"{importances[idx]:.4f}")

    # Permutation importance con el label v3.
    print(f"\n{'='*60}")
    print("PERMUTATION IMPORTANCE (F1, label v3)")
    print(f"{'='*60}")
    result = permutation_importance(
        modelo, df_test[FEATURES], df_test[LABEL],
        n_repeats=10, random_state=42, scoring="f1",
        n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    print(f"{'Feature':>25s} | {'Δ F1':>8s} | {'std':>6s}")
    print("-" * 48)
    for idx in order:
        print(f"{FEATURES[idx]:>25s} | "
              f"{result.importances_mean[idx]:>+8.4f} | "
              f"{result.importances_std[idx]:>6.4f}")

    # Gráfico.
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_idx = np.argsort(importances)
    ax.barh(
        [FEATURES[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#2196F3"
    )
    ax.set_xlabel("Importancia")
    ax.set_title("Feature Importance v3 (label R≥0.5)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGráfico: {IMPORTANCE_CHART}")


# ==========================================================
# FUNCIÓN 6 - Comparativa directa v1 podado vs v3
# ==========================================================
def comparar_con_v1(df_test, proba_v3, metricas_v3):
    """
    Comparación justa: ambos modelos evaluados sobre el MISMO
    test set, usando la MISMA métrica de trading (PnL real).

    Las métricas de clasificación (F1, precision, recall) NO
    son directamente comparables porque los labels son
    distintos. Por eso la comparación principal es la
    simulación de trading.
    """
    if not os.path.exists(V1_MODEL_FILE):
        print(f"\n[INFO] No se encontró {V1_MODEL_FILE}.")
        return

    print(f"\n{'='*60}")
    print("COMPARATIVA DIRECTA: v1 podado vs v3")
    print(f"{'='*60}")

    try:
        modelo_v1 = joblib.load(V1_MODEL_FILE)
    except Exception as e:
        print(f"\n[WARN] No se pudo cargar el modelo v1 "
              f"(posible incompatibilidad de versión sklearn):")
        print(f"  {e}")
        print(f"\nComparación manual con resultados documentados "
              f"del v1 podado (presentation_notes.md).")
        return
    proba_v1 = modelo_v1.predict_proba(df_test[FEATURES])[:, 1]

    # Tabla comparativa de PnL a cada umbral.
    print(f"\n{'Umbral':>8s} | {'--- v1 podado ---':>20s} | "
          f"{'--- v3 (R≥0.5) ---':>20s} | {'Δ PnL':>12s}")
    print(f"{'':>8s} | {'Trades':>7s} {'PnL':>12s} | "
          f"{'Trades':>7s} {'PnL':>12s} | {'':>12s}")
    print("-" * 75)

    for umbral in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        # v1 podado
        f1 = df_test[proba_v1 >= umbral]
        n1 = len(f1)
        pnl1 = f1["pnl_abs"].sum() if n1 > 0 else 0

        # v3
        f3 = df_test[proba_v3 >= umbral]
        n3 = len(f3)
        pnl3 = f3["pnl_abs"].sum() if n3 > 0 else 0

        delta = pnl3 - pnl1
        marker = " ✓" if delta > 0 else ""
        print(f"{umbral:>8.2f} | {n1:>7d} {pnl1:>11,.0f} | "
              f"{n3:>7d} {pnl3:>11,.0f} | "
              f"{delta:>+11,.0f}{marker}")

    # Resumen rápido de métricas de clasificación (informativo,
    # NO decisivo porque los labels son distintos).
    print(f"\nMétricas de clasificación (informativas):")
    y_pred_v1 = modelo_v1.predict(df_test[FEATURES])
    m_v1 = {
        "f1": f1_score(df_test[LABEL_V1], y_pred_v1,
                       zero_division=0),
        "precision": precision_score(df_test[LABEL_V1], y_pred_v1,
                                      zero_division=0),
        "recall": recall_score(df_test[LABEL_V1], y_pred_v1,
                                zero_division=0),
    }
    print(f"  v1 podado (label pnl>0):  F1={m_v1['f1']:.3f}  "
          f"Prec={m_v1['precision']:.3f}  "
          f"Rec={m_v1['recall']:.3f}")
    print(f"  v3 (label R≥0.5):         F1={metricas_v3['f1']:.3f}  "
          f"Prec={metricas_v3['precision']:.3f}  "
          f"Rec={metricas_v3['recall']:.3f}")
    print(f"  (Labels distintos → F1 no comparable directamente)")


# ==========================================================
# FUNCIÓN 7 - Análisis de MAE por probabilidad
# ==========================================================
def analizar_mae(df_test, y_proba):
    """
    Analiza si el modelo v3 selecciona trades con mejor MAE
    (menos sufrimiento durante el trade). Esto es un bonus:
    si el modelo filtra trades que además de ganar, ganan
    de forma más limpia, es un indicador de calidad.
    """
    df = df_test.copy()
    df["y_proba"] = y_proba

    print(f"\n{'='*60}")
    print("ANÁLISIS MAE POR UMBRAL DE PROBABILIDAD")
    print(f"{'='*60}")
    print(f"{'Umbral':>8s} | {'Trades':>7s} | {'MAE medio':>10s} | "
          f"{'MAE ganad.':>11s} | {'MAE perd.':>10s}")
    print("-" * 60)

    for umbral in [0.0, 0.30, 0.40, 0.50, 0.55]:
        f = df[df["y_proba"] >= umbral]
        if len(f) == 0:
            continue
        mae_all = f["mae_r"].mean()
        gan = f[f["pnl_abs"] > 0]
        per = f[f["pnl_abs"] <= 0]
        mae_g = gan["mae_r"].mean() if len(gan) > 0 else 0
        mae_p = per["mae_r"].mean() if len(per) > 0 else 0
        print(f"{umbral:>8.2f} | {len(f):>7d} | {mae_all:>+9.3f}R | "
              f"{mae_g:>+10.3f}R | {mae_p:>+9.3f}R")


# ==========================================================
# FUNCIÓN 8 - Guardar resultados
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
    print(f"{'='*60}")
    print("ITERACIÓN 2 — MODELO V3 (label R-multiple ≥ 0.5)")
    print(f"{'='*60}\n")

    # 1. Cargar y split
    df_train, df_test = cargar_y_split()

    # 2. Entrenar v3
    modelo = entrenar_modelo(df_train)

    # 3. Métricas de clasificación (contra su propio label v3)
    y_pred, y_proba, metricas = evaluar_clasificacion(
        modelo, df_test, FEATURES, LABEL,
        titulo="v3 (label R≥0.5, test set)"
    )

    # 4. Simulación de trading (la métrica REAL)
    simular_con_filtro(df_test, y_proba, titulo="v3")

    # 5. Importancias
    mostrar_importancias(modelo, df_test)

    # 6. Comparativa directa con v1 podado
    comparar_con_v1(df_test, y_proba, metricas)

    # 7. Análisis de MAE
    analizar_mae(df_test, y_proba)

    # 8. Guardar
    guardar_resultados(df_test, y_pred, y_proba)

    print(f"\n{'='*60}")
    print("ENTRENAMIENTO V3 COMPLETADO")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
