"""
==========================================================
FASE 11 (Iteración 3) - Comparación de algoritmos
==========================================================
Mismo dataset, mismas features, mismo label (R≥0.5), mismo
split temporal. El ÚNICO cambio es el algoritmo.

Candidatos:
  1. GradientBoostingClassifier (sklearn) — el actual (v3)
  2. XGBClassifier (xgboost)
  3. LGBMClassifier (lightgbm)

La comparación se hace por SIMULACIÓN DE TRADING (PnL real),
no solo por métricas de clasificación.

Al final se muestra una tabla resumen y un veredicto claro.
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
    f1_score, confusion_matrix
)
import joblib

# Intentar importar XGBoost y LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost no instalado. Saltando XGBoost.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARN] lightgbm no instalado. Saltando LightGBM.")


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
OUTPUT_DIR = "outputs"
DATASET_FILE = os.path.join(OUTPUT_DIR, "ml_dataset.csv")
TRAIN_CUTOFF = "2024-12-31"

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

LABEL = "label"  # R >= 0.5


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

    print(f"Dataset: {len(df)} filas")
    print(f"Train: {len(df_train)} | Test: {len(df_test)}")
    print(f"Label (R≥0.5): train {df_train[LABEL].sum()} gan. "
          f"({df_train[LABEL].mean()*100:.1f}%) | "
          f"test {df_test[LABEL].sum()} gan. "
          f"({df_test[LABEL].mean()*100:.1f}%)")
    return df_train, df_test


# ==========================================================
# FUNCIÓN 2 - Definir modelos candidatos
# ==========================================================
def definir_modelos():
    """
    Define los tres modelos con parámetros comparables.

    Los parámetros se eligen para que sean equivalentes
    entre los tres algoritmos:
      - 200 árboles
      - Profundidad máxima 4
      - Learning rate 0.05
      - Subsample 0.8
      - Regularización moderada

    Así la comparación es sobre el ALGORITMO, no sobre
    los parámetros.
    """
    modelos = {}

    # 1. GradientBoosting (sklearn) — línea base actual
    modelos["GBM_sklearn"] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )

    # 2. XGBoost
    if HAS_XGB:
        modelos["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=20,     # ~ min_samples_leaf
            subsample=0.8,
            colsample_bytree=0.8,    # regularización extra
            reg_alpha=0.1,           # L1 regularización
            reg_lambda=1.0,          # L2 regularización
            random_state=42,
            eval_metric="logloss",
            verbosity=0,             # silenciar warnings
        )

    # 3. LightGBM
    if HAS_LGBM:
        modelos["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_child_samples=20,    # ~ min_samples_leaf
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,              # silenciar output
        )

    return modelos


# ==========================================================
# FUNCIÓN 3 - Entrenar y evaluar un modelo
# ==========================================================
def entrenar_y_evaluar(nombre, modelo, X_train, y_train,
                       X_test, y_test, df_test):
    """
    Entrena un modelo y devuelve sus métricas + simulación.
    """
    print(f"\n{'─'*50}")
    print(f"  {nombre}")
    print(f"{'─'*50}")

    # Entrenar
    modelo.fit(X_train, y_train)

    # Predecir
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    # Métricas de clasificación
    metricas = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred,
                                      zero_division=0),
        "recall":    recall_score(y_test, y_pred,
                                   zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"  Accuracy: {metricas['accuracy']:.3f}  "
          f"Precision: {metricas['precision']:.3f}  "
          f"Recall: {metricas['recall']:.3f}  "
          f"F1: {metricas['f1']:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusión: TN={cm[0][0]} FP={cm[0][1]} "
          f"FN={cm[1][0]} TP={cm[1][1]}")

    # Simulación de trading
    sim = simular_trading(df_test, y_proba)

    # Importancia de features
    if hasattr(modelo, "feature_importances_"):
        importances = modelo.feature_importances_
        top5_idx = np.argsort(importances)[::-1][:5]
        print(f"  Top 5 features: ", end="")
        print(", ".join(
            f"{FEATURES[i]}({importances[i]:.3f})"
            for i in top5_idx
        ))

    return {
        "nombre": nombre,
        "modelo": modelo,
        "metricas": metricas,
        "simulacion": sim,
        "y_proba": y_proba,
    }


# ==========================================================
# FUNCIÓN 4 - Simulación de trading
# ==========================================================
def simular_trading(df_test, y_proba):
    """
    Simula PnL real a distintos umbrales.
    Devuelve lista de diccionarios con resultados.
    """
    df = df_test.copy()
    df["y_proba"] = y_proba

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
        r_m = f["r_multiple"].mean() if "r_multiple" in f.columns else 0
        resultados.append({
            "umbral": umbral, "trades": n, "wr": wr,
            "pnl": pnl, "pnl_trade": pnl_m, "r_medio": r_m,
        })

    # Mostrar tabla
    print(f"  {'Umbral':>7s} | {'Trades':>6s} | {'WR':>5s} | "
          f"{'PnL':>11s} | {'PnL/t':>8s}")
    print(f"  {'-'*45}")
    for r in resultados:
        print(f"  {r['umbral']:>7.2f} | {r['trades']:>6d} | "
              f"{r['wr']:>4.1f}% | {r['pnl']:>10,.0f} | "
              f"{r['pnl_trade']:>7,.0f}")

    return resultados


# ==========================================================
# FUNCIÓN 5 - Tabla comparativa final
# ==========================================================
def tabla_comparativa(resultados_todos):
    """
    Muestra la tabla definitiva lado a lado.
    """
    nombres = [r["nombre"] for r in resultados_todos]

    print(f"\n{'='*70}")
    print("TABLA COMPARATIVA FINAL")
    print(f"{'='*70}")

    # --- Métricas de clasificación ---
    print(f"\n{'Métrica':>12s}", end="")
    for n in nombres:
        print(f" | {n:>14s}", end="")
    print()
    print("-" * (14 + 17 * len(nombres)))

    for m in ["accuracy", "precision", "recall", "f1"]:
        print(f"{m:>12s}", end="")
        valores = []
        for r in resultados_todos:
            v = r["metricas"][m]
            valores.append(v)
        mejor = max(valores)
        for v in valores:
            marca = " ★" if v == mejor and valores.count(v) == 1 else ""
            print(f" | {v:>12.3f}{marca}", end="")
        print()

    # --- Simulación de trading a umbrales clave ---
    umbrales_clave = [0.30, 0.35, 0.40, 0.50]

    print(f"\n{'--- PnL total (USD) ---':>30s}")
    print(f"{'Umbral':>12s}", end="")
    for n in nombres:
        print(f" | {n:>14s}", end="")
    print()
    print("-" * (14 + 17 * len(nombres)))

    for u in umbrales_clave:
        print(f"{u:>12.2f}", end="")
        valores = []
        for r in resultados_todos:
            sim = r["simulacion"]
            match = [s for s in sim if s["umbral"] == u]
            if match:
                v = match[0]["pnl"]
            else:
                v = 0
            valores.append(v)
        mejor = max(valores)
        for v in valores:
            marca = " ★" if v == mejor and valores.count(v) == 1 else ""
            print(f" | {v:>12,.0f}{marca}", end="")
        print()

    # --- PnL/trade ---
    print(f"\n{'--- PnL/trade (USD) ---':>30s}")
    print(f"{'Umbral':>12s}", end="")
    for n in nombres:
        print(f" | {n:>14s}", end="")
    print()
    print("-" * (14 + 17 * len(nombres)))

    for u in umbrales_clave:
        print(f"{u:>12.2f}", end="")
        valores = []
        for r in resultados_todos:
            sim = r["simulacion"]
            match = [s for s in sim if s["umbral"] == u]
            if match:
                v = match[0]["pnl_trade"]
            else:
                v = 0
            valores.append(v)
        mejor = max(valores)
        for v in valores:
            marca = " ★" if v == mejor and valores.count(v) == 1 else ""
            print(f" | {v:>12,.0f}{marca}", end="")
        print()

    # --- Win Rate ---
    print(f"\n{'--- Win Rate (%) ---':>30s}")
    print(f"{'Umbral':>12s}", end="")
    for n in nombres:
        print(f" | {n:>14s}", end="")
    print()
    print("-" * (14 + 17 * len(nombres)))

    for u in umbrales_clave:
        print(f"{u:>12.2f}", end="")
        valores = []
        for r in resultados_todos:
            sim = r["simulacion"]
            match = [s for s in sim if s["umbral"] == u]
            if match:
                v = match[0]["wr"]
            else:
                v = 0
            valores.append(v)
        mejor = max(valores)
        for v in valores:
            marca = " ★" if v == mejor and valores.count(v) == 1 else ""
            print(f" | {v:>11.1f}%{marca}", end="")
        print()

    # --- Veredicto automático ---
    print(f"\n{'='*70}")
    print("VEREDICTO")
    print(f"{'='*70}")

    # Criterio: mejor PnL total al umbral 0.35 (sweet spot)
    pnl_035 = {}
    for r in resultados_todos:
        sim = r["simulacion"]
        match = [s for s in sim if s["umbral"] == 0.35]
        pnl_035[r["nombre"]] = match[0]["pnl"] if match else 0

    ganador = max(pnl_035, key=pnl_035.get)
    print(f"\nMejor PnL al umbral 0.35 (sweet spot): "
          f"{ganador} con {pnl_035[ganador]:,.0f} USD")

    # Criterio secundario: mejor PnL medio ponderado
    # (promedio de PnL a los 4 umbrales clave)
    pnl_medio = {}
    for r in resultados_todos:
        sim = r["simulacion"]
        pnls = [s["pnl"] for s in sim
                if s["umbral"] in umbrales_clave]
        pnl_medio[r["nombre"]] = np.mean(pnls) if pnls else 0

    ganador_medio = max(pnl_medio, key=pnl_medio.get)
    print(f"Mejor PnL medio (4 umbrales): "
          f"{ganador_medio} con {pnl_medio[ganador_medio]:,.0f} USD")

    for nombre in pnl_035:
        print(f"  {nombre}: "
              f"PnL@0.35={pnl_035[nombre]:>10,.0f} | "
              f"PnL_medio={pnl_medio[nombre]:>10,.0f}")

    return ganador, pnl_035


# ==========================================================
# FUNCIÓN 6 - Guardar modelo ganador
# ==========================================================
def guardar_ganador(resultados_todos, ganador_nombre):
    """
    Guarda el modelo ganador con sufijo _comparison.
    """
    for r in resultados_todos:
        if r["nombre"] == ganador_nombre:
            path = os.path.join(
                OUTPUT_DIR,
                f"model_iter3_{ganador_nombre.lower()}.joblib"
            )
            joblib.dump(r["modelo"], path)
            print(f"\nModelo ganador guardado: {path}")
            return path
    return None


# ==========================================================
# MAIN
# ==========================================================
def main():
    print(f"{'='*70}")
    print("ITERACIÓN 3 — COMPARACIÓN DE ALGORITMOS")
    print("GradientBoosting vs XGBoost vs LightGBM")
    print(f"{'='*70}\n")

    # 1. Cargar datos
    df_train, df_test = cargar_y_split()
    X_train = df_train[FEATURES]
    y_train = df_train[LABEL]
    X_test = df_test[FEATURES]
    y_test = df_test[LABEL]

    # 2. Definir modelos
    modelos = definir_modelos()
    print(f"\nModelos a comparar: {list(modelos.keys())}")

    # 3. Entrenar y evaluar cada uno
    resultados_todos = []
    for nombre, modelo in modelos.items():
        resultado = entrenar_y_evaluar(
            nombre, modelo, X_train, y_train,
            X_test, y_test, df_test
        )
        resultados_todos.append(resultado)

    # 4. Tabla comparativa
    ganador, pnl_035 = tabla_comparativa(resultados_todos)

    # 5. Guardar modelo ganador
    guardar_ganador(resultados_todos, ganador)

    print(f"\n{'='*70}")
    print("COMPARACIÓN COMPLETADA")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
