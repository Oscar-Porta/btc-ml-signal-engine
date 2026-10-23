"""
==========================================================
FASE 12 (Iteración 4) - Walk-Forward Validation
==========================================================
Valida el modelo reentrenándolo cada 6 meses con ventana
expandible (siempre desde el inicio de los datos).

Diseño:
  - Fold 1: Train hasta 2022-12 → Predict 2023-01 a 2023-06
  - Fold 2: Train hasta 2023-06 → Predict 2023-07 a 2023-12
  - Fold 3: Train hasta 2023-12 → Predict 2024-01 a 2024-06
  - Fold 4: Train hasta 2024-06 → Predict 2024-07 a 2024-12
  - Fold 5: Train hasta 2024-12 → Predict 2025-01 a 2025-06
  - Fold 6: Train hasta 2025-06 → Predict 2025-07 a 2025-12
  - Fold 7: Train hasta 2025-12 → Predict 2026-01 a 2026-04

Cada fold entrena desde cero con TODOS los datos disponibles
hasta ese momento (ventana expandible, no deslizante).

Al final compara los resultados agregados walk-forward con
el split fijo de v3 (train ≤2024-12, test >2024-12).
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

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
OUTPUT_DIR = "outputs"
DATASET_FILE = os.path.join(OUTPUT_DIR, "ml_dataset.csv")

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

# Parámetros del modelo (idénticos a v3)
GBM_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=42,
)

# Definición de folds: (train_end, test_start, test_end)
# train siempre empieza desde el inicio de los datos
FOLDS = [
    ("2022-12-31", "2023-01-01", "2023-06-30"),
    ("2023-06-30", "2023-07-01", "2023-12-31"),
    ("2023-12-31", "2024-01-01", "2024-06-30"),
    ("2024-06-30", "2024-07-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2025-06-30"),
    ("2025-06-30", "2025-07-01", "2025-12-31"),
    ("2025-12-31", "2026-01-01", "2026-12-31"),  # hasta fin de datos
]


# ==========================================================
# FUNCIÓN 1 - Cargar datos
# ==========================================================
def cargar_datos():
    df = pd.read_csv(DATASET_FILE)
    df["open_time"] = pd.to_datetime(
        df["open_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")
    print(f"Dataset total: {len(df)} filas")
    print(f"Rango: {df['open_time'].min()} → {df['open_time'].max()}")
    print(f"Labels positivos: {df[LABEL].sum()} "
          f"({df[LABEL].mean()*100:.1f}%)\n")
    return df


# ==========================================================
# FUNCIÓN 2 - Ejecutar un fold
# ==========================================================
def ejecutar_fold(fold_num, df, train_end, test_start, test_end):
    """
    Entrena con datos ≤ train_end, predice test_start..test_end.
    Devuelve DataFrame de test con predicciones añadidas.
    """
    tz = df["open_time"].dt.tz

    t_end = pd.Timestamp(train_end, tz=tz)
    p_start = pd.Timestamp(test_start, tz=tz)
    p_end = pd.Timestamp(test_end, tz=tz)

    df_train = df[df["open_time"] <= t_end].copy()
    df_test = df[
        (df["open_time"] >= p_start) & (df["open_time"] <= p_end)
    ].copy()

    if len(df_test) == 0:
        print(f"  Fold {fold_num}: sin datos de test → saltando")
        return None

    X_train = df_train[FEATURES]
    y_train = df_train[LABEL]
    X_test = df_test[FEATURES]
    y_test = df_test[LABEL]

    # Entrenar
    modelo = GradientBoostingClassifier(**GBM_PARAMS)
    modelo.fit(X_train, y_train)

    # Predecir
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"  Fold {fold_num}: train={len(df_train):>4d} | "
          f"test={len(df_test):>3d} | "
          f"Acc={acc:.3f} Prec={prec:.3f} "
          f"Rec={rec:.3f} F1={f1:.3f}")

    # Guardar predicciones en df_test
    df_test = df_test.copy()
    df_test["y_proba_wf"] = y_proba
    df_test["y_pred_wf"] = y_pred
    df_test["fold"] = fold_num

    return df_test


# ==========================================================
# FUNCIÓN 3 - Simulación de trading
# ==========================================================
def simular_trading(df, proba_col, label=""):
    """
    Simula PnL a distintos umbrales usando la columna de
    probabilidades indicada. Devuelve lista de resultados.
    """
    resultados = []
    for umbral in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                    0.55, 0.60]:
        f = df[df[proba_col] >= umbral]
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

    if label:
        print(f"\n  {label}")
    print(f"  {'Umbral':>7s} | {'Trades':>6s} | {'WR':>5s} | "
          f"{'PnL':>11s} | {'PnL/t':>8s} | {'R medio':>7s}")
    print(f"  {'-'*55}")
    for r in resultados:
        print(f"  {r['umbral']:>7.2f} | {r['trades']:>6d} | "
              f"{r['wr']:>4.1f}% | {r['pnl']:>10,.0f} | "
              f"{r['pnl_trade']:>7,.0f} | {r['r_medio']:>7.2f}")

    return resultados


# ==========================================================
# FUNCIÓN 4 - Comparar walk-forward vs split fijo
# ==========================================================
def comparar_wf_vs_fijo(df_wf_all, df_original):
    """
    Compara las predicciones walk-forward agregadas con
    el modelo v3 de split fijo (train ≤2024-12).

    Para hacer la comparación justa, entrenamos el modelo
    fijo aquí mismo con el mismo código.
    """
    print(f"\n{'='*70}")
    print("COMPARACIÓN: WALK-FORWARD vs SPLIT FIJO (v3)")
    print(f"{'='*70}")

    # --- Modelo fijo: entrenar con datos ≤ 2024-12 ---
    tz = df_original["open_time"].dt.tz
    cutoff = pd.Timestamp("2024-12-31", tz=tz)

    df_train_fijo = df_original[df_original["open_time"] <= cutoff]
    df_test_fijo = df_original[df_original["open_time"] > cutoff]

    modelo_fijo = GradientBoostingClassifier(**GBM_PARAMS)
    modelo_fijo.fit(df_train_fijo[FEATURES], df_train_fijo[LABEL])
    y_proba_fijo = modelo_fijo.predict_proba(
        df_test_fijo[FEATURES]
    )[:, 1]

    df_test_fijo = df_test_fijo.copy()
    df_test_fijo["y_proba_fijo"] = y_proba_fijo

    # --- Walk-forward: solo los mismos meses que el test fijo ---
    # El test fijo es >2024-12, o sea 2025-01 en adelante
    # Walk-forward tiene folds 5,6,7 que cubren esos meses
    df_wf_post2024 = df_wf_all[df_wf_all["open_time"] > cutoff].copy()

    print(f"\nPeríodo de comparación: >2024-12-31")
    print(f"  Walk-forward (folds 5-7): {len(df_wf_post2024)} trades")
    print(f"  Split fijo (test v3):     {len(df_test_fijo)} trades")

    # --- Simulación lado a lado (mismo período) ---
    print(f"\n{'─'*60}")
    sim_wf = simular_trading(
        df_wf_post2024, "y_proba_wf",
        "WALK-FORWARD (reentrenado cada 6 meses):"
    )
    sim_fijo = simular_trading(
        df_test_fijo, "y_proba_fijo",
        "SPLIT FIJO (entrenado una vez con datos ≤2024-12):"
    )

    # --- Tabla comparativa a umbrales clave ---
    umbrales_clave = [0.30, 0.35, 0.40, 0.50]

    print(f"\n{'='*70}")
    print("TABLA COMPARATIVA (mismo período >2024-12)")
    print(f"{'='*70}")

    print(f"\n{'--- PnL total (USD) ---':>35s}")
    print(f"{'Umbral':>8s} | {'Walk-Forward':>14s} | "
          f"{'Split Fijo':>14s} | {'Diferencia':>12s}")
    print(f"  {'-'*55}")

    for u in umbrales_clave:
        wf_match = [s for s in sim_wf if s["umbral"] == u]
        fj_match = [s for s in sim_fijo if s["umbral"] == u]
        wf_pnl = wf_match[0]["pnl"] if wf_match else 0
        fj_pnl = fj_match[0]["pnl"] if fj_match else 0
        diff = wf_pnl - fj_pnl
        marca = " ★" if wf_pnl > fj_pnl else ""
        print(f"  {u:>6.2f} | {wf_pnl:>13,.0f} | "
              f"{fj_pnl:>13,.0f} | {diff:>+11,.0f}{marca}")

    print(f"\n{'--- Win Rate (%) ---':>35s}")
    print(f"{'Umbral':>8s} | {'Walk-Forward':>14s} | "
          f"{'Split Fijo':>14s} | {'Diferencia':>12s}")
    print(f"  {'-'*55}")

    for u in umbrales_clave:
        wf_match = [s for s in sim_wf if s["umbral"] == u]
        fj_match = [s for s in sim_fijo if s["umbral"] == u]
        wf_wr = wf_match[0]["wr"] if wf_match else 0
        fj_wr = fj_match[0]["wr"] if fj_match else 0
        diff = wf_wr - fj_wr
        marca = " ★" if wf_wr > fj_wr else ""
        print(f"  {u:>6.2f} | {wf_wr:>13.1f}% | "
              f"{fj_wr:>13.1f}% | {diff:>+10.1f}pp{marca}")

    print(f"\n{'--- Trades (n) ---':>35s}")
    print(f"{'Umbral':>8s} | {'Walk-Forward':>14s} | "
          f"{'Split Fijo':>14s}")
    print(f"  {'-'*40}")

    for u in umbrales_clave:
        wf_match = [s for s in sim_wf if s["umbral"] == u]
        fj_match = [s for s in sim_fijo if s["umbral"] == u]
        wf_n = wf_match[0]["trades"] if wf_match else 0
        fj_n = fj_match[0]["trades"] if fj_match else 0
        print(f"  {u:>6.2f} | {wf_n:>14d} | {fj_n:>14d}")

    # --- Walk-forward COMPLETO (todos los folds) ---
    print(f"\n{'='*70}")
    print("WALK-FORWARD COMPLETO (todos los folds, 2023-2026)")
    print(f"{'='*70}")
    print(f"Total predicciones out-of-sample: {len(df_wf_all)}")

    sim_wf_total = simular_trading(
        df_wf_all, "y_proba_wf",
        "WALK-FORWARD — todos los folds agregados:"
    )

    return sim_wf, sim_fijo, sim_wf_total


# ==========================================================
# FUNCIÓN 5 - Análisis por fold
# ==========================================================
def analisis_por_fold(df_wf_all):
    """
    Muestra el PnL y WR por fold para detectar si el modelo
    se degrada con el tiempo.
    """
    print(f"\n{'='*70}")
    print("ANÁLISIS POR FOLD (estabilidad temporal)")
    print(f"{'='*70}")

    umbral = 0.35  # sweet spot

    print(f"\n  Umbral: {umbral}")
    print(f"  {'Fold':>5s} | {'Período':>20s} | {'Trades':>6s} | "
          f"{'WR':>5s} | {'PnL':>11s} | {'PnL/t':>8s}")
    print(f"  {'-'*65}")

    for fold_num in sorted(df_wf_all["fold"].unique()):
        df_fold = df_wf_all[df_wf_all["fold"] == fold_num]
        f = df_fold[df_fold["y_proba_wf"] >= umbral]
        n = len(f)
        if n == 0:
            continue

        periodo_ini = f["open_time"].min().strftime("%Y-%m")
        periodo_fin = f["open_time"].max().strftime("%Y-%m")
        periodo = f"{periodo_ini} → {periodo_fin}"

        wr = (f["pnl_abs"] > 0).mean() * 100
        pnl = f["pnl_abs"].sum()
        pnl_m = f["pnl_abs"].mean()

        print(f"  {fold_num:>5d} | {periodo:>20s} | {n:>6d} | "
              f"{wr:>4.1f}% | {pnl:>10,.0f} | {pnl_m:>7,.0f}")

    # Total
    f_total = df_wf_all[df_wf_all["y_proba_wf"] >= umbral]
    n_total = len(f_total)
    wr_total = (f_total["pnl_abs"] > 0).mean() * 100
    pnl_total = f_total["pnl_abs"].sum()
    pnl_m_total = f_total["pnl_abs"].mean()
    print(f"  {'-'*65}")
    print(f"  {'TOTAL':>5s} | {'2023-01 → 2026-04':>20s} | "
          f"{n_total:>6d} | {wr_total:>4.1f}% | "
          f"{pnl_total:>10,.0f} | {pnl_m_total:>7,.0f}")


# ==========================================================
# FUNCIÓN 6 - Gráfico de equity curve
# ==========================================================
def grafico_equity(df_wf_all, umbral=0.35):
    """
    Genera gráfico de equity curve walk-forward.
    """
    df = df_wf_all[df_wf_all["y_proba_wf"] >= umbral].copy()
    df = df.sort_values("open_time")
    df["equity"] = df["pnl_abs"].cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Equity curve
    ax1 = axes[0]
    ax1.plot(df["open_time"], df["equity"], linewidth=1.2,
             color="steelblue")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title(f"Walk-Forward Equity Curve (umbral ≥ {umbral})",
                  fontsize=13)
    ax1.set_ylabel("PnL acumulado (USD)")
    ax1.grid(True, alpha=0.3)

    # Marcar fronteras de folds
    for fold_num in sorted(df["fold"].unique()):
        fold_start = df[df["fold"] == fold_num]["open_time"].min()
        ax1.axvline(fold_start, color="red", linestyle=":",
                    alpha=0.4, label=f"Fold {fold_num}" if fold_num <= 3 else "")

    # PnL por trade
    ax2 = axes[1]
    colors = ["green" if x > 0 else "red" for x in df["pnl_abs"]]
    ax2.bar(range(len(df)), df["pnl_abs"].values, color=colors,
            alpha=0.6, width=1.0)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("PnL por trade (walk-forward)")
    ax2.set_ylabel("PnL (USD)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "walk_forward_equity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nGráfico guardado: {path}")


# ==========================================================
# FUNCIÓN 7 - Veredicto
# ==========================================================
def veredicto(sim_wf, sim_fijo, sim_wf_total):
    """
    Emite un veredicto basado en los resultados.
    """
    print(f"\n{'='*70}")
    print("VEREDICTO — ITERACIÓN 4 (Walk-Forward)")
    print(f"{'='*70}")

    # Comparar PnL al umbral 0.35
    wf_035 = [s for s in sim_wf if s["umbral"] == 0.35]
    fj_035 = [s for s in sim_fijo if s["umbral"] == 0.35]

    wf_pnl = wf_035[0]["pnl"] if wf_035 else 0
    fj_pnl = fj_035[0]["pnl"] if fj_035 else 0
    wf_wr = wf_035[0]["wr"] if wf_035 else 0
    fj_wr = fj_035[0]["wr"] if fj_035 else 0

    print(f"\n  Comparación al umbral 0.35 (período >2024-12):")
    print(f"  Walk-Forward: PnL={wf_pnl:>10,.0f} | WR={wf_wr:.1f}%")
    print(f"  Split Fijo:   PnL={fj_pnl:>10,.0f} | WR={fj_wr:.1f}%")

    if wf_pnl > fj_pnl:
        diff_pct = ((wf_pnl - fj_pnl) / abs(fj_pnl) * 100
                    if fj_pnl != 0 else 0)
        print(f"\n  ✓ Walk-forward GANA por {wf_pnl - fj_pnl:,.0f} USD "
              f"({diff_pct:+.1f}%)")
        print(f"  → El reentrenamiento periódico MEJORA el modelo.")
        print(f"  → Recomendación: adoptar walk-forward en producción")
        print(f"    (reentrenar cada 6 meses con datos nuevos).")
    elif wf_pnl < fj_pnl:
        diff_pct = ((fj_pnl - wf_pnl) / abs(wf_pnl) * 100
                    if wf_pnl != 0 else 0)
        print(f"\n  ✗ Split fijo GANA por {fj_pnl - wf_pnl:,.0f} USD "
              f"({diff_pct:+.1f}%)")
        print(f"  → El modelo v3 con entrenamiento único es robusto.")
        print(f"  → El reentrenamiento no aporta mejora significativa.")
        print(f"  → Recomendación: mantener split fijo, pero")
        print(f"    monitorizar degradación y reentrenar si baja.")
    else:
        print(f"\n  ≈ Resultados equivalentes.")
        print(f"  → El modelo es estable. Mantener split fijo.")

    # Walk-forward total
    wf_total_035 = [s for s in sim_wf_total if s["umbral"] == 0.35]
    if wf_total_035:
        print(f"\n  Walk-forward COMPLETO (2023-2026):")
        print(f"  {wf_total_035[0]['trades']} trades | "
              f"WR={wf_total_035[0]['wr']:.1f}% | "
              f"PnL={wf_total_035[0]['pnl']:,.0f} USD | "
              f"PnL/trade={wf_total_035[0]['pnl_trade']:,.0f} USD")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print(f"{'='*70}")
    print("ITERACIÓN 4 — WALK-FORWARD VALIDATION")
    print("Ventana expandible, reentrenamiento cada 6 meses")
    print(f"{'='*70}\n")

    # 1. Cargar datos
    df = cargar_datos()

    # 2. Ejecutar cada fold
    print("Ejecutando folds walk-forward:")
    print(f"{'─'*60}")
    dfs_test = []
    for i, (train_end, test_start, test_end) in enumerate(FOLDS, 1):
        df_fold = ejecutar_fold(i, df, train_end, test_start, test_end)
        if df_fold is not None:
            dfs_test.append(df_fold)

    # 3. Concatenar todas las predicciones out-of-sample
    df_wf_all = pd.concat(dfs_test, ignore_index=True)
    print(f"\nTotal predicciones walk-forward: {len(df_wf_all)}")

    # 4. Análisis por fold
    analisis_por_fold(df_wf_all)

    # 5. Comparar con split fijo
    sim_wf, sim_fijo, sim_wf_total = comparar_wf_vs_fijo(
        df_wf_all, df
    )

    # 6. Gráfico
    grafico_equity(df_wf_all, umbral=0.35)

    # 7. Veredicto
    veredicto(sim_wf, sim_fijo, sim_wf_total)

    print(f"\n{'='*70}")
    print("WALK-FORWARD VALIDATION COMPLETADA")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
