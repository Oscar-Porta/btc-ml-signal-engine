"""
==========================================================
FASE 9 (Iteración 1B) - Entrenamiento del modelo V2
==========================================================
Copia de 07_train_model.py (v1) con cambios MÍNIMOS:

  1. Lee el dataset nuevo: outputs/ml_dataset_v2.csv
     (generado por 06_prepare_dataset_v2.py).
  2. La lista FEATURES incluye las 4 features externas:
     eth_roc_5, eth_roc_20, btc_eth_corr_20, dxy_roc_5d.
  3. Guarda el modelo como outputs/model_v2.joblib
     (NO pisa model.joblib del v1).
  4. Guarda resultados como outputs/model_results_v2.csv.
  5. Gráfico de importancia en outputs/feature_importance_v2.png.

Extras exclusivos de esta versión (respecto al v1):

  A. Permutation importance (scikit-learn).
     Calcula importancia "real": para cada feature, baraja
     sus valores en el test set y mide cuánto baja el F1.
     Es más fiable que feature_importances_ del modelo,
     que puede estar sesgada hacia features con más
     cardinalidad. Útil para decidir qué podar.

  B. Matriz de correlaciones entre features.
     Detecta pares con correlación >0.9 (redundantes).
     Candidatos a eliminar una de las dos.

  C. Tabla comparativa final v1 vs v2.
     Lee el modelo v1 (si existe) y reevalúa ambos sobre
     EL MISMO test set del v2. Imprime diferencias F1,
     precision, recall.

Criterio de decisión (acordado):
  - Si F1 v2 >= F1 v1 + 5%  → v2 gana, adoptamos v2.
  - Si no                    → nos quedamos con v1 y
    documentamos que las features externas no compensan.
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
DATASET_FILE = os.path.join(OUTPUT_DIR, "ml_dataset_v2.csv")          # v2
MODEL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "model_results_v2.csv")  # v2
MODEL_FILE = os.path.join(OUTPUT_DIR, "model_v2.joblib")               # v2
IMPORTANCE_CHART = os.path.join(OUTPUT_DIR, "feature_importance_v2.png")
PERM_IMPORTANCE_CHART = os.path.join(OUTPUT_DIR, "perm_importance_v2.png")
CORR_CHART = os.path.join(OUTPUT_DIR, "feature_correlation_v2.png")

# Modelo v1 para comparación final (si existe).
V1_MODEL_FILE = os.path.join(OUTPUT_DIR, "model.joblib")

# Mismo split temporal que v1 para comparación justa.
TRAIN_CUTOFF = "2024-12-31"

# Features (las 24 del v1 + las 4 externas = 28).
FEATURES = [
    # Indicadores base
    "rsi_14", "atr_14",
    # Iteración 0
    "dist_ema20_pct", "dist_ema50_pct",
    "ema_spread_pct", "ema_spread_4h_pct",
    "atr_pct", "price_position",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "hour", "day_of_week",
    "risk_pct", "reward_risk_ratio",
    "direction_num", "rule_num",
    # Iteración 1
    "volume_sma20_ratio", "rsi_14_4h", "atr_pct_4h",
    "roc_5", "roc_20", "atr_regime", "candle_range_ratio",
    # --- Iteración 1B (NUEVAS) ---
    "eth_roc_5", "eth_roc_20",
    "btc_eth_corr_20", "dxy_roc_5d",
]

# Features v1 (para poder evaluar el modelo v1 sobre el mismo test).
FEATURES_V1 = [f for f in FEATURES if f not in (
    "eth_roc_5", "eth_roc_20", "btc_eth_corr_20", "dxy_roc_5d"
)]

LABEL = "label"


# ==========================================================
# FUNCIÓN 1 - Cargar dataset y split temporal
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
    print(f"Train (hasta {TRAIN_CUTOFF}): {len(df_train)} filas")
    print(f"Test  (después {TRAIN_CUTOFF}): {len(df_test)} filas")
    print(f"\nTrain label: {df_train[LABEL].value_counts().to_dict()}")
    print(f"Test  label: {df_test[LABEL].value_counts().to_dict()}")
    return df_train, df_test


# ==========================================================
# FUNCIÓN 2 - Entrenar (mismos hiperparámetros que v1)
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

    print("\nEntrenando modelo v2...")
    modelo.fit(X_train, y_train)
    print("Modelo entrenado.")

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
    print(f"MÉTRICAS DE CLASIFICACIÓN — {titulo}")
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

    print("\nReporte completo:")
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
    print("SIMULACIÓN DE TRADING: con y sin filtro IA (v2)")
    print(f"{'='*60}")

    pnl_sin = df["pnl_abs"].sum()
    trades_sin = len(df)
    wr_sin = (df["pnl_abs"] > 0).mean() * 100

    print(f"\nSIN filtro IA:")
    print(f"  Trades: {trades_sin}")
    print(f"  Win rate: {wr_sin:.1f}%")
    print(f"  PnL total: {pnl_sin:,.2f} USD")

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
            "umbral": umbral, "trades": n, "win_rate": wr,
            "pnl_total": pnl, "pnl_medio": pnl_medio,
            "pct_filtradas": pct_filtradas
        })
    return resultados


# ==========================================================
# FUNCIÓN 5 - Feature importance nativa (del modelo)
# ==========================================================
def mostrar_feature_importance_nativa(modelo):
    """
    Importancia 'interna' de GradientBoosting: mide cuántas
    veces y con cuánto 'gain' se usa cada feature para hacer
    los splits en los árboles.

    LIMITACIÓN: esta métrica tiende a inflar features con
    muchos valores posibles (alta cardinalidad), porque dan
    más oportunidades de split. Por eso complementamos con
    permutation importance en la siguiente función.
    """
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n{'='*60}")
    print("IMPORTANCIA DE FEATURES (nativa del modelo)")
    print(f"{'='*60}")
    for i, idx in enumerate(indices):
        print(f"  {i+1:>2}. {FEATURES[idx]:>25s}: {importances[idx]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 7))
    sorted_idx = np.argsort(importances)
    ax.barh(
        [FEATURES[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#2196F3"
    )
    ax.set_xlabel("Importancia")
    ax.set_title("Feature Importance v2 — Gradient Boosting (nativa)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGráfico guardado: {IMPORTANCE_CHART}")


# ==========================================================
# FUNCIÓN 6 - Permutation importance (la buena)
# ==========================================================
def mostrar_permutation_importance(modelo, df_test):
    """
    Permutation importance (Breiman, 2001):

    Procedimiento:
      1. Evalúa el modelo en el test set → F1 base.
      2. Para cada feature f:
         a. Baraja aleatoriamente la columna f (rompe la
            relación entre f y el target, pero mantiene su
            distribución).
         b. Reevalúa el modelo → F1 "barajado".
         c. Importancia(f) = F1_base − F1_barajado.
      3. Repite n_repeats veces con semillas distintas y
         reporta media ± desviación estándar.

    Ventajas sobre la importancia nativa:
      - Mide impacto REAL en el rendimiento predictivo.
      - No sesga hacia features de alta cardinalidad.
      - Detecta features inútiles como 0 (o incluso negativas
        si su ruido empeora el modelo).

    Coste:
      - Más lenta: entrena 0 veces extra, pero reevalúa
        N_FEATURES * N_REPEATS veces. En nuestro caso
        28 * 10 = 280 predicciones sobre ~900 filas de test,
        muy asumible.
    """
    X_test = df_test[FEATURES]
    y_test = df_test[LABEL]

    print(f"\n{'='*60}")
    print("PERMUTATION IMPORTANCE (F1 score, test set)")
    print(f"{'='*60}")
    print("Calculando (n_repeats=10, puede tardar unos segundos)...")

    result = permutation_importance(
        modelo, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1",
        n_jobs=-1,
    )

    # Ordenar por media descendente.
    means = result.importances_mean
    stds = result.importances_std
    order = np.argsort(means)[::-1]

    print(f"\n{'Feature':>25s} | {'Δ F1 medio':>10s} | {'± std':>7s} | "
          f"{'Veredicto':>12s}")
    print("-" * 65)
    for idx in order:
        m, s = means[idx], stds[idx]
        # Veredicto heurístico.
        if m <= 0:
            veredicto = "INÚTIL o RUIDO"
        elif m < 0.005:
            veredicto = "marginal"
        elif m < 0.02:
            veredicto = "útil"
        else:
            veredicto = "MUY útil"
        print(f"{FEATURES[idx]:>25s} | {m:>+10.4f} | {s:>7.4f} | "
              f"{veredicto:>12s}")

    # Gráfico.
    fig, ax = plt.subplots(figsize=(10, 7))
    sorted_idx = np.argsort(means)
    ax.barh(
        [FEATURES[i] for i in sorted_idx],
        means[sorted_idx],
        xerr=stds[sorted_idx],
        color="#4CAF50",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ F1 al barajar la feature (media ± std, 10 repeats)")
    ax.set_title("Permutation Importance v2 — impacto real en F1",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(PERM_IMPORTANCE_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGráfico guardado: {PERM_IMPORTANCE_CHART}")

    # Candidatas a poda: importancia <= 0 o muy pequeña con std alto.
    candidatas_poda = [FEATURES[i] for i in range(len(FEATURES))
                       if means[i] <= 0.001]
    print(f"\nCandidatas a PODAR (Δ F1 ≤ 0.001): {len(candidatas_poda)}")
    for f in candidatas_poda:
        print(f"  - {f}")

    return result, candidatas_poda


# ==========================================================
# FUNCIÓN 7 - Matriz de correlaciones entre features
# ==========================================================
def analizar_correlaciones(df_train):
    """
    Detecta pares de features con correlación >0.9 (Pearson).
    Si dos features son casi iguales, el modelo no gana nada
    usando las dos: una sobra y sólo mete ruido.

    Criterio estándar: |corr| > 0.9 → sospechoso, candidato
    a mantener solo una de las dos.
    """
    print(f"\n{'='*60}")
    print("MATRIZ DE CORRELACIONES (features, train set)")
    print(f"{'='*60}")

    corr = df_train[FEATURES].corr()

    # Pares con |corr| > 0.9 (excluyendo la diagonal).
    pares = []
    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            c = corr.iloc[i, j]
            if abs(c) > 0.9:
                pares.append((FEATURES[i], FEATURES[j], c))

    if pares:
        print(f"\nPares con |correlación| > 0.9 (candidatos a consolidar):")
        for a, b, c in sorted(pares, key=lambda t: -abs(t[2])):
            print(f"  {a:>25s} ↔ {b:<25s}  corr={c:+.3f}")
    else:
        print("\nNo se detectan pares con |correlación| > 0.9. "
              "Features suficientemente independientes.")

    # Heatmap.
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=90, fontsize=8)
    ax.set_yticklabels(FEATURES, fontsize=8)
    ax.set_title("Matriz de correlaciones — features v2",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(CORR_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nHeatmap guardado: {CORR_CHART}")

    return pares


# ==========================================================
# FUNCIÓN 8 - Comparación directa v1 vs v2 sobre mismo test
# ==========================================================
def comparar_v1_vs_v2(df_test, metricas_v2):
    """
    Si existe model.joblib (v1), lo cargamos, lo evaluamos
    sobre EL MISMO test set que v2 (pero solo con las 24
    features de v1) e imprimimos tabla comparativa.

    NOTA: este test set tiene 12 filas menos que el v1 por
    los NaN iniciales de dxy_roc_5d; es un delta mínimo, la
    comparación sigue siendo representativa.
    """
    if not os.path.exists(V1_MODEL_FILE):
        print(f"\n[INFO] No se encontró {V1_MODEL_FILE}. "
              f"Saltando comparación v1 vs v2.")
        return

    print(f"\n{'='*60}")
    print("COMPARATIVA V1 vs V2 (mismo test set)")
    print(f"{'='*60}")

    modelo_v1 = joblib.load(V1_MODEL_FILE)
    X_test_v1 = df_test[FEATURES_V1]
    y_test = df_test[LABEL]
    y_pred_v1 = modelo_v1.predict(X_test_v1)

    metricas_v1 = {
        "accuracy":  accuracy_score(y_test, y_pred_v1),
        "precision": precision_score(y_test, y_pred_v1, zero_division=0),
        "recall":    recall_score(y_test, y_pred_v1, zero_division=0),
        "f1":        f1_score(y_test, y_pred_v1, zero_division=0),
    }

    print(f"\n{'Métrica':>12s} | {'v1':>7s} | {'v2':>7s} | "
          f"{'Δ abs':>8s} | {'Δ %':>7s}")
    print("-" * 55)
    for k in ["accuracy", "precision", "recall", "f1"]:
        v1, v2 = metricas_v1[k], metricas_v2[k]
        delta = v2 - v1
        pct = (delta / v1 * 100) if v1 > 0 else 0
        print(f"{k:>12s} | {v1:>7.3f} | {v2:>7.3f} | "
              f"{delta:>+8.3f} | {pct:>+6.1f}%")

    # Veredicto.
    delta_f1_pct = ((metricas_v2["f1"] - metricas_v1["f1"])
                    / metricas_v1["f1"] * 100) if metricas_v1["f1"] > 0 else 0
    print(f"\n>>> VEREDICTO sobre criterio F1 ≥ +5%:")
    if delta_f1_pct >= 5:
        print(f"    v2 GANA (Δ F1 = {delta_f1_pct:+.1f}%). "
              f"Adoptar v2 y actualizar 08_live_signal.py.")
    else:
        print(f"    v2 NO supera el criterio (Δ F1 = {delta_f1_pct:+.1f}%). "
              f"Mantener v1. Documentar por qué las features externas "
              f"no aportaron en esta iteración.")


# ==========================================================
# FUNCIÓN 9 - Guardar resultados del test set
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
    print("Entrenando modelo v2 (con features externas)...\n")

    # 1. Cargar + split
    df_train, df_test = cargar_y_split()

    # 2. Entrenar
    modelo = entrenar_modelo(df_train)

    # 3. Evaluar clasificación v2
    y_pred, y_proba, metricas_v2 = evaluar_clasificacion(
        modelo, df_test, FEATURES, titulo="v2 (test set)"
    )

    # 4. Simulación trading
    simular_con_filtro(df_test, y_pred, y_proba)

    # 5. Feature importance NATIVA (del modelo)
    mostrar_feature_importance_nativa(modelo)

    # 6. Permutation importance (la buena)
    mostrar_permutation_importance(modelo, df_test)

    # 7. Correlaciones entre features
    analizar_correlaciones(df_train)

    # 8. Comparativa v1 vs v2
    comparar_v1_vs_v2(df_test, metricas_v2)

    # 9. Guardar resultados
    guardar_resultados(df_test, y_pred, y_proba)

    print(f"\n{'='*60}")
    print("ENTRENAMIENTO V2 COMPLETADO")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
