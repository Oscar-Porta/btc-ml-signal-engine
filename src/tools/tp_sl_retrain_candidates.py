"""
==========================================================
REENTRENAMIENTO DE CANDIDATAS TP/SL
==========================================================
Para cada configuración candidata del barrido (Anexo II):
  1. Recalcula trades con los nuevos niveles de TP/SL
  2. Genera el dataset ML con features + etiquetas nuevas
  3. Reentrena el modelo XGBoost
  4. Evalúa en múltiples umbrales
  5. Compara con la configuración actual

NO modifica archivos de producción. Todo se guarda en
outputs/tp_sl_optimization/candidates/

Uso:
  python src/tp_sl_retrain_candidates.py
==========================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib

# Importar la simulación vectorizada del optimizador
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tp_sl_optimizer import cargar_y_preparar, simular_todos


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SIGNALS_FILE = os.path.join(OUTPUT_DIR, "signals.csv")
CANDIDATES_DIR = os.path.join(OUTPUT_DIR, "tp_sl_optimization", "candidates")

SYMBOL = "BTCUSDT"
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

# Candidatas del barrido + configuración actual
CONFIGS = {
    "actual": {
        "sl_atr_mult": 1.5, "tp1_r_mult": 1.0,
        "tp2_r_mult": 2.0, "partial_close": 0.5,
        "max_bars": 24,
        "desc": "Producción actual"
    },
    "A": {
        "sl_atr_mult": 1.25, "tp1_r_mult": 2.0,
        "tp2_r_mult": 4.0, "partial_close": 1.0,
        "max_bars": 24,
        "desc": "SL apretado, TP1 lejano, 100% en TP1"
    },
    "B": {
        "sl_atr_mult": 1.25, "tp1_r_mult": 1.5,
        "tp2_r_mult": 3.0, "partial_close": 1.0,
        "max_bars": 24,
        "desc": "SL apretado, TP1 intermedio, 100% en TP1"
    },
    "C": {
        "sl_atr_mult": 1.75, "tp1_r_mult": 1.5,
        "tp2_r_mult": 3.0, "partial_close": 1.0,
        "max_bars": 24,
        "desc": "SL holgado, TP1 intermedio, 100% en TP1"
    },
}


# ==========================================================
# FUNCIÓN 1 - Generar trades para una configuración
# ==========================================================
def generar_trades(entry_prices, atrs, directions,
                   candle_highs, candle_lows, candle_closes,
                   valid_bars, config):
    """
    Ejecuta la simulación vectorizada para una config y
    devuelve arrays de resultados.
    """
    pnl_abs, pnl_pct, r_mult, cr, bars = simular_todos(
        entry_prices, atrs, directions,
        candle_highs, candle_lows, candle_closes, valid_bars,
        config["sl_atr_mult"], config["tp1_r_mult"],
        config["tp2_r_mult"], config["partial_close"],
        config["max_bars"]
    )
    return pnl_abs, pnl_pct, r_mult, cr, bars


# ==========================================================
# FUNCIÓN 2 - Construir dataset ML
# ==========================================================
def construir_dataset(df_signals, pnl_abs, pnl_pct, r_mult,
                      close_reason, bars_in_trade):
    """
    Construye el dataset ML: features del momento de la señal
    + resultado del trade con la nueva configuración.

    Replica la lógica de 06_prepare_dataset.py pero en memoria.
    """
    df = df_signals.copy()

    # Añadir resultados del trade
    reason_map = {1: "stop_loss", 2: "take_profit_1",
                  3: "take_profit_2", 4: "breakeven",
                  5: "time_exit"}
    df["pnl_abs"] = pnl_abs
    df["pnl_pct"] = pnl_pct
    df["r_multiple"] = r_mult
    df["close_reason"] = [reason_map.get(c, "unknown")
                          for c in close_reason]
    df["bars_in_trade"] = bars_in_trade

    # Label: R >= 0.5
    df["label"] = (df["r_multiple"] >= 0.5).astype(int)
    df["label_v1"] = (df["pnl_abs"] > 0).astype(int)

    return df


# ==========================================================
# FUNCIÓN 3 - Crear features derivadas
# ==========================================================
def crear_features(df, df_1h_ctx, df_4h_ctx):
    """
    Añade features derivadas. Replica 06_prepare_dataset.py.
    """
    df = df.copy()

    # Merge con contexto rolling de 1h
    df = pd.merge_asof(
        df.sort_values("open_time"),
        df_1h_ctx.sort_values("open_time"),
        on="open_time",
        direction="backward"
    )

    # Merge con contexto de 4h
    df = pd.merge_asof(
        df.sort_values("open_time"),
        df_4h_ctx.sort_values("open_time"),
        on="open_time",
        direction="backward"
    )

    # Features derivadas
    df["dist_ema20_pct"] = (df["close"] - df["ema_20"]) / df["ema_20"] * 100
    df["dist_ema50_pct"] = (df["close"] - df["ema_50"]) / df["ema_50"] * 100
    df["ema_spread_pct"] = (df["ema_20"] - df["ema_50"]) / df["ema_50"] * 100

    df["ema_spread_4h_pct"] = np.where(
        df["ema_50_4h"].notna() & (df["ema_50_4h"] > 0),
        (df["ema_20_4h"] - df["ema_50_4h"]) / df["ema_50_4h"] * 100,
        0.0
    )

    df["atr_pct"] = df["atr_14"] / df["close"] * 100

    rango = df["swing_high_20"] - df["swing_low_20"]
    df["price_position"] = np.where(
        rango > 0,
        (df["close"] - df["swing_low_20"]) / rango,
        0.5
    )

    df["body_pct"] = (df["close"] - df["open"]) / df["open"] * 100

    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    full_range = df["high"] - df["low"]

    df["upper_wick_pct"] = np.where(
        full_range > 0,
        (df["high"] - body_top) / full_range * 100, 0
    )
    df["lower_wick_pct"] = np.where(
        full_range > 0,
        (body_bottom - df["low"]) / full_range * 100, 0
    )

    df["hour"] = df["open_time"].dt.hour
    df["day_of_week"] = df["open_time"].dt.dayofweek

    df["volume_sma20_ratio"] = np.where(
        df["volume_sma20"].notna() & (df["volume_sma20"] > 0),
        df["volume"] / df["volume_sma20"], 1.0
    )

    if "rsi_14_4h" not in df.columns:
        df["rsi_14_4h"] = 50.0
    if "atr_14_4h" in df.columns:
        df["atr_pct_4h"] = df["atr_14_4h"] / df["close"] * 100
    else:
        df["atr_pct_4h"] = df["atr_pct"]

    df["roc_5"] = np.where(
        df["close_5_ago"].notna() & (df["close_5_ago"] > 0),
        (df["close"] - df["close_5_ago"]) / df["close_5_ago"] * 100,
        0.0
    )
    df["roc_20"] = np.where(
        df["close_20_ago"].notna() & (df["close_20_ago"] > 0),
        (df["close"] - df["close_20_ago"]) / df["close_20_ago"] * 100,
        0.0
    )

    df["atr_regime"] = np.where(
        df["atr_sma20"].notna() & (df["atr_sma20"] > 0),
        df["atr_14"] / df["atr_sma20"], 1.0
    )

    df["candle_range_ratio"] = np.where(
        df["atr_14"] > 0,
        (df["high"] - df["low"]) / df["atr_14"], 1.0
    )

    return df


# ==========================================================
# FUNCIÓN 4 - Cargar datos de contexto (rolling, 4h)
# ==========================================================
def cargar_contexto():
    """
    Carga las velas procesadas para calcular features rolling
    y de 4h. Igual que 06_prepare_dataset.py.
    """
    df_1h = pd.read_csv(
        os.path.join(PROCESSED_DIR, f"{SYMBOL}_1h_processed.csv")
    )
    df_1h["open_time"] = pd.to_datetime(
        df_1h["open_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")

    df_1h["volume_sma20"] = df_1h["volume"].rolling(20).mean()
    df_1h["close_5_ago"] = df_1h["close"].shift(5)
    df_1h["close_20_ago"] = df_1h["close"].shift(20)
    df_1h["atr_sma20"] = df_1h["atr_14"].rolling(20).mean()

    df_1h_ctx = df_1h[[
        "open_time", "volume_sma20", "close_5_ago",
        "close_20_ago", "atr_sma20"
    ]].copy()

    df_4h = pd.read_csv(
        os.path.join(PROCESSED_DIR, f"{SYMBOL}_4h_processed.csv")
    )
    df_4h["open_time"] = pd.to_datetime(
        df_4h["open_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")

    df_4h_ctx = df_4h[[
        "open_time", "rsi_14", "atr_14", "close"
    ]].copy()
    df_4h_ctx = df_4h_ctx.rename(columns={
        "rsi_14": "rsi_14_4h",
        "atr_14": "atr_14_4h",
        "close": "close_4h"
    })

    return df_1h_ctx, df_4h_ctx


# ==========================================================
# FUNCIÓN 5 - Entrenar y evaluar modelo
# ==========================================================
def entrenar_y_evaluar(df_dataset, config_name, config):
    """
    Entrena XGB con los datos de una configuración y evalúa
    en múltiples umbrales. Devuelve dict con resultados.
    """
    cutoff = pd.Timestamp(TRAIN_CUTOFF, tz="Europe/Madrid")
    df_train = df_dataset[df_dataset["open_time"] <= cutoff].copy()
    df_test = df_dataset[df_dataset["open_time"] > cutoff].copy()

    # Verificar que tenemos features sin NaN
    for f in FEATURES:
        if f not in df_train.columns:
            print(f"  [WARN] Feature {f} no encontrada")
            return None

    df_train = df_train.dropna(subset=FEATURES)
    df_test = df_test.dropna(subset=FEATURES)

    X_train = df_train[FEATURES]
    y_train = df_train["label"]
    X_test = df_test[FEATURES]
    y_test = df_test["label"]

    n_gan_train = y_train.sum()
    n_gan_test = y_test.sum()

    print(f"  Train: {len(df_train)} ({n_gan_train} gan, "
          f"{n_gan_train/len(df_train)*100:.1f}%)")
    print(f"  Test:  {len(df_test)} ({n_gan_test} gan, "
          f"{n_gan_test/len(df_test)*100:.1f}%)")

    # Entrenar
    modelo = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )
    modelo.fit(X_train, y_train)

    # Predecir
    y_proba = modelo.predict_proba(X_test)[:, 1]
    y_pred = modelo.predict(X_test)

    # Métricas de clasificación
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    print(f"  F1: {f1:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f}")

    # Evaluar por umbrales (usando PnL real del backtest)
    df_eval = df_test.copy()
    df_eval["y_proba"] = y_proba

    resultados_umbrales = []
    for umbral in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        f = df_eval[df_eval["y_proba"] >= umbral]
        n = len(f)
        if n == 0:
            continue

        ganadores = f[f["pnl_abs"] > 0]
        perdedores = f[f["pnl_abs"] < 0]
        n_win = len(ganadores)
        n_loss = len(perdedores)

        wr = n_win / n * 100
        pnl = f["pnl_abs"].sum()
        pnl_m = f["pnl_abs"].mean()
        r_m = f["r_multiple"].mean()

        avg_win = abs(ganadores["pnl_abs"].mean()) if n_win > 0 else 0
        avg_loss = abs(perdedores["pnl_abs"].mean()) if n_loss > 0 else 0
        payoff = avg_win / avg_loss if avg_loss > 0 else 0
        be_wr = (1 / (1 + payoff) * 100) if payoff > 0 else 50.0
        margen = wr - be_wr

        gross_p = ganadores["pnl_abs"].sum() if n_win > 0 else 0
        gross_l = abs(perdedores["pnl_abs"].sum()) if n_loss > 0 else 1
        pf = gross_p / gross_l if gross_l > 0 else 0

        # Max drawdown
        cum = f["pnl_abs"].cumsum()
        dd = (cum - cum.cummax()).min()

        resultados_umbrales.append({
            "config": config_name,
            "umbral": umbral,
            "trades": n,
            "win_rate": round(wr, 2),
            "payoff": round(payoff, 2),
            "pnl_total": round(pnl, 2),
            "pnl_medio": round(pnl_m, 2),
            "r_medio": round(r_m, 4),
            "breakeven_wr": round(be_wr, 1),
            "margen": round(margen, 1),
            "profit_factor": round(pf, 2),
            "max_drawdown": round(dd, 2),
        })

    # Guardar modelo
    model_file = os.path.join(
        CANDIDATES_DIR, f"model_{config_name}.joblib"
    )
    joblib.dump(modelo, model_file)

    return resultados_umbrales


# ==========================================================
# FUNCIÓN 6 - Mostrar tabla comparativa
# ==========================================================
def mostrar_comparativa(all_results, umbral_ref=0.55):
    """
    Muestra tabla comparativa de todas las configs a un umbral.
    """
    print(f"\n{'=' * 80}")
    print(f"COMPARATIVA FINAL (umbral {umbral_ref})")
    print(f"{'=' * 80}")

    rows = []
    for r in all_results:
        match = [x for x in r if x["umbral"] == umbral_ref]
        if match:
            rows.append(match[0])

    if not rows:
        print("No hay resultados para este umbral.")
        return

    df = pd.DataFrame(rows)
    cols = ["config", "trades", "win_rate", "payoff", "pnl_total",
            "r_medio", "margen", "profit_factor", "max_drawdown"]
    print(df[cols].to_string(index=False))

    # Ganadora por PnL
    best_pnl = df.loc[df["pnl_total"].idxmax()]
    print(f"\nMejor por PnL: Config {best_pnl['config']} "
          f"(${best_pnl['pnl_total']:,.0f})")

    # Ganadora por margen
    best_margin = df.loc[df["margen"].idxmax()]
    print(f"Mejor por margen: Config {best_margin['config']} "
          f"({best_margin['margen']:+.1f}pp)")

    return df


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("=" * 60)
    print("REENTRENAMIENTO DE CANDIDATAS TP/SL")
    print("=" * 60)

    # 1. Cargar datos base
    print("\n[1/5] Cargando datos...")
    (entry_prices, atrs, directions,
     candle_highs, candle_lows, candle_closes,
     valid_bars) = cargar_y_preparar()

    # Cargar signals para features
    df_signals = pd.read_csv(SIGNALS_FILE)
    df_signals["open_time"] = pd.to_datetime(
        df_signals["open_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")
    df_signals = df_signals.sort_values("open_time").reset_index(drop=True)

    # Cargar contexto para features rolling
    print("\n[2/5] Cargando datos de contexto...")
    df_1h_ctx, df_4h_ctx = cargar_contexto()

    os.makedirs(CANDIDATES_DIR, exist_ok=True)

    # 3. Para cada configuración: generar trades + entrenar + evaluar
    print("\n[3/5] Procesando candidatas...")
    all_results = []
    all_threshold_results = []

    for name, config in CONFIGS.items():
        print(f"\n{'─' * 60}")
        print(f"Config {name}: {config['desc']}")
        print(f"  SL: {config['sl_atr_mult']}x ATR | "
              f"TP1: {config['tp1_r_mult']}R | "
              f"TP2: {config['tp2_r_mult']}R | "
              f"Parcial: {config['partial_close']*100:.0f}%")
        print(f"{'─' * 60}")

        # Generar trades
        pnl_abs, pnl_pct, r_mult, cr, bars = generar_trades(
            entry_prices, atrs, directions,
            candle_highs, candle_lows, candle_closes,
            valid_bars, config
        )

        # Construir dataset
        df_dataset = construir_dataset(
            df_signals, pnl_abs, pnl_pct, r_mult, cr, bars
        )

        # Crear features
        df_dataset = crear_features(df_dataset, df_1h_ctx, df_4h_ctx)

        # Info del dataset
        n_label1 = df_dataset["label"].sum()
        print(f"  Dataset: {len(df_dataset)} trades | "
              f"Label=1: {n_label1} ({n_label1/len(df_dataset)*100:.1f}%)")

        # Entrenar y evaluar
        results = entrenar_y_evaluar(df_dataset, name, config)
        if results:
            all_results.append(results)
            all_threshold_results.extend(results)

    # 4. Tabla comparativa
    print("\n[4/5] Generando comparativas...")
    df_comp = mostrar_comparativa(all_results, umbral_ref=0.55)

    # Tabla completa por umbrales
    print(f"\n{'=' * 80}")
    print("TABLA COMPLETA POR UMBRALES")
    print(f"{'=' * 80}")

    df_all = pd.DataFrame(all_threshold_results)
    cols = ["config", "umbral", "trades", "win_rate", "payoff",
            "pnl_total", "r_medio", "margen", "profit_factor",
            "max_drawdown"]
    for name in CONFIGS:
        subset = df_all[df_all["config"] == name]
        if len(subset) > 0:
            print(f"\n--- Config {name}: {CONFIGS[name]['desc']} ---")
            print(subset[cols[1:]].to_string(index=False))

    # 5. Guardar resultados
    print(f"\n[5/5] Guardando...")
    df_all.to_csv(
        os.path.join(CANDIDATES_DIR, "comparison_all_thresholds.csv"),
        index=False
    )
    if df_comp is not None:
        df_comp.to_csv(
            os.path.join(CANDIDATES_DIR, "comparison_055.csv"),
            index=False
        )

    print(f"\nResultados en: {CANDIDATES_DIR}")
    print(f"\n{'=' * 60}")
    print("REENTRENAMIENTO COMPLETADO")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
