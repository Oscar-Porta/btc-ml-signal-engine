"""
==========================================================
FASE 9 (Iteración 1B) - Preparación del dataset V2
==========================================================
Copia de 06_prepare_dataset.py (v1) con un único cambio
conceptual: añade al dataset las 4 features externas
generadas por 02b_external_features.py.

    - eth_roc_5
    - eth_roc_20
    - btc_eth_corr_20
    - dxy_roc_5d

Todo lo demás queda EXACTAMENTE igual que en v1. Así la
comparación v1 vs v2 será limpia: la única diferencia
son estas 4 features.

Salida:
  - outputs/ml_dataset_v2.csv  (NO sobrescribe el v1)
==========================================================
"""

import os
import pandas as pd
import numpy as np


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
OUTPUT_DIR = "outputs"
SIGNALS_FILE = os.path.join(OUTPUT_DIR, "signals.csv")
TRADES_FILE = os.path.join(OUTPUT_DIR, "trades.csv")
DATASET_FILE = os.path.join(OUTPUT_DIR, "ml_dataset_v2.csv")  # << v2

# Datos procesados
PROCESSED_DIR = "data/processed"
SYMBOL = "BTCUSDT"
PROCESSED_1H = os.path.join(PROCESSED_DIR, f"{SYMBOL}_1h_processed.csv")
PROCESSED_4H = os.path.join(PROCESSED_DIR, f"{SYMBOL}_4h_processed.csv")

# NUEVO en v2: features externas generadas por 02b_external_features.py
EXTERNAL_FEATURES_FILE = os.path.join(PROCESSED_DIR, "external_features.csv")


# ==========================================================
# FUNCIÓN 1 - Cargar y unir señales con trades
# ==========================================================
def cargar_y_unir():
    df_signals = pd.read_csv(SIGNALS_FILE)
    df_trades = pd.read_csv(TRADES_FILE)

    df_signals["open_time"] = pd.to_datetime(
        df_signals["open_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")

    df_trades["signal_time"] = pd.to_datetime(
        df_trades["signal_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")

    df = pd.merge(
        df_signals,
        df_trades[["signal_time", "direction", "rule",
                    "exit_price", "close_reason",
                    "bars_in_trade", "pnl_abs", "pnl_pct"]],
        left_on=["open_time", "direction", "rule"],
        right_on=["signal_time", "direction", "rule"],
        how="inner"
    )
    df = df.drop(columns=["signal_time"], errors="ignore")

    print(f"Señales: {len(df_signals)}")
    print(f"Trades: {len(df_trades)}")
    print(f"Unidas: {len(df)}")
    return df


# ==========================================================
# FUNCIÓN 1B - Cargar datos procesados para features rolling
# ==========================================================
def cargar_datos_procesados():
    df_1h = pd.read_csv(PROCESSED_1H)
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

    df_4h = pd.read_csv(PROCESSED_4H)
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

    print(f"Datos 1h cargados: {len(df_1h)}")
    print(f"Datos 4h cargados: {len(df_4h)}")
    return df_1h_ctx, df_4h_ctx


# ==========================================================
# FUNCIÓN 1C (NUEVA en v2) - Cargar features externas
# ==========================================================
def cargar_features_externas():
    """
    Carga el CSV generado por 02b_external_features.py con
    las 4 features externas ya alineadas a la granularidad
    de 1h de BTC (ETH_1h + DXY con forward-fill).

    Nota: pd.read_csv no parsea bien open_time con tz en el
    string. Forzamos conversión a datetime64[ns, Europe/Madrid].
    """
    df = pd.read_csv(EXTERNAL_FEATURES_FILE)
    df["open_time"] = pd.to_datetime(
        df["open_time"], utc=True
    ).dt.tz_convert("Europe/Madrid")
    print(f"Features externas cargadas: {len(df)} filas, "
          f"columnas: {list(df.columns)}")
    return df


# ==========================================================
# FUNCIÓN 2 - Crear features derivadas (IDÉNTICA A V1)
# ==========================================================
def crear_features(df):
    df = df.copy()

    df["dist_ema20_pct"] = (
        (df["close"] - df["ema_20"]) / df["ema_20"] * 100
    )
    df["dist_ema50_pct"] = (
        (df["close"] - df["ema_50"]) / df["ema_50"] * 100
    )
    df["ema_spread_pct"] = (
        (df["ema_20"] - df["ema_50"]) / df["ema_50"] * 100
    )
    df["ema_spread_4h_pct"] = (
        (df["ema_20_4h"] - df["ema_50_4h"]) / df["ema_50_4h"] * 100
    )
    df["atr_pct"] = df["atr_14"] / df["close"] * 100

    rango = df["swing_high_20"] - df["swing_low_20"]
    df["price_position"] = np.where(
        rango > 0,
        (df["close"] - df["swing_low_20"]) / rango,
        0.5
    )
    df["body_pct"] = (
        (df["close"] - df["open"]) / df["open"] * 100
    )
    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    full_range = df["high"] - df["low"]
    df["upper_wick_pct"] = np.where(
        full_range > 0,
        (df["high"] - body_top) / full_range * 100,
        0
    )
    df["lower_wick_pct"] = np.where(
        full_range > 0,
        (body_bottom - df["low"]) / full_range * 100,
        0
    )
    df["hour"] = df["open_time"].dt.hour
    df["day_of_week"] = df["open_time"].dt.dayofweek
    df["direction_num"] = (df["direction"] == "LONG").astype(int)
    df["rule_num"] = (df["rule"] == "breakout").astype(int)

    dist_tp2 = (df["tp2"] - df["entry_price"]).abs()
    dist_sl = (df["entry_price"] - df["stop_loss"]).abs()
    df["reward_risk_ratio"] = np.where(
        dist_sl > 0, dist_tp2 / dist_sl, 0
    )

    df["volume_sma20_ratio"] = np.where(
        df["volume_sma20"].notna() & (df["volume_sma20"] > 0),
        df["volume"] / df["volume_sma20"],
        1.0
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
        df["atr_14"] / df["atr_sma20"],
        1.0
    )
    df["candle_range_ratio"] = np.where(
        df["atr_14"] > 0,
        (df["high"] - df["low"]) / df["atr_14"],
        1.0
    )

    print(f"Features añadidas. Total columnas: {len(df.columns)}")
    return df


# ==========================================================
# FUNCIÓN 3 - Crear label
# ==========================================================
def crear_label(df):
    df = df.copy()
    df["label"] = (df["pnl_abs"] > 0).astype(int)
    n_positivos = df["label"].sum()
    n_negativos = len(df) - n_positivos
    ratio = n_positivos / len(df) * 100
    print(f"\nDistribución del label:")
    print(f"  Ganadores (1): {n_positivos} ({ratio:.1f}%)")
    print(f"  Perdedores/neutros (0): {n_negativos} ({100-ratio:.1f}%)")
    return df


# ==========================================================
# FUNCIÓN 4 - Seleccionar features finales (V2: +4 externas)
# ==========================================================
def seleccionar_features(df):
    features = [
        # Indicadores base
        "rsi_14",
        "atr_14",
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

        # --- NUEVAS FEATURES (Iteración 1B) ---
        "eth_roc_5",         # momentum ETH 5h
        "eth_roc_20",        # momentum ETH 20h
        "btc_eth_corr_20",   # correlación móvil BTC-ETH
        "dxy_roc_5d",        # momentum DXY 5 días
    ]

    metadata = [
        "open_time", "direction", "rule",
        "entry_price", "stop_loss", "tp1", "tp2",
        "close_reason", "pnl_abs", "pnl_pct",
        "bars_in_trade",
    ]
    label = "label"

    print(f"\nFeatures para el modelo: {len(features)}")
    for f in features:
        print(f"  - {f}")
    return features, label, metadata


# ==========================================================
# FUNCIÓN 5 - Guardar dataset
# ==========================================================
def guardar_dataset(df, features, label, metadata):
    cols_existentes = [c for c in metadata + features + [label]
                       if c in df.columns]
    df_out = df[cols_existentes].copy()

    nan_features = df_out[features].isna().sum()
    nan_features = nan_features[nan_features > 0]
    if not nan_features.empty:
        print(f"\nAdvertencia: NaN en features:")
        for col, n in nan_features.items():
            print(f"  {col}: {n}")
        antes = len(df_out)
        df_out = df_out.dropna(subset=features)
        print(f"  Eliminadas {antes - len(df_out)} filas con NaN")

    df_out.to_csv(DATASET_FILE, index=False)
    print(f"\nDataset guardado: {DATASET_FILE}")
    print(f"Filas: {len(df_out)}, Columnas: {len(df_out.columns)}")
    return df_out


# ==========================================================
# FUNCIÓN 6 - Resumen features (comparación medias por clase)
# ==========================================================
def mostrar_resumen_features(df, features, label):
    print(f"\n{'='*60}")
    print("RESUMEN DE FEATURES: media por clase")
    print(f"{'='*60}")
    print(f"{'Feature':>25s} | {'Perdedor':>10s} | {'Ganador':>10s} | {'Dif %':>8s}")
    print("-" * 60)
    for f in features:
        media_0 = df[df[label] == 0][f].mean()
        media_1 = df[df[label] == 1][f].mean()
        if media_0 != 0:
            dif_pct = (media_1 - media_0) / abs(media_0) * 100
        else:
            dif_pct = 0
        print(f"{f:>25s} | {media_0:>10.3f} | {media_1:>10.3f} | {dif_pct:>+7.1f}%")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("Preparando dataset V2 (con features externas)...\n")

    # 1. Cargar y unir señales con trades
    df = cargar_y_unir()

    # 1B. Cargar procesados (rolling features + 4h context)
    df_1h_ctx, df_4h_ctx = cargar_datos_procesados()

    # 1C. (NUEVO) Cargar features externas
    df_ext = cargar_features_externas()

    # Merge 1h rolling features
    df = pd.merge(df, df_1h_ctx, on="open_time", how="left")

    # Merge 4h con merge_asof
    df = df.sort_values("open_time").reset_index(drop=True)
    df_4h_ctx = df_4h_ctx.sort_values("open_time").reset_index(drop=True)
    df = pd.merge_asof(
        df, df_4h_ctx, on="open_time", direction="backward"
    )

    # (NUEVO) Merge de features externas (alineadas a 1h exacto)
    df = pd.merge(df, df_ext, on="open_time", how="left")

    print(f"Merges completados. Filas: {len(df)}")

    # 2. Features derivadas
    df = crear_features(df)

    # 3. Label
    df = crear_label(df)

    # 4. Selección
    features, label, metadata = seleccionar_features(df)

    # 5. Guardar
    df = guardar_dataset(df, features, label, metadata)

    # 6. Resumen
    mostrar_resumen_features(df, features, label)


if __name__ == "__main__":
    main()
