"""
==========================================================
FASE 6A - Preparación del dataset para IA
==========================================================
Este script toma las señales (outputs/signals.csv) y los
trades (outputs/trades.csv), los une y crea un dataset
enriquecido con features adicionales para entrenar un
modelo de machine learning.

El modelo deberá responder a la pregunta:
  "Dado el contexto del mercado en el momento de esta
  señal, ¿es probable que el trade sea ganador?"

Features que añadimos (más allá de los indicadores base):
  - Distancia del precio a las EMAs (% de retroceso)
  - Fuerza de la tendencia (separación entre EMAs)
  - Volatilidad relativa (ATR normalizado por precio)
  - Posición del precio en el rango reciente
  - Volumen relativo (vs media)
  - Hora del día y día de la semana
  - Dirección y regla codificadas numéricamente

Label:
  - 1 = trade ganador (pnl_abs > 0)
  - 0 = trade perdedor o neutro (pnl_abs <= 0)
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
DATASET_FILE = os.path.join(OUTPUT_DIR, "ml_dataset.csv")

# Datos procesados (para features que necesitan contexto histórico)
PROCESSED_DIR = "data/processed"
SYMBOL = "BTCUSDT"
PROCESSED_1H = os.path.join(PROCESSED_DIR, f"{SYMBOL}_1h_processed.csv")
PROCESSED_4H = os.path.join(PROCESSED_DIR, f"{SYMBOL}_4h_processed.csv")


# ==========================================================
# FUNCIÓN 1 - Cargar y unir señales con trades
# ==========================================================
def cargar_y_unir():
    """
    Carga señales y trades, y los une para tener en cada
    fila: las features del momento de la señal + el resultado
    del trade.

    La unión se hace por signal_time + direction + rule,
    que identifica cada señal de forma única.
    """
    df_signals = pd.read_csv(SIGNALS_FILE)
    df_trades = pd.read_csv(TRADES_FILE)

    # Convertir fechas
    df_signals["open_time"] = pd.to_datetime(
        df_signals["open_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")

    df_trades["signal_time"] = pd.to_datetime(
        df_trades["signal_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")

    # Unir: cada señal con su resultado
    df = pd.merge(
        df_signals,
        df_trades[["signal_time", "direction", "rule",
                    "exit_price", "close_reason",
                    "bars_in_trade", "pnl_abs", "pnl_pct",
                    "r_multiple", "mae_r"]],
        left_on=["open_time", "direction", "rule"],
        right_on=["signal_time", "direction", "rule"],
        how="inner"
    )

    # Limpieza: quitar columna duplicada
    df = df.drop(columns=["signal_time"], errors="ignore")

    print(f"Señales: {len(df_signals)}")
    print(f"Trades: {len(df_trades)}")
    print(f"Unidas: {len(df)} (deben coincidir)")

    return df


# ==========================================================
# FUNCIÓN 1B - Cargar datos procesados para features rolling
# ==========================================================
def cargar_datos_procesados():
    """
    Carga los CSV procesados de 1h y 4h para calcular features
    que necesitan contexto histórico (volumen relativo, momentum,
    régimen de volatilidad, RSI de 4h, etc.).

    Precalcula columnas rolling sobre el dataset completo de 1h
    y prepara el de 4h para merge_asof.
    """
    # --- 1h: features que necesitan rolling ---
    df_1h = pd.read_csv(PROCESSED_1H)
    df_1h["open_time"] = pd.to_datetime(
        df_1h["open_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")

    # Volumen: media móvil simple de 20 velas
    df_1h["volume_sma20"] = df_1h["volume"].rolling(20).mean()

    # Momentum: precio hace 5 y 20 velas
    df_1h["close_5_ago"] = df_1h["close"].shift(5)
    df_1h["close_20_ago"] = df_1h["close"].shift(20)

    # Régimen de volatilidad: ATR vs su media de 20 periodos
    df_1h["atr_sma20"] = df_1h["atr_14"].rolling(20).mean()

    # Seleccionar solo las columnas que necesitamos para el merge
    df_1h_ctx = df_1h[[
        "open_time", "volume_sma20", "close_5_ago",
        "close_20_ago", "atr_sma20"
    ]].copy()

    # --- 4h: RSI y ATR para merge_asof ---
    # CORRECCIÓN LOOKAHEAD (jul 2026): usamos close_time (no
    # open_time) para emparejar luego solo velas 4h ya cerradas
    # en el momento de la señal.
    df_4h = pd.read_csv(PROCESSED_4H)
    df_4h["close_time"] = pd.to_datetime(
        df_4h["close_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")

    df_4h_ctx = df_4h[[
        "close_time", "rsi_14", "atr_14", "close"
    ]].copy()
    df_4h_ctx = df_4h_ctx.rename(columns={
        "close_time": "close_time_4h",
        "rsi_14": "rsi_14_4h",
        "atr_14": "atr_14_4h",
        "close": "close_4h"
    })

    print(f"Datos 1h cargados: {len(df_1h)} filas (con rolling features)")
    print(f"Datos 4h cargados: {len(df_4h)} filas (RSI + ATR)")

    return df_1h_ctx, df_4h_ctx


# ==========================================================
# FUNCIÓN 2 - Crear features derivadas
# ==========================================================
def crear_features(df):
    """
    Añade features calculadas a partir de los datos existentes.
    Todas se calculan con información disponible en el momento
    de la señal (sin mirar al futuro).
    """
    df = df.copy()

    # --- Distancia del precio a las EMAs ---
    # Mide "cuánto ha retrocedido el precio" respecto a la EMA.
    # Positivo = precio por encima, negativo = por debajo.
    df["dist_ema20_pct"] = (
        (df["close"] - df["ema_20"]) / df["ema_20"] * 100
    )
    df["dist_ema50_pct"] = (
        (df["close"] - df["ema_50"]) / df["ema_50"] * 100
    )

    # --- Fuerza de la tendencia en 1h ---
    # Cuánto se separan las EMAs. Más separación = tendencia
    # más fuerte y definida.
    df["ema_spread_pct"] = (
        (df["ema_20"] - df["ema_50"]) / df["ema_50"] * 100
    )

    # --- Fuerza de la tendencia en 4h ---
    df["ema_spread_4h_pct"] = (
        (df["ema_20_4h"] - df["ema_50_4h"]) / df["ema_50_4h"] * 100
    )

    # --- Volatilidad relativa ---
    # ATR como porcentaje del precio. Permite comparar la
    # volatilidad entre periodos con precios muy distintos
    # (BTC a 16k vs BTC a 100k).
    df["atr_pct"] = df["atr_14"] / df["close"] * 100

    # --- Posición del precio en el rango reciente ---
    # 1.0 = precio está en el máximo de las últimas 20 velas.
    # 0.0 = precio está en el mínimo.
    # Captura algo similar a lo que Fibonacci intenta medir.
    rango = df["swing_high_20"] - df["swing_low_20"]
    df["price_position"] = np.where(
        rango > 0,
        (df["close"] - df["swing_low_20"]) / rango,
        0.5  # si el rango es 0, posición neutra
    )

    # --- Tamaño de la vela de señal ---
    # Velas con cuerpo grande pueden indicar más convicción.
    df["body_pct"] = (
        (df["close"] - df["open"]) / df["open"] * 100
    )

    # --- Ratio mecha superior / inferior ---
    # Mechas largas pueden indicar rechazo de precio.
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

    # --- Hora del día ---
    # El mercado cripto tiene patrones intradiarios
    # (más volumen en horas europeas/americanas).
    df["hour"] = df["open_time"].dt.hour

    # --- Día de la semana ---
    # 0 = lunes, 6 = domingo. Los fines de semana suelen
    # tener menos volumen.
    df["day_of_week"] = df["open_time"].dt.dayofweek

    # --- Codificación numérica de dirección y regla ---
    df["direction_num"] = (df["direction"] == "LONG").astype(int)
    df["rule_num"] = (df["rule"] == "breakout").astype(int)

    # --- Riesgo relativo ---
    # risk_pct ya lo tenemos, pero añadimos el ratio
    # distancia a TP2 / distancia a SL (siempre debería
    # ser ~2 por diseño, pero puede variar ligeramente).
    dist_tp2 = (df["tp2"] - df["entry_price"]).abs()
    dist_sl = (df["entry_price"] - df["stop_loss"]).abs()
    df["reward_risk_ratio"] = np.where(
        dist_sl > 0, dist_tp2 / dist_sl, 0
    )

    # ==========================================================
    # NUEVAS FEATURES — Iteración 1
    # ==========================================================

    # --- Volumen relativo (NUEVA) ---
    # Volumen de la vela de señal dividido por la media de las
    # últimas 20 velas. >1 = volumen superior al habitual,
    # lo que puede confirmar la señal. Nunca usábamos el volumen.
    df["volume_sma20_ratio"] = np.where(
        df["volume_sma20"].notna() & (df["volume_sma20"] > 0),
        df["volume"] / df["volume_sma20"],
        1.0  # valor neutro si no hay dato
    )

    # --- RSI de 4h (NUEVA) ---
    # Hasta ahora solo usábamos la tendencia de 4h (up/down).
    # El RSI de 4h da información continua sobre el momentum
    # en el marco temporal superior. Un RSI_4h de 70 no es lo
    # mismo que uno de 55, aunque ambos estén en tendencia "up".
    # (rsi_14_4h ya viene del merge en main, solo validamos)
    if "rsi_14_4h" not in df.columns:
        df["rsi_14_4h"] = 50.0  # fallback neutro

    # --- ATR de 4h normalizado (NUEVA) ---
    # Volatilidad del marco temporal superior como % del precio.
    if "atr_14_4h" in df.columns:
        df["atr_pct_4h"] = df["atr_14_4h"] / df["close"] * 100
    else:
        df["atr_pct_4h"] = df["atr_pct"]  # fallback: usar 1h

    # --- Momentum: Rate of Change 5 velas (NUEVA) ---
    # Cuánto se ha movido el precio en las últimas 5 horas.
    # Positivo = subiendo, negativo = bajando. Captura la
    # velocidad y dirección del movimiento reciente.
    df["roc_5"] = np.where(
        df["close_5_ago"].notna() & (df["close_5_ago"] > 0),
        (df["close"] - df["close_5_ago"]) / df["close_5_ago"] * 100,
        0.0
    )

    # --- Momentum: Rate of Change 20 velas (NUEVA) ---
    # Lo mismo pero a 20 horas. Captura el momentum de
    # medio plazo.
    df["roc_20"] = np.where(
        df["close_20_ago"].notna() & (df["close_20_ago"] > 0),
        (df["close"] - df["close_20_ago"]) / df["close_20_ago"] * 100,
        0.0
    )

    # --- Régimen de volatilidad (NUEVA) ---
    # ATR actual / media del ATR en las últimas 20 velas.
    # >1 = volatilidad expandiéndose (mercado más activo)
    # <1 = volatilidad contrayéndose (mercado más calmado)
    # Un modelo podría aprender que señales en volatilidad
    # en contracción funcionan peor (mercado lateral).
    df["atr_regime"] = np.where(
        df["atr_sma20"].notna() & (df["atr_sma20"] > 0),
        df["atr_14"] / df["atr_sma20"],
        1.0
    )

    # --- Tamaño de vela vs normal (NUEVA) ---
    # (high - low) / ATR. Si >1, la vela de señal es más
    # grande de lo normal. Velas muy grandes pueden indicar
    # impulso fuerte o volatilidad puntual.
    df["candle_range_ratio"] = np.where(
        df["atr_14"] > 0,
        (df["high"] - df["low"]) / df["atr_14"],
        1.0
    )

    print(f"Features añadidas. Total columnas: {len(df.columns)}")

    return df


# ==========================================================
# FUNCIÓN 3 - Crear label (variable objetivo)
# ==========================================================
def crear_label(df):
    """
    Crea la columna 'label' basada en R-multiple.

    Iteración 2: etiquetas mejoradas.
    En lugar de usar pnl_abs > 0 (que trata un time_exit de
    +0.1R igual que un TP2 de +2R), usamos un umbral sobre
    el R-multiple para definir "ganador".

    Umbral elegido: R >= 0.5
    - Excluye trades que "ganan" por los pelos (time_exits
      con beneficio mínimo que añaden ruido al label).
    - Mantiene TP1 (~+2R con config A) y time_exits con
      beneficio significativo.

    También se conserva el label original (label_v1) y el
    r_multiple para poder comparar y experimentar con
    distintos umbrales en el entrenamiento.
    """
    df = df.copy()

    # Label v1 original (para comparación)
    df["label_v1"] = (df["pnl_abs"] > 0).astype(int)

    # Label v3: basado en R-multiple >= 0.5
    R_THRESHOLD = 0.5
    df["label"] = (df["r_multiple"] >= R_THRESHOLD).astype(int)

    n_v1 = df["label_v1"].sum()
    n_v3 = df["label"].sum()
    r_v1 = n_v1 / len(df) * 100
    r_v3 = n_v3 / len(df) * 100

    print(f"\nDistribución del label:")
    print(f"  Label v1 (pnl>0):   Ganadores={n_v1} ({r_v1:.1f}%)")
    print(f"  Label v3 (R≥{R_THRESHOLD}):  Ganadores={n_v3} ({r_v3:.1f}%)")
    print(f"  Diferencia: {n_v1 - n_v3} trades reclasificados "
          f"de ganador a perdedor")

    return df


# ==========================================================
# FUNCIÓN 4 - Seleccionar features finales
# ==========================================================
def seleccionar_features(df):
    """
    Define qué columnas son features (input del modelo),
    cuál es el label (output) y cuáles son solo metadata
    (no entran al modelo pero se guardan para análisis).
    """
    features = [
        # Indicadores base
        "rsi_14",
        "atr_14",

        # Features derivadas (Iteración 0, podadas en 1B.2)
        # Eliminadas: risk_pct (=atr_pct), reward_risk_ratio
        #             (invariante ~2), direction_num, rule_num
        #             (importancia 0)
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

        # Features Iteración 1
        "volume_sma20_ratio",   # volumen relativo
        "rsi_14_4h",            # RSI del timeframe 4h
        "atr_pct_4h",           # ATR 4h normalizado
        "roc_5",                # momentum corto plazo
        "roc_20",               # momentum medio plazo
        "atr_regime",           # régimen de volatilidad
        "candle_range_ratio",   # tamaño de vela vs normal

        # NOTA: mae_r NO es feature porque se calcula DURANTE
        # el trade (lookahead bias). Se guarda en metadata para
        # análisis post-hoc.
    ]

    metadata = [
        "open_time", "direction", "rule",
        "entry_price", "stop_loss", "tp1", "tp2",
        "close_reason", "pnl_abs", "pnl_pct",
        "bars_in_trade", "r_multiple", "mae_r", "label_v1",
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
    """
    Guarda el dataset con features, label y metadata.
    Las columnas van en este orden: metadata → features → label.
    """
    cols_existentes = [c for c in metadata + features + [label]
                       if c in df.columns]

    df_out = df[cols_existentes].copy()

    # Verificar que no haya NaN en features
    nan_features = df_out[features].isna().sum()
    nan_features = nan_features[nan_features > 0]
    if not nan_features.empty:
        print(f"\nAdvertencia: NaN en features:")
        for col, n in nan_features.items():
            print(f"  {col}: {n}")
        # Eliminar filas con NaN en features
        antes = len(df_out)
        df_out = df_out.dropna(subset=features)
        print(f"  Eliminadas {antes - len(df_out)} filas con NaN")

    df_out.to_csv(DATASET_FILE, index=False)
    print(f"\nDataset guardado: {DATASET_FILE}")
    print(f"Filas: {len(df_out)}, Columnas: {len(df_out.columns)}")

    return df_out


# ==========================================================
# FUNCIÓN 6 - Mostrar resumen estadístico de features
# ==========================================================
def mostrar_resumen_features(df, features, label):
    """
    Muestra estadísticas básicas de cada feature, separando
    entre trades ganadores y perdedores. Esto da una primera
    idea de qué features podrían ser útiles para el modelo.
    """
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
    print("Preparando dataset para IA...\n")

    # 1. Cargar y unir señales con trades
    df = cargar_y_unir()

    # 1B. Cargar datos procesados para features que necesitan
    # contexto histórico (volumen rolling, momentum, 4h RSI/ATR)
    df_1h_ctx, df_4h_ctx = cargar_datos_procesados()

    # Merge 1h rolling features por open_time (inner join exacto)
    df = pd.merge(df, df_1h_ctx, on="open_time", how="left")

    # Merge 4h RSI y ATR por merge_asof (temporal)
    # CORRECCIÓN LOOKAHEAD (jul 2026): la señal se decide al
    # CIERRE de su vela 1h (open_time + 1h). Emparejamos con la
    # última vela 4h cuyo close_time sea <= ese momento, es
    # decir, la última vela 4h YA CERRADA al decidir. Antes se
    # asignaba la vela 4h en curso (hasta 3h de futuro).
    df["decision_time"] = df["open_time"] + pd.Timedelta(hours=1)
    df = df.sort_values("decision_time").reset_index(drop=True)
    df_4h_ctx = df_4h_ctx.sort_values("close_time_4h").reset_index(drop=True)
    df = pd.merge_asof(
        df, df_4h_ctx,
        left_on="decision_time", right_on="close_time_4h",
        direction="backward"
    )
    df = df.drop(columns=["decision_time", "close_time_4h"])
    df = df.sort_values("open_time").reset_index(drop=True)

    print(f"Merges completados. Filas: {len(df)}")

    # 2. Crear features
    df = crear_features(df)

    # 3. Crear label
    df = crear_label(df)

    # 4. Seleccionar features
    features, label, metadata = seleccionar_features(df)

    # 5. Guardar
    df = guardar_dataset(df, features, label, metadata)

    # 6. Resumen
    mostrar_resumen_features(df, features, label)


if __name__ == "__main__":
    main()
