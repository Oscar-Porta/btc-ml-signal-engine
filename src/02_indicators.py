"""
==========================================================
FASE 2 - Cálculo de indicadores técnicos
==========================================================
Este script lee los CSV "crudos" de 'data/raw/' (producidos
por 01_download_data.py), añade columnas con los indicadores
técnicos que usará el sistema y guarda el resultado en
'data/processed/'.

Indicadores calculados:
  - EMA 20 (media móvil exponencial rápida)
  - EMA 50 (media móvil exponencial lenta)
  - RSI 14 (oscilador de fuerza relativa)
  - ATR 14 (volatilidad media)
  - Swing high / swing low de 20 velas (estructura)
  - Columna derivada 'trend' con 'up' o 'down'

No usamos librerías de terceros tipo ta / pandas-ta.
Solo pandas y numpy. Las fórmulas son las estándar.
==========================================================
"""

import os
import pandas as pd
import numpy as np


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
SYMBOL = "BTCUSDT"

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

INTERVALS = ["4h", "1h"]

# Parámetros de los indicadores.
# Son los valores estándar; cambiarlos afecta a todo el sistema.
EMA_FAST = 20
EMA_SLOW = 50
RSI_WINDOW = 14
ATR_WINDOW = 14
STRUCT_WINDOW = 20


# ==========================================================
# FUNCIÓN 1 - EMA (media móvil exponencial)
# ==========================================================
def calcular_ema(serie, periodo):
    """
    EMA estándar usando pandas.

    adjust=False usa la fórmula recursiva clásica:
        EMA_hoy = precio * alpha + EMA_ayer * (1 - alpha)
    donde alpha = 2 / (periodo + 1).

    min_periods=periodo obliga a tener al menos 'periodo'
    velas reales antes de devolver un valor. Antes de eso,
    devuelve NaN. Esto evita confiar en valores de
    calentamiento que todavía no son representativos.
    """
    return serie.ewm(span=periodo, adjust=False, min_periods=periodo).mean()


# ==========================================================
# FUNCIÓN 2 - RSI (Relative Strength Index)
# ==========================================================
def calcular_rsi(close, periodo=14):
    """
    RSI con suavizado de Wilder (el estándar).

    Pasos:
      1. delta = diferencia entre cierres consecutivos
      2. separar en ganancias (delta>0) y pérdidas (delta<0)
      3. suavizar ganancias y pérdidas con EMA de Wilder
         (alpha = 1/periodo)
      4. rs = avg_gain / avg_loss
      5. rsi = 100 - 100 / (1 + rs)
    """
    delta = close.diff()

    ganancia = delta.clip(lower=0)      # pone en 0 los negativos
    perdida = -delta.clip(upper=0)      # pone en 0 los positivos y cambia signo

    # min_periods=periodo fuerza NaN en las primeras filas
    # hasta tener suficiente historial para un RSI honesto.
    avg_gain = ganancia.ewm(
        alpha=1 / periodo, adjust=False, min_periods=periodo
    ).mean()
    avg_loss = perdida.ewm(
        alpha=1 / periodo, adjust=False, min_periods=periodo
    ).mean()

    # Evitamos división por cero sustituyendo 0 por NaN.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ==========================================================
# FUNCIÓN 3 - ATR (Average True Range)
# ==========================================================
def calcular_atr(df, periodo=14):
    """
    ATR con suavizado de Wilder sobre el True Range.

    True Range de una vela = máximo de:
      - high - low
      - |high - close_anterior|
      - |low - close_anterior|

    Luego suavizamos con EMA de Wilder (alpha = 1/periodo).
    """
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    # Para cada fila cogemos el mayor de los tres componentes.
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # min_periods=periodo fuerza NaN en las primeras filas
    # hasta tener suficiente historial para un ATR honesto.
    atr = tr.ewm(
        alpha=1 / periodo, adjust=False, min_periods=periodo
    ).mean()
    return atr


# ==========================================================
# FUNCIÓN 4 - Estructura: swing high / swing low rolling
# ==========================================================
def calcular_estructura(df, periodo=20):
    """
    Máximo y mínimo rolling de las últimas 'periodo' velas.

    Importante: 'rolling' en pandas incluye la vela actual,
    es decir, swing_high_20 de la fila i es el máximo de los
    'high' entre las filas [i-19, i]. Esto NO mira al futuro.
    """
    swing_high = df["high"].rolling(window=periodo).max()
    swing_low = df["low"].rolling(window=periodo).min()
    return swing_high, swing_low


# ==========================================================
# FUNCIÓN 5 - Tendencia derivada ('up' / 'down')
# ==========================================================
def calcular_tendencia(df, col_fast, col_slow):
    """
    Devuelve una serie con 'up' si EMA rápida > EMA lenta,
    'down' en caso contrario. Si alguna EMA es NaN, devuelve NaN.
    """
    up = df[col_fast] > df[col_slow]
    # np.where devuelve un array; lo pasamos a Series con el mismo índice.
    trend = pd.Series(
        np.where(up, "up", "down"),
        index=df.index
    )
    # Marcamos como NaN las filas donde las EMAs aún no tienen valor.
    trend[df[col_fast].isna() | df[col_slow].isna()] = np.nan
    return trend


# ==========================================================
# FUNCIÓN 6 - Procesar una temporalidad entera
# ==========================================================
def procesar_temporalidad(interval):
    """
    Lee el CSV crudo de una temporalidad, calcula todos los
    indicadores, valida y guarda el CSV procesado.
    """
    path_in = os.path.join(RAW_DIR, f"{SYMBOL}_{interval}.csv")
    path_out = os.path.join(
        PROCESSED_DIR, f"{SYMBOL}_{interval}_processed.csv"
    )

    print(f"\n>>> Procesando temporalidad {interval}...")

    # --- Lectura ---
    df = pd.read_csv(path_in, parse_dates=["open_time", "close_time"])

    # --- Cálculo de indicadores ---
    col_ema_fast = f"ema_{EMA_FAST}"
    col_ema_slow = f"ema_{EMA_SLOW}"
    col_rsi = f"rsi_{RSI_WINDOW}"
    col_atr = f"atr_{ATR_WINDOW}"
    col_swing_high = f"swing_high_{STRUCT_WINDOW}"
    col_swing_low = f"swing_low_{STRUCT_WINDOW}"

    df[col_ema_fast] = calcular_ema(df["close"], EMA_FAST)
    df[col_ema_slow] = calcular_ema(df["close"], EMA_SLOW)
    df[col_rsi] = calcular_rsi(df["close"], RSI_WINDOW)
    df[col_atr] = calcular_atr(df, ATR_WINDOW)
    df[col_swing_high], df[col_swing_low] = calcular_estructura(
        df, STRUCT_WINDOW
    )
    df["trend"] = calcular_tendencia(df, col_ema_fast, col_ema_slow)

    # --- Validación ---
    print(f"Total filas: {len(df)}")
    print(f"Columnas finales: {list(df.columns)}")

    # Contamos NaN por columna (esperamos algunos en las
    # primeras filas por los periodos de calentamiento).
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        print("NaN por columna (solo columnas con NaN):")
        for col, n in nan_counts.items():
            print(f"  {col}: {n}")

    # ¿En qué fila empiezan a estar todos los indicadores listos?
    cols_indicadores = [
        col_ema_fast, col_ema_slow, col_rsi,
        col_atr, col_swing_high, col_swing_low, "trend"
    ]
    df_listos = df.dropna(subset=cols_indicadores)
    if not df_listos.empty:
        primera = df_listos.iloc[0]
        print(
            f"Primera vela con TODOS los indicadores calculados: "
            f"fila {df_listos.index[0]}, fecha {primera['open_time']}"
        )

    # --- Guardado ---
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(path_out, index=False)
    print(f"Guardado: {path_out}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("Calculando indicadores técnicos sobre los datos crudos...")

    for interval in INTERVALS:
        procesar_temporalidad(interval)

    print("\nProcesamiento terminado.")


if __name__ == "__main__":
    main()
