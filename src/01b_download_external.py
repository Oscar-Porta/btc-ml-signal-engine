"""
==========================================================
FASE 9 (Iteración 1B) - Descarga de datos externos
==========================================================
Este script amplía la Fase 1 descargando datos de activos
externos al par BTC/USDT, para poder generar features que
aporten contexto macro al modelo:

    - ETH/USDT en 1h desde Binance:
        segundo cripto más grande. Nos servirá para calcular
        correlación y momentum relativo frente a BTC.

    - DXY (índice del dólar) en diario desde Yahoo Finance:
        cuando el dólar se fortalece, los activos de riesgo
        (BTC incluido) suelen sufrir. Lo usaremos como filtro
        de régimen macro.

Por qué DXY diario y no intra-día:
    Yahoo no ofrece DXY intra-día fiable y su influencia
    sobre BTC es de régimen, no de timing. El último valor
    diario disponible es suficiente.

Los datos se guardan en 'data/raw/' como CSV, en el mismo
formato que los archivos de BTC para simplificar los merges
posteriores.

Este script NO modifica nada existente: es puramente aditivo.
==========================================================
"""

import os
import pandas as pd
import yfinance as yf
from binance.client import Client


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
# --- ETH ---
ETH_SYMBOL = "ETHUSDT"
ETH_INTERVAL = Client.KLINE_INTERVAL_1HOUR
ETH_INTERVAL_NAME = "1h"

# --- DXY ---
# Ticker de Yahoo para el índice del dólar (US Dollar Index).
# Historia: yfinance 0.2.51 estaba roto y devolvía vacío para
# TODOS los tickers (incluidos AAPL y BTC-USD). Al actualizar
# a yfinance 1.2.2 (que incluye curl_cffi para evitar bloqueos
# de Yahoo) volvieron a funcionar. Volvemos al ticker original
# "DX-Y.NYB" (DXY spot de ICE), que es el índice DXY "de
# verdad" — no los futuros. Es el valor que se publica en los
# medios financieros.
DXY_TICKER = "DX-Y.NYB"
DXY_INTERVAL = "1d"
DXY_INTERVAL_NAME = "1d"

# Ventana de tiempo. Debe cubrir TODA la historia que tenemos
# de BTC (5 años). Yahoo acepta "period" o fechas explícitas.
LOOKBACK_BINANCE = "5 years ago UTC"
LOOKBACK_YAHOO = "5y"

# Carpeta de salida (misma que BTC para simplificar merges).
OUTPUT_DIR = "data/raw"

# Zona horaria local.
TIMEZONE = "Europe/Madrid"


# ==========================================================
# FUNCIÓN 1 - Descargar ETH desde Binance
# ==========================================================
def descargar_eth_binance():
    """
    Descarga velas de ETHUSDT en 1h desde Binance.
    Formato idéntico al de BTC (para que los scripts posteriores
    puedan tratarlos con la misma lógica).
    """
    print(f"\n>>> Descargando {ETH_SYMBOL} en {ETH_INTERVAL_NAME} "
          f"desde '{LOOKBACK_BINANCE}'...")

    client = Client()  # cliente público, sin claves
    klines = client.get_historical_klines(
        ETH_SYMBOL, ETH_INTERVAL, LOOKBACK_BINANCE
    )

    columnas = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(klines, columns=columnas)

    # Conversión de tiempos a zona horaria local (misma lógica
    # que 01_download_data.py).
    df["open_time"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
          .dt.tz_convert(TIMEZONE)
    )
    df["close_time"] = (
        pd.to_datetime(df["close_time"], unit="ms", utc=True)
          .dt.tz_convert(TIMEZONE)
    )

    # Conversión de precios y volúmenes a float.
    cols_numericas = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base", "taker_buy_quote"
    ]
    for col in cols_numericas:
        df[col] = df[col].astype(float)

    # Nos quedamos solo con las columnas útiles.
    df = df[[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "number_of_trades"
    ]]

    return df


# ==========================================================
# FUNCIÓN 2 - Descargar DXY desde Yahoo Finance
# ==========================================================
def descargar_dxy_yahoo():
    """
    Descarga el índice DXY diario desde Yahoo Finance.

    yfinance devuelve un DataFrame con:
      - Índice: fechas tipo Timestamp (por defecto tz-naive)
      - Columnas: Open, High, Low, Close, Volume, Dividends,
        Stock Splits

    Lo normalizamos para que tenga las mismas columnas y
    convenciones que los CSVs de Binance (open_time, open,
    high, low, close, volume).
    """
    print(f"\n>>> Descargando DXY ({DXY_TICKER}) en "
          f"{DXY_INTERVAL_NAME} desde Yahoo Finance...")

    ticker = yf.Ticker(DXY_TICKER)
    df_yf = ticker.history(period=LOOKBACK_YAHOO, interval=DXY_INTERVAL)

    if df_yf.empty:
        raise RuntimeError(
            f"Yahoo devolvió 0 filas para {DXY_TICKER}. "
            f"Revisa conexión o ticker."
        )

    # --- Normalización para que se parezca al formato Binance ---
    df = df_yf.reset_index()

    # Yahoo nombra la columna de fecha "Date" (diario) o
    # "Datetime" (intra-día). Normalizamos a "open_time".
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "open_time"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "open_time"})

    # Columnas OHLCV a minúsculas (estilo Binance).
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Dividends y Stock Splits no aplican al DXY, los quitamos
    # si existen.
    df = df.drop(
        columns=[c for c in ["Dividends", "Stock Splits"] if c in df.columns]
    )

    # --- Zona horaria ---
    # Yahoo devuelve las fechas diarias en UTC (o naive, según
    # versión). Las normalizamos a Europe/Madrid para coherencia
    # con los datos de Binance.
    if df["open_time"].dt.tz is None:
        df["open_time"] = df["open_time"].dt.tz_localize("UTC")
    df["open_time"] = df["open_time"].dt.tz_convert(TIMEZONE)

    # Aseguramos que los precios son float (yfinance ya lo hace,
    # pero por si acaso).
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Ordenamos columnas.
    columnas_finales = ["open_time", "open", "high", "low", "close", "volume"]
    df = df[columnas_finales].reset_index(drop=True)

    return df


# ==========================================================
# FUNCIÓN 3 - Validar DataFrame
# ==========================================================
def validar_velas(df, nombre):
    """
    Mismo formato de validación que 01_download_data.py:
    cuántas velas, rango temporal, duplicados, huecos.
    """
    print(f"\n--- Validación {nombre} ---")
    print(f"Total filas: {len(df)}")
    print(f"Desde: {df['open_time'].min()}")
    print(f"Hasta: {df['open_time'].max()}")

    duplicados = df["open_time"].duplicated().sum()
    print(f"Duplicados en open_time: {duplicados}")

    diffs = df["open_time"].diff().dropna()
    if len(diffs) > 0:
        diferencia_esperada = diffs.mode()[0]
        huecos = (diffs != diferencia_esperada).sum()
        print(f"Diferencia esperada entre filas: {diferencia_esperada}")
        # Solo añadimos la aclaración de fines de semana cuando
        # son datos diarios (caso DXY). En datos 1h (ETH/BTC) no
        # aplica.
        if "DXY" in nombre or "1d" in nombre:
            print(f"Huecos detectados: {huecos} "
                  f"(normal en DXY por fines de semana/festivos)")
        else:
            print(f"Huecos detectados: {huecos}")


# ==========================================================
# FUNCIÓN 4 - Guardar CSV
# ==========================================================
def guardar_csv(df, nombre_archivo):
    """Guarda un DataFrame en data/raw/<nombre_archivo>."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ruta = os.path.join(OUTPUT_DIR, nombre_archivo)
    df.to_csv(ruta, index=False)
    print(f"Guardado: {ruta}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("=" * 60)
    print("FASE 9 - Descarga de datos externos (ETH + DXY)")
    print("=" * 60)

    # --- ETH ---
    df_eth = descargar_eth_binance()
    validar_velas(df_eth, f"{ETH_SYMBOL} {ETH_INTERVAL_NAME}")
    guardar_csv(df_eth, f"{ETH_SYMBOL}_{ETH_INTERVAL_NAME}.csv")

    # --- DXY ---
    df_dxy = descargar_dxy_yahoo()
    validar_velas(df_dxy, f"DXY {DXY_INTERVAL_NAME}")
    guardar_csv(df_dxy, f"DXY_{DXY_INTERVAL_NAME}.csv")

    print("\n" + "=" * 60)
    print("Descarga externa terminada.")
    print("=" * 60)


if __name__ == "__main__":
    main()
