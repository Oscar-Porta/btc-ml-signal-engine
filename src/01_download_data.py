"""
==========================================================
FASE 1 - Descarga de datos históricos desde Binance
==========================================================
Este script se conecta al API público de Binance y descarga
las velas OHLCV (open, high, low, close, volume) del par
BTC/USDT en dos temporalidades:

    - 4h: la usaremos como contexto / filtro de tendencia.
    - 1h: la usaremos para generar señales de entrada.

Los datos se guardan en 'data/raw/' como archivos CSV.
No se necesitan claves de API para datos públicos históricos.
==========================================================
"""

import os
import pandas as pd
from binance.client import Client


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
SYMBOL = "BTCUSDT"

# Cuánta historia queremos descargar.
# Fijamos 5 años como constante del proyecto: cubre un ciclo
# completo del halving de BTC y contiene todos los regímenes
# de mercado relevantes (techo 2021, bear 2022, halving 2024,
# bull post-halving). Es el mejor equilibrio entre volumen de
# datos y relevancia para el BTC actual.
LOOKBACK = "5 years ago UTC"

# Temporalidades que vamos a descargar.
# La clave (str) es lo que usamos para el nombre del archivo.
# El valor es la constante que entiende python-binance.
INTERVALS = {
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1h": Client.KLINE_INTERVAL_1HOUR,
}

# Carpeta de salida (relativa a la raíz del proyecto).
OUTPUT_DIR = "data/raw"

# Zona horaria local: convertimos desde UTC a Madrid.
TIMEZONE = "Europe/Madrid"


# ==========================================================
# FUNCIÓN 1 - Descargar velas desde Binance
# ==========================================================
def descargar_velas(symbol, interval, lookback):
    """
    Descarga velas OHLCV desde Binance para un símbolo,
    un intervalo y un periodo de tiempo hacia atrás.

    Devuelve un DataFrame de pandas con los datos ya limpios
    y con los tiempos convertidos a la zona horaria local.
    """
    client = Client()  # cliente público, sin claves

    # get_historical_klines se encarga de paginar por nosotros
    # si el periodo es grande (Binance devuelve máximo 1000
    # velas por petición).
    klines = client.get_historical_klines(symbol, interval, lookback)

    # Las 12 columnas que devuelve Binance, en orden.
    columnas = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(klines, columns=columnas)

    # --- Conversión de tiempos ---
    # Los tiempos vienen como milisegundos desde 1970 (epoch).
    # Los pasamos a datetime en UTC y luego a Europe/Madrid.
    df["open_time"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
          .dt.tz_convert(TIMEZONE)
    )
    df["close_time"] = (
        pd.to_datetime(df["close_time"], unit="ms", utc=True)
          .dt.tz_convert(TIMEZONE)
    )

    # --- Conversión de precios y volúmenes a float ---
    # Binance los devuelve como strings, los queremos numéricos.
    cols_numericas = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base", "taker_buy_quote"
    ]
    for col in cols_numericas:
        df[col] = df[col].astype(float)

    # Nos quedamos solo con las columnas que realmente usaremos.
    df = df[[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "number_of_trades"
    ]]

    return df


# ==========================================================
# FUNCIÓN 2 - Validar el DataFrame descargado
# ==========================================================
def validar_velas(df, nombre):
    """
    Comprueba que los datos descargados son consistentes:
    - cuántas velas hay
    - rango temporal
    - duplicados en open_time
    - huecos (velas que faltan)

    No modifica el DataFrame, solo imprime un resumen.
    """
    print(f"\n--- Validación {nombre} ---")
    print(f"Total velas: {len(df)}")
    print(f"Desde: {df['open_time'].min()}")
    print(f"Hasta: {df['open_time'].max()}")

    duplicados = df["open_time"].duplicated().sum()
    print(f"Duplicados en open_time: {duplicados}")

    # Huecos: la diferencia entre dos velas consecutivas debería
    # ser siempre la misma (1h, 4h...). Si no lo es, falta alguna.
    diffs = df["open_time"].diff().dropna()
    if len(diffs) > 0:
        diferencia_esperada = diffs.mode()[0]
        huecos = (diffs != diferencia_esperada).sum()
        print(f"Diferencia esperada entre velas: {diferencia_esperada}")
        print(f"Huecos detectados: {huecos}")


# ==========================================================
# FUNCIÓN 3 - Guardar el DataFrame como CSV
# ==========================================================
def guardar_csv(df, interval_nombre):
    """
    Guarda el DataFrame en 'data/raw/BTCUSDT_<interval>.csv'.
    Crea la carpeta si no existe.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nombre_archivo = f"{SYMBOL}_{interval_nombre}.csv"
    ruta = os.path.join(OUTPUT_DIR, nombre_archivo)

    df.to_csv(ruta, index=False)
    print(f"Guardado: {ruta}")


# ==========================================================
# MAIN - Flujo principal del script
# ==========================================================
def main():
    print(f"Descargando {SYMBOL} desde '{LOOKBACK}'...")

    for nombre, interval_binance in INTERVALS.items():
        print(f"\n>>> Descargando temporalidad {nombre}...")
        df = descargar_velas(SYMBOL, interval_binance, LOOKBACK)
        validar_velas(df, nombre)
        guardar_csv(df, nombre)

    print("\nDescarga terminada.")


if __name__ == "__main__":
    main()
