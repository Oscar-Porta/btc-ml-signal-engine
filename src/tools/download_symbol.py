"""
==========================================================
Descarga histórica de velas para CUALQUIER símbolo
==========================================================
Uso (desde la raíz del proyecto):
  python src/tools/download_symbol.py SOLUSDT 1h
  python src/tools/download_symbol.py SOLUSDT 1h 5   # años

Guarda data/raw/{SYMBOL}_{interval}.csv con EXACTAMENTE el
mismo formato que los históricos de BTC/ETH:
  open_time, open, high, low, close, volume,
  close_time, number_of_trades
(fechas en Europe/Madrid, ISO8601)

Creado en la revisión jul-2026 (Fase D: pooling multi-activo).
==========================================================
"""

import os
import sys
import pandas as pd
from binance.client import Client

TZ = "Europe/Madrid"


def main():
    if len(sys.argv) < 3:
        print("Uso: python src/tools/download_symbol.py SYMBOL INTERVALO [AÑOS]")
        print("Ej.:  python src/tools/download_symbol.py SOLUSDT 1h 5")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    interval = sys.argv[2]
    anios = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    print(f"Descargando {symbol} {interval} ({anios} años)...")
    client = Client()
    klines = client.get_historical_klines(
        symbol, interval, f"{anios} year ago UTC"
    )
    print(f"  {len(klines)} velas recibidas")

    columnas = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(klines, columns=columnas)

    for c in ["open_time", "close_time"]:
        df[c] = pd.to_datetime(df[c], unit="ms", utc=True).dt.tz_convert(TZ)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)

    # Descartar la última vela si aún está en formación
    ahora = pd.Timestamp.now(tz=TZ)
    df = df[df["close_time"] <= ahora].reset_index(drop=True)

    # Mismo esquema que BTCUSDT_1h.csv / ETHUSDT_1h.csv
    df = df[["open_time", "open", "high", "low", "close", "volume",
             "close_time", "number_of_trades"]]

    os.makedirs("data/raw", exist_ok=True)
    ruta = f"data/raw/{symbol}_{interval}.csv"
    df.to_csv(ruta, index=False)
    print(f"  Guardado: {ruta}")
    print(f"  Rango: {df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}")


if __name__ == "__main__":
    main()
