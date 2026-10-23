"""
==========================================================
FASE 9 (Iteración 1B) - Features externas
==========================================================
A partir de los CSV descargados por 01b_download_external.py:

    - data/raw/ETHUSDT_1h.csv
    - data/raw/DXY_1d.csv

Y usando también los datos de BTC ya existentes:

    - data/raw/BTCUSDT_1h.csv

Este script calcula 4 features que aportan contexto macro
al modelo y las deja alineadas a la granularidad de BTC
(1 vela = 1 hora), en un único archivo:

    data/processed/external_features.csv

Las 4 features son:

  1. eth_roc_5    → momentum de ETH en las últimas 5 velas
                    (5 horas). ROC = Rate of Change.
  2. eth_roc_20   → momentum de ETH en las últimas 20 velas
                    (~1 día).
  3. btc_eth_corr_20
                 → correlación móvil BTC-ETH sobre retornos
                    de las últimas 20 velas. Cerca de 1 →
                    mercado cripto "unido". Cerca de 0 o
                    negativo → desacople, señal más frágil.
  4. dxy_roc_5d   → ROC del DXY a 5 días. Positivo = dólar
                    fortaleciéndose (bearish para cripto).

Tratamiento del DXY (frecuencia diferente):
  Este script combina datos horarios (BTC, ETH) con datos
  diarios (DXY). Para alinearlos usamos FORWARD-FILL:
  para cada hora, aplicamos el último valor diario conocido
  del DXY. Esto replica exactamente lo que ve un trader
  mirando su terminal: durante las 24h del lunes se usa la
  cotización DXY del lunes, el fin de semana se usa la del
  viernes (porque ICE cierra). NO interpolamos porque eso
  inventaría datos que el mercado no produjo.

Salida:
  CSV con columnas [open_time, eth_roc_5, eth_roc_20,
  btc_eth_corr_20, dxy_roc_5d], una fila por cada vela 1h
  de BTC. Luego 06_prepare_dataset_v2.py hará el merge final.
==========================================================
"""

import os
import numpy as np
import pandas as pd


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
BTC_CSV = "data/raw/BTCUSDT_1h.csv"
ETH_CSV = "data/raw/ETHUSDT_1h.csv"
DXY_CSV = "data/raw/DXY_1d.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "external_features.csv")

# Ventanas de cálculo (en número de velas).
ETH_ROC_SHORT = 5      # 5 horas
ETH_ROC_LONG = 20      # 20 horas (~1 día)
CORR_WINDOW = 20       # 20 velas para correlación rolling BTC-ETH
DXY_ROC_DAYS = 5       # 5 días para ROC del DXY


# ==========================================================
# FUNCIÓN 1 - Cargar los CSVs
# ==========================================================
def cargar_csv(ruta, nombre):
    """
    Carga un CSV del proyecto y parsea 'open_time' como datetime
    con zona horaria Europe/Madrid.

    Nota técnica: pd.read_csv con parse_dates NO maneja bien
    strings con tz (+02:00) y a veces los deja como 'object'
    (strings). Forzamos el parseo con pd.to_datetime(utc=True)
    y convertimos después a Europe/Madrid. Así garantizamos
    dtype datetime64[ns, Europe/Madrid], que sí es compatible
    con pd.merge_asof().
    """
    print(f"  Cargando {nombre} desde {ruta}...")
    df = pd.read_csv(ruta)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True).dt.tz_convert(
        "Europe/Madrid"
    )
    # Aseguramos que open_time está ordenado (por si acaso).
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"    {len(df)} filas leídas.")
    return df


# ==========================================================
# FUNCIÓN 2 - Features horarias (ETH + correlación BTC/ETH)
# ==========================================================
def calcular_features_horarias(df_btc, df_eth):
    """
    Calcula las 3 features que viven en granularidad 1h:
      - eth_roc_5
      - eth_roc_20
      - btc_eth_corr_20

    Alinea BTC y ETH por 'open_time'. Si alguna vela falta
    en un lado, la descartamos (inner join) para no inventar
    datos. Esto puede dejar unas pocas filas fuera en los
    huecos de ETH (recuerda: 5 huecos sobre 43.812).
    """
    print("\n--- Calculando features horarias (ETH + correlación) ---")

    # Nos quedamos solo con lo necesario para reducir memoria.
    btc = df_btc[["open_time", "close"]].rename(columns={"close": "btc_close"})
    eth = df_eth[["open_time", "close"]].rename(columns={"close": "eth_close"})

    # Merge en la intersección temporal (INNER): solo horas que
    # existen en ambos.
    df = pd.merge(btc, eth, on="open_time", how="inner")
    print(f"  Filas tras merge BTC∩ETH: {len(df)}")

    # --- ROC de ETH ---
    # ROC = rate of change = (precio_actual / precio_hace_N) - 1.
    # Ejemplo: si eth_close pasa de 3000 a 3150 en 5h, ROC_5 = 0.05 (=+5%).
    df["eth_roc_5"] = df["eth_close"].pct_change(ETH_ROC_SHORT)
    df["eth_roc_20"] = df["eth_close"].pct_change(ETH_ROC_LONG)

    # --- Correlación móvil BTC-ETH ---
    # La correlación la calculamos sobre RETORNOS (pct_change),
    # no sobre precios absolutos: dos activos con tendencia suben
    # "a la vez" por escala, pero lo que nos interesa es si sus
    # movimientos hora a hora van al unísono.
    btc_ret = df["btc_close"].pct_change()
    eth_ret = df["eth_close"].pct_change()
    df["btc_eth_corr_20"] = btc_ret.rolling(CORR_WINDOW).corr(eth_ret)

    # Ya no necesitamos los closes crudos (solo queremos features).
    df = df.drop(columns=["btc_close", "eth_close"])

    print(f"  Features horarias calculadas: eth_roc_5, eth_roc_20, "
          f"btc_eth_corr_20")
    return df


# ==========================================================
# FUNCIÓN 3 - Feature diaria (DXY) expandida a horario
# ==========================================================
def calcular_feature_dxy(df_dxy, horas_objetivo):
    """
    Calcula dxy_roc_5d en granularidad diaria y luego lo
    expande a granularidad horaria con forward-fill, usando
    la lista de 'open_time' horarios como objetivo.

    Parámetros
    ----------
    df_dxy : DataFrame
        DXY diario con columnas [open_time, ..., close, ...].
    horas_objetivo : pd.Series
        Serie de timestamps horarios (los open_time de BTC)
        a los que queremos mapear el valor diario del DXY.
    """
    print("\n--- Calculando feature DXY (diario → horario con forward-fill) ---")

    # ROC 5 días sobre el close diario del DXY.
    dxy = df_dxy[["open_time", "close"]].copy()
    dxy = dxy.sort_values("open_time").reset_index(drop=True)
    dxy["dxy_roc_5d"] = dxy["close"].pct_change(DXY_ROC_DAYS)

    # Nos quedamos solo con [open_time, dxy_roc_5d].
    dxy = dxy[["open_time", "dxy_roc_5d"]]
    print(f"  Filas DXY diarias: {len(dxy)}")

    # --- Forward-fill a horario ---
    # Idea:
    #   1. Tomamos el DataFrame de horas objetivo (las de BTC).
    #   2. Lo mergeamos con DXY por "asof" — para cada hora,
    #      coge el último valor diario de DXY cuya fecha sea
    #      <= esa hora.
    #   3. pd.merge_asof hace exactamente eso.
    #
    # Detalle: merge_asof exige que ambos lados estén ordenados
    # por la clave y que la clave sea del mismo tipo. Los dos
    # son Timestamp con tz (Europe/Madrid), así que casa.
    horas_df = pd.DataFrame({"open_time": horas_objetivo}).sort_values(
        "open_time"
    )

    resultado = pd.merge_asof(
        horas_df,
        dxy,
        on="open_time",
        direction="backward",   # último valor diario anterior o igual
    )

    print(f"  Filas horarias con DXY asignado: {len(resultado)}")
    nans = resultado["dxy_roc_5d"].isna().sum()
    print(f"  NaN en dxy_roc_5d tras forward-fill: {nans} "
          f"(normal al principio: ROC necesita 5 días previos)")

    return resultado


# ==========================================================
# FUNCIÓN 4 - Combinar y guardar
# ==========================================================
def combinar_y_guardar(df_horarias, df_dxy_horaria):
    """Merge final de las 3 features horarias + la del DXY."""
    print("\n--- Combinando todo y guardando ---")
    df = pd.merge(df_horarias, df_dxy_horaria, on="open_time", how="left")

    columnas_finales = [
        "open_time",
        "eth_roc_5", "eth_roc_20",
        "btc_eth_corr_20",
        "dxy_roc_5d",
    ]
    df = df[columnas_finales]

    # Diagnóstico.
    print(f"  Filas totales: {len(df)}")
    print("  NaN por feature (esperable al inicio de la serie):")
    print(df.isna().sum().to_string())

    # Guardamos.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nGuardado: {OUTPUT_FILE}")

    # Pequeña muestra para ver que todo tiene pinta razonable.
    print("\nÚltimas 5 filas:")
    print(df.tail(5).to_string(index=False))


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("=" * 60)
    print("FASE 9 - Features externas (ETH + DXY)")
    print("=" * 60)

    # 1. Cargar.
    df_btc = cargar_csv(BTC_CSV, "BTC 1h")
    df_eth = cargar_csv(ETH_CSV, "ETH 1h")
    df_dxy = cargar_csv(DXY_CSV, "DXY 1d")

    # 2. Features horarias (ETH + correlación BTC-ETH).
    df_horarias = calcular_features_horarias(df_btc, df_eth)

    # 3. Feature DXY (diaria → forward-fill a horario).
    df_dxy_horaria = calcular_feature_dxy(df_dxy, df_horarias["open_time"])

    # 4. Merge + guardar.
    combinar_y_guardar(df_horarias, df_dxy_horaria)

    print("\n" + "=" * 60)
    print("Features externas generadas correctamente.")
    print("=" * 60)


if __name__ == "__main__":
    main()
