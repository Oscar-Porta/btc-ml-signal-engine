"""
==========================================================
FASE 3 - Reglas de entrada y generación de señales
==========================================================
Este script lee los CSV procesados (con indicadores) de
'data/processed/', aplica las reglas de entrada y genera
un CSV de señales en 'outputs/signals.csv'.

Arquitectura:
  1. Carga datos procesados de 4h y 1h.
  2. Alinea el contexto de 4h sobre cada vela de 1h
     (merge_asof: "¿qué pasaba en 4h cuando se formó
     esta vela de 1h?").
  3. Aplica reglas de entrada:
     - Regla 1: Pullback a EMA 20 en tendencia.
     - Regla 2: Breakout de estructura (swing high/low).
  4. Calcula niveles de SL, TP1 y TP2 en base al ATR.
  5. Guarda el CSV de señales con todo el contexto.

Filtro obligatorio:
  - Solo señales LONG si la tendencia 4h es 'up'.
  - Solo señales SHORT si la tendencia 4h es 'down'.
==========================================================
"""

import os
import pandas as pd
import numpy as np


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
SYMBOL = "BTCUSDT"

PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs"
SIGNALS_FILE = os.path.join(OUTPUT_DIR, "signals.csv")

# --- Parámetros de las reglas ---

# Regla Pullback: umbral de RSI para confirmar retroceso.
# Originalmente RSI < 45 / > 55, pero el análisis demostró
# que un pullback sano a la EMA 20 en tendencia tiene RSI
# en zona 45-55 (neutra, no débil). Relajamos a < 55 / > 45
# para no contradecir las otras condiciones de la regla.
PULLBACK_RSI_LONG_MAX = 55    # long: RSI debe estar por debajo
PULLBACK_RSI_SHORT_MIN = 45   # short: RSI debe estar por encima

# Regla Breakout: rango de RSI aceptable.
BREAKOUT_RSI_LONG_MIN = 40
BREAKOUT_RSI_LONG_MAX = 75
BREAKOUT_RSI_SHORT_MIN = 25
BREAKOUT_RSI_SHORT_MAX = 60

# --- Parámetros de gestión de riesgo ---

# Multiplicador de ATR para el stop loss.
# SL = entrada ± (ATR * SL_ATR_MULT)
SL_ATR_MULT = 1.25

# Multiplicador para take profit, expresado en "R".
# R = distancia entre entrada y SL.
# TP1 = entrada ± (R * TP1_R_MULT)  →  ratio 1:2
# Se cierra el 100% de la posición en TP1 (sin segunda fase).
# Decisión basada en el barrido del Anexo II: config A.
TP1_R_MULT = 2.0

# TP2 ya no se usa (cierre 100% en TP1). Se mantiene la
# variable para compatibilidad con scripts de análisis.
TP2_R_MULT = TP1_R_MULT


# ==========================================================
# FUNCIÓN 1 - Cargar datos procesados
# ==========================================================
def cargar_datos_procesados():
    """
    Lee los CSV procesados de 4h y 1h.
    Devuelve dos DataFrames con open_time como datetime.
    """
    df_4h = pd.read_csv(
        os.path.join(PROCESSED_DIR, f"{SYMBOL}_4h_processed.csv")
    )
    df_1h = pd.read_csv(
        os.path.join(PROCESSED_DIR, f"{SYMBOL}_1h_processed.csv")
    )

    # Forzamos la conversión a datetime.
    # parse_dates no siempre reconoce zonas horarias (+02:00),
    # así que usamos pd.to_datetime explícitamente con utc=True
    # y luego convertimos a Europe/Madrid.
    # Forzamos formato ISO8601 porque las fechas en el CSV
    # tienen formatos ligeramente distintos (unas con
    # microsegundos y otras sin). format='ISO8601' maneja
    # ambos casos correctamente.
    for df in [df_4h, df_1h]:
        for col in ["open_time", "close_time"]:
            df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True)
            df[col] = df[col].dt.tz_convert("Europe/Madrid")

    # Eliminamos filas con NaN en indicadores (periodo de
    # calentamiento). No tienen sentido para generar señales.
    cols_necesarias_4h = ["ema_20", "ema_50", "trend"]
    cols_necesarias_1h = [
        "ema_20", "ema_50", "rsi_14", "atr_14",
        "swing_high_20", "swing_low_20", "trend"
    ]

    df_4h = df_4h.dropna(subset=cols_necesarias_4h).reset_index(drop=True)
    df_1h = df_1h.dropna(subset=cols_necesarias_1h).reset_index(drop=True)

    print(f"Datos 4h cargados: {len(df_4h)} filas válidas")
    print(f"Datos 1h cargados: {len(df_1h)} filas válidas")

    return df_4h, df_1h


# ==========================================================
# FUNCIÓN 2 - Alinear contexto de 4h sobre 1h
# ==========================================================
def alinear_contexto_4h(df_1h, df_4h):
    """
    Para cada vela de 1h, busca la última vela de 4h que ya
    había CERRADO en el momento de decisión (el cierre de la
    vela de 1h, que es cuando se genera la señal).

    CORRECCIÓN LOOKAHEAD (revisión jul 2026):
    Antes se emparejaba por open_time <= open_time, lo que
    asignaba la vela 4h EN CURSO. Sus indicadores se calculan
    con su cierre, hasta 3 horas en el futuro respecto a la
    decisión → el backtest veía el futuro (75% de las señales
    afectadas). Ahora emparejamos por close_time de la vela 4h
    <= momento de decisión: solo velas 4h completadas, igual
    que en el sistema live.

    Usamos pd.merge_asof que hace exactamente este tipo
    de join temporal ("as-of merge").

    Añade columnas con sufijo '_4h' al DataFrame de 1h.
    """
    df_1h = df_1h.sort_values("open_time").reset_index(drop=True)
    df_4h_ctx = df_4h[["close_time", "trend", "ema_20", "ema_50"]].copy()

    # Renombramos antes del merge para evitar colisiones.
    df_4h_ctx = df_4h_ctx.rename(columns={
        "close_time": "close_time_4h",
        "trend": "trend_4h",
        "ema_20": "ema_20_4h",
        "ema_50": "ema_50_4h"
    })
    # merge_asof necesita ambos DataFrames ordenados por la
    # columna de merge.
    df_4h_ctx = df_4h_ctx.sort_values("close_time_4h").reset_index(drop=True)

    # Momento de decisión: la señal se evalúa al CIERRE de la
    # vela de 1h (open_time + 1h).
    df_1h["decision_time"] = df_1h["open_time"] + pd.Timedelta(hours=1)

    df = pd.merge_asof(
        df_1h,
        df_4h_ctx,
        left_on="decision_time",
        right_on="close_time_4h",
        direction="backward"   # última 4h CERRADA antes de decidir
    )

    # Columnas auxiliares del merge: fuera.
    df = df.drop(columns=["decision_time", "close_time_4h"])

    # Renombramos la tendencia original de 1h para claridad.
    df = df.rename(columns={"trend": "trend_1h"})

    print(f"Contexto 4h alineado. Filas resultantes: {len(df)}")

    return df


# ==========================================================
# FUNCIÓN 3 - Regla Pullback a EMA 20
# ==========================================================
def regla_pullback(df):
    """
    Regla 1: Pullback a la EMA 20 en tendencia.

    LONG:
      - trend_4h = 'up'  (tendencia macro alcista)
      - trend_1h = 'up'  (tendencia local alcista)
      - low <= ema_20    (la vela tocó la EMA por debajo)
      - close > ema_20   (pero cerró por encima → rebote)
      - rsi_14 < 45      (viene de retroceder, no sobrecomprado)

    SHORT (espejo):
      - trend_4h = 'down'
      - trend_1h = 'down'
      - high >= ema_20   (la vela tocó la EMA por arriba)
      - close < ema_20   (pero cerró por debajo → rechazo)
      - rsi_14 > 55      (viene de subir, no sobrevendido)

    Devuelve dos Series booleanas: (señales_long, señales_short)
    """
    # --- LONG ---
    long_mask = (
        (df["trend_4h"] == "up") &
        (df["trend_1h"] == "up") &
        (df["low"] <= df["ema_20"]) &
        (df["close"] > df["ema_20"]) &
        (df["rsi_14"] < PULLBACK_RSI_LONG_MAX)
    )

    # --- SHORT ---
    short_mask = (
        (df["trend_4h"] == "down") &
        (df["trend_1h"] == "down") &
        (df["high"] >= df["ema_20"]) &
        (df["close"] < df["ema_20"]) &
        (df["rsi_14"] > PULLBACK_RSI_SHORT_MIN)
    )

    return long_mask, short_mask


# ==========================================================
# FUNCIÓN 4 - Regla Breakout de estructura
# ==========================================================
def regla_breakout(df):
    """
    Regla 2: Breakout de estructura (ruptura de máximos/mínimos).

    LONG:
      - trend_4h = 'up'
      - close > swing_high_20 de la vela ANTERIOR
        (ruptura confirmada al cierre, no durante la vela)
      - rsi_14 entre 40 y 75 (ni débil ni agotado)

    SHORT (espejo):
      - trend_4h = 'down'
      - close < swing_low_20 de la vela ANTERIOR
      - rsi_14 entre 25 y 60

    Usamos .shift(1) para comparar con la vela anterior.
    Esto evita "lookahead bias": la referencia de estructura
    se fijó ANTES de que la vela actual se formara.

    Devuelve dos Series booleanas: (señales_long, señales_short)
    """
    # Niveles de estructura de la vela anterior.
    prev_swing_high = df["swing_high_20"].shift(1)
    prev_swing_low = df["swing_low_20"].shift(1)

    # --- LONG ---
    long_mask = (
        (df["trend_4h"] == "up") &
        (df["close"] > prev_swing_high) &
        (df["rsi_14"] >= BREAKOUT_RSI_LONG_MIN) &
        (df["rsi_14"] <= BREAKOUT_RSI_LONG_MAX)
    )

    # --- SHORT ---
    short_mask = (
        (df["trend_4h"] == "down") &
        (df["close"] < prev_swing_low) &
        (df["rsi_14"] >= BREAKOUT_RSI_SHORT_MIN) &
        (df["rsi_14"] <= BREAKOUT_RSI_SHORT_MAX)
    )

    return long_mask, short_mask


# ==========================================================
# FUNCIÓN 5 - Calcular niveles de SL, TP1 y TP2
# ==========================================================
def calcular_niveles(entry_price, atr, direction):
    """
    Calcula stop loss, take profit 1 y take profit 2
    a partir del precio de entrada, el ATR y la dirección.

    Para LONG:
      SL  = entry - (ATR * SL_ATR_MULT)
      TP1 = entry + (riesgo * TP1_R_MULT)
      TP2 = entry + (riesgo * TP2_R_MULT)
    donde riesgo = entry - SL.

    Para SHORT: al revés.

    Parámetros y devuelve: todo en Series de pandas para
    poder aplicarlo a muchas filas a la vez.
    """
    riesgo = atr * SL_ATR_MULT

    if direction == "LONG":
        sl = entry_price - riesgo
        tp1 = entry_price + (riesgo * TP1_R_MULT)
        tp2 = entry_price + (riesgo * TP2_R_MULT)
    else:  # SHORT
        sl = entry_price + riesgo
        tp1 = entry_price - (riesgo * TP1_R_MULT)
        tp2 = entry_price - (riesgo * TP2_R_MULT)

    risk_pct = (riesgo / entry_price) * 100

    return sl, tp1, tp2, riesgo, risk_pct


# ==========================================================
# FUNCIÓN 6 - Construir DataFrame de señales para una regla
# ==========================================================
def construir_senales(df, long_mask, short_mask, rule_name):
    """
    A partir de las máscaras booleanas de una regla, extrae
    las filas correspondientes y calcula los niveles de
    SL/TP para cada señal.

    Devuelve un DataFrame con todas las señales de esa regla.
    """
    senales = []

    # --- LONG ---
    df_long = df[long_mask].copy()
    if len(df_long) > 0:
        sl, tp1, tp2, risk, risk_pct = calcular_niveles(
            df_long["close"], df_long["atr_14"], "LONG"
        )
        df_long = df_long.assign(
            direction="LONG",
            rule=rule_name,
            entry_price=df_long["close"],
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            risk_usd=risk,
            risk_pct=risk_pct
        )
        senales.append(df_long)

    # --- SHORT ---
    df_short = df[short_mask].copy()
    if len(df_short) > 0:
        sl, tp1, tp2, risk, risk_pct = calcular_niveles(
            df_short["close"], df_short["atr_14"], "SHORT"
        )
        df_short = df_short.assign(
            direction="SHORT",
            rule=rule_name,
            entry_price=df_short["close"],
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            risk_usd=risk,
            risk_pct=risk_pct
        )
        senales.append(df_short)

    if senales:
        return pd.concat(senales, ignore_index=True)
    else:
        return pd.DataFrame()


# ==========================================================
# FUNCIÓN 7 - Generar todas las señales
# ==========================================================
def generar_senales(df):
    """
    Aplica todas las reglas de entrada y devuelve un único
    DataFrame con todas las señales, ordenadas por fecha.
    """
    # Regla 1: Pullback
    pb_long, pb_short = regla_pullback(df)
    df_pullback = construir_senales(df, pb_long, pb_short, "pullback")

    # Regla 2: Breakout
    bo_long, bo_short = regla_breakout(df)
    df_breakout = construir_senales(df, bo_long, bo_short, "breakout")

    # Unir todas las señales
    all_signals = pd.concat(
        [df_pullback, df_breakout], ignore_index=True
    )

    if all_signals.empty:
        print("No se generaron señales.")
        return all_signals

    # Ordenar por fecha
    all_signals = all_signals.sort_values("open_time").reset_index(drop=True)

    return all_signals


# ==========================================================
# FUNCIÓN 8 - Seleccionar columnas de salida
# ==========================================================
def preparar_salida(df_signals):
    """
    Selecciona y ordena las columnas que queremos en el CSV
    de señales. Incluye contexto de indicadores para que
    la IA pueda usarlos como features en la Fase 8.
    """
    columnas = [
        # --- Identificación de la señal ---
        "open_time",
        "direction",
        "rule",

        # --- Niveles del trade ---
        "entry_price",
        "stop_loss",
        "tp1",
        "tp2",
        "risk_usd",
        "risk_pct",

        # --- Contexto del mercado (features para IA) ---
        "open", "high", "low", "close", "volume",
        "ema_20", "ema_50", "rsi_14", "atr_14",
        "swing_high_20", "swing_low_20",
        "trend_1h", "trend_4h",
        "ema_20_4h", "ema_50_4h",
    ]

    # Solo incluimos las que existen (por seguridad).
    columnas_existentes = [c for c in columnas if c in df_signals.columns]

    return df_signals[columnas_existentes]


# ==========================================================
# FUNCIÓN 9 - Mostrar resumen de señales
# ==========================================================
def mostrar_resumen(df_signals):
    """
    Imprime un resumen rápido de las señales generadas
    para que el usuario pueda validar a simple vista.
    """
    print(f"\n{'='*60}")
    print("RESUMEN DE SEÑALES GENERADAS")
    print(f"{'='*60}")

    total = len(df_signals)
    print(f"Total señales: {total}")

    if total == 0:
        return

    # Por dirección
    print(f"\nPor dirección:")
    for d, count in df_signals["direction"].value_counts().items():
        pct = count / total * 100
        print(f"  {d}: {count} ({pct:.1f}%)")

    # Por regla
    print(f"\nPor regla:")
    for r, count in df_signals["rule"].value_counts().items():
        pct = count / total * 100
        print(f"  {r}: {count} ({pct:.1f}%)")

    # Por combinación dirección + regla
    print(f"\nPor dirección + regla:")
    combo = df_signals.groupby(["direction", "rule"]).size()
    for (d, r), count in combo.items():
        pct = count / total * 100
        print(f"  {d} / {r}: {count} ({pct:.1f}%)")

    # Rango temporal
    print(f"\nRango temporal:")
    print(f"  Primera señal: {df_signals['open_time'].min()}")
    print(f"  Última señal:  {df_signals['open_time'].max()}")

    # Riesgo medio
    print(f"\nRiesgo medio por señal:")
    print(f"  En USD: {df_signals['risk_usd'].mean():.2f}")
    print(f"  En %:   {df_signals['risk_pct'].mean():.2f}%")

    # Frecuencia aproximada
    rango_dias = (
        df_signals["open_time"].max() - df_signals["open_time"].min()
    ).days
    if rango_dias > 0:
        freq = total / rango_dias
        print(f"\nFrecuencia: ~{freq:.1f} señales/día "
              f"(~{freq*7:.0f} señales/semana)")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("Generando señales de entrada...\n")

    # 1. Cargar datos
    df_4h, df_1h = cargar_datos_procesados()

    # 2. Alinear contexto 4h → 1h
    df = alinear_contexto_4h(df_1h, df_4h)

    # 3. Generar señales
    df_signals = generar_senales(df)

    if df_signals.empty:
        return

    # 4. Preparar salida
    df_signals = preparar_salida(df_signals)

    # 5. Mostrar resumen
    mostrar_resumen(df_signals)

    # 6. Guardar CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_signals.to_csv(SIGNALS_FILE, index=False)
    print(f"\nSeñales guardadas en: {SIGNALS_FILE}")


if __name__ == "__main__":
    main()
