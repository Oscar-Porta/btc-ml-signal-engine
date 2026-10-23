"""
==========================================================
FASE 7 - Señales en vivo con notificación por Telegram
==========================================================
Este script descarga las velas más recientes de Binance,
calcula los indicadores técnicos, comprueba si la última
vela cerrada cumple alguna regla de entrada y, si hay
señal, envía una notificación a Telegram.

Diseño:
  - Se ejecuta una vez cada hora (al cierre de la vela 1h).
  - Descarga suficientes velas para que los indicadores
    tengan un calentamiento correcto (~100 velas).
  - Reutiliza EXACTAMENTE las mismas fórmulas de cálculo
    que los scripts del histórico (02, 03) para garantizar
    consistencia.
  - Si existe un modelo ML entrenado (outputs/model.joblib),
    lo carga y añade la probabilidad de éxito a la señal.
  - Envía un mensaje a Telegram con los detalles completos.
  - Si no hay señal, no envía nada (silencioso).

Ejecución manual (una única comprobación):
  python src/08_live_signal.py

Ejecución en modo bucle continuo (se queda corriendo,
comprueba automáticamente al cierre de cada vela 1h):
  python src/08_live_signal.py --loop

Para detener el bucle: Ctrl+C

Ejecución automática (cada hora, ejemplo con cron):
  1 * * * * cd /ruta/al/proyecto && .venv/bin/python src/08_live_signal.py

Requiere archivo .env con:
  TELEGRAM_BOT_TOKEN=tu_token
  TELEGRAM_CHAT_ID=tu_chat_id
==========================================================
"""

import os
import sys
import json
import time
import logging
import traceback
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from binance.client import Client
from dotenv import load_dotenv

# Paper trading en testnet (opcional)
try:
    from testnet_trader import ejecutar_orden_testnet, testnet_configurado
except ImportError:
    # Si el módulo no existe, testnet desactivado
    def ejecutar_orden_testnet(signal_info):
        return None
    def testnet_configurado():
        return False

# Cargar variables de entorno (.env)
load_dotenv()


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
SYMBOL = "BTCUSDT"
TIMEZONE = "Europe/Madrid"

# Cuántas velas descargar para el cálculo de indicadores.
# Necesitamos al menos 50 (para EMA 50) + margen.
# 150 velas de 1h = ~6 días, sobra de sobra.
# 80 velas de 4h = ~13 días, sobra también.
CANDLES_1H = 150
CANDLES_4H = 80

# Parámetros de indicadores (idénticos a 02_indicators.py)
EMA_FAST = 20
EMA_SLOW = 50
RSI_WINDOW = 14
ATR_WINDOW = 14
STRUCT_WINDOW = 20

# Parámetros de reglas (idénticos a 03_rules.py)
PULLBACK_RSI_LONG_MAX = 55
PULLBACK_RSI_SHORT_MIN = 45
BREAKOUT_RSI_LONG_MIN = 40
BREAKOUT_RSI_LONG_MAX = 75
BREAKOUT_RSI_SHORT_MIN = 25
BREAKOUT_RSI_SHORT_MAX = 60

# Parámetros de riesgo (idénticos a 03_rules.py — Config A, Anexo II)
SL_ATR_MULT = 1.25
TP1_R_MULT = 2.0
TP2_R_MULT = TP1_R_MULT  # compatibilidad, no se usa (cierre 100% en TP1)

# Umbral de probabilidad del modelo ML.
# Solo se muestra en el mensaje, NO filtra señales.
# El operador decide si operar o no.
AI_THRESHOLD = 0.55

# Modo bucle continuo: segundos de margen tras el cierre
# de la vela para asegurarnos de que Binance ya la tiene
# disponible en su API.
LOOP_MARGIN_SECONDS = 30

# Archivos
# DESPLIEGUE jul-2026: nombre EXPLÍCITO del modelo (lección del bug
# 10.1: model.joblib genérico provocó 54 días con el modelo viejo).
# model_limit_btc.joblib = entrenado con ejecución LIMIT, etiquetas
# netas y sin lookahead 4h (sklearn 1.5.2, la del VPS).
MODEL_FILE = os.getenv("MODEL_FILE", "outputs/model_limit_btc.joblib")
LOG_FILE = "outputs/live_signal.log"
LAST_CHECK_FILE = "outputs/last_check.json"
OPEN_POSITIONS_FILE = "outputs/open_positions.json"
# Órdenes limit pendientes de ejecutarse (nuevo flujo limit jul-2026).
# El monitor (10) comprueba si se ejecutan o expiran.
OPEN_ORDERS_FILE = "outputs/open_orders.json"
# Validez de la orden limit: 1 vela de 1h (igual que el backtest
# validado: colocación al close de la señal, ventana W=1).
ORDER_VALIDITY_HOURS = 1

# Resumen diario: hora CET a la que se envía (8:00)
DAILY_SUMMARY_HOUR = 8

# URL pública del dashboard (para incluir en mensajes)
DASHBOARD_URL = os.getenv(
    "DASHBOARD_URL", "http://localhost:8501"
)

# Telegram — canal always-on (monitoreo interno, sin filtro de umbral)
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ==========================================================
# ROUTING MULTICANAL — Bloque C1
# ==========================================================
# REVISIÓN jul-2026: durante la validación live, un ÚNICO canal de
# producto (BTC · Selectivo · 24h, umbral 0.60). Motivos:
#   - El umbral "Activo" (0.45) no es rentable con métricas honestas
#     (PF 0.97 a jornada completa) → retirado.
#   - Segmentar por franja horaria deja muestras diminutas (28-36
#     trades en 3 años) sin fiabilidad estadística → se pospone.
# El canal ALWAYS-ON (interno) sigue recibiendo TODAS las señales con
# su probabilidad — sirve al operador para auditar el routing en vivo.
# Reactivar franjas/Activo cuando el live acumule muestra suficiente.
MULTICANAL_ENABLED = True

# Umbral único de producto (el validado en walk-forward: +0.098R, 3/3 años)
UMBRAL_SELECTIVO = 0.60
UMBRAL_ACTIVO    = UMBRAL_SELECTIVO  # compat.: "Activo" retirado, apunta a 0.60

# Franjas horarias: solo jornada completa durante la validación.
FRANJAS = {
    "fulltime": {"horas": range(0, 24),  "solo_laborables": False},
}

# Chat ID del único canal de producto activo.
CANALES_PRODUCTO = {
    ("fulltime", "selectivo"): os.getenv("TELEGRAM_BTC_FULLTIME_SELECTIVO", ""),
}


def get_canales_activos(ai_proba, hora_madrid, dia_semana):
    """
    Devuelve la lista de chat_ids de los canales de producto que deben
    recibir esta señal, según la franja horaria activa y el umbral ML.

    Parámetros:
      ai_proba   — probabilidad del modelo (float 0-1). Si es None,
                   solo se envía al canal always-on (sin routing).
      hora_madrid — hora actual en Madrid (int 0-23)
      dia_semana  — día de la semana (0=lunes … 6=domingo)

    Lógica:
      - Full Time: activo todos los días, las 24h.
      - Mañana / América: solo lunes–viernes.
      - Umbral Activo (≥0.45): señal pasa el primer filtro.
      - Umbral Selectivo (≥0.60): señal pasa el filtro más estricto.
      - Si ai_proba es None (modelo no disponible), no se envía a
        ningún canal de producto (no podemos evaluar el umbral).
    """
    if not MULTICANAL_ENABLED:
        return []

    if ai_proba is None:
        log("[Routing] Modelo ML no disponible — no se envía a canales de producto.",
            "warning")
        return []

    canales = []
    es_laborable = dia_semana < 5  # lunes=0 … viernes=4

    for (franja, umbral), chat_id in CANALES_PRODUCTO.items():
        if not chat_id:
            continue

        cfg = FRANJAS[franja]

        # Comprobar franja horaria
        if cfg["solo_laborables"] and not es_laborable:
            continue
        if hora_madrid not in cfg["horas"]:
            continue

        # Comprobar umbral
        thresh = UMBRAL_SELECTIVO if umbral == "selectivo" else UMBRAL_ACTIVO
        if ai_proba >= thresh:
            canales.append((chat_id, franja, umbral))

    return canales


# ==========================================================
# CONFIGURACIÓN DEL LOGGER
# ==========================================================
# Creamos un logger con dos salidas simultáneas:
#   - Consola (stdout): para ver en tiempo real.
#   - Archivo outputs/live_signal.log: para consultar después.
# Cada línea lleva timestamp y nivel (INFO, WARNING, ERROR).
os.makedirs("outputs", exist_ok=True)

logger = logging.getLogger("live_signal")
logger.setLevel(logging.INFO)
# Evitar duplicados si el módulo se recarga
logger.handlers.clear()

_formato = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Handler de consola
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(_formato)
logger.addHandler(_console)

# Handler de archivo (modo append, no sobrescribe)
_archivo = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
_archivo.setFormatter(_formato)
logger.addHandler(_archivo)


def log(mensaje, nivel="info"):
    """
    Envoltorio para escribir al logger. Simplifica sustituir
    los print() existentes. Niveles válidos: info, warning, error.
    """
    if nivel == "warning":
        logger.warning(mensaje)
    elif nivel == "error":
        logger.error(mensaje)
    else:
        logger.info(mensaje)


# ==========================================================
# BLOQUE 1 - Descarga de velas recientes (con reintentos)
# ==========================================================
# Configuración de reintentos para la API de Binance
BINANCE_MAX_RETRIES = 3
BINANCE_RETRY_BACKOFF = [5, 10, 20]  # segundos entre reintentos


def descargar_velas_recientes(symbol, interval, limit):
    """
    Descarga las últimas 'limit' velas de Binance.
    Devuelve un DataFrame limpio con la misma estructura
    que los CSV del histórico.

    Incluye reintentos automáticos ante errores transitorios
    (timeout, fallos de red, 5xx de Binance). Si tras
    BINANCE_MAX_RETRIES intentos sigue fallando, lanza la
    excepción para que la capa superior decida qué hacer.
    """
    client = Client()

    ultimo_error = None
    for intento in range(1, BINANCE_MAX_RETRIES + 1):
        try:
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            break  # éxito, salimos del bucle de reintentos
        except Exception as e:
            ultimo_error = e
            if intento < BINANCE_MAX_RETRIES:
                espera = BINANCE_RETRY_BACKOFF[intento - 1]
                log(f"[Binance {interval}] Intento {intento} falló "
                    f"({type(e).__name__}: {e}). "
                    f"Reintentando en {espera}s...", "warning")
                time.sleep(espera)
            else:
                log(f"[Binance {interval}] Fallaron "
                    f"{BINANCE_MAX_RETRIES} intentos consecutivos.",
                    "error")
                raise ultimo_error

    columnas = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(klines, columns=columnas)

    # Conversión de tiempos
    df["open_time"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
          .dt.tz_convert(TIMEZONE)
    )
    df["close_time"] = (
        pd.to_datetime(df["close_time"], unit="ms", utc=True)
          .dt.tz_convert(TIMEZONE)
    )

    # Conversión de precios a float
    cols_numericas = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base", "taker_buy_quote"
    ]
    for col in cols_numericas:
        df[col] = df[col].astype(float)

    df = df[[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "number_of_trades"
    ]]

    return df


# ==========================================================
# BLOQUE 2 - Cálculo de indicadores
# (Mismas fórmulas que 02_indicators.py)
# ==========================================================
def calcular_ema(serie, periodo):
    """EMA estándar con min_periods para evitar calentamiento."""
    return serie.ewm(
        span=periodo, adjust=False, min_periods=periodo
    ).mean()


def calcular_rsi(close, periodo=14):
    """RSI con suavizado de Wilder."""
    delta = close.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)

    avg_gain = ganancia.ewm(
        alpha=1 / periodo, adjust=False, min_periods=periodo
    ).mean()
    avg_loss = perdida.ewm(
        alpha=1 / periodo, adjust=False, min_periods=periodo
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calcular_atr(df, periodo=14):
    """ATR con suavizado de Wilder."""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(
        alpha=1 / periodo, adjust=False, min_periods=periodo
    ).mean()


def calcular_estructura(df, periodo=20):
    """Swing high/low rolling."""
    swing_high = df["high"].rolling(window=periodo).max()
    swing_low = df["low"].rolling(window=periodo).min()
    return swing_high, swing_low


def calcular_tendencia(df, col_fast, col_slow):
    """Tendencia: 'up' si EMA rápida > lenta, 'down' si no."""
    up = df[col_fast] > df[col_slow]
    trend = pd.Series(np.where(up, "up", "down"), index=df.index)
    trend[df[col_fast].isna() | df[col_slow].isna()] = np.nan
    return trend


def procesar_indicadores(df):
    """Calcula todos los indicadores sobre un DataFrame de velas."""
    df = df.copy()
    df[f"ema_{EMA_FAST}"] = calcular_ema(df["close"], EMA_FAST)
    df[f"ema_{EMA_SLOW}"] = calcular_ema(df["close"], EMA_SLOW)
    df[f"rsi_{RSI_WINDOW}"] = calcular_rsi(df["close"], RSI_WINDOW)
    df[f"atr_{ATR_WINDOW}"] = calcular_atr(df, ATR_WINDOW)
    df[f"swing_high_{STRUCT_WINDOW}"], df[f"swing_low_{STRUCT_WINDOW}"] = (
        calcular_estructura(df, STRUCT_WINDOW)
    )
    df["trend"] = calcular_tendencia(
        df, f"ema_{EMA_FAST}", f"ema_{EMA_SLOW}"
    )
    return df


# ==========================================================
# BLOQUE 3 - Alineación de contexto 4h sobre 1h
# (Misma lógica que 03_rules.py)
# ==========================================================
def alinear_contexto_4h(df_1h, df_4h):
    """
    Para cada vela de 1h, busca la última vela de 4h cuyo
    open_time sea <= al open_time de la vela 1h.
    """
    df_1h = df_1h.sort_values("open_time").reset_index(drop=True)

    df_4h_ctx = df_4h[[
        "open_time", "trend", "ema_20", "ema_50", "rsi_14", "atr_14"
    ]].copy()
    df_4h_ctx = df_4h_ctx.sort_values("open_time").reset_index(drop=True)
    df_4h_ctx = df_4h_ctx.rename(columns={
        "trend": "trend_4h",
        "ema_20": "ema_20_4h",
        "ema_50": "ema_50_4h",
        "rsi_14": "rsi_14_4h",
        "atr_14": "atr_14_4h"
    })

    df = pd.merge_asof(
        df_1h, df_4h_ctx,
        on="open_time",
        direction="backward"
    )

    df = df.rename(columns={"trend": "trend_1h"})
    return df


# ==========================================================
# BLOQUE 4 - Reglas de entrada
# (Misma lógica que 03_rules.py, aplicada a UNA fila)
# ==========================================================
def comprobar_reglas(row, prev_row):
    """
    Comprueba si una vela cumple alguna regla de entrada.
    Recibe la fila actual (última vela cerrada) y la anterior.
    Devuelve una lista de señales encontradas (puede ser 0, 1 o 2).
    """
    senales = []

    trend_4h = row.get("trend_4h", None)
    trend_1h = row.get("trend_1h", None)
    close = row["close"]
    low = row["low"]
    high = row["high"]
    ema_20 = row["ema_20"]
    rsi = row["rsi_14"]

    # ---- PULLBACK ----
    # LONG
    if (trend_4h == "up" and trend_1h == "up" and
            low <= ema_20 and close > ema_20 and
            rsi < PULLBACK_RSI_LONG_MAX):
        senales.append({"direction": "LONG", "rule": "pullback"})

    # SHORT
    if (trend_4h == "down" and trend_1h == "down" and
            high >= ema_20 and close < ema_20 and
            rsi > PULLBACK_RSI_SHORT_MIN):
        senales.append({"direction": "SHORT", "rule": "pullback"})

    # ---- BREAKOUT ----
    if prev_row is not None:
        prev_swing_high = prev_row["swing_high_20"]
        prev_swing_low = prev_row["swing_low_20"]

        # LONG
        if (trend_4h == "up" and
                close > prev_swing_high and
                BREAKOUT_RSI_LONG_MIN <= rsi <= BREAKOUT_RSI_LONG_MAX):
            senales.append({"direction": "LONG", "rule": "breakout"})

        # SHORT
        if (trend_4h == "down" and
                close < prev_swing_low and
                BREAKOUT_RSI_SHORT_MIN <= rsi <= BREAKOUT_RSI_SHORT_MAX):
            senales.append({"direction": "SHORT", "rule": "breakout"})

    return senales


def calcular_niveles(entry_price, atr, direction):
    """
    Calcula SL, TP1, TP2 a partir del precio de entrada,
    el ATR y la dirección. Idéntico a 03_rules.py.
    """
    riesgo = atr * SL_ATR_MULT

    if direction == "LONG":
        sl = entry_price - riesgo
        tp1 = entry_price + (riesgo * TP1_R_MULT)
        tp2 = entry_price + (riesgo * TP2_R_MULT)
    else:
        sl = entry_price + riesgo
        tp1 = entry_price - (riesgo * TP1_R_MULT)
        tp2 = entry_price - (riesgo * TP2_R_MULT)

    risk_pct = (riesgo / entry_price) * 100
    return sl, tp1, tp2, riesgo, risk_pct


# ==========================================================
# BLOQUE 5 - Modelo ML (carga y predicción)
# ==========================================================
def cargar_modelo():
    """
    Carga el modelo entrenado si existe.
    Devuelve el modelo o None si no existe.
    """
    if os.path.exists(MODEL_FILE):
        import joblib
        modelo = joblib.load(MODEL_FILE)
        log(f"Modelo ML cargado: {MODEL_FILE}")
        return modelo
    else:
        log("No se encontró modelo ML. Se enviarán señales sin filtro.",
            "warning")
        return None


def crear_features_para_modelo(row):
    """
    Crea el vector de features para una señal, en el mismo
    orden que 06_prepare_dataset.py.
    """
    entry_price = row["close"]
    ema_20 = row["ema_20"]
    ema_50 = row["ema_50"]
    ema_20_4h = row.get("ema_20_4h", ema_20)
    ema_50_4h = row.get("ema_50_4h", ema_50)
    atr = row["atr_14"]
    swing_high = row["swing_high_20"]
    swing_low = row["swing_low_20"]
    rango = swing_high - swing_low

    # Distancias a EMAs
    dist_ema20_pct = (entry_price - ema_20) / ema_20 * 100
    dist_ema50_pct = (entry_price - ema_50) / ema_50 * 100

    # Fuerza de tendencia
    ema_spread_pct = (ema_20 - ema_50) / ema_50 * 100
    ema_spread_4h_pct = (ema_20_4h - ema_50_4h) / ema_50_4h * 100

    # Volatilidad relativa
    atr_pct = atr / entry_price * 100

    # Posición en rango
    price_position = (
        (entry_price - swing_low) / rango if rango > 0 else 0.5
    )

    # Vela de señal
    body_pct = (row["close"] - row["open"]) / row["open"] * 100
    body_top = max(row["open"], row["close"])
    body_bottom = min(row["open"], row["close"])
    full_range = row["high"] - row["low"]

    upper_wick_pct = (
        (row["high"] - body_top) / full_range * 100
        if full_range > 0 else 0
    )
    lower_wick_pct = (
        (body_bottom - row["low"]) / full_range * 100
        if full_range > 0 else 0
    )

    # Temporales
    hour = row["open_time"].hour
    day_of_week = row["open_time"].dayofweek

    # --- Nuevas features Iteración 1 ---
    # Volumen relativo
    vol_sma20 = row.get("volume_sma20", 0)
    volume_sma20_ratio = (
        row["volume"] / vol_sma20
        if vol_sma20 and vol_sma20 > 0 else 1.0
    )

    # RSI 4h
    rsi_14_4h = row.get("rsi_14_4h", 50.0)

    # ATR 4h normalizado
    atr_14_4h = row.get("atr_14_4h", atr)
    atr_pct_4h = atr_14_4h / entry_price * 100

    # Momentum
    close_5_ago = row.get("close_5_ago", entry_price)
    roc_5 = (
        (entry_price - close_5_ago) / close_5_ago * 100
        if close_5_ago and close_5_ago > 0 else 0.0
    )
    close_20_ago = row.get("close_20_ago", entry_price)
    roc_20 = (
        (entry_price - close_20_ago) / close_20_ago * 100
        if close_20_ago and close_20_ago > 0 else 0.0
    )

    # Régimen de volatilidad
    atr_sma20 = row.get("atr_sma20", atr)
    atr_regime = atr / atr_sma20 if atr_sma20 and atr_sma20 > 0 else 1.0

    # Tamaño de vela vs normal
    candle_range_ratio = (
        (row["high"] - row["low"]) / atr if atr > 0 else 1.0
    )

    return {
        "rsi_14": row["rsi_14"],
        "atr_14": atr,
        "dist_ema20_pct": dist_ema20_pct,
        "dist_ema50_pct": dist_ema50_pct,
        "ema_spread_pct": ema_spread_pct,
        "ema_spread_4h_pct": ema_spread_4h_pct,
        "atr_pct": atr_pct,
        "price_position": price_position,
        "body_pct": body_pct,
        "upper_wick_pct": upper_wick_pct,
        "lower_wick_pct": lower_wick_pct,
        "hour": hour,
        "day_of_week": day_of_week,
        # Nota Iteración 1B.2 (15-abr-2026): se eliminaron 4 features
        # del modelo porque eran duplicadas o inútiles:
        #   - risk_pct (duplicada 100% con atr_pct)
        #   - reward_risk_ratio (invariante = 2.0)
        #   - direction_num (importancia 0)
        #   - rule_num (importancia 0)
        # El modelo en producción es ahora el v1_pruned (20 features).
        # Nuevas features Iteración 1
        "volume_sma20_ratio": volume_sma20_ratio,
        "rsi_14_4h": rsi_14_4h,
        "atr_pct_4h": atr_pct_4h,
        "roc_5": roc_5,
        "roc_20": roc_20,
        "atr_regime": atr_regime,
        "candle_range_ratio": candle_range_ratio,
    }


def predecir_probabilidad(modelo, features_dict, direction, rule,
                          risk_pct):
    """
    Pasa las features por el modelo y devuelve la probabilidad
    de que el trade sea ganador.

    Nota Iteración 1B.2 (15-abr-2026): el modelo en producción
    es el 'v1 podado' (20 features). Se eliminaron risk_pct,
    reward_risk_ratio, direction_num y rule_num por ser
    duplicadas/inútiles. Los parámetros 'direction', 'rule' y
    'risk_pct' siguen recibiéndose por compatibilidad de firma
    (otras partes del código los pasan) pero no se usan para
    predecir.
    """
    # Orden de features (debe coincidir con el entrenamiento
    # del modelo v3 — 20 features, mismas que v1 podado.
    # Ver 07_train_model_v3.py. Label cambió a R≥0.5 pero
    # las features de entrada son idénticas.)
    orden = [
        "rsi_14", "atr_14", "dist_ema20_pct", "dist_ema50_pct",
        "ema_spread_pct", "ema_spread_4h_pct", "atr_pct",
        "price_position", "body_pct", "upper_wick_pct",
        "lower_wick_pct", "hour", "day_of_week",
        # Features Iteración 1
        "volume_sma20_ratio", "rsi_14_4h", "atr_pct_4h",
        "roc_5", "roc_20", "atr_regime", "candle_range_ratio",
    ]

    X = pd.DataFrame([{k: features_dict[k] for k in orden}])
    proba = modelo.predict_proba(X)[0][1]  # probabilidad de clase 1

    return proba


# ==========================================================
# BLOQUE 6 - Envío de mensaje a Telegram
# ==========================================================
# Configuración de reintentos para Telegram
TELEGRAM_MAX_RETRIES = 2
TELEGRAM_RETRY_BACKOFF = 3  # segundos entre reintentos


def enviar_telegram(mensaje):
    """
    Envía un mensaje de texto a Telegram usando la API HTTP.

    Con reintentos: si falla el primer intento, prueba una
    vez más tras una breve espera. Si sigue fallando, NO
    abortamos la señal (ya está calculada y registrada en
    CSV). Simplemente registramos el error en consola y
    devolvemos False.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram no configurado. Mensaje solo en consola.", "warning")
        log(mensaje)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje,
        "parse_mode": "HTML"
    }

    for intento in range(1, TELEGRAM_MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                log("Mensaje enviado a Telegram.")
                return True
            else:
                log(f"[Telegram] Intento {intento} respondió "
                    f"HTTP {response.status_code}: {response.text[:200]}",
                    "warning")
        except Exception as e:
            log(f"[Telegram] Intento {intento} falló: "
                f"{type(e).__name__}: {e}", "warning")

        if intento < TELEGRAM_MAX_RETRIES:
            time.sleep(TELEGRAM_RETRY_BACKOFF)

    log(f"[Telegram] Fallaron {TELEGRAM_MAX_RETRIES} intentos. "
        f"La señal está registrada en CSV pero NO se envió el aviso.",
        "error")
    return False


def _enviar_telegram_a(mensaje, chat_id):
    """
    Igual que enviar_telegram() pero a un chat_id específico.
    Se usa para el routing multicanal (canales de producto).
    """
    if not TELEGRAM_TOKEN or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": mensaje, "parse_mode": "HTML"}

    for intento in range(1, TELEGRAM_MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            else:
                log(f"[Routing] Intento {intento} → chat {chat_id}: "
                    f"HTTP {response.status_code}", "warning")
        except Exception as e:
            log(f"[Routing] Intento {intento} → chat {chat_id}: {e}", "warning")
        if intento < TELEGRAM_MAX_RETRIES:
            time.sleep(TELEGRAM_RETRY_BACKOFF)

    log(f"[Routing] Fallaron {TELEGRAM_MAX_RETRIES} intentos para chat {chat_id}.", "error")
    return False


def formatear_mensaje(signal_info):
    """
    Formatea el mensaje de Telegram con los detalles de la señal.
    Diseño: claro, directo, sin USD absolutos (solo % + ejemplo $100).
    """
    s = signal_info

    dir_emoji = "\U0001f7e2" if s["direction"] == "LONG" else "\U0001f534"
    dir_text = "LONG (compra)" if s["direction"] == "LONG" else "SHORT (venta)"

    # Hora limpia sin timezone
    time_str = str(s["time"])[:16].replace("T", " ") + " (Madrid)"

    # Ejemplo con $100
    risk_100 = abs(s["risk_pct"])  # con $100, el riesgo en $ = risk_pct

    msg = (
        f"<b>\U0001f514 SEÑAL BTC/USDT</b>\n"
        f"\n"
        f"{dir_emoji} <b>{dir_text}</b> — {s['rule']}\n"
        f"\U0001f4c5 {time_str}\n"
        f"\n"
        f"━━━ Cómo entrar ━━━\n"
        f"\U0001f4cc <b>Orden LIMIT a ${s['entry']:,.2f}</b>\n"
        f"\u23f3 Válida 1 hora — si no se ejecuta, cancélala\n"
        f"\n"
        f"━━━ Niveles (desde la entrada) ━━━\n"
        f"\U0001f6d1 Stop Loss: ${s['sl']:,.2f}\n"
        f"\u2705 TP1 (limit): ${s['tp1']:,.2f} (cierre 100%)\n"
        f"\n"
        f"━━━ Riesgo ━━━\n"
        f"\U0001f4d0 {s['risk_pct']:.2f}%"
        f" (con $100 → ${risk_100:.2f})\n"
    )

    # ML: solo confianza, sin mencionar umbrales
    if s.get("ai_proba") is not None:
        proba = s["ai_proba"]
        msg += f"\U0001f916 ML: {proba:.1%} de confianza\n"

    msg += (
        f"\n"
        f"━━━ Contexto ━━━\n"
        f"RSI: {s['rsi']:.1f} | ATR: ${s['atr']:,.2f}\n"
        f"Trend 1h: {s['trend_1h']} | Trend 4h: {s['trend_4h']}\n"
        f"\n"
        f"\U0001f4ca <a href=\"{DASHBOARD_URL}\">Síguela en vivo en el dashboard</a>\n"
        f"\n"
        f"<i>Sistema informativo. No es consejo financiero.</i>"
    )

    return msg


# ==========================================================
# BLOQUE 7 - Registro de señales en CSV
# ==========================================================
def registrar_senal(signal_info):
    """
    Guarda la señal en un CSV acumulativo para ir construyendo
    el histórico de señales en vivo. Cada ejecución añade una
    fila (append), no sobreescribe.
    """
    log_file = "outputs/live_signals_log.csv"
    os.makedirs("outputs", exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_time": str(signal_info["time"]),
        "direction": signal_info["direction"],
        "rule": signal_info["rule"],
        "entry_price": signal_info["entry"],
        "stop_loss": signal_info["sl"],
        "tp1": signal_info["tp1"],
        "tp2": signal_info["tp2"],
        "risk_pct": signal_info["risk_pct"],
        "risk_usd": signal_info["risk_usd"],
        "rsi": signal_info["rsi"],
        "atr": signal_info["atr"],
        "trend_1h": signal_info["trend_1h"],
        "trend_4h": signal_info["trend_4h"],
        "ai_proba": signal_info.get("ai_proba", None),
    }

    df = pd.DataFrame([row])

    if os.path.exists(log_file):
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

    log(f"Señal registrada en: {log_file}")


def registrar_senal_supabase(signal_info, ahora_madrid):
    """
    Inserta la señal en Supabase (tabla signals).
    Devuelve el UUID generado por Supabase, o None si falla.
    La función es tolerante a fallos: si Supabase no está disponible,
    el motor sigue funcionando con normalidad.
    """
    from supabase_client import get_supabase
    sb = get_supabase()
    if sb is None:
        log("[Supabase] Cliente no disponible — señal no registrada en BD.", "warning")
        return None

    try:
        data = {
            "instrument":   "BTC",
            "direction":    signal_info["direction"],
            "rule":         signal_info["rule"],
            "entry_price":  float(signal_info["entry"]),
            "stop_loss":    float(signal_info["sl"]),
            "tp1":          float(signal_info["tp1"]),
            "risk_pct":     float(signal_info["risk_pct"]) if signal_info.get("risk_pct") else None,
            "ai_proba":     float(signal_info["ai_proba"]) if signal_info.get("ai_proba") is not None else None,
            "hora_madrid":  ahora_madrid.hour,
            "dia_semana":   ahora_madrid.weekday(),
            "status":       "open",
        }
        result = sb.table("signals").insert(data).execute()
        supabase_id = result.data[0]["id"]
        log(f"[Supabase] Señal insertada — id: {supabase_id}")
        return supabase_id
    except Exception as e:
        log(f"[Supabase] Error al insertar señal: {e}", "warning")
        return None


def guardar_last_check(candle_time, close_price, signal_found,
                       direction=None, rule=None):
    """
    Guarda info de la última vela analizada en un JSON pequeño.
    El dashboard lo lee para mostrar que el bot sigue vivo,
    incluso cuando no hay señales.
    """
    data = {
        "candle_time": str(candle_time),
        "close_price": float(close_price),
        "signal_found": signal_found,
        "direction": direction,
        "rule": rule,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs("outputs", exist_ok=True)
    with open(LAST_CHECK_FILE, "w") as f:
        json.dump(data, f)


def registrar_orden_pendiente(signal_info):
    """
    FLUJO LIMIT (jul-2026): en vez de asumir entrada inmediata,
    la señal genera una ORDEN LIMIT pendiente. El monitor (10)
    comprueba cada ciclo si el precio la toca:
      - Si la toca antes de expirar → se convierte en posición.
      - Si expira sin tocarse → se cancela y se notifica.
    """
    ordenes = []
    if os.path.exists(OPEN_ORDERS_FILE):
        try:
            with open(OPEN_ORDERS_FILE) as f:
                ordenes = json.load(f)
        except Exception:
            ordenes = []

    orden_id = f"{signal_info['time']}_{signal_info['direction']}_{signal_info['rule']}"
    expiry = pd.Timestamp(signal_info["time"]) + pd.Timedelta(
        hours=1 + ORDER_VALIDITY_HOURS)
    # Nota: time = open_time de la vela de señal; la señal nace a su
    # cierre (+1h) y la orden vale ORDER_VALIDITY_HOURS más.

    ordenes.append({
        "order_id": orden_id,
        "signal_time": str(signal_info["time"]),
        "expiry": str(expiry),
        "direction": signal_info["direction"],
        "rule": signal_info["rule"],
        "limit_price": signal_info["entry"],
        "stop_loss": signal_info["sl"],
        "tp1": signal_info["tp1"],
        "risk_pct": signal_info["risk_pct"],
        "ai_proba": signal_info.get("ai_proba"),
        "supabase_id": signal_info.get("supabase_id"),
    })
    with open(OPEN_ORDERS_FILE, "w") as f:
        json.dump(ordenes, f, indent=2, default=str)
    log(f"Orden limit pendiente registrada: {orden_id} "
        f"(expira {expiry})")


def registrar_posicion_abierta(signal_info):
    """
    Añade una posición al archivo open_positions.json para que
    el monitor de posiciones (10_position_monitor.py) la recoja
    y empiece a vigilarla.
    """
    # Cargar posiciones existentes
    posiciones = []
    if os.path.exists(OPEN_POSITIONS_FILE):
        try:
            with open(OPEN_POSITIONS_FILE) as f:
                posiciones = json.load(f)
        except (json.JSONDecodeError, Exception):
            posiciones = []

    # Crear ID único para la posición
    pos_id = (
        f"{signal_info['direction']}_{signal_info['rule']}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )

    posicion = {
        "id": pos_id,
        "supabase_id": signal_info.get("supabase_id"),  # UUID para actualizar al cerrar
        "open_time": datetime.now(timezone.utc).isoformat(),
        "direction": signal_info["direction"],
        "rule": signal_info["rule"],
        "entry_price": signal_info["entry"],
        "stop_loss": signal_info["sl"],
        "tp1": signal_info["tp1"],
        "tp2": signal_info["tp2"],  # compatibilidad (= tp1)
        "risk_pct": signal_info["risk_pct"],
        "ai_proba": signal_info.get("ai_proba"),
    }

    posiciones.append(posicion)

    with open(OPEN_POSITIONS_FILE, "w") as f:
        json.dump(posiciones, f, indent=2, default=str)

    log(f"Posición abierta registrada: {pos_id}")


def _actualizar_supabase_id_en_posicion(pos_id, supabase_id):
    """
    Actualiza el supabase_id en open_positions.json una vez obtenido.
    Se llama justo después de registrar la señal en Supabase.
    """
    if not os.path.exists(OPEN_POSITIONS_FILE):
        return
    try:
        with open(OPEN_POSITIONS_FILE) as f:
            posiciones = json.load(f)
        for pos in posiciones:
            if pos.get("id") == pos_id:
                pos["supabase_id"] = supabase_id
                break
        with open(OPEN_POSITIONS_FILE, "w") as f:
            json.dump(posiciones, f, indent=2, default=str)
    except Exception as e:
        log(f"[Supabase] No se pudo actualizar supabase_id en JSON: {e}", "warning")


# ==========================================================
# BLOQUE 8 - Ejecución única (una comprobación)
# ==========================================================
def ejecutar_comprobacion():
    """
    Ejecuta UNA comprobación completa:
      1. Descarga velas de Binance
      2. Calcula indicadores
      3. Comprueba reglas sobre la última vela cerrada
      4. Si hay señal, envía a Telegram y registra en CSV

    Esta función es la "unidad de trabajo" del script.
    En modo único se ejecuta una vez y el script termina.
    En modo bucle se ejecuta cada hora al cierre de vela.
    """
    ahora = datetime.now(timezone.utc).astimezone(
        pd.Timestamp.now(tz=TIMEZONE).tzinfo
    )
    log("=" * 60)
    log(f"SEÑALES EN VIVO — {ahora.strftime('%Y-%m-%d %H:%M')} ({TIMEZONE})")
    log("=" * 60)

    # --- 1. Descargar velas recientes ---
    log("Descargando velas recientes de Binance...")
    df_1h = descargar_velas_recientes(
        SYMBOL, Client.KLINE_INTERVAL_1HOUR, CANDLES_1H
    )
    df_4h = descargar_velas_recientes(
        SYMBOL, Client.KLINE_INTERVAL_4HOUR, CANDLES_4H
    )

    # CORRECCIÓN LOOKAHEAD (jul 2026): Binance devuelve como
    # última vela la que está EN FORMACIÓN. Sus indicadores
    # cambiarían hasta su cierre y el modelo debe ver solo
    # velas 4h cerradas (igual que el backtest corregido).
    # Nos quedamos únicamente con velas cuyo cierre ya pasó.
    n_antes = len(df_4h)
    df_4h = df_4h[df_4h["close_time"] <= ahora].reset_index(drop=True)
    if len(df_4h) < n_antes:
        log(f"  4h: descartada la vela en formación "
            f"(quedan {len(df_4h)} velas cerradas)")

    log(f"  1h: {len(df_1h)} velas "
        f"({df_1h['open_time'].min()} → {df_1h['open_time'].max()})")
    log(f"  4h: {len(df_4h)} velas "
        f"({df_4h['open_time'].min()} → {df_4h['open_time'].max()})")

    # --- 2. Calcular indicadores ---
    log("Calculando indicadores...")
    df_1h = procesar_indicadores(df_1h)
    df_4h = procesar_indicadores(df_4h)

    # --- 3. Alinear contexto 4h ---
    df = alinear_contexto_4h(df_1h, df_4h)

    # --- 3B. Calcular features rolling para el modelo ---
    df["volume_sma20"] = df["volume"].rolling(20).mean()
    df["close_5_ago"] = df["close"].shift(5)
    df["close_20_ago"] = df["close"].shift(20)
    df["atr_sma20"] = df["atr_14"].rolling(20).mean()

    # Eliminar filas con NaN (calentamiento)
    cols_necesarias = [
        "ema_20", "ema_50", "rsi_14", "atr_14",
        "swing_high_20", "swing_low_20", "trend_1h", "trend_4h"
    ]
    df = df.dropna(subset=cols_necesarias).reset_index(drop=True)

    if len(df) < 2:
        log("No hay suficientes velas con indicadores calculados.",
            "warning")
        return

    # --- 4. La última vela cerrada es la penúltima ---
    # (la última de Binance es la vela actual, aún sin cerrar)
    ultima = df.iloc[-2]  # última vela cerrada
    penultima = df.iloc[-3] if len(df) >= 3 else None  # para breakout

    log(f"Última vela cerrada: {ultima['open_time']}")
    log(f"  Close: {ultima['close']:,.2f} | "
        f"RSI: {ultima['rsi_14']:.1f} | "
        f"ATR: {ultima['atr_14']:,.2f}")
    log(f"  Trend 1h: {ultima['trend_1h']} | "
        f"Trend 4h: {ultima['trend_4h']}")

    # --- 5. Comprobar reglas ---
    senales = comprobar_reglas(ultima, penultima)

    if not senales:
        log("No hay señal en esta vela. Sin acción.")
        guardar_last_check(
            ultima["open_time"], ultima["close"],
            signal_found=False
        )
        return

    # --- 5B. Comprobar si ya hay posición abierta ---
    # Una posición a la vez por instrumento (diseñado para multi-producto)
    posiciones_abiertas = []
    if os.path.exists(OPEN_POSITIONS_FILE):
        try:
            with open(OPEN_POSITIONS_FILE) as f:
                posiciones_abiertas = json.load(f)
        except (json.JSONDecodeError, Exception):
            posiciones_abiertas = []

    if len(posiciones_abiertas) > 0:
        pos_abierta = posiciones_abiertas[0]
        log(f"Señal ignorada: ya hay posición abierta "
            f"({pos_abierta['direction']} {pos_abierta['rule']} "
            f"desde {pos_abierta.get('open_time', '?')})")
        guardar_last_check(
            ultima["open_time"], ultima["close"],
            signal_found=False
        )
        return

    # FLUJO LIMIT: también bloquea si hay una orden pendiente sin
    # resolver (misma política: una operación a la vez).
    ordenes_pendientes = []
    if os.path.exists(OPEN_ORDERS_FILE):
        try:
            with open(OPEN_ORDERS_FILE) as f:
                ordenes_pendientes = json.load(f)
        except Exception:
            ordenes_pendientes = []
    if len(ordenes_pendientes) > 0:
        log(f"Señal ignorada: ya hay una orden limit pendiente "
            f"({ordenes_pendientes[0].get('order_id', '?')})")
        guardar_last_check(
            ultima["open_time"], ultima["close"],
            signal_found=False
        )
        return

    # --- 6. Cargar modelo ML (si existe) ---
    modelo = cargar_modelo()

    # --- 7. Procesar cada señal encontrada ---
    for senal in senales:
        direction = senal["direction"]
        rule = senal["rule"]

        entry_price = ultima["close"]
        atr = ultima["atr_14"]

        sl, tp1, tp2, risk_usd, risk_pct = calcular_niveles(
            entry_price, atr, direction
        )

        # Probabilidad ML
        ai_proba = None
        if modelo is not None:
            features = crear_features_para_modelo(ultima)
            ai_proba = predecir_probabilidad(
                modelo, features, direction, rule, risk_pct
            )

        signal_info = {
            "time": ultima["open_time"],
            "direction": direction,
            "rule": rule,
            "entry": entry_price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "risk_pct": risk_pct,
            "risk_usd": risk_usd,
            "rsi": ultima["rsi_14"],
            "atr": atr,
            "trend_1h": ultima["trend_1h"],
            "trend_4h": ultima["trend_4h"],
            "ai_proba": ai_proba,
        }

        # Mostrar en log
        log("=" * 60)
        log(f"SEÑAL DETECTADA: {direction} / {rule}")
        log("=" * 60)
        log(f"  Entrada: {entry_price:,.2f} USD")
        log(f"  SL: {sl:,.2f} | TP1: {tp1:,.2f} (cierre 100%)")
        log(f"  Riesgo: {risk_pct:.2f}% ({risk_usd:,.2f} USD)")
        if ai_proba is not None:
            log(f"  ML Prob: {ai_proba:.1%}")

        # --- Enviar a Telegram ---
        mensaje = formatear_mensaje(signal_info)

        # 1. Canal always-on (siempre, sin filtro de umbral)
        enviar_telegram(mensaje)

        # 2. Canales de producto (routing por franja + umbral)
        ahora_madrid = datetime.now(timezone.utc).astimezone(
            pd.Timestamp.now(tz=TIMEZONE).tzinfo
        )
        canales_activos = get_canales_activos(
            ai_proba,
            ahora_madrid.hour,
            ahora_madrid.weekday()
        )

        if canales_activos:
            log(f"[Routing] Enviando a {len(canales_activos)} canal(es) de producto:")
            for chat_id, franja, umbral in canales_activos:
                log(f"  → {franja} · {umbral} ({chat_id})")
                # Enlace opcional a una página pública de historial (/track).
                # Configurable con TRACK_BASE_URL; si no se define, no se añade.
                track_base = os.getenv("TRACK_BASE_URL", "").rstrip("/")
                if track_base:
                    track_link = f"{track_base}/track/btc-{franja}-{umbral}"
                    mensaje_con_link = mensaje + f"\n\n📊 Historial → {track_link}"
                else:
                    mensaje_con_link = mensaje
                _enviar_telegram_a(mensaje_con_link, chat_id)
        else:
            prob_str = f"{ai_proba:.2f}" if ai_proba is not None else "N/A"
            log(f"[Routing] Ningún canal de producto activo para esta señal "
                f"(hora={ahora_madrid.hour}, prob={prob_str}).")

        # Registrar en CSV
        registrar_senal(signal_info)

        # FLUJO LIMIT (jul-2026): registrar ORDEN PENDIENTE para el
        # monitor (antes: posición abierta directa). El monitor decide
        # si se ejecuta o expira.
        # IMPORTANTE: va ANTES de Supabase para que un error en Supabase
        # nunca impida que el monitor empiece a vigilar la orden.
        registrar_orden_pendiente(signal_info)

        # Registrar en Supabase (defensivo: si falla, el monitor sigue activo)
        try:
            supabase_id = registrar_senal_supabase(signal_info, ahora_madrid)
            if supabase_id:
                signal_info["supabase_id"] = supabase_id
                # Actualizar el JSON con el supabase_id recién obtenido
                _actualizar_supabase_id_en_posicion(signal_info["id"], supabase_id)
        except Exception as e:
            log(f"[Supabase] Error registrando señal (posición ya guardada): {e}", "warning")

        # Paper trading en testnet (desactivado temporalmente)
        # if testnet_configurado():
        #     log("[Testnet] Colocando orden en paper trading...")
        #     ejecutar_orden_testnet(signal_info)

        # Guardar last check (con info de señal)
        guardar_last_check(
            ultima["open_time"], ultima["close"],
            signal_found=True,
            direction=direction, rule=rule
        )

    log("=" * 60)
    log("EJECUCIÓN COMPLETADA")
    log("=" * 60)


# ==========================================================
# BLOQUE 9 - Modo bucle continuo
# ==========================================================
def segundos_hasta_proxima_vela():
    """
    Calcula cuántos segundos faltan hasta el próximo cierre
    de vela de 1h (minuto 0 de la próxima hora) más un margen
    de seguridad (LOOP_MARGIN_SECONDS).

    Ejemplos:
      Si ahora son las 14:23:15, la próxima vela cierra a las
      15:00:00 y querremos despertar a las 15:00:30.
      Resultado: (15:00:30 - 14:23:15) = 2235 segundos.
    """
    ahora = datetime.now(timezone.utc).astimezone(
        pd.Timestamp.now(tz=TIMEZONE).tzinfo
    )
    # Próxima hora exacta (minuto 0, segundo 0)
    proxima_hora = (
        ahora.replace(minute=0, second=0, microsecond=0)
        + timedelta(hours=1)
    )
    # Añadir margen
    objetivo = proxima_hora + timedelta(seconds=LOOP_MARGIN_SECONDS)
    segundos = (objetivo - ahora).total_seconds()
    return segundos, objetivo


def bucle_continuo():
    """
    Modo bucle: el script se queda corriendo indefinidamente.
    En cada iteración:
      1. Calcula cuántos segundos faltan hasta el próximo
         cierre de vela de 1h + margen.
      2. Duerme hasta ese momento.
      3. Ejecuta la comprobación (protegida por try/except
         para que un error puntual no tumbe el bucle).
      4. Vuelve al paso 1.

    Para detener el bucle: Ctrl+C.
    """
    log("=" * 60)
    log("MODO BUCLE CONTINUO ACTIVADO")
    log("=" * 60)
    log("El script se quedará corriendo y comprobará automáticamente")
    log("al cierre de cada vela de 1h. Pulsa Ctrl+C para detener.")

    # --- Ping de arranque por Telegram ---
    ahora_arranque = datetime.now(timezone.utc).astimezone(
        pd.Timestamp.now(tz=TIMEZONE).tzinfo
    )
    enviar_telegram(
        f"\U0001f7e2 <b>Sistema ARRANCADO</b>\n"
        f"\n"
        f"\u23f0 {ahora_arranque.strftime('%Y-%m-%d %H:%M:%S')} "
        f"({TIMEZONE})\n"
        f"\U0001f4bb Modo: bucle continuo\n"
        f"\n"
        f"Comprobando cada cierre de vela 1h. "
        f"Te avisaré cuando se detenga."
    )

    # --- Comprobación inmediata al arrancar ---
    # Aprovechamos la última vela cerrada disponible ahora mismo,
    # en lugar de hacer esperar al usuario hasta el próximo cierre.
    log("-" * 60)
    log("[Arranque] Ejecutando comprobación inicial con la última "
        "vela cerrada disponible...")
    log("-" * 60)
    try:
        ejecutar_comprobacion()
    except KeyboardInterrupt:
        log("Interrupción detectada. Saliendo del bucle.", "warning")
        _ping_parada(motivo="Ctrl+C durante comprobación inicial")
        return
    except Exception as e:
        ahora = datetime.now(timezone.utc).astimezone(
            pd.Timestamp.now(tz=TIMEZONE).tzinfo
        )
        log(f"[ERROR en comprobación inicial @ "
            f"{ahora.strftime('%Y-%m-%d %H:%M:%S')}]", "error")
        log(f"  Tipo: {type(e).__name__}", "error")
        log(f"  Mensaje: {e}", "error")
        log(f"  Traceback:\n{traceback.format_exc()}", "error")
        log("El bucle continúa. Esperando próxima vela.", "warning")

    # --- Bucle de esperas hasta cada cierre de vela ---
    iteracion = 0
    motivo_parada = "Fin inesperado del bucle"

    # Contadores para el resumen diario
    velas_hoy = 1  # la comprobación inicial cuenta
    senales_hoy = 0
    resumen_enviado_hoy = False

    # Contar señal si la comprobación inicial encontró algo
    if os.path.exists(LAST_CHECK_FILE):
        try:
            with open(LAST_CHECK_FILE) as f:
                lc = json.load(f)
            if lc.get("signal_found"):
                senales_hoy += 1
        except Exception:
            pass

    try:
        while True:
            iteracion += 1
            segundos, objetivo = segundos_hasta_proxima_vela()

            log(f"[Iteración {iteracion}] "
                f"Esperando hasta {objetivo.strftime('%Y-%m-%d %H:%M:%S')} "
                f"({segundos:.0f} segundos)...")

            try:
                time.sleep(segundos)
            except KeyboardInterrupt:
                motivo_parada = "Ctrl+C durante espera"
                log("Interrupción detectada. Saliendo del bucle.",
                    "warning")
                break

            # --- Comprobar si toca resumen diario ---
            ahora_check = datetime.now(timezone.utc).astimezone(
                pd.Timestamp.now(tz=TIMEZONE).tzinfo
            )
            hora_actual = ahora_check.hour

            if hora_actual == DAILY_SUMMARY_HOUR and not resumen_enviado_hoy:
                try:
                    _enviar_resumen_diario(velas_hoy, senales_hoy)
                except Exception as e:
                    log(f"[Resumen diario] Error: {e}", "error")
                resumen_enviado_hoy = True
                # Resetear contadores
                velas_hoy = 0
                senales_hoy = 0

            # Resetear flag cuando pasamos de la hora del resumen
            if hora_actual != DAILY_SUMMARY_HOUR:
                resumen_enviado_hoy = False

            try:
                ejecutar_comprobacion()
                velas_hoy += 1

                # Contar si hubo señal
                if os.path.exists(LAST_CHECK_FILE):
                    try:
                        with open(LAST_CHECK_FILE) as f:
                            lc = json.load(f)
                        if lc.get("signal_found"):
                            senales_hoy += 1
                    except Exception:
                        pass

            except KeyboardInterrupt:
                motivo_parada = "Ctrl+C durante comprobación"
                log("Interrupción detectada. Saliendo del bucle.",
                    "warning")
                break
            except Exception as e:
                # Capturamos CUALQUIER error para no tumbar el bucle.
                # Los errores transitorios de Binance y Telegram ya tienen
                # sus propios reintentos internos; si llegan hasta aquí es
                # que son errores más graves. Los registramos con detalle.
                ahora_err = datetime.now(timezone.utc).astimezone(
                    pd.Timestamp.now(tz=TIMEZONE).tzinfo
                )
                log(f"[ERROR en iteración {iteracion} @ "
                    f"{ahora_err.strftime('%Y-%m-%d %H:%M:%S')}]", "error")
                log(f"  Tipo: {type(e).__name__}", "error")
                log(f"  Mensaje: {e}", "error")
                log(f"  Traceback:\n{traceback.format_exc()}", "error")
                log("El bucle continúa. Esperando próxima vela.",
                    "warning")
                velas_hoy += 1  # contamos el intento
    finally:
        _ping_parada(motivo=motivo_parada)


def _enviar_resumen_diario(iteraciones_hoy, senales_hoy):
    """
    Envía un resumen diario al canal de Telegram para confirmar
    que el bot sigue activo, incluso si no hubo señales.
    Se llama automáticamente cada día a las DAILY_SUMMARY_HOUR CET.
    """
    ahora = datetime.now(timezone.utc).astimezone(
        pd.Timestamp.now(tz=TIMEZONE).tzinfo
    )
    ayer = ahora - timedelta(days=1)
    fecha_ayer = ayer.strftime("%d %b")
    fecha_hoy = ahora.strftime("%d %b %Y")
    hora_resumen = f"{DAILY_SUMMARY_HOUR:02d}:00"

    msg = (
        f"\U0001f916 <b>Resumen últimas 24h</b>\n"
        f"Desde {fecha_ayer} {hora_resumen} hasta "
        f"{fecha_hoy} {hora_resumen} (Madrid)\n"
        f"\n"
        f"\U0001f50d Velas analizadas: <b>{iteraciones_hoy}</b>\n"
        f"\U0001f514 Señales enviadas: <b>{senales_hoy}</b>\n"
        f"\u2705 Bot: <b>activo</b>\n"
        f"\n"
        f"\U0001f4ca <a href=\"{DASHBOARD_URL}\">Ver dashboard</a>"
    )
    enviar_telegram(msg)
    log(f"[Resumen diario] Enviado: {iteraciones_hoy} velas, "
        f"{senales_hoy} señales.")


def _ping_parada(motivo="Parada normal"):
    """
    Envía un mensaje de Telegram avisando de que el sistema
    se ha detenido. Se llama desde bucle_continuo() al salir.
    """
    ahora_parada = datetime.now(timezone.utc).astimezone(
        pd.Timestamp.now(tz=TIMEZONE).tzinfo
    )
    enviar_telegram(
        f"\U0001f534 <b>Sistema DETENIDO</b>\n"
        f"\n"
        f"\u23f0 {ahora_parada.strftime('%Y-%m-%d %H:%M:%S')} "
        f"({TIMEZONE})\n"
        f"\U0001f4cc Motivo: {motivo}\n"
        f"\n"
        f"No se comprobarán nuevas señales hasta que reinicies "
        f"el script."
    )


# ==========================================================
# PUNTO DE ENTRADA
# ==========================================================
if __name__ == "__main__":
    # Si se pasa el argumento --loop, entrar en modo bucle.
    # Si no, ejecutar una única comprobación y salir.
    if "--loop" in sys.argv:
        bucle_continuo()
    else:
        ejecutar_comprobacion()
