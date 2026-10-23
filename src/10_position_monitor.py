"""
==========================================================
FASE 15 - Monitor de posiciones SL/TP en tiempo real
==========================================================
Monitorea las operaciones abiertas por el bot de señales.
Cada minuto consulta el precio actual de BTC/USDT y comprueba
si alguna posición ha tocado SL, TP1 o si ha expirado
por time exit (24h).

Diseño (config A — Anexo II):
  - Lee posiciones abiertas de outputs/open_positions.json
  - Consulta precio actual de Binance cada 60 segundos
  - Al tocar TP1: cierre total (100%) de la posición
  - Al tocar SL o 24h: cierre total
  - Envía notificación a Telegram en cada evento
  - Guarda resultados en outputs/position_results.csv
  - El dashboard lee ese archivo para mostrar estado en vivo

Config: SL 1.25x ATR, TP1 2.0R, cierre 100% en TP1.
No hay cierre parcial, TP2 ni breakeven.

Ejecución:
  python src/10_position_monitor.py

En modo servicio (systemd):
  Se ejecuta como bucle continuo que revisa cada minuto.
==========================================================
"""

import os
import sys
import json
import time
import logging
import traceback
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
SYMBOL = "BTCUSDT"
TIMEZONE = "Europe/Madrid"

# Intervalo de comprobación (segundos)
CHECK_INTERVAL = 60

# Time exit: horas máximas que una posición puede estar abierta
TIME_EXIT_HOURS = 24

# Archivos
OPEN_POSITIONS_FILE = "outputs/open_positions.json"
POSITION_RESULTS_FILE = "outputs/position_results.csv"
MONITOR_LOG_FILE = "outputs/position_monitor.log"
MONITOR_STATE_FILE = "outputs/monitor_state.json"
# FLUJO LIMIT (jul-2026): órdenes limit pendientes creadas por 08.
# El monitor las convierte en posición (si el precio las toca) o
# las cancela (si expiran). Registro de órdenes canceladas en CSV
# para poder medir el fill-rate real vs el 98.4% del backtest.
OPEN_ORDERS_FILE = "outputs/open_orders.json"
CANCELLED_ORDERS_FILE = "outputs/cancelled_orders.csv"

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DASHBOARD_URL = os.getenv(
    "DASHBOARD_URL", "http://localhost:8501"
)

# ==========================================================
# LOGGER
# ==========================================================
os.makedirs("outputs", exist_ok=True)

logger = logging.getLogger("position_monitor")
logger.setLevel(logging.INFO)
logger.handlers.clear()

_formato = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(_formato)
logger.addHandler(_console)

_archivo = logging.FileHandler(
    MONITOR_LOG_FILE, mode="a", encoding="utf-8"
)
_archivo.setFormatter(_formato)
logger.addHandler(_archivo)


def log(mensaje, nivel="info"):
    if nivel == "warning":
        logger.warning(mensaje)
    elif nivel == "error":
        logger.error(mensaje)
    else:
        logger.info(mensaje)


# ==========================================================
# PRECIO ACTUAL
# ==========================================================
def obtener_precio_actual():
    """
    Obtiene el precio actual de BTC/USDT desde Binance.
    Usa el endpoint de ticker que es ligero y rápido.
    """
    try:
        client = Client()
        ticker = client.get_symbol_ticker(symbol=SYMBOL)
        return float(ticker["price"])
    except Exception as e:
        log(f"Error obteniendo precio: {e}", "error")
        return None


# ==========================================================
# GESTIÓN DE POSICIONES ABIERTAS
# ==========================================================
def cargar_posiciones():
    """Carga las posiciones abiertas desde el archivo JSON."""
    if not os.path.exists(OPEN_POSITIONS_FILE):
        return []
    try:
        with open(OPEN_POSITIONS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return []


def guardar_posiciones(posiciones):
    """Guarda las posiciones abiertas en el archivo JSON."""
    with open(OPEN_POSITIONS_FILE, "w") as f:
        json.dump(posiciones, f, indent=2, default=str)


def guardar_estado_monitor(precio_actual, posiciones_abiertas):
    """
    Guarda el estado actual del monitor para que el dashboard
    pueda mostrar información en tiempo real.
    """
    estado = {
        "last_check": datetime.now(timezone.utc).isoformat(),
        "precio_actual": precio_actual,
        "posiciones_abiertas": len(posiciones_abiertas),
        "posiciones": []
    }

    for pos in posiciones_abiertas:
        entry = pos["entry_price"]
        direction = pos["direction"]

        # Calcular PnL flotante
        if direction == "LONG":
            pnl_pct = (precio_actual - entry) / entry * 100
            pnl_usd = precio_actual - entry
            dist_sl = (pos["stop_loss"] - precio_actual) / precio_actual * 100
            dist_tp = (pos["tp1"] - precio_actual) / precio_actual * 100
        else:
            pnl_pct = (entry - precio_actual) / entry * 100
            pnl_usd = entry - precio_actual
            dist_sl = (precio_actual - pos["stop_loss"]) / precio_actual * 100
            dist_tp = (precio_actual - pos["tp1"]) / precio_actual * 100

        # Tiempo transcurrido
        open_time = pd.Timestamp(pos["open_time"]).to_pydatetime()
        if open_time.tzinfo is None:
            open_time = open_time.replace(tzinfo=timezone.utc)
        elapsed = datetime.now(timezone.utc) - open_time
        hours_open = elapsed.total_seconds() / 3600

        # Progreso hacia TP1 (0% = entrada, 100% = TP1)
        if direction == "LONG":
            total_range = pos["tp1"] - entry
            current_progress = precio_actual - entry
        else:
            total_range = entry - pos["tp1"]
            current_progress = entry - precio_actual
        progress_pct = (current_progress / total_range * 100
                        if total_range != 0 else 0)

        estado["posiciones"].append({
            "id": pos["id"],
            "direction": direction,
            "rule": pos["rule"],
            "entry_price": entry,
            "stop_loss": pos["stop_loss"],
            "tp1": pos["tp1"],
            "precio_actual": precio_actual,
            "pnl_pct": round(pnl_pct, 3),
            "pnl_usd": round(pnl_usd, 2),
            "dist_sl_pct": round(dist_sl, 3),
            "dist_tp_pct": round(dist_tp, 3),
            "progress_pct": round(progress_pct, 1),
            "hours_open": round(hours_open, 1),
            "ai_proba": pos.get("ai_proba"),
            "open_time": pos["open_time"],
        })

    with open(MONITOR_STATE_FILE, "w") as f:
        json.dump(estado, f, indent=2)


# ==========================================================
# REGISTRO DE RESULTADOS
# ==========================================================
def registrar_resultado(pos, close_price, close_reason):
    """
    Registra el cierre de una posición en CSV.
    Cierre siempre al 100% (sin parciales).
    """
    entry = pos["entry_price"]
    direction = pos["direction"]

    if direction == "LONG":
        pnl_usd = close_price - entry
        pnl_pct = (close_price - entry) / entry * 100
    else:
        pnl_usd = entry - close_price
        pnl_pct = (entry - close_price) / entry * 100

    # R-multiple
    risk = abs(entry - pos["stop_loss"])
    r_multiple = pnl_usd / risk if risk > 0 else 0

    open_time = pd.Timestamp(pos["open_time"]).to_pydatetime()
    if open_time.tzinfo is None:
        open_time = open_time.replace(tzinfo=timezone.utc)
    close_time = datetime.now(timezone.utc)
    duration_hours = (close_time - open_time).total_seconds() / 3600

    row = {
        "id": pos["id"],
        "open_time": pos["open_time"],
        "close_time": close_time.isoformat(),
        "direction": direction,
        "rule": pos["rule"],
        "entry_price": entry,
        "close_price": close_price,
        "stop_loss": pos["stop_loss"],
        "tp1": pos["tp1"],
        "close_reason": close_reason,
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 4),
        "r_multiple": round(r_multiple, 4),
        "duration_hours": round(duration_hours, 2),
        "ai_proba": pos.get("ai_proba"),
    }

    df = pd.DataFrame([row])

    if os.path.exists(POSITION_RESULTS_FILE):
        df.to_csv(POSITION_RESULTS_FILE, mode="a",
                  header=False, index=False)
    else:
        df.to_csv(POSITION_RESULTS_FILE, index=False)

    log(f"Resultado registrado: {close_reason} | "
        f"PnL: ${pnl_usd:+,.2f} ({pnl_pct:+.2f}%) | "
        f"R: {r_multiple:+.2f}")

    # Actualizar en Supabase
    actualizar_cierre_supabase(pos, close_price, close_reason,
                               pnl_usd, r_multiple, close_time)


def actualizar_cierre_supabase(pos, close_price, close_reason,
                                pnl_usd, r_multiple, close_time):
    """
    Actualiza la fila en Supabase con el resultado del trade.
    Si la posición no tiene supabase_id (señales anteriores a D1),
    la función simplemente omite la actualización sin error.
    """
    supabase_id = pos.get("supabase_id")
    if not supabase_id:
        return

    # Mapear close_reason al status de la tabla
    status_map = {"TP1": "win", "SL": "loss", "TIME_EXIT": "timeout"}
    status = status_map.get(close_reason, "timeout")

    sys.path.insert(0, os.path.dirname(__file__))
    from supabase_client import get_supabase
    sb = get_supabase()
    if sb is None:
        log("[Supabase] Cliente no disponible — cierre no actualizado en BD.", "warning")
        return

    try:
        sb.table("signals").update({
            "status":      status,
            "close_price": float(close_price),
            "close_reason": close_reason,
            "pnl_usd":     round(float(pnl_usd), 2),
            "r_multiple":  round(float(r_multiple), 4),
            "closed_at":   close_time.isoformat(),
        }).eq("id", supabase_id).execute()
        log(f"[Supabase] Cierre actualizado — {status} | R: {r_multiple:+.2f}")
    except Exception as e:
        log(f"[Supabase] Error al actualizar cierre: {e}", "warning")


# ==========================================================
# TELEGRAM
# ==========================================================
def enviar_telegram(mensaje):
    """Envía un mensaje a Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram no configurado.", "warning")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return True
        else:
            log(f"Telegram HTTP {response.status_code}", "warning")
    except Exception as e:
        log(f"Telegram error: {e}", "error")
    return False


def notificar_cierre(pos, close_price, close_reason):
    """
    Envia un mensaje a Telegram notificando el cierre
    de una posicion. Cierre siempre al 100%.
    """
    entry = pos["entry_price"]
    direction = pos["direction"]

    if direction == "LONG":
        pnl_usd = close_price - entry
        pnl_pct = (close_price - entry) / entry * 100
    else:
        pnl_usd = entry - close_price
        pnl_pct = (entry - close_price) / entry * 100

    # R-multiple
    risk = abs(entry - pos["stop_loss"])
    r_multiple = pnl_usd / risk if risk > 0 else 0

    # Resultado
    if pnl_pct > 0:
        result_emoji = "✅"
        result_text = "GANADOR"
    else:
        result_emoji = "❌"
        result_text = "PERDEDOR"

    # Razon de cierre
    reason_map = {
        "stop_loss": "\U0001f6d1 Stop Loss",
        "take_profit_1": "\U0001f3af TP1 (cierre 100%)",
        "time_exit": "⏰ Time Exit (24h)",
    }
    reason_text = reason_map.get(close_reason, close_reason)

    title = f"{result_emoji} OPERACION CERRADA — {result_text}"

    open_time = pd.Timestamp(pos["open_time"]).to_pydatetime()
    if open_time.tzinfo is None:
        open_time = open_time.replace(tzinfo=timezone.utc)
    duration = datetime.now(timezone.utc) - open_time
    hours = duration.total_seconds() / 3600

    open_madrid = open_time.astimezone(ZoneInfo(TIMEZONE))
    open_str = open_madrid.strftime("%d %b %Y — %H:%M (Madrid)")

    arrow = "↗️" if direction == "LONG" else "↘️"
    pnl_dot = "\U0001f7e2" if pnl_usd >= 0 else "\U0001f534"

    msg = (
        f"<b>{title}</b>\n"
        f"\n"
        f"{arrow} <b>{direction}</b> — {pos['rule']}\n"
        f"\U0001f4c5 Apertura: {open_str}\n"
        f"\U0001f4b0 Razon: {reason_text}\n"
        f"\n"
        f"━━━ Resultado ━━━\n"
        f"\U0001f4b0 Entrada: ${entry:,.2f}\n"
        f"\U0001f3f7️ Cierre: ${close_price:,.2f}\n"
        f"{pnl_dot} <b>PnL: {pnl_pct:+.2f}%</b>"
        f" (con $100 → ${pnl_pct:+.2f})\n"
        f"\U0001f4cf R-multiple: {r_multiple:+.2f}R\n"
        f"⏱️ Duracion: {hours:.1f}h\n"
    )

    if pos.get("ai_proba") is not None:
        msg += f"\U0001f916 ML: {pos['ai_proba']:.0%} de confianza\n"

    msg += (
        f"\n"
        f"\U0001f4ca <a href=\"{DASHBOARD_URL}\">Ver dashboard</a>"
    )

    enviar_telegram(msg)


# ==========================================================
# LOGICA DE MONITOREO
# ==========================================================
def comprobar_posicion(pos, precio_actual):
    """
    Comprueba si una posicion ha tocado SL, TP1 o time exit.
    Devuelve (accion, close_reason) donde accion puede ser:
      - None: nada que hacer, sigue abierta
      - "close": cierre total (SL, TP1 o time exit)
    """
    direction = pos["direction"]

    # --- Time exit ---
    open_time = pd.Timestamp(pos["open_time"]).to_pydatetime()
    if open_time.tzinfo is None:
        open_time = open_time.replace(tzinfo=timezone.utc)
    elapsed = datetime.now(timezone.utc) - open_time
    if elapsed.total_seconds() >= TIME_EXIT_HOURS * 3600:
        return "close", "time_exit"

    if direction == "LONG":
        # SL: precio cae por debajo del stop loss
        if precio_actual <= pos["stop_loss"]:
            return "close", "stop_loss"

        # TP1: precio sube hasta TP1 (cierre total)
        if precio_actual >= pos["tp1"]:
            return "close", "take_profit_1"

    else:  # SHORT
        # SL: precio sube por encima del stop loss
        if precio_actual >= pos["stop_loss"]:
            return "close", "stop_loss"

        # TP1: precio baja hasta TP1 (cierre total)
        if precio_actual <= pos["tp1"]:
            return "close", "take_profit_1"

    return None, None


def cargar_ordenes():
    """Carga las órdenes limit pendientes (flujo limit jul-2026)."""
    if not os.path.exists(OPEN_ORDERS_FILE):
        return []
    try:
        with open(OPEN_ORDERS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return []


def guardar_ordenes(ordenes):
    with open(OPEN_ORDERS_FILE, "w") as f:
        json.dump(ordenes, f, indent=2, default=str)


def procesar_ordenes_pendientes(precio):
    """
    FLUJO LIMIT (jul-2026): comprueba cada orden pendiente.

      - EJECUTADA: el precio actual toca el precio limit
        (LONG: precio <= limit; SHORT: precio >= limit).
        Se convierte en posición abierta (SL/TP ya calculados
        desde el precio limit) y se notifica.
      - EXPIRADA: pasó la validez sin tocarse. Se cancela,
        se registra en CSV (para medir fill-rate real) y se
        notifica.

    Nota de honestidad: el monitor sondea cada CHECK_INTERVAL
    segundos, así que un toque muy breve del precio puede
    escapársele — es decir, el fill-rate real será COMO MUCHO
    el del backtest (98.4%), nunca mejor. Sesgo conservador.
    """
    ordenes = cargar_ordenes()
    if not ordenes:
        return

    ahora = datetime.now(timezone.utc)
    restantes = []
    for orden in ordenes:
        d = orden["direction"]
        limit = float(orden["limit_price"])
        tocada = (precio <= limit) if d == "LONG" else (precio >= limit)

        if tocada:
            # → CONVERTIR EN POSICIÓN ABIERTA
            # Mismo esquema que registrar_posicion_abierta() de 08
            posiciones = cargar_posiciones()
            posiciones.append({
                "id": orden["order_id"],
                "supabase_id": orden.get("supabase_id"),
                "open_time": ahora.isoformat(),
                "signal_time": orden.get("signal_time"),
                "direction": d,
                "rule": orden.get("rule"),
                "entry_price": limit,
                "stop_loss": orden["stop_loss"],
                "tp1": orden["tp1"],
                "tp2": orden["tp1"],  # compatibilidad (= tp1)
                "risk_pct": orden.get("risk_pct"),
                "ai_proba": orden.get("ai_proba"),
            })
            guardar_posiciones(posiciones)
            log(f"ORDEN EJECUTADA: {orden['order_id']} a ${limit:,.2f}")
            enviar_telegram(
                f"✅ <b>Orden limit EJECUTADA</b>\n\n"
                f"{d} {orden.get('rule','')} a ${limit:,.2f}\n"
                f"SL ${float(orden['stop_loss']):,.2f} | "
                f"TP ${float(orden['tp1']):,.2f}\n"
                f"El monitor vigila la posición."
            )
            continue

        expiry = pd.Timestamp(orden["expiry"])
        if expiry.tzinfo is None:
            expiry = expiry.tz_localize("UTC")
        if pd.Timestamp(ahora) >= expiry:
            # → CANCELAR POR EXPIRACIÓN
            log(f"ORDEN EXPIRADA sin ejecutarse: {orden['order_id']}")
            try:
                nueva = pd.DataFrame([{**orden, "cancelled_at": str(ahora)}])
                header = not os.path.exists(CANCELLED_ORDERS_FILE)
                nueva.to_csv(CANCELLED_ORDERS_FILE, mode="a",
                             header=header, index=False)
            except Exception as e:
                log(f"No se pudo registrar cancelación: {e}", "warning")
            enviar_telegram(
                f"⏹ <b>Orden limit CANCELADA</b> (no se ejecutó)\n\n"
                f"{d} {orden.get('rule','')} a "
                f"${limit:,.2f} — el precio no volvió a ese nivel "
                f"en 1h.\nSi la tenías puesta, cancélala."
            )
            continue

        restantes.append(orden)

    guardar_ordenes(restantes)


def ciclo_monitoreo():
    """
    Un ciclo de monitoreo: (1) órdenes limit pendientes,
    (2) posiciones abiertas.
    """
    posiciones = cargar_posiciones()
    ordenes = cargar_ordenes()

    if not posiciones and not ordenes:
        return  # Nada que vigilar

    precio = obtener_precio_actual()
    if precio is None:
        log("No se pudo obtener precio. Saltando ciclo.", "warning")
        return

    # FLUJO LIMIT: primero resolver órdenes pendientes
    if ordenes:
        procesar_ordenes_pendientes(precio)
        posiciones = cargar_posiciones()  # pueden haberse creado

    if not posiciones:
        guardar_estado_monitor(precio, [])
        return

    posiciones_actualizadas = []
    hubo_cambios = False

    for pos in posiciones:
        accion, reason = comprobar_posicion(pos, precio)

        if accion == "close":
            # Cierre total
            log(f"CIERRE: {pos['direction']} {pos['rule']} | "
                f"Razon: {reason} | Precio: ${precio:,.2f}")
            registrar_resultado(pos, precio, reason)
            notificar_cierre(pos, precio, reason)
            hubo_cambios = True
            # No anadir a la lista = posicion eliminada

        else:
            # Sin cambio, sigue abierta
            posiciones_actualizadas.append(pos)

    if hubo_cambios:
        guardar_posiciones(posiciones_actualizadas)

    # Actualizar estado para el dashboard
    guardar_estado_monitor(precio, posiciones_actualizadas)


# ==========================================================
# BUCLE PRINCIPAL
# ==========================================================
def bucle_monitor():
    """
    Bucle continuo que comprueba posiciones cada minuto.
    Pensado para correr como servicio systemd.
    """
    log("=" * 60)
    log("MONITOR DE POSICIONES — ARRANCADO")
    log(f"Intervalo: {CHECK_INTERVAL}s | "
        f"Time exit: {TIME_EXIT_HOURS}h | "
        f"TP1 → cierre 100%")
    log("=" * 60)

    enviar_telegram(
        "\U0001f50d <b>Monitor de posiciones ARRANCADO</b>\n"
        f"\n"
        f"Vigilando operaciones cada {CHECK_INTERVAL}s.\n"
        f"TP1 → cierre 100%.\n"
        f"Time exit: {TIME_EXIT_HOURS}h."
    )

    while True:
        try:
            ciclo_monitoreo()
        except KeyboardInterrupt:
            log("Monitor detenido por el usuario.")
            enviar_telegram(
                "\U0001f534 <b>Monitor de posiciones DETENIDO</b>"
            )
            break
        except Exception as e:
            log(f"Error en ciclo: {type(e).__name__}: {e}", "error")
            log(traceback.format_exc(), "error")

        try:
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            log("Monitor detenido por el usuario.")
            enviar_telegram(
                "\U0001f534 <b>Monitor de posiciones DETENIDO</b>"
            )
            break


# ==========================================================
# PUNTO DE ENTRADA
# ==========================================================
if __name__ == "__main__":
    bucle_monitor()
