"""
==========================================================
FASE 18 - Paper trading en Binance Futures Testnet
==========================================================
Coloca órdenes reales en Binance Futures Testnet (dinero
virtual) cada vez que el sistema detecta una señal.

Objetivo: validar el pipeline completo midiendo slippage,
ejecución real de SL/TP y latencia. No arriesga dinero real.

Funcionamiento:
  - Se importa desde 08_live_signal.py
  - Cuando hay señal, ejecutar_orden_testnet() coloca:
    1. Orden de mercado (entrada)
    2. STOP_MARKET (stop loss)
    3. TAKE_PROFIT_MARKET (take profit)
  - Envía notificación al canal de Telegram de testnet
  - Si las claves de testnet no están en .env, no hace nada

Configuración necesaria en .env:
  BINANCE_TESTNET_API_KEY=tu_key
  BINANCE_TESTNET_API_SECRET=tu_secret
  TELEGRAM_TESTNET_CHAT_ID=tu_chat_id

No modifica nada del sistema de producción.
==========================================================
"""

import os
import json
import logging
from datetime import datetime, timezone
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
import requests

load_dotenv()

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
SYMBOL = "BTCUSDT"
TIMEZONE = "Europe/Madrid"

# Tamaño de posición en USDT por trade.
# Con 100 USDT y BTC a ~100.000, la cantidad será ~0.001 BTC.
POSITION_SIZE_USDT = 100

# Cantidad mínima de BTC en Binance Futures
MIN_QTY = 0.001
# Paso de cantidad (3 decimales para BTC)
QTY_STEP = 0.001

# Claves de testnet
TESTNET_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
TESTNET_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")

# Telegram testnet
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_TESTNET_CHAT_ID = os.getenv("TELEGRAM_TESTNET_CHAT_ID", "")

# Archivo para registrar trades de testnet
TESTNET_LOG_FILE = "outputs/testnet_trades.csv"

# Logger
logger = logging.getLogger("live_signal")


# ==========================================================
# FUNCIÓN 1 - Verificar si testnet está configurado
# ==========================================================
def testnet_configurado():
    """
    Devuelve True si las claves de testnet están en el .env.
    Si no lo están, el paper trading simplemente no se activa.
    """
    return bool(TESTNET_API_KEY and TESTNET_API_SECRET)


# ==========================================================
# FUNCIÓN 2 - Crear cliente de testnet
# ==========================================================
def crear_cliente_testnet():
    """
    Crea un cliente de Binance conectado al testnet de Futures.
    testnet=True configura automáticamente las URLs de testnet.
    """
    client = Client(
        TESTNET_API_KEY,
        TESTNET_API_SECRET,
        testnet=True
    )
    return client


# ==========================================================
# FUNCIÓN 3 - Calcular cantidad de BTC
# ==========================================================
def calcular_cantidad(precio_actual):
    """
    Calcula la cantidad de BTC a operar basándose en
    POSITION_SIZE_USDT y el precio actual.

    Redondea hacia abajo al paso mínimo (0.001 BTC).
    Si la cantidad es menor que MIN_QTY, usa MIN_QTY.
    """
    cantidad = POSITION_SIZE_USDT / precio_actual

    # Redondear al paso de cantidad (3 decimales)
    cantidad = int(cantidad / QTY_STEP) * QTY_STEP

    # Asegurar mínimo
    if cantidad < MIN_QTY:
        cantidad = MIN_QTY

    return round(cantidad, 3)


# ==========================================================
# FUNCIÓN 4 - Redondear precio para Binance
# ==========================================================
def redondear_precio(precio):
    """
    Binance Futures BTCUSDT acepta precios con 1 decimal.
    """
    return round(precio, 1)


# ==========================================================
# FUNCIÓN 5 - Ejecutar orden en testnet
# ==========================================================
def ejecutar_orden_testnet(signal_info):
    """
    Coloca una orden de mercado en Binance Futures Testnet
    con SL y TP automáticos.

    Pasos:
      1. Conectar al testnet
      2. Orden de mercado (entrada)
      3. STOP_MARKET (stop loss)
      4. TAKE_PROFIT_MARKET (take profit)
      5. Notificar por Telegram
      6. Registrar en CSV

    Si algo falla, registra el error pero no interrumpe
    el sistema de señales.
    """
    if not testnet_configurado():
        return None

    try:
        client = crear_cliente_testnet()

        # Apalancamiento 1x = sin apalancamiento real.
        # Margen completo, sin riesgo de liquidación amplificada.
        client.futures_change_leverage(symbol=SYMBOL, leverage=1)

        direction = signal_info["direction"]
        entry_price = signal_info["entry"]
        sl_price = redondear_precio(signal_info["sl"])
        tp_price = redondear_precio(signal_info["tp1"])

        # Calcular cantidad
        cantidad = calcular_cantidad(entry_price)

        # Determinar lado de la orden
        # LONG = comprar para abrir, SHORT = vender para abrir
        side_entry = SIDE_BUY if direction == "LONG" else SIDE_SELL
        side_exit = SIDE_SELL if direction == "LONG" else SIDE_BUY

        logger.info(f"[Testnet] Colocando orden {direction} "
                     f"{cantidad} BTC @ mercado...")

        # --- 1. Orden de mercado (entrada) ---
        orden_entrada = client.futures_create_order(
            symbol=SYMBOL,
            side=side_entry,
            type=ORDER_TYPE_MARKET,
            quantity=cantidad,
        )

        # Precio real de ejecución
        # El testnet a veces devuelve avgPrice=0; en ese caso
        # consultamos el precio actual del ticker como fallback.
        fill_price = float(orden_entrada.get("avgPrice", 0))
        if fill_price == 0:
            ticker = client.futures_symbol_ticker(symbol=SYMBOL)
            fill_price = float(ticker["price"])
        order_id = orden_entrada.get("orderId", "?")

        logger.info(f"[Testnet] Entrada ejecutada: "
                     f"ID={order_id}, precio={fill_price:,.2f}")

        # Calcular slippage
        slippage = fill_price - entry_price
        slippage_pct = (slippage / entry_price) * 100

        # --- 2. Stop Loss (STOP_MARKET) via Algo Order API ---
        # Desde dic 2025, Binance requiere usar el endpoint
        # /fapi/v1/algoOrder para órdenes condicionales.
        # python-binance no lo tiene integrado, así que
        # usamos su método interno _request_futures_api.
        orden_sl = client._request_futures_api(
            'post', 'algoOrder', signed=True, data={
                "symbol": SYMBOL,
                "side": side_exit,
                "algoType": "CONDITIONAL",
                "type": "STOP_MARKET",
                "triggerPrice": str(sl_price),
                "closePosition": "true",
                "workingType": "MARK_PRICE",
            }
        )
        sl_order_id = orden_sl.get("algoId", "?")
        logger.info(f"[Testnet] SL colocado: ID={sl_order_id}, "
                     f"precio={sl_price:,.2f}")

        # --- 3. Take Profit (TAKE_PROFIT_MARKET) via Algo Order API ---
        orden_tp = client._request_futures_api(
            'post', 'algoOrder', signed=True, data={
                "symbol": SYMBOL,
                "side": side_exit,
                "algoType": "CONDITIONAL",
                "type": "TAKE_PROFIT_MARKET",
                "triggerPrice": str(tp_price),
                "closePosition": "true",
                "workingType": "MARK_PRICE",
            }
        )
        tp_order_id = orden_tp.get("algoId", "?")
        logger.info(f"[Testnet] TP colocado: ID={tp_order_id}, "
                     f"precio={tp_price:,.2f}")

        # --- 4. Preparar resultado ---
        resultado = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": direction,
            "rule": signal_info["rule"],
            "signal_price": entry_price,
            "fill_price": fill_price,
            "slippage": slippage,
            "slippage_pct": slippage_pct,
            "quantity": cantidad,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "entry_order_id": order_id,
            "sl_order_id": sl_order_id,
            "tp_order_id": tp_order_id,
            "ai_proba": signal_info.get("ai_proba"),
            "status": "OPEN",
        }

        # --- 5. Notificar por Telegram ---
        notificar_testnet(resultado)

        # --- 6. Registrar en CSV ---
        registrar_trade_testnet(resultado)

        return resultado

    except Exception as e:
        logger.error(f"[Testnet] Error al ejecutar orden: "
                      f"{type(e).__name__}: {e}")
        notificar_error_testnet(signal_info, e)
        return None


# ==========================================================
# FUNCIÓN 6 - Notificar ejecución por Telegram
# ==========================================================
def notificar_testnet(resultado):
    """
    Envía un mensaje al canal de testnet con los detalles
    de la orden ejecutada.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_TESTNET_CHAT_ID:
        return

    dir_emoji = "\U0001f7e2" if resultado["direction"] == "LONG" else "\U0001f534"
    slip_emoji = "✅" if abs(resultado["slippage_pct"]) < 0.05 else "⚠️"

    msg = (
        f"{dir_emoji} <b>ORDEN TESTNET EJECUTADA</b>\n"
        f"\n"
        f"<b>{resultado['direction']}</b> — {resultado['rule']}\n"
        f"\n"
        f"Precio senal: ${resultado['signal_price']:,.2f}\n"
        f"Precio real:  ${resultado['fill_price']:,.2f}\n"
        f"{slip_emoji} Slippage: ${resultado['slippage']:+,.2f} "
        f"({resultado['slippage_pct']:+.4f}%)\n"
        f"\n"
        f"Cantidad: {resultado['quantity']} BTC "
        f"(~${resultado['quantity'] * resultado['fill_price']:,.0f})\n"
        f"SL: ${resultado['sl_price']:,.2f}\n"
        f"TP: ${resultado['tp_price']:,.2f}\n"
    )

    if resultado.get("ai_proba") is not None:
        msg += f"ML: {resultado['ai_proba']:.1%}\n"

    msg += f"\n<i>Paper trading — dinero virtual</i>"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": TELEGRAM_TESTNET_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
    except Exception as e:
        logger.error(f"[Testnet Telegram] Error: {e}")


def notificar_error_testnet(signal_info, error):
    """
    Notifica al canal de testnet si una orden falló.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_TESTNET_CHAT_ID:
        return

    msg = (
        f"❌ <b>ERROR TESTNET</b>\n"
        f"\n"
        f"Senal: {signal_info['direction']} {signal_info['rule']}\n"
        f"Error: {type(error).__name__}: {error}\n"
        f"\n"
        f"<i>La senal se envio por Telegram pero la orden "
        f"en testnet fallo.</i>"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": TELEGRAM_TESTNET_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
    except Exception:
        pass


# ==========================================================
# FUNCIÓN 7 - Registrar trade en CSV
# ==========================================================
def registrar_trade_testnet(resultado):
    """
    Guarda el trade de testnet en un CSV acumulativo.
    """
    import pandas as pd

    os.makedirs("outputs", exist_ok=True)

    row = {
        "timestamp": resultado["timestamp"],
        "direction": resultado["direction"],
        "rule": resultado["rule"],
        "signal_price": resultado["signal_price"],
        "fill_price": resultado["fill_price"],
        "slippage": resultado["slippage"],
        "slippage_pct": resultado["slippage_pct"],
        "quantity": resultado["quantity"],
        "sl_price": resultado["sl_price"],
        "tp_price": resultado["tp_price"],
        "entry_order_id": resultado["entry_order_id"],
        "sl_order_id": resultado["sl_order_id"],
        "tp_order_id": resultado["tp_order_id"],
        "ai_proba": resultado["ai_proba"],
        "status": resultado["status"],
    }

    df = pd.DataFrame([row])

    if os.path.exists(TESTNET_LOG_FILE):
        df.to_csv(TESTNET_LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(TESTNET_LOG_FILE, index=False)

    logger.info(f"[Testnet] Trade registrado en {TESTNET_LOG_FILE}")


# ==========================================================
# FUNCIÓN 8 - Consultar balance de testnet
# ==========================================================
def consultar_balance():
    """
    Consulta y muestra el balance de la cuenta testnet.
    Útil para verificar la conexión y ver cuánto
    dinero virtual tenemos disponible.
    """
    if not testnet_configurado():
        print("Testnet no configurado. Revisa el .env.")
        return None

    client = crear_cliente_testnet()

    # Balance de futuros
    balances = client.futures_account_balance()
    for b in balances:
        if float(b["balance"]) > 0:
            print(f"  {b['asset']}: {float(b['balance']):,.2f} "
                  f"(disponible: {float(b['availableBalance']):,.2f})")

    return balances


# ==========================================================
# FUNCIÓN 9 - Test de conexión
# ==========================================================
def test_conexion():
    """
    Prueba rápida: conectar al testnet, consultar balance,
    y enviar un mensaje al canal de Telegram de testnet.
    """
    print("=" * 50)
    print("TEST DE CONEXION BINANCE TESTNET")
    print("=" * 50)

    if not testnet_configurado():
        print("\nERROR: Claves de testnet no configuradas en .env")
        print("Necesitas BINANCE_TESTNET_API_KEY y "
              "BINANCE_TESTNET_API_SECRET")
        return False

    # 1. Conexión
    print("\n1. Conectando al testnet...")
    try:
        client = crear_cliente_testnet()
        server_time = client.futures_time()
        print(f"   Conectado. Server time: {server_time}")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")
        return False

    # 2. Balance
    print("\n2. Consultando balance...")
    try:
        consultar_balance()
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")
        return False

    # 3. Precio actual
    print("\n3. Precio actual de BTCUSDT...")
    try:
        ticker = client.futures_symbol_ticker(symbol=SYMBOL)
        precio = float(ticker["price"])
        print(f"   BTCUSDT: ${precio:,.2f}")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")
        return False

    # 4. Telegram
    print("\n4. Enviando mensaje de prueba a Telegram testnet...")
    if TELEGRAM_TOKEN and TELEGRAM_TESTNET_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try:
            resp = requests.post(url, json={
                "chat_id": TELEGRAM_TESTNET_CHAT_ID,
                "text": (
                    "\U0001f7e2 <b>Test de conexion testnet</b>\n"
                    "\n"
                    f"Binance Testnet: conectado\n"
                    f"BTCUSDT: ${precio:,.2f}\n"
                    f"\n"
                    "<i>Paper trading listo.</i>"
                ),
                "parse_mode": "HTML"
            }, timeout=10)
            if resp.status_code == 200:
                print("   Mensaje enviado al canal testnet.")
            else:
                print(f"   ERROR Telegram: {resp.status_code}")
        except Exception as e:
            print(f"   ERROR: {e}")
    else:
        print("   Telegram testnet no configurado (sin CHAT_ID)")

    print("\n" + "=" * 50)
    print("TEST COMPLETADO")
    print("=" * 50)
    return True


# ==========================================================
# MAIN (para ejecutar test de conexión)
# ==========================================================
if __name__ == "__main__":
    test_conexion()
