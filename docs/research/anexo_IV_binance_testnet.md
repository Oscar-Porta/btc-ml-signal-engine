# Anexo IV — Paper Trading en Binance Futures Testnet

> Validar el pipeline completo de señales colocando órdenes
> reales en Binance Futures Testnet (dinero virtual), midiendo
> slippage, ejecución de SL/TP y latencia.

---

## 1. Objetivo

El sistema de señales genera entradas con SL y TP, pero
hasta ahora solo se envían por Telegram como texto. No hay
forma de saber si la ejecución real hubiera sido rentable
porque no se mide:

- **Slippage:** diferencia entre el precio de la señal y el
  precio real de llenado.
- **Ejecución de SL/TP:** si Binance los ejecuta al precio
  indicado o hay desviación.
- **Latencia:** tiempo entre la detección de la señal y la
  colocación de la orden.

Binance Futures Testnet permite operar con dinero virtual
usando la misma API que producción. Las órdenes se llenan
con precios reales de mercado.

---

## 2. Infraestructura

### 2.1 Cuenta testnet

Binance Futures Testnet es un entorno separado de la cuenta
real. Se accede desde `testnet.binancefuture.com`. Requiere
crear API keys propias (no sirven las de producción).

La cuenta viene con 5.000 USDT virtuales de balance inicial.

### 2.2 Telegram separado

Se creó un grupo de Telegram exclusivo para notificaciones
de testnet, independiente del canal de señales de producción.
Esto evita mezclar órdenes virtuales con señales reales.

Variables de entorno necesarias:

```
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_API_SECRET=...
TELEGRAM_TESTNET_CHAT_ID=...
```

Si estas variables no están en `.env`, el paper trading
simplemente no se activa. No afecta al sistema de producción.

---

## 3. Implementación

### 3.1 Script: `src/testnet_trader.py`

Script independiente que gestiona todo el paper trading.
No modifica ningún archivo del pipeline de producción.

**Funciones principales:**

| Función | Descripción |
|---------|-------------|
| `testnet_configurado()` | Verifica si las claves de testnet están en `.env` |
| `crear_cliente_testnet()` | Crea cliente Binance con `testnet=True` |
| `calcular_cantidad(precio)` | Calcula qty en BTC según `POSITION_SIZE_USDT` |
| `ejecutar_orden_testnet(signal)` | Coloca entrada + SL + TP en testnet |
| `notificar_testnet(resultado)` | Envía mensaje al grupo de Telegram testnet |
| `notificar_error_testnet(signal, error)` | Notifica si una orden falla |
| `registrar_trade_testnet(resultado)` | Guarda el trade en CSV |
| `consultar_balance()` | Muestra balance de la cuenta testnet |
| `test_conexion()` | Prueba rápida de conectividad |

**Flujo de ejecución cuando hay señal:**

1. Orden de mercado (entrada) — captura el precio real de
   llenado para medir slippage.
2. STOP_MARKET (stop loss) — vía Algo Order API con
   `closePosition=true`.
3. TAKE_PROFIT_MARKET (take profit) — mismo endpoint.
4. Notificación a Telegram testnet con detalles y slippage.
5. Registro en `outputs/testnet_trades.csv`.

**Configuración por defecto:**

- Tamaño de posición: 100 USDT por trade (~0.001 BTC)
- Cantidad mínima: 0.001 BTC (mínimo de Binance Futures)
- Precios redondeados a 1 decimal (requisito de BTCUSDT)

### 3.2 Integración en `src/08_live_signal.py`

Se añadió una llamada a `ejecutar_orden_testnet()` después
de registrar cada señal. La integración usa `try/except
ImportError` para que si el script de testnet no existe
o falla, el sistema de producción siga funcionando.

```python
try:
    from testnet_trader import ejecutar_orden_testnet, testnet_configurado
except ImportError:
    def ejecutar_orden_testnet(signal_info):
        return None
    def testnet_configurado():
        return False
```

---

## 4. API de Binance: Algo Order

Desde diciembre de 2025, Binance migró las órdenes
condicionales (STOP_MARKET, TAKE_PROFIT_MARKET) del endpoint
clásico `/fapi/v1/order` al nuevo `/fapi/v1/algoOrder`.

El endpoint clásico devuelve error `-4120` para estos tipos
de orden. La librería `python-binance` no tiene soporte
nativo para el nuevo endpoint, así que se usa su método
interno `_request_futures_api`.

**Parámetros requeridos:**

```python
client._request_futures_api(
    'post', 'algoOrder', signed=True, data={
        "symbol": "BTCUSDT",
        "side": "SELL",            # contrario a la entrada
        "algoType": "CONDITIONAL", # obligatorio
        "type": "STOP_MARKET",     # o TAKE_PROFIT_MARKET
        "triggerPrice": "77793.4",
        "closePosition": "true",
        "workingType": "MARK_PRICE",
    }
)
```

La respuesta incluye un `algoId` en lugar de `orderId`.

---

## 5. Prueba de validación

Se ejecutó una orden manual de prueba el 2 de mayo de 2026:

| Campo | Valor |
|-------|-------|
| Dirección | LONG |
| Precio señal | $78,143.08 |
| Cantidad | 0.001 BTC (~$78) |
| Stop Loss | $77,793.40 |
| Take Profit | $78,842.50 |
| Entry Order ID | 13100297208 |
| SL Order ID | 1000000064196666 |
| TP Order ID | 1000000064196667 |

Las tres órdenes se colocaron correctamente en testnet.
La notificación de Telegram se recibió en el grupo de testnet.

**Nota sobre el testnet:** el campo `avgPrice` de la respuesta
a veces devuelve 0 en testnet (en producción funciona
correctamente). Se implementó un fallback que consulta el
precio del ticker en esos casos.

---

## 6. Registro de trades

Los trades de testnet se acumulan en
`outputs/testnet_trades.csv` con las columnas:

```
timestamp, direction, rule, signal_price, fill_price,
slippage, slippage_pct, quantity, sl_price, tp_price,
entry_order_id, sl_order_id, tp_order_id, ai_proba, status
```

Este CSV permitirá analizar el rendimiento real del paper
trading: slippage medio, tasa de aciertos en SL/TP, y
comparar con el backtest.

---

**Rama:** `feature/binance-testnet`

*Fecha: 2 de mayo de 2026.*
