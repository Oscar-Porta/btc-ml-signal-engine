"""
==========================================================
OPTIMIZADOR DE TP/SL - Fase A: Barrido de combinaciones
==========================================================
Este script prueba muchas combinaciones de parámetros de
gestión de riesgo (SL, TP1, TP2, cierre parcial) manteniendo
todo lo demás igual (mismas señales, mismas velas).

NO modifica ningún archivo de producción:
  - NO toca signals.csv, trades.csv, ni el modelo
  - Lee signals.csv + velas 1h procesadas
  - Recalcula SL/TP internamente para cada combinación
  - Guarda resultados en outputs/tp_sl_optimization/

Versión optimizada: pre-extrae las velas posteriores a cada
señal en arrays numpy una sola vez. Después, cada combinación
solo recalcula niveles (vectorizado) y simula con numpy puro.

Uso:
  python src/tp_sl_optimizer.py

Parámetros que se barren:
  - SL_ATR_MULT:     distancia del SL en múltiplos de ATR
  - TP1_R_MULT:      TP1 en múltiplos de R (riesgo)
  - TP2_R_MULT:      TP2 en múltiplos de R
  - PARTIAL_CLOSE:   % de posición que se cierra en TP1
  - MAX_BARS:        máximo de velas antes de time_exit
==========================================================
"""

import os
import itertools
import time
import pandas as pd
import numpy as np
from datetime import datetime


# ==========================================================
# RUTAS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SIGNALS_FILE = os.path.join(OUTPUT_DIR, "signals.csv")
OPTIM_DIR = os.path.join(OUTPUT_DIR, "tp_sl_optimization")

SYMBOL = "BTCUSDT"


# ==========================================================
# PARÁMETROS A BARRER
# ==========================================================
PARAM_GRID = {
    "sl_atr_mult":    [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
    "tp1_r_mult":     [0.75, 1.0, 1.25, 1.5, 2.0],
    "tp2_r_mult":     [1.5, 2.0, 2.5, 3.0, 4.0],
    "partial_close":  [0.0, 0.3, 0.5, 0.7, 1.0],
    "max_bars":       [24],
}

# Configuración actual de producción (para referencia)
PRODUCCION = {
    "sl_atr_mult": 1.5,
    "tp1_r_mult": 1.0,
    "tp2_r_mult": 2.0,
    "partial_close": 0.5,
    "max_bars": 24,
}


# ==========================================================
# FUNCIÓN 1 - Cargar datos y pre-extraer velas
# ==========================================================
def cargar_y_preparar():
    """
    Carga signals.csv y velas 1h. Pre-extrae las 24 velas
    posteriores a cada señal en arrays numpy para que la
    simulación sea rápida.

    Devuelve:
      - entry_prices: array (N,) con precios de entrada
      - atrs: array (N,) con ATR de cada señal
      - directions: array (N,) con 1=LONG, -1=SHORT
      - candle_highs: array (N, MAX_BARS) highs post-señal
      - candle_lows: array (N, MAX_BARS) lows post-señal
      - candle_closes: array (N, MAX_BARS) closes post-señal
      - valid_bars: array (N,) cuántas velas válidas hay
    """
    df_signals = pd.read_csv(SIGNALS_FILE)
    df_1h = pd.read_csv(
        os.path.join(PROCESSED_DIR, f"{SYMBOL}_1h_processed.csv")
    )

    # Convertir fechas
    df_signals["open_time"] = pd.to_datetime(
        df_signals["open_time"], format="ISO8601", utc=True
    ).dt.tz_convert("Europe/Madrid")

    for col in ["open_time", "close_time"]:
        df_1h[col] = pd.to_datetime(
            df_1h[col], format="ISO8601", utc=True
        ).dt.tz_convert("Europe/Madrid")

    df_signals = df_signals.sort_values("open_time").reset_index(drop=True)
    df_1h = df_1h.sort_values("open_time").reset_index(drop=True)

    print(f"Señales cargadas: {len(df_signals)}")
    print(f"Velas 1h cargadas: {len(df_1h)}")

    # Índice temporal
    time_to_idx = {}
    for i, t in enumerate(df_1h["open_time"]):
        time_to_idx[t] = i

    # Arrays de velas 1h completos
    all_highs = df_1h["high"].values
    all_lows = df_1h["low"].values
    all_closes = df_1h["close"].values
    n_candles = len(df_1h)

    max_bars = max(PARAM_GRID["max_bars"])

    # Pre-extraer velas para cada señal
    N = len(df_signals)
    entry_prices = np.zeros(N)
    atrs = np.zeros(N)
    directions = np.zeros(N, dtype=np.int8)
    candle_highs = np.zeros((N, max_bars))
    candle_lows = np.zeros((N, max_bars))
    candle_closes = np.zeros((N, max_bars))
    valid_bars = np.zeros(N, dtype=np.int32)
    valid_mask = np.zeros(N, dtype=bool)

    print("Pre-extrayendo velas posteriores a cada señal...")
    for i in range(N):
        sig_time = df_signals.iloc[i]["open_time"]

        if sig_time not in time_to_idx:
            continue

        idx = time_to_idx[sig_time]
        start = idx + 1
        end = min(start + max_bars, n_candles)
        n_bars = end - start

        if n_bars <= 0:
            continue

        valid_mask[i] = True
        entry_prices[i] = df_signals.iloc[i]["entry_price"]
        atrs[i] = df_signals.iloc[i]["atr_14"]
        directions[i] = 1 if df_signals.iloc[i]["direction"] == "LONG" else -1
        candle_highs[i, :n_bars] = all_highs[start:end]
        candle_lows[i, :n_bars] = all_lows[start:end]
        candle_closes[i, :n_bars] = all_closes[start:end]
        valid_bars[i] = n_bars

    # Filtrar solo señales válidas
    entry_prices = entry_prices[valid_mask]
    atrs = atrs[valid_mask]
    directions = directions[valid_mask]
    candle_highs = candle_highs[valid_mask]
    candle_lows = candle_lows[valid_mask]
    candle_closes = candle_closes[valid_mask]
    valid_bars = valid_bars[valid_mask]

    print(f"Señales válidas (con velas post): {len(entry_prices)}")

    return (entry_prices, atrs, directions,
            candle_highs, candle_lows, candle_closes, valid_bars)


# ==========================================================
# FUNCIÓN 2 - Simular todos los trades para una combinación
# ==========================================================
def simular_todos(entry_prices, atrs, directions,
                  candle_highs, candle_lows, candle_closes,
                  valid_bars, sl_atr_mult, tp1_r_mult,
                  tp2_r_mult, partial_close, max_bars):
    """
    Simula todos los trades de golpe para una combinación.
    Usa un bucle por vela (máx 24 iteraciones) pero dentro
    de cada iteración opera vectorizado sobre todas las señales.

    Devuelve arrays con:
      - pnl_abs (N,)
      - pnl_pct (N,)
      - r_multiple (N,)
      - close_reason (N,) enteros: 0=pendiente, 1=SL, 2=TP1,
                                   3=TP2, 4=breakeven, 5=time_exit
      - bars_in_trade (N,)
    """
    N = len(entry_prices)

    # Recalcular niveles
    riesgo = atrs * sl_atr_mult
    is_long = directions == 1

    sl = np.where(is_long,
                  entry_prices - riesgo,
                  entry_prices + riesgo)
    tp1 = np.where(is_long,
                   entry_prices + riesgo * tp1_r_mult,
                   entry_prices - riesgo * tp1_r_mult)
    tp2 = np.where(is_long,
                   entry_prices + riesgo * tp2_r_mult,
                   entry_prices - riesgo * tp2_r_mult)

    # Estado del trade
    # close_reason: 0=abierto, 1=SL, 2=TP1, 3=TP2, 4=BE, 5=time_exit
    close_reason = np.zeros(N, dtype=np.int8)
    exit_price = np.zeros(N)
    bars_in_trade = np.zeros(N, dtype=np.int32)
    tp1_hit = np.zeros(N, dtype=bool)
    current_sl = sl.copy()

    # Recorrer vela a vela
    for bar in range(max_bars):
        # Señales aún abiertas y con velas disponibles
        abiertos = (close_reason == 0) & (bar < valid_bars)
        if not abiertos.any():
            break

        h = candle_highs[:, bar]
        l = candle_lows[:, bar]
        c = candle_closes[:, bar]

        # Trades que acaban de tocar TP1 en ESTA vela.
        # No deben evaluarse en Fase B hasta la siguiente vela
        # (igual que en 04_backtest.py donde el loop continúa).
        just_hit_tp1 = np.zeros(N, dtype=bool)

        # === FASE A: buscando TP1 (tp1_hit == False) ===
        fase_a = abiertos & ~tp1_hit

        if fase_a.any():
            # SL tocado
            sl_hit_long = fase_a & is_long & (l <= current_sl)
            sl_hit_short = fase_a & ~is_long & (h >= current_sl)
            sl_hit = sl_hit_long | sl_hit_short

            # TP1 tocado
            tp1_hit_long = fase_a & is_long & (h >= tp1)
            tp1_hit_short = fase_a & ~is_long & (l <= tp1)
            tp1_tocado = tp1_hit_long | tp1_hit_short

            # Ambos en misma vela → peor caso (SL)
            ambos = sl_hit & tp1_tocado
            solo_sl = sl_hit & ~tp1_tocado
            solo_tp1 = tp1_tocado & ~sl_hit

            # Cerrar por SL (ambos + solo SL)
            cierra_sl = ambos | solo_sl
            close_reason[cierra_sl] = 1
            exit_price[cierra_sl] = current_sl[cierra_sl]
            bars_in_trade[cierra_sl] = bar + 1

            # TP1 tocado (sin SL)
            if partial_close >= 1.0:
                # 100% cierra en TP1
                close_reason[solo_tp1] = 2
                exit_price[solo_tp1] = tp1[solo_tp1]
                bars_in_trade[solo_tp1] = bar + 1
                tp1_hit[solo_tp1] = True
            else:
                # Cierre parcial: TP1 hit, SL sube a breakeven
                tp1_hit[solo_tp1] = True
                current_sl[solo_tp1] = entry_prices[solo_tp1]
                just_hit_tp1 = solo_tp1  # marcar para excluir de fase B

        # === FASE B: TP1 ya tocado, buscando TP2 ===
        # Excluir trades que acaban de tocar TP1 en esta misma vela
        fase_b = abiertos & tp1_hit & (close_reason == 0) & ~just_hit_tp1

        if fase_b.any():
            # SL (breakeven) tocado
            be_hit_long = fase_b & is_long & (l <= current_sl)
            be_hit_short = fase_b & ~is_long & (h >= current_sl)
            be_hit = be_hit_long | be_hit_short

            # TP2 tocado
            tp2_hit_long = fase_b & is_long & (h >= tp2)
            tp2_hit_short = fase_b & ~is_long & (l <= tp2)
            tp2_tocado = tp2_hit_long | tp2_hit_short

            # Ambos → peor caso (breakeven)
            ambos_b = be_hit & tp2_tocado
            solo_be = be_hit & ~tp2_tocado
            solo_tp2 = tp2_tocado & ~be_hit

            cierra_be = ambos_b | solo_be
            close_reason[cierra_be] = 4
            exit_price[cierra_be] = current_sl[cierra_be]
            bars_in_trade[cierra_be] = bar + 1

            close_reason[solo_tp2] = 3
            exit_price[solo_tp2] = tp2[solo_tp2]
            bars_in_trade[solo_tp2] = bar + 1

    # TIME EXIT: los que quedaron abiertos
    time_exit_mask = close_reason == 0
    if time_exit_mask.any():
        # Usar el close de la última vela disponible
        last_bar_idx = np.minimum(valid_bars - 1, max_bars - 1)
        last_bar_idx = np.maximum(last_bar_idx, 0)
        for i in np.where(time_exit_mask)[0]:
            exit_price[i] = candle_closes[i, last_bar_idx[i]]
            bars_in_trade[i] = min(valid_bars[i], max_bars)
        close_reason[time_exit_mask] = 5

    # Calcular PnL
    final_pnl_abs = np.where(is_long,
                             exit_price - entry_prices,
                             entry_prices - exit_price)
    final_pnl_pct = np.where(is_long,
                             (exit_price / entry_prices - 1) * 100,
                             (entry_prices / exit_price - 1) * 100)

    tp1_pnl_abs = np.where(is_long,
                           tp1 - entry_prices,
                           entry_prices - tp1)
    tp1_pnl_pct = np.where(is_long,
                           (tp1 / entry_prices - 1) * 100,
                           (entry_prices / tp1 - 1) * 100)

    # PnL combinado según cierre parcial
    if partial_close > 0:
        combinar = tp1_hit
        pnl_abs = np.where(
            combinar,
            tp1_pnl_abs * partial_close + final_pnl_abs * (1 - partial_close),
            final_pnl_abs
        )
        pnl_pct = np.where(
            combinar,
            tp1_pnl_pct * partial_close + final_pnl_pct * (1 - partial_close),
            final_pnl_pct
        )
    else:
        pnl_abs = final_pnl_abs
        pnl_pct = final_pnl_pct

    r_multiple = np.where(riesgo > 0, pnl_abs / riesgo, 0.0)

    return pnl_abs, pnl_pct, r_multiple, close_reason, bars_in_trade


# ==========================================================
# FUNCIÓN 3 - Calcular métricas de un conjunto de trades
# ==========================================================
def calcular_metricas(pnl_abs, pnl_pct, r_multiple,
                      close_reason, bars_in_trade):
    """
    Calcula métricas clave a partir de arrays numpy.
    """
    total = len(pnl_abs)
    if total == 0:
        return None

    ganadores = pnl_abs > 0
    perdedores = pnl_abs < 0
    n_win = ganadores.sum()
    n_loss = perdedores.sum()

    win_rate = n_win / total * 100
    pnl_total = pnl_abs.sum()
    pnl_medio = pnl_abs.mean()
    r_medio = r_multiple.mean()

    avg_win = abs(pnl_abs[ganadores].mean()) if n_win > 0 else 0
    avg_loss = abs(pnl_abs[perdedores].mean()) if n_loss > 0 else 0
    payoff = avg_win / avg_loss if avg_loss > 0 else 0

    breakeven_wr = (1 / (1 + payoff) * 100) if payoff > 0 else 50.0
    margen = win_rate - breakeven_wr

    gross_profit = pnl_abs[ganadores].sum() if n_win > 0 else 0
    gross_loss = abs(pnl_abs[perdedores].sum()) if n_loss > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max drawdown
    cumulative = np.cumsum(pnl_abs)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = drawdown.min()

    # Desglose por close_reason
    # 1=SL, 2=TP1, 3=TP2, 4=BE, 5=time_exit
    n_sl = (close_reason == 1).sum()
    n_tp1 = (close_reason == 2).sum()
    n_tp2 = (close_reason == 3).sum()
    n_be = (close_reason == 4).sum()
    n_te = (close_reason == 5).sum()

    dur_media = bars_in_trade.mean()

    return {
        "total_trades": int(total),
        "win_rate": round(float(win_rate), 2),
        "pnl_total": round(float(pnl_total), 2),
        "pnl_medio": round(float(pnl_medio), 2),
        "r_medio": round(float(r_medio), 4),
        "payoff": round(float(payoff), 2),
        "breakeven_wr": round(float(breakeven_wr), 1),
        "margen": round(float(margen), 1),
        "profit_factor": round(float(profit_factor), 2),
        "max_drawdown_usd": round(float(max_dd), 2),
        "dur_media_bars": round(float(dur_media), 1),
        "n_stop_loss": int(n_sl),
        "n_tp1": int(n_tp1),
        "n_tp2": int(n_tp2),
        "n_breakeven": int(n_be),
        "n_time_exit": int(n_te),
    }


# ==========================================================
# FUNCIÓN 4 - Generar combinaciones válidas
# ==========================================================
def generar_combinaciones():
    """
    Genera combinaciones del grid. Filtra las inválidas:
    - TP2 debe ser > TP1
    - Si partial_close=0 (sin cierre parcial), TP1 no importa
      → fijamos TP1 a un solo valor para no duplicar
    """
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    combos = []
    for vals in itertools.product(*values):
        params = dict(zip(keys, vals))

        if params["tp2_r_mult"] <= params["tp1_r_mult"]:
            continue

        if params["partial_close"] == 0.0 and params["tp1_r_mult"] != 1.0:
            continue

        combos.append(params)

    return combos


# ==========================================================
# FUNCIÓN 5 - Ejecutar optimización completa
# ==========================================================
def ejecutar_optimizacion():
    """
    Función principal. Carga datos, genera combinaciones,
    ejecuta backtests vectorizados y guarda resultados.
    """
    print("=" * 60)
    print("OPTIMIZADOR TP/SL - FASE A: BARRIDO DE COMBINACIONES")
    print("=" * 60)

    # 1. Cargar y preparar datos
    print("\n[1/4] Cargando datos y pre-extrayendo velas...")
    (entry_prices, atrs, directions,
     candle_highs, candle_lows, candle_closes,
     valid_bars) = cargar_y_preparar()

    # 2. Generar combinaciones
    print("\n[2/4] Generando combinaciones...")
    combos = generar_combinaciones()
    print(f"  Combinaciones válidas: {len(combos)}")
    print(f"\n  Configuración actual (producción):")
    for k, v in PRODUCCION.items():
        print(f"    {k}: {v}")

    # 3. Ejecutar backtests
    print(f"\n[3/4] Ejecutando {len(combos)} backtests vectorizados...")
    print(f"  ({len(entry_prices)} señales × {max(PARAM_GRID['max_bars'])} "
          f"velas máx)\n")

    resultados = []
    t_inicio = time.time()

    for i, params in enumerate(combos):
        if (i + 1) % 50 == 0 or (i + 1) == len(combos) or i == 0:
            elapsed = time.time() - t_inicio
            if i > 0:
                eta = (elapsed / (i + 1)) * (len(combos) - i - 1)
                print(f"  [{i+1}/{len(combos)}] "
                      f"{elapsed:.1f}s | ETA: {eta:.0f}s")
            else:
                print(f"  [{i+1}/{len(combos)}] Iniciando...")

        pnl_abs, pnl_pct, r_mult, cr, bars = simular_todos(
            entry_prices, atrs, directions,
            candle_highs, candle_lows, candle_closes,
            valid_bars,
            params["sl_atr_mult"],
            params["tp1_r_mult"],
            params["tp2_r_mult"],
            params["partial_close"],
            params["max_bars"]
        )

        metricas = calcular_metricas(pnl_abs, pnl_pct, r_mult, cr, bars)
        if metricas is None:
            continue

        fila = {**params, **metricas}
        fila["es_produccion"] = all(
            params[k] == PRODUCCION[k] for k in PRODUCCION
        )
        resultados.append(fila)

    t_total = time.time() - t_inicio
    print(f"\n  Completado en {t_total:.1f}s")

    # 4. Guardar
    print(f"\n[4/4] Guardando resultados...")
    df_results = pd.DataFrame(resultados)
    df_results = df_results.sort_values("pnl_total", ascending=False)
    df_results = df_results.reset_index(drop=True)

    os.makedirs(OPTIM_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(OPTIM_DIR, f"sweep_results_{timestamp}.csv")
    df_results.to_csv(csv_file, index=False)
    print(f"  Guardado en: {csv_file}")

    latest_file = os.path.join(OPTIM_DIR, "sweep_results_latest.csv")
    df_results.to_csv(latest_file, index=False)
    print(f"  Guardado en: {latest_file}")

    # 5. Resumen
    mostrar_resumen(df_results)

    return df_results


# ==========================================================
# FUNCIÓN 6 - Mostrar resumen
# ==========================================================
def mostrar_resumen(df):
    """
    Muestra las mejores combinaciones y compara con producción.
    """
    print(f"\n{'=' * 60}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'=' * 60}")

    # Producción
    prod = df[df["es_produccion"] == True]
    if len(prod) > 0:
        p = prod.iloc[0]
        print(f"\n--- CONFIGURACIÓN ACTUAL (producción) ---")
        print(f"  SL: {p['sl_atr_mult']}x ATR | "
              f"TP1: {p['tp1_r_mult']}R | "
              f"TP2: {p['tp2_r_mult']}R | "
              f"Parcial: {p['partial_close']*100:.0f}%")
        print(f"  Trades: {int(p['total_trades'])} | "
              f"WR: {p['win_rate']:.1f}% | "
              f"Payoff: {p['payoff']:.2f} | "
              f"PnL: ${p['pnl_total']:,.0f} | "
              f"R medio: {p['r_medio']:+.4f}")
        print(f"  BE WR: {p['breakeven_wr']:.1f}% | "
              f"Margen: {p['margen']:+.1f}pp | "
              f"PF: {p['profit_factor']:.2f} | "
              f"Max DD: ${p['max_drawdown_usd']:,.0f}")

    cols = ["sl_atr_mult", "tp1_r_mult", "tp2_r_mult",
            "partial_close", "total_trades", "win_rate",
            "payoff", "pnl_total", "r_medio", "margen",
            "profit_factor", "max_drawdown_usd"]

    # Top 10 por PnL
    print(f"\n--- TOP 10 POR PNL TOTAL ---")
    print(df[cols].head(10).to_string(index=False))

    # Top 10 por margen
    print(f"\n--- TOP 10 POR MARGEN DE SEGURIDAD ---")
    df_m = df.sort_values("margen", ascending=False)
    print(df_m[cols].head(10).to_string(index=False))

    # Top 10 por profit factor
    print(f"\n--- TOP 10 POR PROFIT FACTOR ---")
    df_pf = df.sort_values("profit_factor", ascending=False)
    print(df_pf[cols].head(10).to_string(index=False))

    # Top 10 por R medio
    print(f"\n--- TOP 10 POR R MEDIO ---")
    df_r = df.sort_values("r_medio", ascending=False)
    print(df_r[cols].head(10).to_string(index=False))

    # Estadísticas
    print(f"\n--- ESTADÍSTICAS DEL BARRIDO ---")
    print(f"  Combinaciones probadas: {len(df)}")
    rent = (df['pnl_total'] > 0).sum()
    print(f"  Rentables (PnL > 0): {rent} ({rent/len(df)*100:.1f}%)")
    marg = (df['margen'] > 0).sum()
    print(f"  Con margen > 0: {marg} ({marg/len(df)*100:.1f}%)")
    print(f"  PnL rango: ${df['pnl_total'].min():,.0f} "
          f"a ${df['pnl_total'].max():,.0f}")
    print(f"  WR rango: {df['win_rate'].min():.1f}% "
          f"a {df['win_rate'].max():.1f}%")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    ejecutar_optimizacion()
