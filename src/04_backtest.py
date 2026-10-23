"""
==========================================================
FASE 4 - Motor de backtest con detección fiable de SL/TP
==========================================================
Este script toma las señales generadas en la Fase 3
(outputs/signals.csv) y simula cada trade recorriendo las
velas posteriores de 1h una a una.

Para cada señal:
  1. Busca la vela de entrada (la siguiente a la señal).
  2. Recorre velas posteriores comprobando high/low contra
     los niveles de SL, TP1 y TP2.
  3. Gestión en dos fases:
     - Se busca TP1. Si se toca, se cierra el 100% de la posición.
     - No hay segunda fase (Fase B eliminada, Anexo II config A).
  4. Si en la misma vela tocan SL y TP, asumimos el peor caso.
  5. Límite temporal: 24 velas (24h). Si no se cierra, se
     sale al close de la vela 24.

Produce: outputs/trades.csv con el resultado de cada trade.
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
TRADES_FILE = os.path.join(OUTPUT_DIR, "trades.csv")

# Máximo de velas que dejamos un trade abierto.
# 24 velas de 1h = 24 horas. Pasado ese tiempo, cerramos
# al precio de cierre de la última vela (time_exit).
MAX_BARS_IN_TRADE = 24

# --- Fricción del trade (A3, revisión jul 2026) ---
# El backtest antiguo no descontaba costes: cada trade se
# simulaba "gratis". En la realidad, un usuario que sigue la
# señal en futuros de Binance paga:
#   - Comisión taker: 0.05% del nocional POR LADO (entrar y salir)
#   - Slippage: ~0.02% por lado (el precio se mueve entre la
#     señal de Telegram y la ejecución de la orden)
# Total: 0.07% por lado = 0.14% por trade completo.
# Decisión de diseño (3 jul 2026): el PnL, el R-multiple y las
# etiquetas del modelo se calculan en NETO (después de costes).
FEE_PCT_PER_SIDE = 0.0005        # 0.05% comisión taker futuros
SLIPPAGE_PCT_PER_SIDE = 0.0002   # 0.02% slippage estimado
FRICTION_PER_SIDE = FEE_PCT_PER_SIDE + SLIPPAGE_PCT_PER_SIDE


# ==========================================================
# FUNCIÓN 1 - Cargar datos necesarios
# ==========================================================
def cargar_datos():
    """
    Carga las señales y las velas de 1h procesadas.
    Las velas se usan para recorrer el mercado tras la señal.
    """
    df_signals = pd.read_csv(SIGNALS_FILE)
    df_1h = pd.read_csv(
        os.path.join(PROCESSED_DIR, f"{SYMBOL}_1h_processed.csv")
    )

    # Conversión de fechas (mismo patrón que en 03_rules.py)
    for col in ["open_time"]:
        df_signals[col] = pd.to_datetime(
            df_signals[col], format="ISO8601", utc=True
        )
        df_signals[col] = df_signals[col].dt.tz_convert("Europe/Madrid")

    for col in ["open_time", "close_time"]:
        df_1h[col] = pd.to_datetime(
            df_1h[col], format="ISO8601", utc=True
        )
        df_1h[col] = df_1h[col].dt.tz_convert("Europe/Madrid")

    # Ordenar por tiempo
    df_signals = df_signals.sort_values("open_time").reset_index(drop=True)
    df_1h = df_1h.sort_values("open_time").reset_index(drop=True)

    print(f"Señales cargadas: {len(df_signals)}")
    print(f"Velas 1h cargadas: {len(df_1h)}")

    return df_signals, df_1h


# ==========================================================
# FUNCIÓN 2 - Simular un trade individual
# ==========================================================
def simular_trade(signal, df_velas_post):
    """
    Simula un trade a partir de una señal y las velas
    posteriores a ella.

    Parámetros:
      signal:        fila del DataFrame de señales (Series)
      df_velas_post: DataFrame con las velas de 1h que vienen
                     DESPUÉS de la vela de señal.

    Devuelve un diccionario con el resultado del trade, o
    None si no hay suficientes velas para simular.
    """
    direction = signal["direction"]
    entry_price = signal["entry_price"]
    stop_loss = signal["stop_loss"]
    tp1 = signal["tp1"]

    # Si no hay velas posteriores, no podemos simular.
    if len(df_velas_post) == 0:
        return None

    # Riesgo inicial en USD (siempre positivo).
    # Es la distancia entre entrada y SL original.
    risk_amount = abs(entry_price - stop_loss)

    # --- MAE (Maximum Adverse Excursion) ---
    # Rastreamos el peor punto que el precio alcanza EN CONTRA
    # de nuestra dirección durante la vida del trade.
    # Para LONG: el low más bajo (peor punto = precio baja)
    # Para SHORT: el high más alto (peor punto = precio sube)
    worst_adverse = 0.0  # en USD, siempre <= 0

    # Recorremos vela a vela (máximo MAX_BARS_IN_TRADE)
    velas_a_revisar = df_velas_post.head(MAX_BARS_IN_TRADE)

    for bar_num, (idx, vela) in enumerate(velas_a_revisar.iterrows(), 1):
        high = vela["high"]
        low = vela["low"]
        close = vela["close"]
        vela_time = vela["open_time"]

        # --- Actualizar MAE en cada vela ---
        if direction == "LONG":
            adverse = low - entry_price      # negativo si baja
        else:
            adverse = entry_price - high     # negativo si sube
        worst_adverse = min(worst_adverse, adverse)

        # ==================================================
        # LONG
        # ==================================================
        if direction == "LONG":
            sl_tocado = low <= stop_loss
            tp1_tocado = high >= tp1

            if sl_tocado and tp1_tocado:
                # Ambos en la misma vela → peor caso (SL)
                return _resultado(
                    signal, "stop_loss", stop_loss,
                    vela_time, bar_num, entry_price,
                    risk_amount, worst_adverse
                )
            elif sl_tocado:
                return _resultado(
                    signal, "stop_loss", stop_loss,
                    vela_time, bar_num, entry_price,
                    risk_amount, worst_adverse
                )
            elif tp1_tocado:
                # TP1 alcanzado → cierre 100%
                return _resultado(
                    signal, "take_profit_1", tp1,
                    vela_time, bar_num, entry_price,
                    risk_amount, worst_adverse
                )

        # ==================================================
        # SHORT
        # ==================================================
        elif direction == "SHORT":
            sl_tocado = high >= stop_loss
            tp1_tocado = low <= tp1

            if sl_tocado and tp1_tocado:
                return _resultado(
                    signal, "stop_loss", stop_loss,
                    vela_time, bar_num, entry_price,
                    risk_amount, worst_adverse
                )
            elif sl_tocado:
                return _resultado(
                    signal, "stop_loss", stop_loss,
                    vela_time, bar_num, entry_price,
                    risk_amount, worst_adverse
                )
            elif tp1_tocado:
                # TP1 alcanzado → cierre 100%
                return _resultado(
                    signal, "take_profit_1", tp1,
                    vela_time, bar_num, entry_price,
                    risk_amount, worst_adverse
                )

    # ==================================================
    # TIME EXIT: se acabaron las velas sin tocar nada
    # ==================================================
    ultima_vela = velas_a_revisar.iloc[-1]
    return _resultado(
        signal, "time_exit", ultima_vela["close"],
        ultima_vela["open_time"], len(velas_a_revisar),
        entry_price, risk_amount, worst_adverse
    )


# ==========================================================
# FUNCIÓN AUXILIAR - Construir resultado de un trade
# ==========================================================
def _resultado(signal, close_reason, exit_price, exit_time,
               bars_in_trade, entry_price, risk_amount,
               worst_adverse):
    """
    Construye el diccionario de resultado de un trade cerrado.
    Calcula PnL absoluto y porcentual según la dirección.

    Cierre al 100% en el precio de salida (sin cierre parcial).

    Campos:
      - r_multiple: PnL expresado en unidades de riesgo.
      - mae_r: Maximum Adverse Excursion en unidades de riesgo.
    """
    direction = signal["direction"]

    # PnL del cierre (entry → exit_price), BRUTO
    if direction == "LONG":
        pnl_bruto = exit_price - entry_price
    else:  # SHORT
        pnl_bruto = entry_price - exit_price

    # --- Fricción (A3): comisión + slippage, por los dos lados ---
    # Se paga un % del precio al entrar y otro al salir.
    friccion = (entry_price + exit_price) * FRICTION_PER_SIDE

    # PnL NETO: lo que de verdad se lleva el usuario.
    pnl_abs = pnl_bruto - friccion
    pnl_pct = pnl_abs / entry_price * 100

    # R-multiple: PnL NETO / riesgo inicial.
    # (Decisión 3 jul 2026: las etiquetas del modelo se basan
    # en el resultado neto, no en el bruto.)
    if risk_amount > 0:
        r_multiple = pnl_abs / risk_amount
    else:
        r_multiple = 0.0

    # MAE en unidades de riesgo (siempre <= 0).
    if risk_amount > 0:
        mae_r = worst_adverse / risk_amount
    else:
        mae_r = 0.0

    return {
        # Identificación
        "signal_time": signal["open_time"],
        "direction": direction,
        "rule": signal["rule"],

        # Niveles
        "entry_price": entry_price,
        "stop_loss": signal["stop_loss"],
        "tp1": signal["tp1"],
        "tp2": signal.get("tp2"),  # compatibilidad

        # Resultado
        "exit_time": exit_time,
        "exit_price": exit_price,
        "close_reason": close_reason,
        "bars_in_trade": bars_in_trade,
        "pnl_abs": round(pnl_abs, 2),
        "pnl_pct": round(pnl_pct, 4),
        "pnl_bruto": round(pnl_bruto, 2),
        "friccion_usd": round(friccion, 2),
        "r_multiple": round(r_multiple, 4),
        "mae_r": round(mae_r, 4),

        # Contexto (features para ML)
        "rsi_14": signal.get("rsi_14"),
        "atr_14": signal.get("atr_14"),
        "trend_1h": signal.get("trend_1h"),
        "trend_4h": signal.get("trend_4h"),
    }


# ==========================================================
# FUNCIÓN 3 - Ejecutar backtest completo
# ==========================================================
def ejecutar_backtest(df_signals, df_1h):
    """
    Recorre todas las señales, simula cada trade y devuelve
    un DataFrame con los resultados.
    """
    # Creamos un índice de open_time → posición en df_1h
    # para buscar rápidamente la vela de cada señal.
    time_to_idx = pd.Series(
        df_1h.index, index=df_1h["open_time"]
    )

    trades = []
    no_simulados = 0

    total = len(df_signals)

    for i, (_, signal) in enumerate(df_signals.iterrows()):
        # Progreso cada 500 señales
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  Procesando señal {i+1}/{total}...")

        signal_time = signal["open_time"]

        # Buscar la posición de la vela de señal en df_1h.
        if signal_time not in time_to_idx.index:
            no_simulados += 1
            continue

        idx_signal = time_to_idx[signal_time]

        # Las velas posteriores a la señal (excluyendo la
        # propia vela de señal, porque la señal se genera
        # al cierre de esa vela).
        # Optimización (jul 2026): solo hacen falta como máximo
        # MAX_BARS_IN_TRADE velas — simular_trade no mira más
        # allá. Antes se copiaba el resto del histórico entero
        # (~40k filas por señal). Resultado idéntico, mucho
        # más rápido.
        df_post = df_1h.iloc[
            idx_signal + 1 : idx_signal + 1 + MAX_BARS_IN_TRADE
        ]

        resultado = simular_trade(signal, df_post)

        if resultado is not None:
            trades.append(resultado)
        else:
            no_simulados += 1

    print(f"\nTrades simulados: {len(trades)}")
    if no_simulados > 0:
        print(f"Señales no simulables (sin velas post): {no_simulados}")

    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(trades)


# ==========================================================
# FUNCIÓN 4 - Mostrar resumen del backtest
# ==========================================================
def mostrar_resumen(df_trades):
    """
    Imprime un resumen rápido de los resultados del backtest.
    """
    print(f"\n{'='*60}")
    print("RESUMEN DEL BACKTEST")
    print(f"{'='*60}")

    total = len(df_trades)
    if total == 0:
        print("No hay trades para mostrar.")
        return

    # --- Resultados generales ---
    ganadores = df_trades[df_trades["pnl_abs"] > 0]
    perdedores = df_trades[df_trades["pnl_abs"] < 0]
    neutros = df_trades[df_trades["pnl_abs"] == 0]

    win_rate = len(ganadores) / total * 100
    pnl_total = df_trades["pnl_abs"].sum()
    pnl_medio = df_trades["pnl_abs"].mean()

    print(f"\nTotal trades: {total}")
    print(f"Ganadores: {len(ganadores)} ({win_rate:.1f}%)")
    print(f"Perdedores: {len(perdedores)} ({len(perdedores)/total*100:.1f}%)")
    print(f"Neutros: {len(neutros)} ({len(neutros)/total*100:.1f}%)")
    print(f"PnL total: {pnl_total:,.2f} USD")
    print(f"PnL medio por trade: {pnl_medio:,.2f} USD")

    if len(ganadores) > 0:
        print(f"Ganancia media: {ganadores['pnl_abs'].mean():,.2f} USD")
    if len(perdedores) > 0:
        print(f"Pérdida media: {perdedores['pnl_abs'].mean():,.2f} USD")

    # Profit factor
    gross_profit = ganadores["pnl_abs"].sum() if len(ganadores) > 0 else 0
    gross_loss = abs(perdedores["pnl_abs"].sum()) if len(perdedores) > 0 else 0
    if gross_loss > 0:
        print(f"Profit factor: {gross_profit / gross_loss:.2f}")

    # --- Por motivo de cierre ---
    print(f"\nPor motivo de cierre:")
    for reason, group in df_trades.groupby("close_reason"):
        n = len(group)
        pnl = group["pnl_abs"].sum()
        print(f"  {reason:20s}: {n:>5} trades | PnL: {pnl:>12,.2f} USD")

    # --- Por regla ---
    print(f"\nPor regla:")
    for rule, group in df_trades.groupby("rule"):
        n = len(group)
        wr = (group["pnl_abs"] > 0).mean() * 100
        pnl = group["pnl_abs"].sum()
        print(f"  {rule:20s}: {n:>5} trades | "
              f"Win rate: {wr:.1f}% | PnL: {pnl:>12,.2f} USD")

    # --- Por dirección ---
    print(f"\nPor dirección:")
    for d, group in df_trades.groupby("direction"):
        n = len(group)
        wr = (group["pnl_abs"] > 0).mean() * 100
        pnl = group["pnl_abs"].sum()
        print(f"  {d:20s}: {n:>5} trades | "
              f"Win rate: {wr:.1f}% | PnL: {pnl:>12,.2f} USD")

    # --- Duración ---
    dur_media = df_trades["bars_in_trade"].mean()
    dur_mediana = df_trades["bars_in_trade"].median()
    print(f"\nDuración media: {dur_media:.1f} velas ({dur_media:.1f}h)")
    print(f"Duración mediana: {dur_mediana:.0f} velas ({dur_mediana:.0f}h)")

    # --- R-multiple y MAE (Iteración 2) ---
    if "r_multiple" in df_trades.columns:
        print(f"\n--- R-multiple y MAE ---")
        print(f"R-multiple medio: {df_trades['r_multiple'].mean():+.3f}")
        print(f"R-multiple mediano: {df_trades['r_multiple'].median():+.3f}")
        print(f"R > 0.5 (ganadores claros): {(df_trades['r_multiple'] > 0.5).sum()}")
        print(f"R entre -0.3 y 0.3 (neutros): "
              f"{((df_trades['r_multiple'] >= -0.3) & (df_trades['r_multiple'] <= 0.3)).sum()}")
        print(f"R < -0.5 (perdedores claros): {(df_trades['r_multiple'] < -0.5).sum()}")
        print(f"MAE medio: {df_trades['mae_r'].mean():.3f} R")
        print(f"MAE mediano: {df_trades['mae_r'].median():.3f} R")
        print(f"MAE peor (trade más castigado): {df_trades['mae_r'].min():.3f} R")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("Ejecutando backtest...\n")

    # 1. Cargar datos
    df_signals, df_1h = cargar_datos()

    # 2. Ejecutar backtest
    df_trades = ejecutar_backtest(df_signals, df_1h)

    if df_trades.empty:
        print("No se generaron trades.")
        return

    # 3. Mostrar resumen
    mostrar_resumen(df_trades)

    # 4. Guardar CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_trades.to_csv(TRADES_FILE, index=False)
    print(f"\nTrades guardados en: {TRADES_FILE}")


if __name__ == "__main__":
    main()
