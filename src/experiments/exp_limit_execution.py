"""
==========================================================
EXPERIMENTO A6.1 - Simulador honesto de entradas LIMIT
==========================================================
Pregunta: ¿el sistema 1h vuelve a ser rentable si el usuario
entra con orden limit (comisión maker, sin slippage) en vez
de a mercado?

El escenario S1 del 3-jul asumía que la limit SIEMPRE se
ejecuta (optimista). Aquí lo simulamos de verdad:

  1. La señal se genera al cierre de la vela (como siempre).
  2. Se coloca una orden limit a un precio P.
  3. Si en las siguientes W velas el precio toca P, la orden
     se ejecuta (entrada maker 0.02%, sin slippage).
  4. Si no lo toca, la orden se cancela: TRADE PERDIDO.
  5. Desde el precio de ejecución F se recalculan SL (1.25xATR
     de la señal) y TP (2R). Salida por TP1 = limit (maker
     0.02%); salida por SL o time-exit = mercado (taker 0.05%
     + slippage 0.02%).

Reglas de honestidad (peor caso):
  - Si la vela de ejecución también toca el SL, asumimos SL.
  - Fill: si el open de la vela ya es mejor que P, se ejecuta
    al open; si no, hace falta que el low/high SUPERE P
    (estrictamente) para asumir ejecución (tocar justo P no
    garantiza fill en la cola del libro).
  - Time exit: 24 velas desde la EJECUCIÓN.

Grid de escenarios (decisión 4-jul: probar opciones
y quedarse con la más robusta, documentando):
  - Colocación P: close | ema20 | close -/+ 0.15*ATR
  - Ventana W: 1, 2, 3 velas
  - Breakout: 'market' (entra a mercado como ahora, peaje
    taker) o 'limit' (misma limit que pullback)

Salida: tabla por escenario (fill rate, PnL neto test,
R medio, y con filtro ML 0.45/0.50) + CSV de resultados.
==========================================================
"""

import os
import sys
import pandas as pd
import numpy as np

OUTPUT_DIR = "outputs"
EXP_DIR = os.path.join(OUTPUT_DIR, "experiments")
os.makedirs(EXP_DIR, exist_ok=True)

SIGNALS_FILE = os.path.join(OUTPUT_DIR, "signals.csv")
RESULTS_V3 = os.path.join(OUTPUT_DIR, "model_results_v3.csv")
PROCESSED_1H = "data/processed/BTCUSDT_1h_processed.csv"

SL_ATR_MULT = 1.25
TP_R_MULT = 2.0
MAX_BARS = 24

MAKER = 0.0002          # 0.02% limit
TAKER_SLIP = 0.0007     # 0.05% taker + 0.02% slippage

TEST_CUTOFF = pd.Timestamp("2024-12-31", tz="Europe/Madrid")


def cargar():
    sig = pd.read_csv(SIGNALS_FILE)
    sig["open_time"] = pd.to_datetime(sig["open_time"], format="ISO8601", utc=True).dt.tz_convert("Europe/Madrid")
    df = pd.read_csv(PROCESSED_1H)
    df["open_time"] = pd.to_datetime(df["open_time"], format="ISO8601", utc=True).dt.tz_convert("Europe/Madrid")
    df = df.sort_values("open_time").reset_index(drop=True)
    r = pd.read_csv(RESULTS_V3)
    r["open_time"] = pd.to_datetime(r["open_time"], format="ISO8601", utc=True).dt.tz_convert("Europe/Madrid")
    probas = r.set_index(["open_time", "direction", "rule"])["model_proba"]
    return sig, df, probas


def precio_limit(row, modo):
    """Precio de la orden limit según el modo de colocación."""
    if modo == "close":
        return row["close"]
    if modo == "ema20":
        return row["ema_20"]
    if modo == "atr015":
        off = 0.15 * row["atr_14"]
        return row["close"] - off if row["direction"] == "LONG" else row["close"] + off
    raise ValueError(modo)


def simular_trade_limit(sig_row, velas, P, W, es_market):
    """
    Simula un trade. Devuelve dict con resultado o None si la
    limit no se ejecutó. 'velas' = array de velas posteriores
    (open, high, low, close), suficiente para W + MAX_BARS.
    """
    d = 1 if sig_row["direction"] == "LONG" else -1
    atr = sig_row["atr_14"]
    riesgo = SL_ATR_MULT * atr

    # ---- 1. Ejecución de la entrada ----
    if es_market:
        F, k_fill = sig_row["close"], -1  # entra "al cierre de señal" (como backtest base)
        fric_in = F * TAKER_SLIP
    else:
        F, k_fill, fric_in = None, None, None
        for k in range(min(W, len(velas))):
            o, h, l, c = velas[k]
            if d == 1:
                if o <= P:
                    F = o
                elif l < P:
                    F = P
            else:
                if o >= P:
                    F = o
                elif h > P:
                    F = P
            if F is not None:
                k_fill = k
                fric_in = F * MAKER
                break
        if F is None:
            return None  # orden cancelada, trade perdido

    sl = F - d * riesgo
    tp = F + d * TP_R_MULT * riesgo

    # ---- 2. Vida del trade desde la vela de ejecución ----
    # En la propia vela de fill (si es limit) también puede tocar SL/TP:
    # peor caso -> SL primero.
    start = k_fill if not es_market else 0
    fin = min(start + MAX_BARS + (0 if es_market else 0), len(velas))
    exit_price, reason = None, None
    n_bars = 0
    for k in range(start, min(start + MAX_BARS, len(velas))):
        o, h, l, c = velas[k]
        n_bars += 1
        sl_hit = (l <= sl) if d == 1 else (h >= sl)
        tp_hit = (h >= tp) if d == 1 else (l <= tp)
        if sl_hit:                    # peor caso: SL manda
            exit_price, reason = sl, "stop_loss"
            break
        if tp_hit:
            exit_price, reason = tp, "take_profit_1"
            break
    if exit_price is None:
        if n_bars == 0:
            return None
        exit_price, reason = velas[min(start + MAX_BARS, len(velas)) - 1][3], "time_exit"

    # ---- 3. PnL neto ----
    pnl_bruto = d * (exit_price - F)
    fric_out = exit_price * (MAKER if reason == "take_profit_1" else TAKER_SLIP)
    pnl_neto = pnl_bruto - fric_in - fric_out
    return {
        "pnl_neto": pnl_neto,
        "r_neto": pnl_neto / riesgo if riesgo > 0 else 0.0,
        "reason": reason,
    }


def main():
    sig, df, probas = cargar()
    idx_por_tiempo = pd.Series(df.index, index=df["open_time"])
    ohlc = df[["open", "high", "low", "close"]].to_numpy()

    escenarios = []
    for modo in ["close", "ema20", "atr015"]:
        for W in [1, 2, 3]:
            for bo in ["market", "limit"]:
                escenarios.append((modo, W, bo))

    filas = []
    for modo, W, bo in escenarios:
        res, perdidos_r = [], []
        for _, s in sig.iterrows():
            t = s["open_time"]
            if t not in idx_por_tiempo.index:
                continue
            i = idx_por_tiempo[t]
            velas = ohlc[i + 1: i + 1 + W + MAX_BARS]
            if len(velas) == 0:
                continue
            es_market = (s["rule"] == "breakout" and bo == "market")
            P = precio_limit(s, modo)
            r_ = simular_trade_limit(s, velas, P, W, es_market)
            key = (t, s["direction"], s["rule"])
            proba = probas.get(key, np.nan)
            if r_ is None:
                perdidos_r.append({"open_time": t, "proba": proba})
                continue
            r_["open_time"] = t
            r_["proba"] = proba
            r_["rule"] = s["rule"]
            res.append(r_)

        R = pd.DataFrame(res)
        R_test = R[R["open_time"] > TEST_CUTOFF]
        n_sig_test = len(sig[sig["open_time"] > TEST_CUTOFF])
        fila = {
            "colocacion": modo, "ventana": W, "breakout": bo,
            "fill_rate_test": len(R_test) / n_sig_test * 100,
        }
        for u, tag in [(0.0, "sin"), (0.45, "u45"), (0.50, "u50")]:
            F_ = R_test if u == 0 else R_test[R_test["proba"] >= u]
            fila[f"n_{tag}"] = len(F_)
            fila[f"pnl_{tag}"] = F_["pnl_neto"].sum()
            fila[f"r_{tag}"] = F_["r_neto"].mean() if len(F_) else np.nan
            fila[f"wr_{tag}"] = (F_["pnl_neto"] > 0).mean() * 100 if len(F_) else np.nan
        filas.append(fila)
        print(f"{modo:>7} W={W} bo={bo:>6} | fill {fila['fill_rate_test']:5.1f}% | "
              f"sin: {fila['pnl_sin']:>+10,.0f} ({fila['r_sin']:+.3f}R) | "
              f"0.45: n={fila['n_u45']:>3} {fila['pnl_u45']:>+9,.0f} ({fila['r_u45']:+.3f}R) | "
              f"0.50: n={fila['n_u50']:>3} {fila['pnl_u50']:>+9,.0f} ({fila['r_u50']:+.3f}R)")

    out = pd.DataFrame(filas)
    out.to_csv(os.path.join(EXP_DIR, "limit_execution_grid.csv"), index=False)
    print(f"\nGuardado: {EXP_DIR}/limit_execution_grid.csv")


if __name__ == "__main__":
    main()
