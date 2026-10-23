"""
==========================================================
EXPERIMENTO A6.2 - Reentrenamiento end-to-end con ejecución LIMIT
==========================================================
El ganador del grid A6.1 fue: limit al CLOSE de la señal,
ventana 1 vela, breakout también limit (fill ~98%, robusto
en W=1/2/3).

Aquí validamos en serio esa configuración:
  1. Recalculamos el resultado de TODOS los trades (5 años)
     con esa ejecución (mismo simulador honesto que A6.1).
  2. Etiquetas nuevas: ganador = R neto >= 0.5.
  3. Reentrenamos el mismo GB (mismos hiperparámetros) y
     sacamos la tabla de umbrales del test 2025-26.
  4. Walk-forward: 4 folds expansivos (test 2023 / 2024 /
     2025 / 2026), reentrenando en cada fold. La métrica que
     importa: PnL neto y R medio out-of-sample agregado.

Las FEATURES no cambian (describen el momento de la señal);
solo cambian las etiquetas y el PnL (describen el resultado
con la nueva ejecución).
==========================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "exp1", os.path.join(os.path.dirname(__file__), "exp_limit_execution.py"))
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)

OUTPUT_DIR = "outputs"
EXP_DIR = os.path.join(OUTPUT_DIR, "experiments")
DATASET_FILE = os.path.join(OUTPUT_DIR, "ml_dataset.csv")

MODO, W, BO = "close", 1, "limit"   # ganador del grid A6.1

FEATURES = [
    "rsi_14", "atr_14", "dist_ema20_pct", "dist_ema50_pct",
    "ema_spread_pct", "ema_spread_4h_pct", "atr_pct", "price_position",
    "body_pct", "upper_wick_pct", "lower_wick_pct", "hour", "day_of_week",
    "volume_sma20_ratio", "rsi_14_4h", "atr_pct_4h",
    "roc_5", "roc_20", "atr_regime", "candle_range_ratio",
]

UMBRALES = [0.40, 0.45, 0.50, 0.55, 0.60]


def recalcular_trades():
    """Resultado de todos los trades (5 años) con ejecución limit."""
    sig, df, _ = exp1.cargar()
    idx_por_tiempo = pd.Series(df.index, index=df["open_time"])
    ohlc = df[["open", "high", "low", "close"]].to_numpy()

    filas = []
    for _, s in sig.iterrows():
        t = s["open_time"]
        if t not in idx_por_tiempo.index:
            continue
        i = idx_por_tiempo[t]
        velas = ohlc[i + 1: i + 1 + W + exp1.MAX_BARS]
        if len(velas) == 0:
            continue
        P = exp1.precio_limit(s, MODO)
        r_ = exp1.simular_trade_limit(s, velas, P, W, es_market=False)
        filas.append({
            "open_time": t, "direction": s["direction"], "rule": s["rule"],
            "ejecutado": r_ is not None,
            "pnl_neto": r_["pnl_neto"] if r_ else np.nan,
            "r_neto": r_["r_neto"] if r_ else np.nan,
        })
    return pd.DataFrame(filas)


def tabla(df, proba_col, titulo):
    print(f"\n{titulo}")
    print(f"{'Umbral':>7} | {'Trades':>6} | {'WR':>6} | {'PnL neto':>11} | {'R medio':>8}")
    res = {}
    for u in [0.0] + UMBRALES:
        f = df if u == 0 else df[df[proba_col] >= u]
        if len(f) == 0:
            continue
        wr = (f["pnl_neto"] > 0).mean() * 100
        print(f"{u:>7.2f} | {len(f):>6} | {wr:>5.1f}% | {f['pnl_neto'].sum():>+10,.0f} | {f['r_neto'].mean():>+8.3f}")
        res[u] = (len(f), wr, f["pnl_neto"].sum(), f["r_neto"].mean())
    return res


def main():
    print("Recalculando 5 años de trades con ejecución limit "
          f"({MODO}, W={W}, breakout={BO})...")
    tr = recalcular_trades()
    n_no = (~tr["ejecutado"]).sum()
    print(f"Trades: {len(tr)} | no ejecutados (limit cancelada): {n_no} "
          f"({n_no/len(tr)*100:.1f}%)")

    # Unir con las features del dataset actual
    ds = pd.read_csv(DATASET_FILE)
    ds["open_time"] = pd.to_datetime(ds["open_time"], format="ISO8601", utc=True).dt.tz_convert("Europe/Madrid")
    m = ds.merge(tr, on=["open_time", "direction", "rule"], how="inner")
    m = m[m["ejecutado"]].copy()          # solo trades que existieron
    m["label_limit"] = (m["r_neto"] >= 0.5).astype(int)
    print(f"Dataset con ejecución limit: {len(m)} trades | "
          f"ganadores {m['label_limit'].mean()*100:.1f}%")

    # --- Split fijo (comparable con la tabla A3/A5) ---
    cutoff = pd.Timestamp("2024-12-31", tz="Europe/Madrid")
    tr_, te_ = m[m["open_time"] <= cutoff], m[m["open_time"] > cutoff]
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=20, subsample=0.8, random_state=42)
    gb.fit(tr_[FEATURES], tr_["label_limit"])
    te_ = te_.copy()
    te_["proba"] = gb.predict_proba(te_[FEATURES])[:, 1]
    tabla(te_, "proba", "TEST 2025-26 — ejecución limit, modelo reentrenado (PnL NETO):")

    # --- Walk-forward: 4 folds expansivos ---
    folds = ["2023-01-01", "2024-01-01", "2025-01-01", "2025-10-01"]
    oos = []
    for j, ini in enumerate(folds):
        fin = folds[j + 1] if j + 1 < len(folds) else "2027-01-01"
        ini_ts = pd.Timestamp(ini, tz="Europe/Madrid")
        fin_ts = pd.Timestamp(fin, tz="Europe/Madrid")
        tr_f = m[m["open_time"] < ini_ts]
        te_f = m[(m["open_time"] >= ini_ts) & (m["open_time"] < fin_ts)].copy()
        if len(te_f) == 0 or len(tr_f) < 500:
            continue
        gb_f = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42)
        gb_f.fit(tr_f[FEATURES], tr_f["label_limit"])
        te_f["proba"] = gb_f.predict_proba(te_f[FEATURES])[:, 1]
        oos.append(te_f)
        w45 = te_f[te_f["proba"] >= 0.45]
        w50 = te_f[te_f["proba"] >= 0.50]
        print(f"  Fold {ini[:7]}→{fin[:7]}: train {len(tr_f):>4} | "
              f"0.45: n={len(w45):>3} {w45['pnl_neto'].sum():>+9,.0f} ({w45['r_neto'].mean():+.3f}R) | "
              f"0.50: n={len(w50):>3} {w50['pnl_neto'].sum():>+9,.0f} "
              f"({w50['r_neto'].mean():+.3f}R)" if len(w50) else "")
    O = pd.concat(oos)
    tabla(O, "proba", f"WALK-FORWARD agregado ({len(O)} predicciones out-of-sample, 2023-2026):")

    m.to_csv(os.path.join(EXP_DIR, "dataset_limit_execution.csv"), index=False)
    print(f"\nGuardado: {EXP_DIR}/dataset_limit_execution.csv")


if __name__ == "__main__":
    main()
