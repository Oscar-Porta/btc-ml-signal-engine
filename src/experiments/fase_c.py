"""
==========================================================
FASE C - Arnés de experimentos sobre 1h + ejecución limit
==========================================================
Valida cada lote de mejoras con el MISMO walk-forward
(4 folds expansivos) y PnL NETO con ejecución limit.

Criterio de éxito (fijado antes de experimentar, ver
docs/propuesta_fase_c.md): mejorar R medio en 0.45-0.50 sin
romper 0.60, sin volver claramente negativo ningún fold.

Lotes:
  BASE   : las 20 features actuales (referencia)
  L1     : C3+C4 — features relativas al trade ("firmadas")
           + fuera atr_14 absoluto
  L2     : L1 + C1 — régimen de mercado (ER, BB width,
           pendiente EMA50, ADX 4h, velas desde cruce 4h)
  L3     : L2 + C2a — n_trades relativo (flujo)

Anti-lookahead: las features 4h nuevas se alinean por
close_time <= momento de decisión (cierre vela 1h), igual
que el fix de la Fase A. Las rolling de 1h usan solo la
vela de señal y anteriores.
==========================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

EXP_DIR = "outputs/experiments"
DATASET = os.path.join(EXP_DIR, "dataset_limit_execution.csv")
PROCESSED_1H = "data/processed/BTCUSDT_1h_processed.csv"
RAW_4H = "data/raw/BTCUSDT_4h.csv"

BASE = ["rsi_14", "atr_14", "dist_ema20_pct", "dist_ema50_pct", "ema_spread_pct",
        "ema_spread_4h_pct", "atr_pct", "price_position", "body_pct", "upper_wick_pct",
        "lower_wick_pct", "hour", "day_of_week", "volume_sma20_ratio", "rsi_14_4h",
        "atr_pct_4h", "roc_5", "roc_20", "atr_regime", "candle_range_ratio"]

# L1: versión relativa al trade (long y short comparten patrón)
L1 = ["rsi_rel", "dist_ema20_rel", "dist_ema50_rel", "ema_spread_rel",
      "ema_spread_4h_rel", "atr_pct", "price_pos_rel", "body_rel", "wick_favor",
      "wick_against", "hour", "day_of_week", "volume_sma20_ratio", "rsi_4h_rel",
      "atr_pct_4h", "roc_5_rel", "roc_20_rel", "atr_regime", "candle_range_ratio"]

REGIMEN = ["er_20", "er_50", "bb_width_20", "ema50_slope_10",
           "adx_14_4h", "bars_since_cross_4h"]
L2 = L1 + REGIMEN
L3 = L2 + ["n_trades_ratio"]

FOLDS = ["2023-01-01", "2024-01-01", "2025-01-01", "2025-10-01"]
UMBRALES = [0.45, 0.50, 0.55, 0.60]


def ema(s, n): return s.ewm(span=n, adjust=False, min_periods=n).mean()


def adx_14(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    cp = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-cp).abs(), (df["low"]-cp).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def cargar():
    m = pd.read_csv(DATASET)
    m["open_time"] = pd.to_datetime(m["open_time"], utc=True)
    m["label"] = (m["r_neto"] >= 0.5).astype(int)
    d = m["direction"].map({"LONG": 1, "SHORT": -1})

    # ---------- L1: features relativas al trade ----------
    m["rsi_rel"] = np.where(d == 1, m["rsi_14"], 100 - m["rsi_14"])
    m["rsi_4h_rel"] = np.where(d == 1, m["rsi_14_4h"], 100 - m["rsi_14_4h"])
    for c in ["dist_ema20_pct", "dist_ema50_pct", "ema_spread_pct",
              "ema_spread_4h_pct", "body_pct", "roc_5", "roc_20"]:
        m[c.replace("_pct", "").replace("roc", "roc") + "_rel"] = m[c] * d
    m = m.rename(columns={"dist_ema20_rel": "dist_ema20_rel"})
    m["price_pos_rel"] = np.where(d == 1, m["price_position"], 1 - m["price_position"])
    m["wick_favor"] = np.where(d == 1, m["lower_wick_pct"], m["upper_wick_pct"])
    m["wick_against"] = np.where(d == 1, m["upper_wick_pct"], m["lower_wick_pct"])

    # ---------- C1: régimen (1h, solo pasado) ----------
    h = pd.read_csv(PROCESSED_1H)
    h["open_time"] = pd.to_datetime(h["open_time"], format="ISO8601", utc=True)
    h = h.sort_values("open_time").reset_index(drop=True)
    for w in [20, 50]:
        neto = (h["close"] - h["close"].shift(w)).abs()
        ruido = h["close"].diff().abs().rolling(w).sum()
        h[f"er_{w}"] = np.where(ruido > 0, neto / ruido, 0.0)
    sma20, std20 = h["close"].rolling(20).mean(), h["close"].rolling(20).std()
    h["bb_width_20"] = np.where(sma20 > 0, 4 * std20 / sma20 * 100, 0.0)
    h["ema50_slope_10"] = (h["ema_50"] / h["ema_50"].shift(10) - 1) * 100
    if "number_of_trades" in h.columns:
        ntr_sma = h["number_of_trades"].rolling(20).mean()
        h["n_trades_ratio"] = np.where(ntr_sma > 0, h["number_of_trades"] / ntr_sma, 1.0)
    else:
        h["n_trades_ratio"] = 1.0
    cols1h = ["open_time", "er_20", "er_50", "bb_width_20", "ema50_slope_10", "n_trades_ratio"]
    m = m.merge(h[cols1h], on="open_time", how="left")

    # ---------- C1: régimen 4h (alineado sin lookahead) ----------
    q = pd.read_csv(RAW_4H)
    for c in ["open_time", "close_time"]:
        q[c] = pd.to_datetime(q[c], format="ISO8601", utc=True)
    q = q.sort_values("open_time").reset_index(drop=True)
    q["ema_20"], q["ema_50"] = ema(q["close"], 20), ema(q["close"], 50)
    q["adx_14_4h"] = adx_14(q)
    sign = (q["ema_20"] > q["ema_50"]).astype(int)
    grupo = (sign != sign.shift()).cumsum()
    q["bars_since_cross_4h"] = q.groupby(grupo).cumcount().clip(upper=180)
    ctx = q[["close_time", "adx_14_4h", "bars_since_cross_4h"]].rename(
        columns={"close_time": "ct4"}).sort_values("ct4")
    m["decision_time"] = m["open_time"] + pd.Timedelta(hours=1)
    m = pd.merge_asof(m.sort_values("decision_time"), ctx,
                      left_on="decision_time", right_on="ct4", direction="backward")
    m = m.drop(columns=["decision_time", "ct4"]).sort_values("open_time").reset_index(drop=True)
    return m


def walk_forward(m, features, model_factory=None, etiqueta="", pesos=False, verbose=True):
    if model_factory is None:
        model_factory = lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42)
    datos = m.dropna(subset=features).copy()
    oos = []
    for j, ini in enumerate(FOLDS):
        fin = FOLDS[j + 1] if j + 1 < len(FOLDS) else "2027-01-01"
        i_, f_ = pd.Timestamp(ini, tz="UTC"), pd.Timestamp(fin, tz="UTC")
        tr = datos[datos["open_time"] < i_]
        te = datos[(datos["open_time"] >= i_) & (datos["open_time"] < f_)].copy()
        if len(te) == 0 or len(tr) < 500:
            continue
        g = model_factory()
        kw = {"sample_weight": tr["r_neto"].abs().clip(0.1, 2.0)} if pesos else {}
        g.fit(tr[features], tr["label"], **kw)
        te["proba"] = g.predict_proba(te[features])[:, 1]
        te["fold"] = ini[:7]
        oos.append(te)
    O = pd.concat(oos)
    if verbose:
        print(f"\n=== {etiqueta} ({len(O)} preds OOS) ===")
        print(f"{'Umbral':>7} | {'Trades':>6} | {'WR':>6} | {'PnL neto':>10} | {'R medio':>8} | folds R")
        for u in UMBRALES:
            x = O[O["proba"] >= u]
            if len(x) == 0:
                continue
            porfold = " ".join(f"{r:+.2f}" for r in x.groupby("fold")["r_neto"].mean())
            print(f"{u:>7.2f} | {len(x):>6} | {(x['pnl_neto']>0).mean()*100:>5.1f}% | "
                  f"{x['pnl_neto'].sum():>+9,.0f} | {x['r_neto'].mean():>+8.3f} | {porfold}")
    return O


def main():
    m = cargar()
    que = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if que in ("todo", "base"):
        walk_forward(m, BASE, etiqueta="BASE (20 features actuales)")
    if que in ("todo", "l1"):
        walk_forward(m, L1, etiqueta="L1 = C3 firmadas + C4 sin atr_14")
    if que in ("todo", "l2"):
        walk_forward(m, L2, etiqueta="L2 = L1 + C1 régimen")
    if que in ("todo", "l3"):
        walk_forward(m, L3, etiqueta="L3 = L2 + C2a n_trades")

    # ---- Lote 4: C7 pesos + C5 bake-off + ensemble de semillas ----
    # (xgboost/lightgbm no instalables en este entorno: HistGB de
    #  sklearn hace de proxy de LightGBM — histogramas + early stop)
    if que == "pesos":
        walk_forward(m, BASE, etiqueta="C7: BASE + sample_weight=|R|", pesos=True)

    if que == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        hgb = lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=42)
        walk_forward(m, BASE, model_factory=hgb,
                     etiqueta="C5: HistGB (proxy LightGBM, early stopping)")

    if que == "hgbbal":
        from sklearn.ensemble import HistGradientBoostingClassifier
        hgb_bal = lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            class_weight="balanced", random_state=42)
        walk_forward(m, BASE, model_factory=hgb_bal,
                     etiqueta="C5: HistGB + class_weight=balanced")

    if que == "ens":
        # Ensemble de 5 semillas del GB base (promedio de probas)
        datos = m.dropna(subset=BASE).copy()
        oos = []
        for j, ini in enumerate(FOLDS):
            fin = FOLDS[j + 1] if j + 1 < len(FOLDS) else "2027-01-01"
            i_, f_ = pd.Timestamp(ini, tz="UTC"), pd.Timestamp(fin, tz="UTC")
            tr = datos[datos["open_time"] < i_]
            te = datos[(datos["open_time"] >= i_) & (datos["open_time"] < f_)].copy()
            if len(te) == 0 or len(tr) < 500:
                continue
            probs = []
            for seed in [1, 2, 3, 4, 5]:
                g = GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    min_samples_leaf=20, subsample=0.8, random_state=seed)
                g.fit(tr[BASE], tr["label"])
                probs.append(g.predict_proba(te[BASE])[:, 1])
            te["proba"] = np.mean(probs, axis=0)
            te["fold"] = ini[:7]
            oos.append(te)
        O = pd.concat(oos)
        print(f"\n=== C5: ensemble 5 semillas GB ({len(O)} preds OOS) ===")
        for u in UMBRALES:
            x = O[O["proba"] >= u]
            if len(x) == 0:
                continue
            porfold = " ".join(f"{r:+.2f}" for r in x.groupby("fold")["r_neto"].mean())
            print(f"{u:>7.2f} | {len(x):>6} | {(x['pnl_neto']>0).mean()*100:>5.1f}% | "
                  f"{x['pnl_neto'].sum():>+9,.0f} | {x['r_neto'].mean():>+8.3f} | {porfold}")
        O.to_csv(os.path.join(EXP_DIR, "walkforward_ensemble5.csv"), index=False)


if __name__ == "__main__":
    main()
