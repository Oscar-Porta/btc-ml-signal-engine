"""
==========================================================
FASE D (adelantada) - Pooling multi-activo BTC + ETH
==========================================================
Hipótesis: el cuello de botella del modelo es el volumen de
datos (4.4k trades). Al añadir ETH (mismo pipeline, features
porcentuales comparables) el dataset se duplica: si los
patrones son compartidos entre monedas, el modelo pooled
debería predecir MEJOR incluso los trades de BTC.

SOL: pendiente — la API de Binance no es accesible desde
este entorno de trabajo. `01_download_data.py` queda
parametrizado para descargarlo en local.

Método (idéntico al arnés de Fase C para comparabilidad):
  - Pipeline completo por símbolo: indicadores 1h, velas 4h
    (ETH por agregación exacta de 1h, validada contra BTC),
    contexto 4h alineado por close_time (sin lookahead),
    reglas pullback/breakout, ejecución LIMIT (close, W=1),
    PnL neto, 20 features BASE.
  - Modelos comparados con el MISMO walk-forward (4 folds):
      A) BTC-only (la referencia BASE de Fase C)
      B) Pooled BTC+ETH sin feature de símbolo
      C) Pooled BTC+ETH con feature de símbolo (sym_eth)
  - La comparación clave: rendimiento OOS SOBRE LOS TRADES
    DE BTC de cada modelo. Si B o C > A, el pooling funciona.
==========================================================
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

EXP_DIR = "outputs/experiments"
os.makedirs(EXP_DIR, exist_ok=True)
TZ = "Europe/Madrid"

SL_ATR_MULT, TP_R_MULT, MAX_BARS = 1.25, 2.0, 24
MAKER, TAKER_SLIP = 0.0002, 0.0007

BASE = ["rsi_14", "atr_14", "dist_ema20_pct", "dist_ema50_pct", "ema_spread_pct",
        "ema_spread_4h_pct", "atr_pct", "price_position", "body_pct", "upper_wick_pct",
        "lower_wick_pct", "hour", "day_of_week", "volume_sma20_ratio", "rsi_14_4h",
        "atr_pct_4h", "roc_5", "roc_20", "atr_regime", "candle_range_ratio"]

FOLDS = ["2023-01-01", "2024-01-01", "2025-01-01", "2025-10-01"]
UMBRALES = [0.45, 0.50, 0.55, 0.60]


def ema(s, n): return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    p = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + g / p.replace(0, np.nan))

def atr(df, n=14):
    cp = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-cp).abs(),
                    (df["low"]-cp).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def indicadores(df):
    df = df.copy()
    df["ema_20"], df["ema_50"] = ema(df["close"], 20), ema(df["close"], 50)
    df["rsi_14"], df["atr_14"] = rsi(df["close"]), atr(df)
    df["swing_high_20"] = df["high"].rolling(20).max()
    df["swing_low_20"] = df["low"].rolling(20).min()
    df["trend"] = np.where(df["ema_20"] > df["ema_50"], "up", "down")
    df.loc[df["ema_50"].isna(), "trend"] = np.nan
    return df


def cargar_1h(symbol):
    df = pd.read_csv(f"data/raw/{symbol}_1h.csv")
    df["open_time"] = pd.to_datetime(df["open_time"], format="ISO8601", utc=True).dt.tz_convert(TZ)
    return df.sort_values("open_time").reset_index(drop=True)


def velas_4h(symbol, df_1h):
    """BTC: nativas. Otros: agregación exacta de 1h (validada)."""
    if symbol == "BTCUSDT":
        q = pd.read_csv("data/raw/BTCUSDT_4h.csv")
        for c in ["open_time", "close_time"]:
            q[c] = pd.to_datetime(q[c], format="ISO8601", utc=True).dt.tz_convert(TZ)
        return q.sort_values("open_time").reset_index(drop=True)
    h = df_1h.copy()
    h["g"] = h["open_time"].dt.tz_convert("UTC").dt.floor("4h")
    q = h.groupby("g").agg(open=("open", "first"), high=("high", "max"),
                           low=("low", "min"), close=("close", "last"),
                           volume=("volume", "sum"), n=("close", "size")).reset_index()
    q = q[q["n"] == 4].drop(columns="n")
    q["open_time"] = q["g"].dt.tz_convert(TZ)
    q["close_time"] = (q["g"] + pd.Timedelta(hours=4)).dt.tz_convert(TZ)
    return q.drop(columns="g").sort_values("open_time").reset_index(drop=True)


def pipeline_simbolo(symbol):
    h = indicadores(cargar_1h(symbol))
    q = indicadores(velas_4h(symbol, h))

    # Contexto 4h sin lookahead (por close_time <= decisión)
    ctx = q[["close_time", "trend", "ema_20", "ema_50", "rsi_14", "atr_14"]].rename(columns={
        "close_time": "ct4", "trend": "trend_4h", "ema_20": "ema_20_4h",
        "ema_50": "ema_50_4h", "rsi_14": "rsi_14_4h", "atr_14": "atr_14_4h"})
    h["decision_time"] = h["open_time"] + pd.Timedelta(hours=1)
    h = pd.merge_asof(h.sort_values("decision_time"), ctx.sort_values("ct4"),
                      left_on="decision_time", right_on="ct4", direction="backward")
    h = h.rename(columns={"trend": "trend_1h"}).sort_values("open_time").reset_index(drop=True)

    # Reglas (idénticas a 03_rules.py)
    psh, psl = h["swing_high_20"].shift(1), h["swing_low_20"].shift(1)
    masks = [
        ((h["trend_4h"] == "up") & (h["trend_1h"] == "up") & (h["low"] <= h["ema_20"]) &
         (h["close"] > h["ema_20"]) & (h["rsi_14"] < 55), "LONG", "pullback"),
        ((h["trend_4h"] == "down") & (h["trend_1h"] == "down") & (h["high"] >= h["ema_20"]) &
         (h["close"] < h["ema_20"]) & (h["rsi_14"] > 45), "SHORT", "pullback"),
        ((h["trend_4h"] == "up") & (h["close"] > psh) & h["rsi_14"].between(40, 75), "LONG", "breakout"),
        ((h["trend_4h"] == "down") & (h["close"] < psl) & h["rsi_14"].between(25, 60), "SHORT", "breakout"),
    ]
    sig = []
    for m_, d_, r_ in masks:
        s = h[m_ & h["atr_14"].notna()].copy()
        s["direction"], s["rule"] = d_, r_
        sig.append(s)
    sig = pd.concat(sig).sort_values("open_time").reset_index(drop=True)

    # Rolling para features
    h["volume_sma20"] = h["volume"].rolling(20).mean()
    h["close_5_ago"], h["close_20_ago"] = h["close"].shift(5), h["close"].shift(20)
    h["atr_sma20"] = h["atr_14"].rolling(20).mean()
    sig = sig.merge(h[["open_time", "volume_sma20", "close_5_ago", "close_20_ago", "atr_sma20"]],
                    on="open_time", how="left")

    # Ejecución LIMIT (close, W=1, todo limit) + PnL neto
    idx = pd.Series(h.index, index=h["open_time"])
    ohlc = h[["open", "high", "low", "close"]].to_numpy()
    res = []
    for _, s in sig.iterrows():
        i = idx.get(s["open_time"])
        if i is None:
            res.append(np.nan); continue
        velas = ohlc[i + 1: i + 2 + MAX_BARS]
        if len(velas) == 0:
            res.append(np.nan); continue
        d = 1 if s["direction"] == "LONG" else -1
        riesgo = SL_ATR_MULT * s["atr_14"]
        o, hi, lo, c = velas[0]
        F = o if (d == 1 and o <= s["close"]) or (d == -1 and o >= s["close"]) else \
            (s["close"] if (d == 1 and lo < s["close"]) or (d == -1 and hi > s["close"]) else None)
        if F is None:
            res.append(np.nan); continue
        sl, tp = F - d * riesgo, F + d * TP_R_MULT * riesgo
        exit_price, reason = None, None
        for k in range(0, min(MAX_BARS, len(velas))):
            o, hi, lo, c = velas[k]
            if (lo <= sl) if d == 1 else (hi >= sl):
                exit_price, reason = sl, "stop_loss"; break
            if (hi >= tp) if d == 1 else (lo <= tp):
                exit_price, reason = tp, "take_profit_1"; break
        if exit_price is None:
            exit_price, reason = velas[min(MAX_BARS, len(velas)) - 1][3], "time_exit"
        pnl = d * (exit_price - F) - F * MAKER - exit_price * (MAKER if reason == "take_profit_1" else TAKER_SLIP)
        res.append(pnl / riesgo if riesgo > 0 else 0.0)
    sig["r_neto"] = res
    sig = sig[sig["r_neto"].notna()].copy()
    # PnL en USD normalizado: 1R = riesgo% del precio → usamos R como métrica principal
    sig["pnl_neto"] = sig["r_neto"] * SL_ATR_MULT * sig["atr_14"]

    # Features BASE
    f = sig
    f["dist_ema20_pct"] = (f["close"] - f["ema_20"]) / f["ema_20"] * 100
    f["dist_ema50_pct"] = (f["close"] - f["ema_50"]) / f["ema_50"] * 100
    f["ema_spread_pct"] = (f["ema_20"] - f["ema_50"]) / f["ema_50"] * 100
    f["ema_spread_4h_pct"] = (f["ema_20_4h"] - f["ema_50_4h"]) / f["ema_50_4h"] * 100
    f["atr_pct"] = f["atr_14"] / f["close"] * 100
    rng = f["swing_high_20"] - f["swing_low_20"]
    f["price_position"] = np.where(rng > 0, (f["close"] - f["swing_low_20"]) / rng, 0.5)
    f["body_pct"] = (f["close"] - f["open"]) / f["open"] * 100
    bt, bb = f[["open", "close"]].max(axis=1), f[["open", "close"]].min(axis=1)
    fr = f["high"] - f["low"]
    f["upper_wick_pct"] = np.where(fr > 0, (f["high"] - bt) / fr * 100, 0)
    f["lower_wick_pct"] = np.where(fr > 0, (bb - f["low"]) / fr * 100, 0)
    f["hour"] = f["open_time"].dt.hour
    f["day_of_week"] = f["open_time"].dt.dayofweek
    f["volume_sma20_ratio"] = np.where(f["volume_sma20"] > 0, f["volume"] / f["volume_sma20"], 1.0)
    f["atr_pct_4h"] = np.where(f["atr_14_4h"].notna(), f["atr_14_4h"] / f["close"] * 100, f["atr_pct"])
    f["rsi_14_4h"] = f["rsi_14_4h"].fillna(50.0)
    f["roc_5"] = np.where(f["close_5_ago"] > 0, (f["close"] - f["close_5_ago"]) / f["close_5_ago"] * 100, 0.0)
    f["roc_20"] = np.where(f["close_20_ago"] > 0, (f["close"] - f["close_20_ago"]) / f["close_20_ago"] * 100, 0.0)
    f["atr_regime"] = np.where(f["atr_sma20"] > 0, f["atr_14"] / f["atr_sma20"], 1.0)
    f["candle_range_ratio"] = np.where(f["atr_14"] > 0, (f["high"] - f["low"]) / f["atr_14"], 1.0)
    f["symbol"] = symbol
    f["label"] = (f["r_neto"] >= 0.5).astype(int)
    cols = ["open_time", "symbol", "direction", "rule", "r_neto", "pnl_neto", "label"] + BASE
    return f[cols].dropna(subset=BASE).reset_index(drop=True)


def gb():
    return GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                      min_samples_leaf=20, subsample=0.8, random_state=42)


def walk_forward(datos_train, datos_eval, features, etiqueta):
    """Entrena con datos_train (pooled o no) y evalúa OOS sobre datos_eval."""
    oos = []
    for j, ini in enumerate(FOLDS):
        fin = FOLDS[j + 1] if j + 1 < len(FOLDS) else "2027-01-01"
        i_, f_ = pd.Timestamp(ini, tz=TZ), pd.Timestamp(fin, tz=TZ)
        tr = datos_train[datos_train["open_time"] < i_]
        te = datos_eval[(datos_eval["open_time"] >= i_) & (datos_eval["open_time"] < f_)].copy()
        if len(te) == 0 or len(tr) < 500:
            continue
        g = gb()
        g.fit(tr[features], tr["label"])
        te["proba"] = g.predict_proba(te[features])[:, 1]
        te["fold"] = ini[:7]
        oos.append(te)
    O = pd.concat(oos)
    print(f"\n=== {etiqueta} ({len(O)} preds OOS) ===")
    print(f"{'Umbral':>7} | {'Trades':>6} | {'WR':>6} | {'R medio':>8} | folds R")
    for u in UMBRALES:
        x = O[O["proba"] >= u]
        if len(x) == 0:
            continue
        pf = " ".join(f"{r:+.2f}" for r in x.groupby("fold")["r_neto"].mean())
        print(f"{u:>7.2f} | {len(x):>6} | {(x['r_neto']>0).mean()*100:>5.1f}% | "
              f"{x['r_neto'].mean():>+8.3f} | {pf}")
    return O


def main():
    paso = sys.argv[1] if len(sys.argv) > 1 else "datos"
    if paso == "datos":
        for sym in ["BTCUSDT", "ETHUSDT"]:
            d = pipeline_simbolo(sym)
            d.to_csv(os.path.join(EXP_DIR, f"pool_{sym}.csv"), index=False)
            print(f"{sym}: {len(d)} trades | ganadores {d['label'].mean()*100:.1f}% | "
                  f"R medio bruto {d['r_neto'].mean():+.3f}")
        return

    btc = pd.read_csv(os.path.join(EXP_DIR, "pool_BTCUSDT.csv"))
    eth = pd.read_csv(os.path.join(EXP_DIR, "pool_ETHUSDT.csv"))
    for d in (btc, eth):
        d["open_time"] = pd.to_datetime(d["open_time"], utc=True).dt.tz_convert(TZ)
    pool = pd.concat([btc, eth], ignore_index=True).sort_values("open_time")
    pool["sym_eth"] = (pool["symbol"] == "ETHUSDT").astype(int)

    if paso == "a":
        walk_forward(btc, btc, BASE, "A) BTC-only → eval BTC (referencia)")
    if paso.startswith("bfold"):
        # Un fold por ejecución (límite de tiempo del entorno)
        j = int(paso[-1])
        ini = FOLDS[j]
        fin = FOLDS[j + 1] if j + 1 < len(FOLDS) else "2027-01-01"
        i_, f_ = pd.Timestamp(ini, tz=TZ), pd.Timestamp(fin, tz=TZ)
        tr = pool[pool["open_time"] < i_]
        te = pool[(pool["open_time"] >= i_) & (pool["open_time"] < f_)].copy()
        g = gb(); g.fit(tr[BASE], tr["label"])
        te["proba"] = g.predict_proba(te[BASE])[:, 1]
        te["fold"] = ini[:7]
        te.to_csv(os.path.join(EXP_DIR, f"pooled_fold{j}.csv"), index=False)
        print(f"fold {j} ({ini[:7]}): train {len(tr)} → OOS {len(te)} guardado")

    if paso == "bagg":
        O = pd.concat([pd.read_csv(os.path.join(EXP_DIR, f"pooled_fold{j}.csv"))
                       for j in range(4)])
        O["open_time"] = pd.to_datetime(O["open_time"], utc=True)
        O.to_csv(os.path.join(EXP_DIR, "walkforward_pooled.csv"), index=False)
        for nombre, sel in [("TODO", O), ("SOLO BTC", O[O["symbol"] == "BTCUSDT"]),
                            ("SOLO ETH", O[O["symbol"] == "ETHUSDT"])]:
            print(f"\n=== B) Pooled sin símbolo → eval {nombre} ({len(sel)}) ===")
            print(f"{'Umbral':>7} | {'Trades':>6} | {'WR':>6} | {'R medio':>8} | folds R")
            for u in UMBRALES:
                x = sel[sel["proba"] >= u]
                if len(x) == 0:
                    continue
                pf = " ".join(f"{r:+.2f}" for r in x.groupby("fold")["r_neto"].mean())
                print(f"{u:>7.2f} | {len(x):>6} | {(x['r_neto']>0).mean()*100:>5.1f}% | "
                      f"{x['r_neto'].mean():>+8.3f} | {pf}")
    if paso == "c":
        walk_forward(pool, pool.assign(), BASE + ["sym_eth"],
                     "C) Pooled con sym_eth → eval TODO") if False else None
        pool2 = pool.copy()
        walk_forward(pool2, pool2, BASE + ["sym_eth"], "C) Pooled + sym_eth → eval TODO")
        walk_forward(pool2, pool2[pool2["sym_eth"] == 0], BASE + ["sym_eth"],
                     "C) Pooled + sym_eth → eval SOLO BTC")
    if paso == "eth":
        walk_forward(pool, eth.assign(sym_eth=1), BASE, "B) Pooled → eval SOLO ETH")


if __name__ == "__main__":
    main()
