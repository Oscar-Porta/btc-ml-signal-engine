"""
==========================================================
EXPERIMENTO A6.3 - Variante 4h (velas 4h + contexto diario)
==========================================================
Réplica del sistema completo un escalón de zoom más arriba:
  - Señales sobre velas de 4h (antes 1h)
  - Filtro de tendencia con velas DIARIAS (antes 4h)
  - Mismas reglas (pullback EMA20 + breakout swing20),
    mismos parámetros (SL 1.25xATR, TP 2R, peor caso)
  - Time exit: se prueban 12 y 24 velas (2 y 4 días)
  - Fricción idéntica a la del sistema 1h:
      · taker: 0.07% por lado
      · limit-close (ganadora de A6.1): maker 0.02% entrada
        y TP; taker en SL/time-exit; ventana 1 vela
  - Mismo esquema de 20 features, recalculadas en 4h/1d
  - Mismo modelo (GB, mismos hiperparámetros), mismo split
    temporal y mismo walk-forward de 4 folds

Anti-lookahead (lección de la revisión):
  - El contexto diario se alinea por CLOSE_TIME de la vela
    diaria <= momento de decisión (cierre de la vela 4h).
  - Velas diarias construidas sobre días naturales UTC
    (las velas 4h de Binance están alineadas a UTC).
  - Breakout usa el swing de la vela ANTERIOR (shift 1).
==========================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

RAW_4H = "data/raw/BTCUSDT_4h.csv"
EXP_DIR = "outputs/experiments"
os.makedirs(EXP_DIR, exist_ok=True)

SL_ATR_MULT = 1.25
TP_R_MULT = 2.0
MAKER = 0.0002
TAKER_SLIP = 0.0007

FEATURES = [
    "rsi_14", "atr_14", "dist_ema20_pct", "dist_ema50_pct",
    "ema_spread_pct", "ema_spread_ctx_pct", "atr_pct", "price_position",
    "body_pct", "upper_wick_pct", "lower_wick_pct", "hour", "day_of_week",
    "volume_sma20_ratio", "rsi_14_ctx", "atr_pct_ctx",
    "roc_5", "roc_20", "atr_regime", "candle_range_ratio",
]


# ---------- Indicadores (mismas fórmulas del proyecto) ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    p = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = g / p.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    cp = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-cp).abs(), (df["low"]-cp).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def preparar_datos():
    df = pd.read_csv(RAW_4H)
    for c in ["open_time", "close_time"]:
        df[c] = pd.to_datetime(df[c], format="ISO8601", utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)

    # Indicadores 4h
    df["ema_20"], df["ema_50"] = ema(df["close"], 20), ema(df["close"], 50)
    df["rsi_14"], df["atr_14"] = rsi(df["close"]), atr(df)
    df["swing_high_20"] = df["high"].rolling(20).max()
    df["swing_low_20"] = df["low"].rolling(20).min()
    df["trend"] = np.where(df["ema_20"] > df["ema_50"], "up", "down")

    # ---- Velas diarias (días naturales UTC) ----
    d = df.copy()
    d["day"] = d["open_time"].dt.floor("1D")
    dd = d.groupby("day").agg(open=("open", "first"), high=("high", "max"),
                              low=("low", "min"), close=("close", "last"),
                              n=("close", "size")).reset_index()
    dd = dd[dd["n"] == 6].reset_index(drop=True)   # solo días completos
    dd["close_time_1d"] = dd["day"] + pd.Timedelta(days=1)
    dd["ema_20_1d"], dd["ema_50_1d"] = ema(dd["close"], 20), ema(dd["close"], 50)
    dd["rsi_14_1d"] = rsi(dd["close"])
    dd["atr_14_1d"] = atr(dd.rename(columns={"close": "close", "high": "high", "low": "low"}))
    dd["trend_1d"] = np.where(dd["ema_20_1d"] > dd["ema_50_1d"], "up", "down")
    dd.loc[dd["ema_50_1d"].isna(), "trend_1d"] = np.nan

    # ---- Alinear contexto diario SIN lookahead ----
    df["decision_time"] = df["open_time"] + pd.Timedelta(hours=4)
    ctx = dd[["close_time_1d", "trend_1d", "ema_20_1d", "ema_50_1d",
              "rsi_14_1d", "atr_14_1d"]].sort_values("close_time_1d")
    df = pd.merge_asof(df.sort_values("decision_time"), ctx,
                       left_on="decision_time", right_on="close_time_1d",
                       direction="backward")
    return df.reset_index(drop=True)


def generar_senales(df):
    prev_sh, prev_sl = df["swing_high_20"].shift(1), df["swing_low_20"].shift(1)
    pb_long = ((df["trend_1d"] == "up") & (df["trend"] == "up") &
               (df["low"] <= df["ema_20"]) & (df["close"] > df["ema_20"]) &
               (df["rsi_14"] < 55))
    pb_short = ((df["trend_1d"] == "down") & (df["trend"] == "down") &
                (df["high"] >= df["ema_20"]) & (df["close"] < df["ema_20"]) &
                (df["rsi_14"] > 45))
    bo_long = ((df["trend_1d"] == "up") & (df["close"] > prev_sh) &
               (df["rsi_14"].between(40, 75)))
    bo_short = ((df["trend_1d"] == "down") & (df["close"] < prev_sl) &
                (df["rsi_14"].between(25, 60)))
    out = []
    for mask, direction, rule in [(pb_long, "LONG", "pullback"), (pb_short, "SHORT", "pullback"),
                                  (bo_long, "LONG", "breakout"), (bo_short, "SHORT", "breakout")]:
        s = df[mask & df["atr_14"].notna() & df["rsi_14"].notna()].copy()
        s["direction"], s["rule"] = direction, rule
        out.append(s)
    sig = pd.concat(out).sort_values("open_time").reset_index(drop=True)
    return sig


def simular(sig, df, max_bars, ejecucion):
    """ejecucion: 'taker' (al close de señal) o 'limit' (close, W=1)."""
    idx = pd.Series(df.index, index=df["open_time"])
    ohlc = df[["open", "high", "low", "close"]].to_numpy()
    res = []
    for _, s in sig.iterrows():
        i = idx.get(s["open_time"])
        if i is None:
            continue
        velas = ohlc[i + 1: i + 2 + max_bars]
        if len(velas) == 0:
            continue
        d = 1 if s["direction"] == "LONG" else -1
        riesgo = SL_ATR_MULT * s["atr_14"]

        if ejecucion == "taker":
            F, start = s["close"], 0
            fric_in = F * TAKER_SLIP
        else:
            F, start = None, None
            o, h, l, c = velas[0]
            if (d == 1 and o <= s["close"]) or (d == -1 and o >= s["close"]):
                F = o
            elif (d == 1 and l < s["close"]) or (d == -1 and h > s["close"]):
                F = s["close"]
            if F is None:
                res.append({"open_time": s["open_time"], "direction": s["direction"],
                            "rule": s["rule"], "ejecutado": False,
                            "pnl_neto": np.nan, "r_neto": np.nan})
                continue
            start, fric_in = 0, F * MAKER

        sl, tp = F - d * riesgo, F + d * TP_R_MULT * riesgo
        exit_price, reason = None, None
        fin = min(start + max_bars, len(velas))
        for k in range(start, fin):
            o, h, l, c = velas[k]
            if (l <= sl) if d == 1 else (h >= sl):
                exit_price, reason = sl, "stop_loss"; break
            if (h >= tp) if d == 1 else (l <= tp):
                exit_price, reason = tp, "take_profit_1"; break
        if exit_price is None:
            exit_price, reason = velas[fin - 1][3], "time_exit"

        pnl = d * (exit_price - F) - fric_in - exit_price * (MAKER if reason == "take_profit_1" else TAKER_SLIP)
        res.append({"open_time": s["open_time"], "direction": s["direction"],
                    "rule": s["rule"], "ejecutado": True,
                    "pnl_neto": pnl, "r_neto": pnl / riesgo if riesgo > 0 else 0.0})
    return pd.DataFrame(res)


def crear_features(sig, df):
    df = df.copy()
    df["volume_sma20"] = df["volume"].rolling(20).mean()
    df["close_5_ago"] = df["close"].shift(5)
    df["close_20_ago"] = df["close"].shift(20)
    df["atr_sma20"] = df["atr_14"].rolling(20).mean()
    ctx_cols = ["open_time", "volume", "volume_sma20", "close_5_ago", "close_20_ago", "atr_sma20"]
    sig = sig.merge(df[ctx_cols], on="open_time", how="left", suffixes=("", "_ctx2"))

    f = sig
    f["dist_ema20_pct"] = (f["close"] - f["ema_20"]) / f["ema_20"] * 100
    f["dist_ema50_pct"] = (f["close"] - f["ema_50"]) / f["ema_50"] * 100
    f["ema_spread_pct"] = (f["ema_20"] - f["ema_50"]) / f["ema_50"] * 100
    f["ema_spread_ctx_pct"] = (f["ema_20_1d"] - f["ema_50_1d"]) / f["ema_50_1d"] * 100
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
    f["rsi_14_ctx"] = f["rsi_14_1d"].fillna(50.0)
    f["atr_pct_ctx"] = np.where(f["atr_14_1d"].notna(), f["atr_14_1d"] / f["close"] * 100, f["atr_pct"])
    f["roc_5"] = np.where(f["close_5_ago"] > 0, (f["close"] - f["close_5_ago"]) / f["close_5_ago"] * 100, 0.0)
    f["roc_20"] = np.where(f["close_20_ago"] > 0, (f["close"] - f["close_20_ago"]) / f["close_20_ago"] * 100, 0.0)
    f["atr_regime"] = np.where(f["atr_sma20"] > 0, f["atr_14"] / f["atr_sma20"], 1.0)
    f["candle_range_ratio"] = np.where(f["atr_14"] > 0, (f["high"] - f["low"]) / f["atr_14"], 1.0)
    return f


def tabla(df, proba_col, titulo):
    print(f"\n{titulo}")
    print(f"{'Umbral':>7} | {'Trades':>6} | {'WR':>6} | {'PnL neto':>11} | {'R medio':>8}")
    for u in [0.0, 0.40, 0.45, 0.50, 0.55, 0.60]:
        x = df if u == 0 else df[df[proba_col] >= u]
        if len(x) == 0:
            continue
        wr = (x["pnl_neto"] > 0).mean() * 100
        print(f"{u:>7.2f} | {len(x):>6} | {wr:>5.1f}% | {x['pnl_neto'].sum():>+10,.0f} | {x['r_neto'].mean():>+8.3f}")


def evaluar(m, titulo):
    cutoff = pd.Timestamp("2024-12-31", tz="UTC")
    tr_, te_ = m[m["open_time"] <= cutoff], m[m["open_time"] > cutoff].copy()
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    min_samples_leaf=20, subsample=0.8, random_state=42)
    gb.fit(tr_[FEATURES], tr_["label"])
    te_["proba"] = gb.predict_proba(te_[FEATURES])[:, 1]
    tabla(te_, "proba", f"{titulo} — TEST 2025-26 (split fijo):")

    folds = ["2023-01-01", "2024-01-01", "2025-01-01", "2025-10-01"]
    oos = []
    for j, ini in enumerate(folds):
        fin = folds[j + 1] if j + 1 < len(folds) else "2027-01-01"
        ini_ts, fin_ts = pd.Timestamp(ini, tz="UTC"), pd.Timestamp(fin, tz="UTC")
        trf = m[m["open_time"] < ini_ts]
        tef = m[(m["open_time"] >= ini_ts) & (m["open_time"] < fin_ts)].copy()
        if len(tef) == 0 or len(trf) < 250:
            continue
        g = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       min_samples_leaf=20, subsample=0.8, random_state=42)
        g.fit(trf[FEATURES], trf["label"])
        tef["proba"] = g.predict_proba(tef[FEATURES])[:, 1]
        oos.append(tef)
    O = pd.concat(oos)
    tabla(O, "proba", f"{titulo} — WALK-FORWARD ({len(O)} preds OOS):")
    return O


def main():
    df = preparar_datos()
    sig = generar_senales(df)
    n_dias = (sig["open_time"].max() - sig["open_time"].min()).days
    print(f"Señales 4h: {len(sig)} en {n_dias} días (~{len(sig)/n_dias*7:.1f}/semana)")
    print(f"Riesgo medio: {(SL_ATR_MULT*sig['atr_14']/sig['close']*100).mean():.2f}% del precio")

    for max_bars, ejec in [(24, "taker"), (12, "taker"), (24, "limit"), (12, "limit")]:
        r = simular(sig, df, max_bars, ejec)
        ex = r[r["ejecutado"]]
        wr = (ex["pnl_neto"] > 0).mean() * 100
        print(f"\n=== time_exit={max_bars} velas | ejecución={ejec} | "
              f"fills {len(ex)}/{len(r)} | sin filtro: WR {wr:.1f}% "
              f"PnL {ex['pnl_neto'].sum():+,.0f} R {ex['r_neto'].mean():+.3f}")

    # ---- ML sobre la mejor combinación bruta (se elige a la vista de lo anterior) ----
    import sys
    max_bars = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    ejec = sys.argv[2] if len(sys.argv) > 2 else "limit"
    print(f"\n>>> ML con time_exit={max_bars}, ejecución={ejec}")
    r = simular(sig, df, max_bars, ejec)
    feats = crear_features(sig, df)
    m = feats.merge(r[["open_time", "direction", "rule", "ejecutado", "pnl_neto", "r_neto"]],
                    on=["open_time", "direction", "rule"])
    m = m[m["ejecutado"]].dropna(subset=FEATURES).copy()
    m["label"] = (m["r_neto"] >= 0.5).astype(int)
    print(f"Dataset: {len(m)} | ganadores {m['label'].mean()*100:.1f}%")
    O = evaluar(m, f"4H {ejec} te={max_bars}")
    m.to_csv(os.path.join(EXP_DIR, f"dataset_4h_{ejec}_{max_bars}.csv"), index=False)
    O.to_csv(os.path.join(EXP_DIR, f"walkforward_4h_{ejec}_{max_bars}.csv"), index=False)


if __name__ == "__main__":
    main()
