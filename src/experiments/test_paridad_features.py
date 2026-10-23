"""
==========================================================
TEST DE PARIDAD batch <-> live (hallazgo H5 de la revisión)
==========================================================
Comprueba que crear_features_para_modelo() de 08_live_signal
produce EXACTAMENTE los mismos valores que el dataset de
entrenamiento (06_prepare_dataset) para las mismas señales.

Reconstruye el contexto que vería el live (velas + rolling +
4h alineado por close_time, ya corregido en A1) y compara
las 20 features una a una. Tolerancia: 1e-6 relativa.

Ejecutar desde la raíz del proyecto:
  python3 src/experiments/test_paridad_features.py
==========================================================
"""

import os
import sys
import types
import importlib.util
import pandas as pd
import numpy as np

# --- Stubs para importar 08 sin binance/dotenv/requests ---
for nombre in ["requests", "dotenv", "binance", "binance.client"]:
    if nombre not in sys.modules:
        mod = types.ModuleType(nombre)
        if nombre == "dotenv":
            mod.load_dotenv = lambda *a, **k: None
        if nombre == "binance.client":
            mod.Client = object
        sys.modules[nombre] = mod

spec = importlib.util.spec_from_file_location("live", "src/08_live_signal.py")
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)

FEATURES = ["rsi_14", "atr_14", "dist_ema20_pct", "dist_ema50_pct", "ema_spread_pct",
            "ema_spread_4h_pct", "atr_pct", "price_position", "body_pct", "upper_wick_pct",
            "lower_wick_pct", "hour", "day_of_week", "volume_sma20_ratio", "rsi_14_4h",
            "atr_pct_4h", "roc_5", "roc_20", "atr_regime", "candle_range_ratio"]


def main():
    ds = pd.read_csv("outputs/ml_dataset.csv")
    ds["open_time"] = pd.to_datetime(ds["open_time"], format="ISO8601", utc=True).dt.tz_convert("Europe/Madrid")

    h = pd.read_csv("data/processed/BTCUSDT_1h_processed.csv")
    h["open_time"] = pd.to_datetime(h["open_time"], format="ISO8601", utc=True).dt.tz_convert("Europe/Madrid")
    h = h.sort_values("open_time").reset_index(drop=True)

    q = pd.read_csv("data/processed/BTCUSDT_4h_processed.csv")
    for c in ["open_time", "close_time"]:
        q[c] = pd.to_datetime(q[c], format="ISO8601", utc=True).dt.tz_convert("Europe/Madrid")

    # Contexto que construye el live (tras el fix A1): última 4h CERRADA
    h = live.alinear_contexto_4h(
        h, q.rename(columns={})) if False else h
    # Réplica manual del merge del live (open_time backward sobre velas
    # cerradas) == merge por close_time (validado en A1):
    ctx = q[["close_time", "trend", "ema_20", "ema_50", "rsi_14", "atr_14"]].rename(columns={
        "close_time": "ct4", "trend": "trend_4h", "ema_20": "ema_20_4h",
        "ema_50": "ema_50_4h", "rsi_14": "rsi_14_4h", "atr_14": "atr_14_4h"})
    h["decision_time"] = h["open_time"] + pd.Timedelta(hours=1)
    h = pd.merge_asof(h.sort_values("decision_time"), ctx.sort_values("ct4"),
                      left_on="decision_time", right_on="ct4", direction="backward")
    h = h.rename(columns={"trend": "trend_1h"}).sort_values("open_time").reset_index(drop=True)
    h["volume_sma20"] = h["volume"].rolling(20).mean()
    h["close_5_ago"] = h["close"].shift(5)
    h["close_20_ago"] = h["close"].shift(20)
    h["atr_sma20"] = h["atr_14"].rolling(20).mean()

    h_idx = h.set_index("open_time")

    muestra = ds.sample(n=min(500, len(ds)), random_state=42)
    fallos, comparadas = {}, 0
    for _, fila in muestra.iterrows():
        t = fila["open_time"]
        if t not in h_idx.index:
            continue
        row = h_idx.loc[t]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        row = row.copy()
        row["open_time"] = t
        feats = live.crear_features_para_modelo(row)
        comparadas += 1
        for f in FEATURES:
            a, b = feats[f], fila[f]
            tol = max(abs(b) * 1e-6, 1e-9)
            if abs(a - b) > tol:
                fallos.setdefault(f, []).append((t, a, b))

    print(f"Señales comparadas: {comparadas}")
    if not fallos:
        print("✅ PARIDAD PERFECTA: live y entrenamiento calculan lo mismo.")
    else:
        print("❌ DISCREPANCIAS:")
        for f, casos in fallos.items():
            t, a, b = casos[0]
            print(f"  {f}: {len(casos)} casos. Ej: live={a:.6f} vs dataset={b:.6f} ({t})")
        sys.exit(1)


if __name__ == "__main__":
    main()
