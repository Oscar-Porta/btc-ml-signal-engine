"""
generate_metrics.py
-------------------
Genera el archivo configurator_metrics.json para la web (signal-dashboard-web).

Calcula métricas de rendimiento para todas las combinaciones:
  instrumento × franja × umbral × dirección

Uso:
    python src/tools/generate_metrics.py

Salida:
    ../signal-dashboard-web/data/configurator_metrics.json

Relanzar cada vez que el modelo se reentrene.
"""

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]          # trading_project_signal/
DATASET  = BASE_DIR / "outputs" / "ml_dataset.csv"
MODEL    = BASE_DIR / "outputs" / "model_v3.joblib"
OUT_FILE = BASE_DIR.parent / "signal-dashboard-web" / "data" / "configurator_metrics.json"

# ── Parámetros ─────────────────────────────────────────────────────────────────
TEST_START   = "2025-01-01"   # fecha de inicio del período de test (out-of-sample)
POS_SIZE_USD = 10_000         # tamaño de posición asumido para calcular PnL

# ── Definición de franjas ──────────────────────────────────────────────────────
# hour y day_of_week ya están en hora Madrid en ml_dataset.csv
# day_of_week: 0=lunes … 4=viernes, 5=sábado, 6=domingo
FRANJAS = {
    "manana": {
        "name":          "Mañana",
        "hours_madrid":  "06h – 14h",
        "hours_bsas":    "01h – 09h",
        "hours_mx":      "23h – 07h",
        "weekdays_only": True,
        "filter":        lambda df: (df["hour"] >= 6)  & (df["hour"] < 14) & (df["day_of_week"] <= 4),
    },
    "america": {
        "name":          "América",
        "hours_madrid":  "00h – 08h",
        "hours_bsas":    "19h – 03h",
        "hours_mx":      "16h – 00h",
        "weekdays_only": True,
        "filter":        lambda df: (df["hour"] >= 0)  & (df["hour"] < 8)  & (df["day_of_week"] <= 4),
    },
    "fulltime": {
        "name":          "Full Time",
        "hours_madrid":  "00h – 24h",
        "hours_bsas":    "00h – 24h",
        "hours_mx":      "00h – 24h",
        "weekdays_only": False,
        "filter":        lambda df: pd.Series([True] * len(df), index=df.index),
    },
}

# ── Definición de umbrales ─────────────────────────────────────────────────────
UMBRALES = {
    "activo": {
        "name":  "Activo",
        "value": 0.45,
    },
    "selectivo": {
        "name":  "Selectivo",
        "value": 0.60,
    },
}

# ── Instrumentos ───────────────────────────────────────────────────────────────
INSTRUMENTS = {
    "BTC": {"name": "Bitcoin (BTC/USDT)"},
}


# ── Funciones de métricas ──────────────────────────────────────────────────────

def max_drawdown(pnl_series: pd.Series) -> float:
    """Máximo drawdown absoluto sobre la curva de equity."""
    equity = pnl_series.cumsum()
    peak   = equity.cummax()
    return float((equity - peak).min())


def max_consecutive_losses(winner_series: pd.Series) -> int:
    """Racha máxima de pérdidas consecutivas."""
    max_streak = current = 0
    for w in winner_series:
        if w == 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return int(max_streak)


def profit_factor(pnl_series: pd.Series) -> float:
    """Ratio entre suma de ganancias y suma de pérdidas (en valor absoluto)."""
    gains  = pnl_series[pnl_series > 0].sum()
    losses = pnl_series[pnl_series < 0].abs().sum()
    if losses == 0:
        return float("inf")
    return round(float(gains / losses), 2)


def compute_metrics(subset: pd.DataFrame, weeks: float) -> dict:
    """Calcula todas las métricas para un subconjunto de trades."""
    n = len(subset)
    if n == 0:
        return None

    pnl = subset["pnl_abs"]
    win = subset["winner"]

    return {
        "n_signals":              int(n),
        "signals_per_week":       round(n / weeks, 2),
        "win_rate_pct":           round(float(win.mean() * 100), 1),
        "profit_factor":          profit_factor(pnl),
        "pnl_total_usd":          round(float(pnl.sum()), 0),
        "pnl_per_trade_usd":      round(float(pnl.mean()), 0),
        "max_dd_usd":             round(max_drawdown(pnl), 0),
        "max_consecutive_losses": max_consecutive_losses(win),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Cargando datos y modelo...")
    df    = pd.read_csv(DATASET)
    model = joblib.load(MODEL)

    # Parsear fechas y filtrar test
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    test_start_ts   = pd.Timestamp(TEST_START, tz="UTC")
    df = df[df["open_time"] >= test_start_ts].copy()
    print(f"  Período de test: {df['open_time'].min().date()} → {df['open_time'].max().date()}")
    print(f"  Señales en test: {len(df)}")

    # Aplicar modelo
    features    = model.feature_names_in_
    df["prob"]  = model.predict_proba(df[features])[:, 1]
    df["winner"] = (df["label"] == 1).astype(int)

    # Período en semanas
    n_days  = (df["open_time"].max() - df["open_time"].min()).days + 1
    n_weeks = n_days / 7
    period  = {
        "start_date": str(df["open_time"].min().date()),
        "end_date":   str(df["open_time"].max().date()),
        "days":       int(n_days),
        "weeks":      round(n_weeks, 1),
    }
    print(f"  Semanas: {n_weeks:.1f}")

    # ── Construir JSON ─────────────────────────────────────────────────────────
    instruments_out = []

    for instr_id, instr_info in INSTRUMENTS.items():
        # De momento todos los trades son BTC
        df_instr = df.copy()

        franjas_out = []
        for franja_id, franja_def in FRANJAS.items():
            mask_franja  = franja_def["filter"](df_instr)
            df_franja    = df_instr[mask_franja]

            umbrales_out = []
            for umb_id, umb_def in UMBRALES.items():
                df_umb = df_franja[df_franja["prob"] >= umb_def["value"]]

                # Métricas por dirección
                directions_out = {}
                for dir_label, dir_filter in [
                    ("all",   df_umb),
                    ("LONG",  df_umb[df_umb["direction"] == "LONG"]),
                    ("SHORT", df_umb[df_umb["direction"] == "SHORT"]),
                ]:
                    m = compute_metrics(dir_filter, n_weeks)
                    if m is not None:
                        directions_out[dir_label] = m

                umbrales_out.append({
                    "id":          umb_id,
                    "name":        umb_def["name"],
                    "value":       umb_def["value"],
                    "directions":  directions_out,
                })

            franjas_out.append({
                "id":            franja_id,
                "name":          franja_def["name"],
                "hours_madrid":  franja_def["hours_madrid"],
                "hours_bsas":    franja_def["hours_bsas"],
                "hours_mx":      franja_def["hours_mx"],
                "weekdays_only": franja_def["weekdays_only"],
                "umbrales":      umbrales_out,
            })

        instruments_out.append({
            "id":      instr_id,
            "name":    instr_info["name"],
            "franjas": franjas_out,
        })

    output = {
        "_info":        "Generado por trading_project_signal/src/tools/generate_metrics.py",
        "_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "_pos_size_usd": POS_SIZE_USD,
        "_note":        "Métricas sin descuento de comisiones. Binance Spot ~0.1% por operación.",
        "_period":       period,
        "instruments":   instruments_out,
    }

    # ── Guardar ────────────────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nArchivo generado: {OUT_FILE}")
    print("Combinaciones calculadas:")
    for instr in output["instruments"]:
        for franja in instr["franjas"]:
            for umb in franja["umbrales"]:
                n_all = umb["directions"].get("all", {}).get("n_signals", 0)
                wr    = umb["directions"].get("all", {}).get("win_rate_pct", 0)
                print(f"  {instr['id']} · {franja['name']:<12} · {umb['name']:<10} → {n_all:>3} señales, WR {wr}%")


if __name__ == "__main__":
    main()
