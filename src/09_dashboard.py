"""
==========================================================
FASE 13 - Dashboard Streamlit (v3 — El Modelo)
==========================================================
2 tabs:
  1. Señales Live  (sin cambios)
  2. El Modelo     (pipeline + reglas + backtest + dataset
                    + XGBoost + rendimiento test + prod + retrain)
==========================================================
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import joblib

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
OUTPUT_DIR = "outputs"
TRADES_FILE = os.path.join(OUTPUT_DIR, "trades.csv")
EQUITY_FILE = os.path.join(OUTPUT_DIR, "equity_curve.csv")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "model_results_v3.csv")
LIVE_LOG_FILE = os.path.join(OUTPUT_DIR, "live_signals_log.csv")
LAST_CHECK_FILE = os.path.join(OUTPUT_DIR, "last_check.json")
MONITOR_STATE_FILE = os.path.join(OUTPUT_DIR, "monitor_state.json")
POSITION_RESULTS_FILE = os.path.join(OUTPUT_DIR, "position_results.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "model.joblib")

AUTO_REFRESH_MS = 3_600_000

C = {
    "bg":       "#f8f9fb",
    "card":     "#ffffff",
    "border":   "#e2e8f0",
    "text":     "#1e293b",
    "muted":    "#64748b",
    "accent":   "#4f46e5",
    "win":      "#059669",
    "loss":     "#dc2626",
    "warn":     "#d97706",
    "long":     "#059669",
    "short":    "#dc2626",
    "gradient": "#818cf8",
}

FEATURES_DISPLAY = [
    ("rsi_14", "RSI 14", "Fuerza relativa del precio"),
    ("atr_14", "ATR 14", "Volatilidad media"),
    ("dist_ema20_pct", "Dist. EMA20", "Distancia a la media rápida"),
    ("dist_ema50_pct", "Dist. EMA50", "Distancia a la media lenta"),
    ("ema_spread_pct", "EMA Spread 1H", "Separación entre medias"),
    ("ema_spread_4h_pct", "EMA Spread 4H", "Separación entre medias (4H)"),
    ("atr_pct", "ATR %", "Volatilidad relativa al precio"),
    ("price_position", "Posición en rango", "Dónde está el precio en su rango"),
    ("body_pct", "Cuerpo vela %", "Tamaño del cuerpo de la vela"),
    ("upper_wick_pct", "Mecha superior %", "Rechazo al alza"),
    ("lower_wick_pct", "Mecha inferior %", "Rechazo a la baja"),
    ("hour", "Hora", "Hora del día (0-23)"),
    ("day_of_week", "Día semana", "Día de la semana (0=Lun)"),
    ("volume_sma20_ratio", "Vol/SMA20", "Volumen vs su media de 20"),
    ("rsi_14_4h", "RSI 14 (4H)", "RSI en temporalidad 4 horas"),
    ("atr_pct_4h", "ATR % (4H)", "Volatilidad 4H relativa"),
    ("roc_5", "ROC 5", "Cambio de precio en 5 velas"),
    ("roc_20", "ROC 20", "Cambio de precio en 20 velas"),
    ("atr_regime", "Régimen ATR", "Volatilidad actual vs histórica"),
    ("candle_range_ratio", "Rango vela", "Tamaño total de la vela"),
]


# ==========================================================
# CSS PREMIUM
# ==========================================================
def aplicar_css():
    st.markdown("""
    <style>
    /* --- Reset Streamlit defaults --- */
    .stApp { background-color: #f8f9fb; }
    header[data-testid="stHeader"] { background-color: #f8f9fb; }

    /* --- Hide default sidebar --- */
    section[data-testid="stSidebar"] { display: none; }

    /* --- Top navigation bar --- */
    .nav-bar {
        display: flex; align-items: center; gap: 8px;
        padding: 12px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 24px;
    }
    .nav-brand { font-size: 1.3rem; font-weight: 700; color: #1e293b; margin-right: 24px; letter-spacing: -0.5px; }
    .nav-brand span { color: #4f46e5; }
    .nav-badge {
        background: linear-gradient(135deg, #4f46e5, #6366f1); color: white;
        font-size: 0.65rem; padding: 2px 8px; border-radius: 20px;
        margin-left: 8px; font-weight: 600; letter-spacing: 0.5px;
    }

    /* --- Card containers --- */
    .metric-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1e293b; line-height: 1.2; }
    .metric-card .label { font-size: 0.8rem; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .delta-pos { color: #059669; font-size: 0.85rem; }
    .metric-card .delta-neg { color: #dc2626; font-size: 0.85rem; }

    /* --- Signal card --- */
    .signal-card {
        background: #ffffff; border: 1px solid #c7d2fe; border-radius: 16px;
        padding: 24px; margin-bottom: 16px; position: relative; overflow: hidden;
        box-shadow: 0 2px 8px rgba(79,70,229,0.08);
    }
    .signal-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #4f46e5, #818cf8, #4f46e5);
    }
    .signal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
    .signal-dir { font-size: 1.5rem; font-weight: 800; letter-spacing: 1px; }
    .signal-dir.long { color: #059669; }
    .signal-dir.short { color: #dc2626; }
    .signal-prob { font-size: 1.1rem; color: #4f46e5; font-weight: 600; }
    .signal-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .signal-item .label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .signal-item .value { font-size: 1rem; color: #1e293b; font-weight: 600; }
    .signal-time { font-size: 0.75rem; color: #94a3b8; margin-top: 12px; text-align: right; }

    /* --- Section headers (existing) --- */
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #1e293b;
        margin: 32px 0 16px 0; padding-bottom: 8px; border-bottom: 1px solid #e2e8f0;
    }

    /* --- Feature importance bars --- */
    .feat-row { display: flex; align-items: center; margin-bottom: 6px; }
    .feat-name { width: 140px; font-size: 0.8rem; color: #475569; text-align: right; padding-right: 12px; }
    .feat-bar-bg { flex: 1; height: 20px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }
    .feat-bar { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #4f46e5, #818cf8); }
    .feat-val { width: 50px; font-size: 0.75rem; color: #64748b; text-align: right; padding-left: 8px; }

    /* --- Streak badges --- */
    .streak-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 8px 16px; margin: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .streak-badge .num { font-size: 1.4rem; font-weight: 700; }
    .streak-badge .desc { font-size: 0.75rem; color: #64748b; line-height: 1.2; }

    /* --- Info banner --- */
    .info-banner {
        background: #f1f5f9; border-left: 3px solid #4f46e5;
        border-radius: 0 8px 8px 0; padding: 12px 16px;
        font-size: 0.85rem; color: #475569; margin: 16px 0;
    }

    /* --- Plotly chart tweaks --- */
    .js-plotly-plot .plotly .modebar { top: 4px !important; right: 4px !important; }

    /* --- Tab tweaks --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px; background: #ffffff; border-radius: 12px;
        padding: 4px; border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 20px; font-weight: 500; color: #64748b; }
    .stTabs [aria-selected="true"] { background: #4f46e5 !important; color: white !important; }

    /* --- WR info tooltip --- */
    .wr-info { display: inline-flex; align-items: center; gap: 6px; cursor: help; position: relative; }
    .wr-info .info-icon {
        display: inline-flex; align-items: center; justify-content: center;
        width: 16px; height: 16px; border-radius: 50%; background: #e2e8f0;
        color: #64748b; font-size: 0.65rem; font-weight: 700; flex-shrink: 0;
    }
    .wr-info .wr-tooltip {
        display: none; position: absolute; bottom: 100%; left: 50%;
        transform: translateX(-50%); background: #1e293b; color: #e2e8f0;
        padding: 10px 14px; border-radius: 8px; font-size: 0.75rem; line-height: 1.4;
        width: 260px; text-align: left; z-index: 100;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin-bottom: 6px;
    }
    .wr-info .wr-tooltip::after {
        content: ''; position: absolute; top: 100%; left: 50%;
        transform: translateX(-50%); border: 6px solid transparent; border-top-color: #1e293b;
    }
    .wr-info:hover .wr-tooltip { display: block; }

    /* --- Open position card --- */
    .position-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-bottom: 16px; position: relative; overflow: hidden; }
    .position-card.winning { border-left: 4px solid #059669; }
    .position-card.losing { border-left: 4px solid #dc2626; }
    .position-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .position-pnl { font-size: 1.3rem; font-weight: 700; }
    .position-pnl.pos { color: #059669; }
    .position-pnl.neg { color: #dc2626; }
    .progress-bar-bg { width: 100%; height: 8px; background: #e2e8f0; border-radius: 4px; margin: 12px 0 6px 0; position: relative; overflow: visible; }
    .progress-bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
    .progress-bar-fill.pos { background: linear-gradient(90deg, #059669, #34d399); }
    .progress-bar-fill.neg { background: linear-gradient(90deg, #dc2626, #f87171); }
    .progress-labels { display: flex; justify-content: space-between; font-size: 0.7rem; color: #94a3b8; }
    .position-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-top: 12px; }
    .position-item .label { font-size: 0.65rem; color: #94a3b8; text-transform: uppercase; }
    .position-item .value { font-size: 0.9rem; color: #1e293b; font-weight: 600; }

    /* --- Heartbeat --- */
    .heartbeat { display: flex; align-items: center; gap: 8px; padding: 8px 14px; background: #f1f5f9; border-radius: 8px; font-size: 0.78rem; color: #64748b; margin-bottom: 16px; }
    .heartbeat .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .heartbeat .dot.ok { background: #059669; }
    .heartbeat .dot.stale { background: #d97706; }
    .heartbeat .dot.off { background: #dc2626; }

    /* --- Footer --- */
    .footer { text-align: center; color: #94a3b8; font-size: 0.7rem; margin-top: 48px; padding: 16px 0; border-top: 1px solid #e2e8f0; }

    /* ════════════════════════════════════════════════
       NUEVAS CLASES — TAB EL MODELO
    ════════════════════════════════════════════════ */

    /* Section header numerado */
    .step-header {
        display: flex; align-items: flex-start; gap: 16px;
        margin: 40px 0 18px 0; padding: 18px 22px;
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        border-left: 4px solid #4f46e5;
    }
    .sh-num {
        font-size: 1.5rem; font-weight: 800; color: #4f46e5;
        line-height: 1; min-width: 40px;
        border-right: 2px solid #e2e8f0; padding-right: 14px; margin-top: 2px;
    }
    .sh-title { font-size: 1.05rem; font-weight: 700; color: #1e293b; line-height: 1.3; }
    .sh-sub { font-size: 0.82rem; color: #64748b; margin-top: 4px; line-height: 1.5; }

    /* Callout boxes */
    .callout { border-radius: 8px; padding: 12px 16px; margin: 10px 0; font-size: 0.84rem; line-height: 1.6; }
    .callout-info    { background: #eff6ff; border-left: 3px solid #3b82f6; color: #1e3a5f; }
    .callout-warn    { background: #fffbeb; border-left: 3px solid #f59e0b; color: #78350f; }
    .callout-success { background: #f0fdf4; border-left: 3px solid #059669; color: #14532d; }
    .callout-indigo  { background: #eef2ff; border-left: 3px solid #4f46e5; color: #312e81; }

    /* Pipeline cards */
    .pipeline-grid   { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; margin-bottom: 8px; }
    .pipeline-grid-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; }
    .pipe-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 14px; border-left: 3px solid #4f46e5;
    }
    .pipe-card .pc-num   { font-size: 0.68rem; color: #94a3b8; font-weight: 600; margin-bottom: 3px; font-family: monospace; }
    .pipe-card .pc-name  { font-size: 0.88rem; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
    .pipe-card .pc-script{ font-size: 0.68rem; color: #4f46e5; font-family: monospace; margin-bottom: 6px; }
    .pipe-card .pc-desc  { font-size: 0.76rem; color: #64748b; line-height: 1.4; }
    .pipe-card.c-blue   { border-left-color: #3b82f6; }
    .pipe-card.c-green  { border-left-color: #059669; }
    .pipe-card.c-purple { border-left-color: #8b5cf6; }
    .pipe-card.c-amber  { border-left-color: #d97706; }
    .pipe-card.c-cyan   { border-left-color: #06b6d4; }
    .pipe-arrow { text-align: center; color: #94a3b8; font-size: 1.2rem; padding: 2px 0; }

    /* Rule cards */
    .rule-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0; }
    .rule-card-lg {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 20px; border-top: 3px solid #4f46e5;
    }
    .rule-card-lg.green-rule { border-top-color: #059669; }
    .rule-badge { display: inline-block; background: #eef2ff; color: #4f46e5; font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; margin-bottom: 8px; }
    .rule-badge.green { background: #f0fdf4; color: #059669; }
    .rule-title-lg { font-size: 0.95rem; font-weight: 700; color: #1e293b; margin-bottom: 6px; }
    .rule-list { font-size: 0.81rem; color: #475569; line-height: 1.9; }

    /* Train/test split */
    .split-bar-outer { display: flex; border-radius: 8px; overflow: hidden; height: 34px; margin: 10px 0 6px 0; }
    .split-train { background: #4f46e5; display: flex; align-items: center; padding: 0 14px; }
    .split-test  { background: #059669; display: flex; align-items: center; padding: 0 14px; }
    .split-label { font-size: 0.76rem; color: white; font-weight: 600; }
    .split-legend { display: flex; gap: 20px; font-size: 0.76rem; color: #64748b; }
    .split-legend-item { display: flex; align-items: center; gap: 6px; }
    .split-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }

    /* Label comparison cards */
    .lc-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin: 12px 0; }
    .lc-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; text-align: center; }
    .lc-r { font-size: 1.25rem; font-weight: 800; margin-bottom: 5px; }
    .lc-detail { font-size: 0.76rem; color: #64748b; margin-bottom: 8px; line-height: 1.5; }
    .lc-badges { display: flex; justify-content: center; gap: 5px; flex-wrap: wrap; margin-bottom: 8px; }
    .lc-badge { font-size: 0.68rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
    .lc-badge.pass { background: #dcfce7; color: #059669; }
    .lc-badge.fail { background: #fef2f2; color: #dc2626; }
    .lc-label { font-size: 0.82rem; font-weight: 700; padding: 4px 0; border-radius: 6px; }
    .lc-label.one  { background: #eef2ff; color: #4f46e5; }
    .lc-label.zero { background: #f1f5f9; color: #64748b; }

    /* Probability dots grid */
    .prob-dots-row { display: flex; align-items: flex-start; gap: 16px; margin: 10px 0; }
    .prob-dots { display: flex; flex-wrap: wrap; gap: 4px; width: 220px; }
    .prob-dot { width: 16px; height: 16px; border-radius: 3px; }
    .prob-dot.win  { background: #059669; }
    .prob-dot.loss { background: #dc2626; opacity: 0.65; }
    .prob-dots-label { font-size: 0.82rem; color: #475569; line-height: 1.6; padding-top: 2px; }

    /* Walk-forward fold bars */
    .wf-row { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
    .wf-fold-label { font-size: 0.74rem; color: #64748b; width: 55px; flex-shrink: 0; font-family: monospace; }
    .wf-bar-outer { flex: 1; height: 24px; background: #f1f5f9; border-radius: 4px; overflow: hidden; position: relative; }
    .wf-bar-train { height: 100%; background: #818cf8; display: flex; align-items: center; padding: 0 8px; font-size: 0.68rem; color: white; font-weight: 500; }
    .wf-bar-test  { position: absolute; top: 0; right: 0; height: 100%; background: #059669; display: flex; align-items: center; justify-content: flex-end; padding: 0 6px; font-size: 0.68rem; color: white; font-weight: 600; }

    /* XGB config mini-cards */
    .xgb-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 8px; margin: 12px 0; }
    .xgb-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center; }
    .xgb-val { font-size: 1.05rem; font-weight: 700; color: #4f46e5; margin-bottom: 3px; font-family: monospace; }
    .xgb-key { font-size: 0.68rem; color: #64748b; }

    /* TEST DATA banner */
    .test-banner {
        background: linear-gradient(135deg, #fff7ed, #fef3c7);
        border: 1px solid #f59e0b; border-radius: 12px;
        padding: 14px 18px; margin: 0 0 20px 0;
        display: flex; align-items: flex-start; gap: 12px;
    }
    .test-banner .tb-icon { font-size: 1.4rem; flex-shrink: 0; }
    .test-banner .tb-title { font-size: 0.9rem; font-weight: 700; color: #92400e; margin-bottom: 3px; }
    .test-banner .tb-text  { font-size: 0.8rem; color: #78350f; line-height: 1.5; }

    /* Production flow */
    .prod-flow { display: grid; grid-template-columns: repeat(5,1fr); gap: 8px; margin: 12px 0; }
    .prod-step { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 10px; text-align: center; }
    .prod-step .ps-icon { font-size: 1.4rem; margin-bottom: 6px; }
    .prod-step .ps-name { font-size: 0.8rem; font-weight: 700; color: #1e293b; margin-bottom: 3px; }
    .prod-step .ps-desc { font-size: 0.7rem; color: #64748b; line-height: 1.4; }
    .prod-arrow { text-align: center; font-size: 1.1rem; color: #94a3b8; padding-top: 24px; }

    /* Retraining steps */
    .rt-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin: 12px 0; }
    .rt-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; border-top: 2px solid #4f46e5; }
    .rt-num  { font-size: 0.7rem; color: #4f46e5; font-weight: 600; font-family: monospace; margin-bottom: 4px; }
    .rt-name { font-size: 0.88rem; font-weight: 700; color: #1e293b; margin-bottom: 4px; }
    .rt-desc { font-size: 0.76rem; color: #64748b; line-height: 1.4; }

    </style>
    """, unsafe_allow_html=True)


# ==========================================================
# CARGA DE DATOS
# ==========================================================
@st.cache_data(ttl=300)
def cargar_trades():
    df = pd.read_csv(TRADES_FILE)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True).dt.tz_convert("Europe/Madrid")
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True).dt.tz_convert("Europe/Madrid")
    df["resultado"] = df["pnl_abs"].apply(
        lambda x: "Ganador" if x > 0 else "Perdedor"
    )
    return df


@st.cache_data(ttl=300)
def cargar_resultados_modelo():
    df = pd.read_csv(RESULTS_FILE)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True).dt.tz_convert("Europe/Madrid")
    return df


@st.cache_data(ttl=60)
def cargar_live_log():
    if not os.path.exists(LIVE_LOG_FILE):
        return pd.DataFrame()
    df = pd.read_csv(LIVE_LOG_FILE)
    if len(df) == 0:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Madrid")
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True).dt.tz_convert("Europe/Madrid")
    return df


@st.cache_resource
def cargar_modelo():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None


def cargar_last_check():
    if not os.path.exists(LAST_CHECK_FILE):
        return None
    try:
        with open(LAST_CHECK_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def cargar_monitor_state():
    if not os.path.exists(MONITOR_STATE_FILE):
        return None
    try:
        with open(MONITOR_STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=60)
def cargar_position_results():
    if not os.path.exists(POSITION_RESULTS_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(POSITION_RESULTS_FILE)
        if len(df) > 0:
            # Fix robusto para formatos mixtos (con y sin microsegundos).
            # pd.to_datetime() con utc=True falla en pandas 1.x cuando el CSV
            # mezcla "2026-05-06T15:05:23.123456+00:00" y
            # "2026-06-10T12:00:00+00:00". El parsing elemento a elemento
            # (apply) maneja cada fila independientemente y no infiere formato.
            df["open_time"] = df["open_time"].apply(
                lambda x: pd.to_datetime(x, utc=True)
            ).dt.tz_convert("Europe/Madrid")
            df["close_time"] = df["close_time"].apply(
                lambda x: pd.to_datetime(x, utc=True)
            ).dt.tz_convert("Europe/Madrid")
        return df
    except Exception as e:
        import sys
        import traceback
        print(f"[dashboard] ERROR en cargar_position_results: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return pd.DataFrame()


def calcular_wr_breakeven():
    try:
        df = pd.read_csv(RESULTS_FILE)
        if len(df) == 0:
            return 50.0
        winners = df[df["pnl_abs"] > 0]
        losers = df[df["pnl_abs"] < 0]
        if len(winners) == 0 or len(losers) == 0:
            return 50.0
        avg_win = abs(winners["pnl_pct"].mean())
        avg_loss = abs(losers["pnl_pct"].mean())
        payoff = avg_win / avg_loss if avg_loss > 0 else 1
        return round(1 / (1 + payoff) * 100, 1)
    except Exception:
        return 50.0

WR_BREAKEVEN = calcular_wr_breakeven()


# ==========================================================
# HELPERS
# ==========================================================
def metric_card(label, value, delta=None, delta_type="pos", tooltip=None):
    delta_html = ""
    if delta:
        cls = "delta-pos" if delta_type == "pos" else "delta-neg"
        delta_html = f'<div class="{cls}">{delta}</div>'
    if tooltip:
        label_html = (
            f'<span class="wr-info">{label}'
            f'<span class="info-icon">i</span>'
            f'<span class="wr-tooltip">{tooltip}</span>'
            f'</span>'
        )
    else:
        label_html = label
    return f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label_html}</div>
        {delta_html}
    </div>
    """


def calcular_breakeven_payoff(df, pnl_col=None):
    if len(df) == 0:
        return 50.0, 1.0
    if pnl_col is None:
        if "pnl_abs" in df.columns:
            pnl_col = "pnl_abs"
        elif "pnl_usd" in df.columns:
            pnl_col = "pnl_usd"
        else:
            return 50.0, 1.0
    pnl_series = df[pnl_col]
    if pnl_col in ("pnl_abs", "pnl_usd") and "entry_price" in df.columns:
        pnl_series = pnl_series / df["entry_price"] * 100
    winners = pnl_series[pnl_series > 0]
    losers = pnl_series[pnl_series < 0]
    if len(winners) == 0 or len(losers) == 0:
        return 50.0, 1.0
    avg_win = abs(winners.mean())
    avg_loss = abs(losers.mean())
    payoff = avg_win / avg_loss if avg_loss > 0 else 1
    breakeven = round(1 / (1 + payoff) * 100, 1)
    return breakeven, round(payoff, 2)


def wr_delta(wr, breakeven=None):
    be = breakeven if breakeven is not None else WR_BREAKEVEN
    diff = wr - be
    if diff >= 0:
        return f"+{diff:.1f}pp vs {be:.0f}% breakeven", "pos"
    else:
        return f"{diff:.1f}pp vs {be:.0f}% breakeven", "neg"


def wr_metric_card(wr, breakeven=None, payoff=None):
    be = breakeven if breakeven is not None else WR_BREAKEVEN
    diff = wr - be
    cls = "delta-pos" if diff >= 0 else "delta-neg"
    sign = "+" if diff >= 0 else ""
    payoff_text = f" (payoff {payoff:.2f})" if payoff is not None else ""
    return f"""
    <div class="metric-card">
        <div class="value">{wr:.1f}%</div>
        <div class="label">
            <span class="wr-info">
                Win Rate
                <span class="info-icon">i</span>
                <span class="wr-tooltip">
                    El punto de equilibrio se calcula a partir del
                    payoff ratio real de estos trades (ganancia media
                    vs pérdida media). Actualmente el
                    <strong>breakeven es ~{be:.0f}%</strong>
                    con payoff ratio de
                    <strong>{payoff:.2f}</strong>{'' if payoff is None else ' (ganancia media / pérdida media)'}.
                    Cualquier Win Rate por encima genera beneficio neto.
                </span>
            </span>
        </div>
        <div class="{cls}">{sign}{diff:.1f}pp vs {be:.0f}% BE{payoff_text}</div>
    </div>
    """


def plotly_layout(fig, height=400):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.6)",
        font=dict(color="#475569", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        height=height,
        xaxis=dict(gridcolor="rgba(226,232,240,0.8)", zerolinecolor="#e2e8f0"),
        yaxis=dict(gridcolor="rgba(226,232,240,0.8)", zerolinecolor="#e2e8f0"),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_color="#1e293b", bordercolor="#e2e8f0"),
    )
    return fig


def calcular_rachas(series_pnl):
    resultados = (series_pnl > 0).astype(int)
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    prev = None
    for r in resultados:
        if r == prev:
            current_streak += 1
        else:
            current_streak = 1
            prev = r
        if r == 1:
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)
    return max_win_streak, max_loss_streak


# ==========================================================
# NAVEGACIÓN SUPERIOR
# ==========================================================
def render_navbar():
    st.markdown("""
    <div class="nav-bar">
        <div class="nav-brand">
            <span>Signal</span>Lab
            <span class="nav-badge">BTC / USDT</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# TAB 1: SEÑALES LIVE  (sin cambios)
# ==========================================================
def tab_live():
    st.markdown("""
    <div class="info-banner">
        <strong>Señales en tiempo real.</strong> Aquí ves la última señal emitida
        por el modelo ML y, si hay una operación abierta, su evolución en vivo
        con PnL flotante, distancia a los niveles de Stop Loss y Take Profit,
        y tiempo transcurrido. El sistema analiza cada vela de 1 hora y emite
        señales cuando detecta una oportunidad con alta probabilidad.
    </div>
    """, unsafe_allow_html=True)

    last_check = cargar_last_check()
    if last_check:
        candle_time = last_check["candle_time"]
        close_price = last_check["close_price"]
        checked_at = last_check.get("checked_at", "")
        signal_found = last_check.get("signal_found", False)
        try:
            checked_dt = datetime.fromisoformat(checked_at)
            age_hours = (datetime.now(checked_dt.tzinfo) - checked_dt).total_seconds() / 3600
            if age_hours > 6:
                dot_class = "off"; status_text = "Bot sin respuesta"
            elif signal_found:
                dot_class = "ok"; status_text = "Señal detectada"
            else:
                dot_class = "off"; status_text = "Sin señal de entrada"
        except Exception:
            dot_class = "off"; status_text = "Sin señal de entrada"
        try:
            ct_parsed = pd.Timestamp(candle_time)
            candle_display = ct_parsed.strftime("%d %b %Y — %H:%M") + " (Madrid)"
        except Exception:
            candle_display = str(candle_time)
        st.markdown(f"""
        <div class="heartbeat">
            <span class="dot {dot_class}"></span>
            Última vela: <strong>{candle_display}</strong> —
            ${close_price:,.2f} USDT — {status_text}
        </div>
        """, unsafe_allow_html=True)

    monitor = cargar_monitor_state()
    df_live = cargar_live_log()
    df_results_live = cargar_position_results()
    has_open_position = (monitor and monitor.get("posiciones") and len(monitor["posiciones"]) > 0)

    # ── FLUJO LIMIT (jul-2026): órdenes pendientes y fill-rate ──
    ORDERS_FILE = os.path.join(OUTPUT_DIR, "open_orders.json")
    CANCELLED_FILE = os.path.join(OUTPUT_DIR, "cancelled_orders.csv")
    ordenes = []
    if os.path.exists(ORDERS_FILE):
        try:
            with open(ORDERS_FILE) as f:
                ordenes = json.load(f)
        except Exception:
            ordenes = []
    n_cancel = 0
    if os.path.exists(CANCELLED_FILE):
        try:
            n_cancel = len(pd.read_csv(CANCELLED_FILE))
        except Exception:
            n_cancel = 0
    if ordenes:
        o = ordenes[0]
        st.markdown(f"""
        <div class="info-banner">
            ⏳ <strong>Orden limit pendiente:</strong>
            {o.get('direction','?')} {o.get('rule','')} a
            ${float(o.get('limit_price',0)):,.2f}
            — expira {str(o.get('expiry',''))[:16]}
        </div>
        """, unsafe_allow_html=True)
    n_fill = len(df_results_live) if df_results_live is not None else 0
    if (n_fill + n_cancel) >= 5:
        fr = n_fill / (n_fill + n_cancel) * 100
        st.caption(f"Fill-rate de órdenes limit: {fr:.0f}% "
                   f"({n_fill} ejecutadas / {n_cancel} canceladas) · "
                   f"referencia backtest: 98.4%")

    if len(df_live) == 0 and not has_open_position:
        st.markdown("""
        <div class="info-banner">
            No hay señales live registradas todavía.
            Ejecuta <code>python src/08_live_signal.py --loop</code>
            para empezar a generar señales en tiempo real.
        </div>
        """, unsafe_allow_html=True)
        return

    if has_open_position:
        pos = monitor["posiciones"][0]
        pnl_class = "pos" if pos["pnl_usd"] >= 0 else "neg"
        card_border = C["win"] if pos["pnl_usd"] >= 0 else C["loss"]
        progress = max(min(pos["progress_pct"], 100), -100)
        bar_width = max(abs(progress), 2)
        tp1_badge = ""
        dir_arrow = "▲" if pos["direction"] == "LONG" else "▼"
        dir_cls = "long" if pos["direction"] == "LONG" else "short"
        ai_text = (f"{pos['ai_proba']:.0%}" if pos.get("ai_proba") is not None else "—")
        try:
            open_dt = pd.Timestamp(pos.get("open_time", "")).tz_convert("Europe/Madrid")
            open_str = open_dt.strftime("%Y-%m-%d %H:%M") + " (Madrid)"
        except Exception:
            open_str = pos.get("open_time", "")[:16].replace("T", " ") + " (Madrid)"
        pnl_color = "#059669" if pos["pnl_usd"] >= 0 else "#dc2626"
        sl_entry_bar = (
            f'<span style="font-size:0.7rem;color:#dc2626;">SL ${pos["stop_loss"]:,.0f}</span>'
            f'<span style="font-size:0.7rem;color:#475569;">Entrada ${pos["entry_price"]:,.0f}</span>'
        )
        st.markdown(f"""
        <div class="signal-card" style="border-left:4px solid {card_border};">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <div class="signal-dir {dir_cls}" style="margin-bottom:4px;">
                        {dir_arrow} {pos['direction']}
                        <span style="font-size:0.85rem;font-weight:400;color:#64748b;margin-left:8px;">{pos['rule']}{tp1_badge}</span>
                    </div>
                    <div style="font-size:1rem;color:#475569;font-weight:500;">{open_str}</div>
                </div>
                <div style="text-align:right;">
                    <div class="position-pnl {pnl_class}" style="font-size:1.5rem;">${pos['pnl_usd']:+,.2f}</div>
                    <div style="font-size:0.85rem;color:{pnl_color};">{pos['pnl_pct']:+.2f}%</div>
                    <div style="font-size:0.7rem;color:#94a3b8;margin-top:2px;">OPERACIÓN EN CURSO</div>
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin:16px 0 4px 0;">
                {sl_entry_bar}
                <span style="font-size:0.7rem;color:#059669;">TP1 ${pos['tp1']:,.0f}</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill {pnl_class}" style="width:{bar_width:.0f}%;"></div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:14px;">
                <div style="text-align:center;">
                    <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">Precio actual</div>
                    <div style="font-size:0.95rem;color:#1e293b;font-weight:600;">${pos['precio_actual']:,.2f}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">ML Prob</div>
                    <div style="font-size:0.95rem;color:#1e293b;font-weight:600;">{ai_text}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">Dist. SL</div>
                    <div style="font-size:0.95rem;color:#dc2626;font-weight:600;">{pos['dist_sl_pct']:.2f}%</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">Dist. TP1</div>
                    <div style="font-size:0.95rem;color:#059669;font-weight:600;">{pos['dist_tp_pct']:.2f}%</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.65rem;color:#94a3b8;text-transform:uppercase;">Tiempo</div>
                    <div style="font-size:0.95rem;color:#1e293b;font-weight:600;">{pos['hours_open']:.1f}h</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif len(df_live) > 0:
        ultima = df_live.iloc[-1]
        dir_class = "long" if ultima["direction"] == "LONG" else "short"
        dir_arrow = "&#9650;" if ultima["direction"] == "LONG" else "&#9660;"
        ts_str = ultima["signal_time"].strftime("%d %b %Y — %H:%M (Madrid)")
        ai_text = (f'{ultima["ai_proba"]:.0%}' if pd.notna(ultima.get("ai_proba")) else "—")
        result_badge = ""
        result_detail = ""
        if len(df_results_live) > 0:
            df_closed = df_results_live
            if len(df_closed) > 0:
                last_result = df_closed.iloc[-1]
                pnl_result = last_result["pnl_usd"]
                if pnl_result > 0:
                    result_badge = '<span style="background:#dcfce7;color:#059669;font-size:0.7rem;padding:3px 10px;border-radius:12px;font-weight:600;">Operación cerrada · Ganadora ✓</span>'
                else:
                    result_badge = '<span style="background:#fef2f2;color:#dc2626;font-size:0.7rem;padding:3px 10px;border-radius:12px;font-weight:600;">Operación cerrada · Perdedora ✗</span>'
                r_mult = last_result.get("r_multiple", None)
                close_reason = last_result.get("close_reason", "")
                r_text = f" · R: {r_mult:.2f}" if pd.notna(r_mult) else ""
                reason_map = {"stop_loss": "SL", "take_profit_1": "TP1", "take_profit_2": "TP2", "breakeven": "BE", "time_exit": "Time exit"}
                reason_text = reason_map.get(close_reason, close_reason)
                result_detail = (
                    f'<div style="font-size:0.8rem;color:#64748b;margin-top:4px;">'
                    f'PnL: <strong style="color:{"#059669" if pnl_result > 0 else "#dc2626"}">'
                    f'${pnl_result:+,.2f}</strong>{r_text} · Cierre: {reason_text}</div>'
                )
        st.markdown(f"""
        <div class="signal-card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <div class="signal-dir {dir_class}" style="margin-bottom:4px;">
                        {dir_arrow} {ultima["direction"]}
                        <span style="font-size:0.85rem;font-weight:400;color:#64748b;margin-left:8px;">{ultima["rule"]}</span>
                    </div>
                    <div style="font-size:1rem;color:#475569;font-weight:500;">{ts_str}</div>
                    {result_detail}
                </div>
                <div style="text-align:right;">
                    {result_badge}
                    <div class="signal-prob" style="margin-top:8px;">ML: {ai_text}</div>
                </div>
            </div>
            <div class="signal-grid" style="margin-top:16px;">
                <div class="signal-item">
                    <div class="label">Entrada</div>
                    <div class="value">${ultima["entry_price"]:,.2f}</div>
                </div>
                <div class="signal-item">
                    <div class="label">Stop Loss</div>
                    <div class="value" style="color:#dc2626;">${ultima["stop_loss"]:,.2f}</div>
                </div>
                <div class="signal-item">
                    <div class="label">TP1</div>
                    <div class="value" style="color:#059669;">${ultima["tp1"]:,.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if len(df_results_live) > 0:
        df_closed = df_results_live
        if len(df_closed) > 0:
            st.markdown("""
            <div class="info-banner">
                <strong>Nota:</strong> Los resultados que se muestran a continuación no incluyen
                comisiones, spread ni slippage. Estos costes dependen del broker que utilices
                y reducen el rendimiento real entre un 0.1% y un 0.3% por operación.
            </div>
            """, unsafe_allow_html=True)
            primera_fecha = df_closed["open_time"].min()
            fecha_inicio_str = primera_fecha.strftime("%d %b %Y") if pd.notna(primera_fecha) else "—"
            st.markdown(
                f'<div class="section-title">'
                f'<span class="wr-info">Rendimiento live'
                f'<span class="info-icon">i</span>'
                f'<span class="wr-tooltip">Resultados reales de las señales emitidas por el modelo '
                f'en producción. Solo se cuentan trades completamente cerrados. '
                f'El sistema está activo desde {fecha_inicio_str}.</span></span>'
                f'<span style="font-size:0.75rem;color:#64748b;font-weight:400;margin-left:8px;">'
                f'desde {fecha_inicio_str}</span></div>',
                unsafe_allow_html=True)
            pnl_final = df_closed["pnl_pct"] if "pnl_pct" in df_closed.columns else (
                df_closed["pnl_usd"] / df_closed["entry_price"] * 100
            )
            wr_live = (pnl_final > 0).mean() * 100
            df_closed_adj = df_closed.copy()
            df_closed_adj["_pnl_real"] = pnl_final.values
            be_live, payoff_live = calcular_breakeven_payoff(df_closed_adj, pnl_col="_pnl_real")
            pnl_pct_total = pnl_final.sum()
            ejemplo_100 = 100 * (pnl_pct_total / 100)
            ejemplo_1btc = df_closed["pnl_usd"].sum()
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(metric_card("Trades cerrados", f"{len(df_closed)}"), unsafe_allow_html=True)
            c2.markdown(wr_metric_card(wr_live, breakeven=be_live, payoff=payoff_live), unsafe_allow_html=True)
            c3.markdown(metric_card(
                "PnL acumulado", f"{pnl_pct_total:+.2f}%",
                delta=f"Con $100 → ${ejemplo_100:+.2f}",
                delta_type="pos" if pnl_pct_total >= 0 else "neg",
                tooltip="Beneficio acumulado si hubieras entrado con $100 en cada trade."
            ), unsafe_allow_html=True)
            c4.markdown(metric_card(
                "PnL con 1 BTC", f"${ejemplo_1btc:+,.0f}",
                delta="Positivo" if ejemplo_1btc >= 0 else "Negativo",
                delta_type="pos" if ejemplo_1btc >= 0 else "neg",
                tooltip="Beneficio acumulado si hubieras comprado 1 BTC en cada operación."
            ), unsafe_allow_html=True)

            st.markdown('<div class="section-title">Calcula tu resultado</div>', unsafe_allow_html=True)
            st.markdown('<style>.calc-card div[data-testid="stNumberInput"] input { font-size:1.3rem!important; font-weight:700!important; }</style>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="calc-card" style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:20px;">', unsafe_allow_html=True)
                calc_col1, calc_col2 = st.columns([1, 2])
                with calc_col1:
                    capital_usuario = st.number_input("Tu inversión por trade ($)", min_value=1, max_value=1_000_000, value=100, step=10, key="calc_capital")
                with calc_col2:
                    pnl_usuario = capital_usuario * (pnl_pct_total / 100)
                    color_calc = C["win"] if pnl_usuario >= 0 else C["loss"]
                    sign_calc = "+" if pnl_usuario >= 0 else "-"
                    st.markdown(f"""
                    <div style="text-align:center;padding-top:8px;">
                        <div style="font-size:0.8rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Resultado con ${capital_usuario:,} por trade</div>
                        <div style="font-size:1.8rem;font-weight:700;color:{color_calc};line-height:1.2;">{sign_calc}${abs(pnl_usuario):,.2f}</div>
                        <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;">{len(df_closed)} operaciones · {pnl_pct_total:+.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    if len(df_live) > 0:
        st.markdown('<div class="section-title">Historial de señales</div>', unsafe_allow_html=True)
        df_display = df_live[["signal_time", "direction", "rule", "entry_price", "stop_loss", "tp1", "ai_proba"]].copy()
        monitor_start = df_results_live["open_time"].min() if len(df_results_live) > 0 else None

        # Órdenes limit canceladas (flujo limit jul-2026): señales cuya
        # orden no llegó a ejecutarse. Se cargan una vez para marcarlas.
        cancel_keys = set()
        _cancel_file = os.path.join(OUTPUT_DIR, "cancelled_orders.csv")
        if os.path.exists(_cancel_file):
            try:
                _dfc = pd.read_csv(_cancel_file)
                for _, _c in _dfc.iterrows():
                    cancel_keys.add((str(_c.get("direction", "")),
                                     round(float(_c.get("limit_price", 0)), 2)))
            except Exception:
                pass

        def obtener_resultado(row):
            # 1. ¿Hay un resultado CERRADO que casa con esta señal?
            #    (Nota: ya NO cortocircuitamos si no hay cerrados — ese
            #    era el bug que marcaba "Sin datos" a la señal abierta.)
            for _, res in df_results_live.iterrows():
                try:
                    res_entry = res.get("entry_price", 0)
                    sig_entry = row.get("entry_price", -1)
                    res_dir = res.get("direction", "")
                    sig_dir = row.get("direction", "")
                    if res_dir == sig_dir and abs(res_entry - sig_entry) < 0.01:
                        pnl_pct = res.get("pnl_pct", None)
                        if pd.notna(pnl_pct):
                            return f"{pnl_pct:+.2f}%"
                        pnl_usd = res.get("pnl_usd", 0)
                        entry = res.get("entry_price", 1)
                        pct = (pnl_usd / entry) * 100 if entry else 0
                        return f"{pct:+.2f}%"
                except Exception:
                    continue
            # 2. ¿Su orden limit se canceló (no se ejecutó)?
            try:
                if (str(row.get("direction", "")),
                        round(float(row.get("entry_price", 0)), 2)) in cancel_keys:
                    return "Cancelada"
            except Exception:
                pass
            # 3. Señales anteriores al inicio del seguimiento: sin datos.
            if monitor_start is not None:
                try:
                    if row["signal_time"] < monitor_start:
                        return "Sin datos"
                except Exception:
                    pass
            # 4. Emitida y aún sin cerrar → activa.
            return "Abierta"

        df_display["resultado"] = df_live.apply(obtener_resultado, axis=1)
        df_display.columns = ["Fecha", "Dir", "Regla", "Entrada", "Stop Loss", "TP1", "Prob. ML", "Resultado"]
        df_display["Fecha"] = df_display["Fecha"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M (Madrid)") if pd.notna(x) else "—")
        df_display["Prob. ML"] = df_display["Prob. ML"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
        df_display = df_display[["Fecha", "Resultado", "Dir", "Regla", "Entrada", "Stop Loss", "TP1", "Prob. ML"]]
        df_display = df_display.sort_values("Fecha", ascending=False)

        def colorear_resultado(val):
            if val.startswith("+"):
                return "color: #059669; font-weight: 600;"
            elif val.startswith("-"):
                return "color: #dc2626; font-weight: 600;"
            elif val == "Abierta":
                return "color: #4f46e5; font-weight: 600;"
            elif val == "Cancelada":
                return "color: #b45309; font-weight: 600;"
            else:
                return "color: #94a3b8;"

        styled = df_display.style.applymap(colorear_resultado, subset=["Resultado"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=300)


# ==========================================================
# HELPER — Cabecera de sección numerada
# ==========================================================
def section_header(num, title, subtitle=""):
    sub_html = f'<div class="sh-sub">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="step-header">
        <div class="sh-num">{num:02d}</div>
        <div>
            <div class="sh-title">{title}</div>
            {sub_html}
        </div>
    </div>
    """


# ==========================================================
# TAB 2: EL MODELO  (tab unificada)
# ==========================================================
def tab_modelo_combined():
    df_results = cargar_resultados_modelo()   # test set only (post 2024-12-31)
    df_all = cargar_trades()                  # backtest completo 5 años
    modelo = cargar_modelo()

    # ──────────────────────────────────────────────────────
    # SECCIÓN 1 — EL PIPELINE
    # ──────────────────────────────────────────────────────
    st.markdown(section_header(1, "El Pipeline",
        "De datos históricos a señal en vivo: 7 pasos automatizados"), unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline-grid">
        <div class="pipe-card c-blue">
            <div class="pc-num">PASO 01</div>
            <div class="pc-name">Descarga datos</div>
            <div class="pc-script">01_download_data.py</div>
            <div class="pc-desc">Descarga velas OHLCV de Binance para BTC/USDT en 1h y 4h. 5 años de historia.</div>
        </div>
        <div class="pipe-card c-blue">
            <div class="pc-num">PASO 02</div>
            <div class="pc-name">Procesar datos</div>
            <div class="pc-script">02_process_data.py</div>
            <div class="pc-desc">Calcula indicadores técnicos: EMA20, EMA50, RSI14, ATR14, swing highs/lows.</div>
        </div>
        <div class="pipe-card c-cyan">
            <div class="pc-num">PASO 03</div>
            <div class="pc-name">Reglas de entrada</div>
            <div class="pc-script">03_rules.py</div>
            <div class="pc-desc">Detecta señales de pullback y breakout. Calcula SL y TP. Filtra con 4H.</div>
        </div>
        <div class="pipe-card c-cyan">
            <div class="pc-num">PASO 04</div>
            <div class="pc-name">Backtest</div>
            <div class="pc-script">04_backtest.py</div>
            <div class="pc-desc">Simula cada trade vela a vela. Registra salida por SL, TP1 o tiempo (24h).</div>
        </div>
    </div>
    <div class="pipeline-grid" style="grid-template-columns: repeat(3,1fr);">
        <div class="pipe-card c-purple">
            <div class="pc-num">PASO 05</div>
            <div class="pc-name">Dataset ML</div>
            <div class="pc-script">06_prepare_dataset.py</div>
            <div class="pc-desc">Une señales con resultados. Añade 20 features. Define la etiqueta R≥0.5.</div>
        </div>
        <div class="pipe-card c-purple">
            <div class="pc-num">PASO 06</div>
            <div class="pc-name">Entrenar modelo</div>
            <div class="pc-script">07_train_model_v3.py</div>
            <div class="pc-desc">Entrena XGBoost con walk-forward en 7 folds. Guarda model.joblib y resultados test.</div>
        </div>
        <div class="pipe-card c-green">
            <div class="pc-num">PASO 07</div>
            <div class="pc-name">Señal live</div>
            <div class="pc-script">08_live_signal.py</div>
            <div class="pc-desc">Corre en el VPS cada hora. Evalúa la vela actual y envía señal a Telegram.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────
    # SECCIÓN 2 — REGLAS DE ENTRADA
    # ──────────────────────────────────────────────────────
    st.markdown(section_header(2, "Reglas de Entrada",
        "Dos patrones técnicos + filtro de tendencia 4H + SL/TP fijos"), unsafe_allow_html=True)

    st.markdown("""
    <div class="rule-grid">
        <div class="rule-card-lg">
            <div class="rule-badge">PULLBACK</div>
            <div class="rule-title-lg">Retroceso a la EMA20</div>
            <div class="rule-list">
                ▸ Precio retrocede hasta la EMA20 en 1H<br>
                ▸ EMA20 > EMA50 (tendencia alcista)<br>
                ▸ RSI entre 35 y 65 (zona media)<br>
                ▸ Vela de confirmación cierra arriba de la EMA20<br>
                ▸ Confluencia de tendencia 4H activa<br>
                ▸ Entrada: cierre de la vela de señal
            </div>
        </div>
        <div class="rule-card-lg green-rule">
            <div class="rule-badge green">BREAKOUT</div>
            <div class="rule-title-lg">Ruptura de swing structure</div>
            <div class="rule-list">
                ▸ Precio supera el máximo de las últimas 20 velas<br>
                ▸ Volumen superior a la media de 20 períodos<br>
                ▸ EMA20 > EMA50 (momentum alcista activo)<br>
                ▸ ATR en zona normal (sin volatilidad extrema)<br>
                ▸ Confluencia de tendencia 4H activa<br>
                ▸ Entrada: cierre de la vela de ruptura
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:4px;">
        <div class="callout-indigo callout">
            <strong>Stop Loss y Take Profit fijos</strong><br>
            SL = 1.25 × ATR14 (volatilidad de la vela de señal)<br>
            TP1 = 2.0R (el doble de la distancia al SL)<br>
            Cierre 100% de la posición al tocar TP1<br>
            Máximo 24 velas (24h) sin cierre → time exit
        </div>
        <div class="callout-info callout">
            <strong>Filtro de confluencia 4H</strong><br>
            Antes de confirmar una señal 1H, el sistema
            comprueba que la temporalidad 4H esté alineada:
            EMA20 4H > EMA50 4H para LONG (y viceversa para SHORT).
            Este filtro elimina señales a contratendencia.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────
    # SECCIÓN 3 — EL BACKTEST
    # ──────────────────────────────────────────────────────
    st.markdown(section_header(3, "El Backtest",
        "5 años de simulación vela a vela · Metodología conservadora"), unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-info callout" style="margin-bottom:12px;">
        El backtest recorre cada vela de 1H de manera independiente.
        Para cada señal, simula el trade mirando las velas siguientes una a una hasta que se toca
        SL, TP1 o pasan 24 horas. Si en la misma vela se tocan SL y TP, se asume el
        <strong>peor caso (SL)</strong>. Este conservadurismo asegura que las métricas
        no estén infladas.
    </div>
    """, unsafe_allow_html=True)

    if len(df_all) > 0:
        total_bt = len(df_all)
        wr_bt = (df_all["pnl_abs"] > 0).mean() * 100
        be_bt, payoff_bt = calcular_breakeven_payoff(df_all)

        fecha_ini = df_all["signal_time"].min()
        fecha_fin = df_all["signal_time"].max()
        fecha_ini_str = fecha_ini.strftime("%b %Y") if pd.notna(fecha_ini) else "—"
        fecha_fin_str = fecha_fin.strftime("%b %Y") if pd.notna(fecha_fin) else "—"

        n_sl = (df_all["close_reason"] == "stop_loss").sum()
        n_tp = (df_all["close_reason"] == "take_profit_1").sum()
        n_te = (df_all["close_reason"] == "time_exit").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card(
            "Trades simulados", f"{total_bt:,}",
            delta=f"{fecha_ini_str} → {fecha_fin_str}"
        ), unsafe_allow_html=True)
        c2.markdown(metric_card(
            "Win Rate bruto", f"{wr_bt:.1f}%",
            delta=f"Sin filtro ML · BE {be_bt:.0f}%",
            delta_type="pos" if wr_bt > be_bt else "neg"
        ), unsafe_allow_html=True)
        c3.markdown(metric_card(
            "Payoff ratio", f"{payoff_bt:.2f}",
            delta="Ganancia / Pérdida media",
        ), unsafe_allow_html=True)
        c4.markdown(metric_card(
            "Cierre por TP1", f"{n_tp/total_bt*100:.1f}%",
            delta=f"SL {n_sl/total_bt*100:.1f}% · Time {n_te/total_bt*100:.1f}%"
        ), unsafe_allow_html=True)

        # PnL mensual backtest
        st.markdown('<div class="section-title" style="margin-top:24px;">PnL mensual — backtest completo (sin filtro ML)</div>', unsafe_allow_html=True)
        df_m = df_all.copy()
        df_m["mes"] = df_m["signal_time"].dt.to_period("M").astype(str)
        monthly = df_m.groupby("mes").agg(pnl=("pnl_abs", "sum"), trades=("pnl_abs", "count")).reset_index()
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Bar(
            x=monthly["mes"], y=monthly["pnl"],
            marker_color=[C["win"] if x > 0 else C["loss"] for x in monthly["pnl"]],
            hovertemplate="<b>%{x}</b><br>PnL: $%{y:,.0f}<extra></extra>",
        ))
        plotly_layout(fig_bt, height=260)
        fig_bt.update_layout(showlegend=False, xaxis_title="", yaxis_title="PnL ($)")
        st.plotly_chart(fig_bt, use_container_width=True)

    # ──────────────────────────────────────────────────────
    # SECCIÓN 4 — DATASET Y ETIQUETA
    # ──────────────────────────────────────────────────────
    st.markdown(section_header(4, "Dataset y Etiqueta",
        "Cómo se construye el conjunto de entrenamiento y qué aprende el modelo"), unsafe_allow_html=True)

    # Train/test split visual
    st.markdown("""
    <p style="font-size:0.85rem;color:#475569;margin-bottom:8px;">
        Los 5 años de backtest se dividen en <strong>entrenamiento</strong> (hasta 31 dic 2024)
        y <strong>test</strong> (a partir de 1 ene 2025). El modelo <em>nunca</em> ha visto los
        datos de test durante el entrenamiento — son una muestra completamente nueva.
    </p>
    <div class="split-bar-outer">
        <div class="split-train" style="width:80%;">
            <span class="split-label">TRAIN — ~4 años (2020–2024)</span>
        </div>
        <div class="split-test" style="width:20%;">
            <span class="split-label">TEST — ~1 año (2025–2026)</span>
        </div>
    </div>
    <div class="split-legend">
        <div class="split-legend-item">
            <div class="split-dot" style="background:#4f46e5;"></div>
            Entrenamiento: el modelo aprende patrones aquí
        </div>
        <div class="split-legend-item">
            <div class="split-dot" style="background:#059669;"></div>
            Test: evaluación honesta — datos no vistos
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Label explanation
    st.markdown("""
    <p style="font-size:0.85rem;color:#475569;margin:16px 0 8px 0;">
        <strong>¿Por qué la etiqueta es R≥0.5 y no simplemente "ganó dinero"?</strong>
        Un trade que cierra por time_exit con +0.01% "ganó" pero es poco predecible y casi
        sin valor. El modelo necesita aprender a identificar trades con convicción real —
        los que alcanzan al menos la mitad del recorrido hasta el Take Profit.
    </p>
    <div class="lc-grid">
        <div class="lc-card">
            <div class="lc-r" style="color:#059669;">+1.8R</div>
            <div class="lc-detail">Trade que alcanza TP1<br>Cierre limpio a 2.0R</div>
            <div class="lc-badges">
                <span class="lc-badge pass">pnl &gt; 0 ✓</span>
                <span class="lc-badge pass">R ≥ 0.5 ✓</span>
            </div>
            <div class="lc-label one">Label = 1</div>
        </div>
        <div class="lc-card">
            <div class="lc-r" style="color:#d97706;">+0.1R</div>
            <div class="lc-detail">Time exit con mínima<br>ganancia (24h sin TP/SL)</div>
            <div class="lc-badges">
                <span class="lc-badge pass">pnl &gt; 0 ✓</span>
                <span class="lc-badge fail">R ≥ 0.5 ✗</span>
            </div>
            <div class="lc-label zero">Label = 0</div>
        </div>
        <div class="lc-card">
            <div class="lc-r" style="color:#dc2626;">−0.9R</div>
            <div class="lc-detail">Stop Loss alcanzado<br>Pérdida del riesgo completo</div>
            <div class="lc-badges">
                <span class="lc-badge fail">pnl &gt; 0 ✗</span>
                <span class="lc-badge fail">R ≥ 0.5 ✗</span>
            </div>
            <div class="lc-label zero">Label = 0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────
    # SECCIÓN 5 — MODELO XGBOOST
    # ──────────────────────────────────────────────────────
    st.markdown(section_header(5, "Modelo XGBoost",
        "Configuración, features y cómo interpreta la probabilidad"), unsafe_allow_html=True)

    st.markdown("""
    <div class="xgb-grid">
        <div class="xgb-card"><div class="xgb-val">200</div><div class="xgb-key">n_estimators</div></div>
        <div class="xgb-card"><div class="xgb-val">4</div><div class="xgb-key">max_depth</div></div>
        <div class="xgb-card"><div class="xgb-val">0.05</div><div class="xgb-key">learning_rate</div></div>
        <div class="xgb-card"><div class="xgb-val">0.8</div><div class="xgb-key">subsample</div></div>
        <div class="xgb-card"><div class="xgb-val">20</div><div class="xgb-key">min_child_weight</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Walk-forward validation visual
    st.markdown("""
    <p style="font-size:0.85rem;color:#475569;margin:16px 0 10px 0;">
        <strong>Walk-forward validation (7 folds)</strong> — cada fold usa todos los datos
        hasta ese punto como entrenamiento, y evalúa en el siguiente período.
        Esto simula cómo el modelo se comportaría realmente en producción.
    </p>
    <div style="margin-bottom:16px;">
        <div class="wf-row">
            <div class="wf-fold-label">Fold 1</div>
            <div class="wf-bar-outer">
                <div class="wf-bar-train" style="width:40%;">Train 2020–2022</div>
                <div class="wf-bar-test" style="width:10%;">Test</div>
            </div>
        </div>
        <div class="wf-row">
            <div class="wf-fold-label">Fold 2</div>
            <div class="wf-bar-outer">
                <div class="wf-bar-train" style="width:50%;">Train 2020–2022 Q3</div>
                <div class="wf-bar-test" style="width:10%;">Test</div>
            </div>
        </div>
        <div class="wf-row">
            <div class="wf-fold-label">Fold 3</div>
            <div class="wf-bar-outer">
                <div class="wf-bar-train" style="width:60%;">Train 2020–2023</div>
                <div class="wf-bar-test" style="width:10%;">Test</div>
            </div>
        </div>
        <div class="wf-row">
            <div class="wf-fold-label">Fold 4</div>
            <div class="wf-bar-outer">
                <div class="wf-bar-train" style="width:66%;">Train 2020–2023 Q2</div>
                <div class="wf-bar-test" style="width:10%;">Test</div>
            </div>
        </div>
        <div class="wf-row">
            <div class="wf-fold-label">Fold 5</div>
            <div class="wf-bar-outer">
                <div class="wf-bar-train" style="width:72%;">Train 2020–2023 Q4</div>
                <div class="wf-bar-test" style="width:10%;">Test</div>
            </div>
        </div>
        <div class="wf-row">
            <div class="wf-fold-label">Fold 6</div>
            <div class="wf-bar-outer">
                <div class="wf-bar-train" style="width:80%;">Train 2020–2024 Q2</div>
                <div class="wf-bar-test" style="width:10%;">Test</div>
            </div>
        </div>
        <div class="wf-row">
            <div class="wf-fold-label">Fold 7</div>
            <div class="wf-bar-outer">
                <div class="wf-bar-train" style="width:88%;">Train 2020–2024</div>
                <div class="wf-bar-test" style="width:10%;">Test 2025</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probabilidad baja ≠ sistema malo
    st.markdown("""
    <div class="callout-success callout">
        <strong>¿Por qué el modelo puede ser rentable con probabilidades bajas?</strong><br>
        El modelo da una probabilidad continua 0–1, no una decisión binaria. Incluso con
        40 trades ganadores de 100, el sistema puede ser rentable porque cada ganador vale
        ~2R y cada perdedor cuesta ~1R. Con TP de 2.0R y SL de 1.0R:<br><br>
        <strong>40 × (+2R) − 60 × (−1R) = +80R − 60R = +20R neto</strong><br><br>
        La probabilidad del modelo no es un "porcentaje de victoria histórico" directo —
        es una puntuación de calidad de la señal. Umbral más alto = menos trades, mayor
        calidad media. Tú decides dónde filtrar según tu tolerancia al riesgo.
    </div>
    """, unsafe_allow_html=True)

    # Probability dots visual
    st.markdown("""
    <div class="prob-dots-row">
        <div>
            <div style="font-size:0.72rem;color:#64748b;margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">100 trades al azar</div>
            <div class="prob-dots">
    """ + "".join(['<div class="prob-dot win"></div>' for _ in range(40)])
    + "".join(['<div class="prob-dot loss"></div>' for _ in range(60)])
    + """
            </div>
            <div style="display:flex;gap:12px;margin-top:8px;font-size:0.72rem;color:#64748b;">
                <span><span style="display:inline-block;width:10px;height:10px;background:#059669;border-radius:2px;margin-right:4px;"></span>40 ganadores (+2R)</span>
                <span><span style="display:inline-block;width:10px;height:10px;background:#dc2626;border-radius:2px;margin-right:4px;opacity:0.65;"></span>60 perdedores (−1R)</span>
            </div>
        </div>
        <div class="prob-dots-label" style="max-width:300px;">
            <strong>Resultado: +20R neto</strong><br>
            40 × 2R = 80R ganados<br>
            60 × 1R = 60R perdidos<br>
            <strong style="color:#059669;">Balance = +20R</strong>
            <br><br>
            El sistema es rentable <em>no</em> por acertar
            más del 50%, sino porque cuando acierta
            gana el doble de lo que pierde cuando falla.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="section-title">Importancia de features</div>', unsafe_allow_html=True)
    if modelo and hasattr(modelo, "feature_importances_"):
        importances = modelo.feature_importances_
        feat_names = [f[0] for f in FEATURES_DISPLAY]
        feat_labels = [f[1] for f in FEATURES_DISPLAY]
        feat_descs = [f[2] for f in FEATURES_DISPLAY]
        idx_sorted = np.argsort(importances)[::-1]
        max_imp = importances[idx_sorted[0]]
        bars_html = ""
        for rank, i in enumerate(idx_sorted[:10]):
            pct = importances[i] / max_imp * 100
            bars_html += f"""
            <div class="feat-row">
                <div class="feat-name" title="{feat_descs[i]}">{feat_labels[i]}</div>
                <div class="feat-bar-bg"><div class="feat-bar" style="width:{pct:.0f}%"></div></div>
                <div class="feat-val">{importances[i]:.3f}</div>
            </div>
            """
        st.markdown(bars_html, unsafe_allow_html=True)
        with st.expander("Ver las 20 features completas"):
            feat_df = pd.DataFrame([{
                "#": rank + 1, "Feature": feat_labels[i],
                "Nombre técnico": feat_names[i],
                "Descripción": feat_descs[i],
                "Importancia": f"{importances[i]:.4f}",
            } for rank, i in enumerate(idx_sorted)])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
    else:
        st.info("No se pudo cargar el modelo para mostrar importancia de features.")

    # Distribución de probabilidades
    st.markdown('<div class="section-title">Distribución de predicciones (test set)</div>', unsafe_allow_html=True)
    fig_dist = go.Figure()
    df_win = df_results[df_results["pnl_abs"] > 0]
    df_loss = df_results[df_results["pnl_abs"] <= 0]
    fig_dist.add_trace(go.Histogram(x=df_win["model_proba"], nbinsx=40, name="Ganadores", marker_color=C["win"], opacity=0.7))
    fig_dist.add_trace(go.Histogram(x=df_loss["model_proba"], nbinsx=40, name="Perdedores", marker_color=C["loss"], opacity=0.7))
    plotly_layout(fig_dist, height=300)
    fig_dist.update_layout(barmode="overlay", xaxis_title="Probabilidad ML", yaxis_title="Trades")
    st.plotly_chart(fig_dist, use_container_width=True)

    # ──────────────────────────────────────────────────────
    # SECCIÓN 6 — RENDIMIENTO EN TEST
    # ──────────────────────────────────────────────────────
    st.markdown(section_header(6, "Rendimiento en Test",
        "Métricas reales sobre datos no vistos por el modelo durante el entrenamiento"), unsafe_allow_html=True)

    st.markdown("""
    <div class="test-banner">
        <div class="tb-icon">🔬</div>
        <div>
            <div class="tb-title">DATOS DE TEST — Período post 01/01/2025</div>
            <div class="tb-text">
                Todas las métricas de esta sección corresponden exclusivamente al <strong>conjunto de test</strong>:
                datos posteriores al 31 de diciembre de 2024 que el modelo nunca vio durante el entrenamiento.
                Esto refleja el rendimiento real esperado en producción.
                Los datos de entrenamiento (2020–2024) tienen métricas más optimistas por definición — no se muestran.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    umbral = st.slider(
        "Umbral de probabilidad ML",
        min_value=0.20, max_value=0.70, value=0.35, step=0.05,
        help="Filtra trades por probabilidad. Umbral más alto = menos trades, mayor calidad media.",
        key="modelo_umbral"
    )

    df_f = df_results[df_results["model_proba"] >= umbral].sort_values("open_time").copy()

    if len(df_f) == 0:
        st.warning("Sin trades con ese umbral en el test set.")
    else:
        wr = (df_f["pnl_abs"] > 0).mean() * 100
        pnl = df_f["pnl_abs"].sum()
        r_medio = df_f["r_multiple"].mean()
        be_mod, payoff_mod = calcular_breakeven_payoff(df_f)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Trades test", f"{len(df_f)}"), unsafe_allow_html=True)
        c2.markdown(wr_metric_card(wr, breakeven=be_mod, payoff=payoff_mod), unsafe_allow_html=True)
        c3.markdown(metric_card("PnL Total", f"${pnl:,.0f}",
            delta="Positivo" if pnl > 0 else "Negativo",
            delta_type="pos" if pnl > 0 else "neg"), unsafe_allow_html=True)
        c4.markdown(metric_card("R Medio", f"{r_medio:.2f}"), unsafe_allow_html=True)

        # Tabla rendimiento por umbral
        st.markdown('<div class="section-title">Rendimiento por umbral</div>', unsafe_allow_html=True)
        filas = []
        for u in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            f = df_results[df_results["model_proba"] >= u]
            if len(f) == 0:
                continue
            n = len(f)
            gan = f[f["pnl_abs"] > 0]
            perd = f[f["pnl_abs"] < 0]
            wr_u = len(gan) / n * 100
            avg_win = abs(gan["pnl_abs"].mean()) if len(gan) > 0 else 0
            avg_loss = abs(perd["pnl_abs"].mean()) if len(perd) > 0 else 1
            payoff_u = avg_win / avg_loss if avg_loss > 0 else 0
            be_wr = (1 / (1 + payoff_u) * 100) if payoff_u > 0 else 50.0
            margen_u = wr_u - be_wr
            gross_p = gan["pnl_abs"].sum() if len(gan) > 0 else 0
            gross_l = abs(perd["pnl_abs"].sum()) if len(perd) > 0 else 1
            pf_u = gross_p / gross_l if gross_l > 0 else 0
            cum = f["pnl_abs"].cumsum()
            dd_u = (cum - cum.cummax()).min()
            filas.append({
                "Umbral": f"{u:.2f}",
                "Trades": n,
                "Win Rate": f"{wr_u:.1f}%",
                "Payoff": f"{payoff_u:.2f}",
                "PnL": f"${f['pnl_abs'].sum():,.0f}",
                "PnL/Trade": f"${f['pnl_abs'].mean():,.0f}",
                "Margen": f"{margen_u:+.1f}pp",
                "PF": f"{pf_u:.2f}",
                "MaxDD": f"${dd_u:,.0f}",
                "R Medio": f"{f['r_multiple'].mean():.2f}",
            })
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

        # Equity curve
        st.markdown('<div class="section-title">Equity Curve</div>', unsafe_allow_html=True)
        df_f["equity"] = df_f["pnl_abs"].cumsum()
        df_f["equity_peak"] = df_f["equity"].cummax()
        df_f["drawdown"] = df_f["equity"] - df_f["equity_peak"]
        pnl_series = df_f["pnl_abs"]

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=df_f["open_time"], y=df_f["equity"], mode="lines", name="Equity",
            line=dict(color=C["win"], width=2), fill="tozeroy",
            fillcolor="rgba(5,150,105,0.08)",
            hovertemplate="Equity: $%{y:,.0f}<extra></extra>",
        ))
        fig_eq.add_trace(go.Scatter(
            x=df_f["open_time"], y=df_f["equity_peak"], mode="lines", name="ATH",
            line=dict(color=C["muted"], width=1, dash="dot"),
        ))
        plotly_layout(fig_eq, height=380)
        fig_eq.update_layout(title=dict(
            text=f"Equity Curve — ML ≥ {umbral} ({len(df_f)} trades · test 2025–2026)",
            font=dict(size=14)))
        st.plotly_chart(fig_eq, use_container_width=True)

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df_f["open_time"], y=df_f["drawdown"], fill="tozeroy", mode="lines",
            line=dict(color=C["loss"], width=1), fillcolor="rgba(220,38,38,0.1)",
            hovertemplate="DD: $%{y:,.0f}<extra></extra>",
        ))
        plotly_layout(fig_dd, height=180)
        fig_dd.update_layout(title=dict(text="Drawdown", font=dict(size=13)))
        st.plotly_chart(fig_dd, use_container_width=True)

        # Rachas
        st.markdown('<div class="section-title">Rachas y resistencia</div>', unsafe_allow_html=True)
        max_win, max_loss = calcular_rachas(pnl_series)
        max_dd = df_f["drawdown"].min()
        be_eq, payoff_eq = calcular_breakeven_payoff(df_f)

        c_wr1, c_wr2 = st.columns(2)
        c_wr1.markdown(wr_metric_card(wr, breakeven=be_eq, payoff=payoff_eq), unsafe_allow_html=True)
        c_wr2.markdown(metric_card("Payoff Ratio", f"{payoff_eq:.2f}",
            delta=f"BE = {be_eq:.0f}%", delta_type="pos" if payoff_eq >= 1 else "neg",
            tooltip="Ganancia media / pérdida media. Mayor que 1 → ganas más de lo que pierdes por trade."
        ), unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;">
            <div class="streak-badge">
                <span class="num" style="color:{C['win']}">{max_win}</span>
                <span class="desc">Mejor racha<br>ganadora</span>
            </div>
            <div class="streak-badge">
                <span class="num" style="color:{C['loss']}">{max_loss}</span>
                <span class="desc">Peor racha<br>perdedora</span>
            </div>
            <div class="streak-badge">
                <span class="num" style="color:{C['loss']}">${max_dd:,.0f}</span>
                <span class="desc">Máximo<br>drawdown</span>
            </div>
            <div class="streak-badge">
                <span class="num" style="color:{C['text']}">{len(pnl_series):,}</span>
                <span class="desc">Trades<br>test</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Duración de trades
        st.markdown('<div class="section-title">Distribución de duraciones (velas por trade)</div>', unsafe_allow_html=True)
        fig_dur = go.Figure()
        fig_dur.add_trace(go.Histogram(
            x=df_all["bars_in_trade"], nbinsx=50,
            marker_color=C["accent"], opacity=0.8,
            hovertemplate="Duración: %{x} velas<br>Trades: %{y}<extra></extra>"
        ))
        plotly_layout(fig_dur, height=240)
        fig_dur.update_layout(xaxis_title="Velas en el trade", yaxis_title="Trades")
        st.plotly_chart(fig_dur, use_container_width=True)
        mediana = df_all["bars_in_trade"].median()
        p90 = df_all["bars_in_trade"].quantile(0.9)
        st.markdown(f"""
        <div class="info-banner">
            <strong>Duración típica:</strong> La mitad de los trades se cierran en {mediana:.0f} velas o menos.
            El 90% se cierran en {p90:.0f} velas o menos. Una vela = 1 hora.
        </div>
        """, unsafe_allow_html=True)

        # Kelly
        st.markdown('<div class="section-title">Criterio de Kelly — Gestión de capital</div>', unsafe_allow_html=True)
        n_trades_k = len(pnl_series)
        wins_k = (pnl_series > 0).sum()
        losses_k = (pnl_series < 0).sum()
        wr_k = wins_k / n_trades_k if n_trades_k > 0 else 0
        avg_win_k = pnl_series[pnl_series > 0].mean() if wins_k > 0 else 0
        avg_loss_k = abs(pnl_series[pnl_series < 0].mean()) if losses_k > 0 else 1
        payoff_k = avg_win_k / avg_loss_k if avg_loss_k > 0 else 0
        kelly_full = ((wr_k * payoff_k) - (1 - wr_k)) / payoff_k if payoff_k > 0 else 0
        kelly_full = max(kelly_full, 0)
        kelly_half = kelly_full / 2
        kelly_quarter = kelly_full / 4
        if "entry_price" in df_f.columns and "stop_loss" in df_f.columns:
            sl_pct_medio = (abs(df_f["entry_price"] - df_f["stop_loss"]) / df_f["entry_price"]).mean() * 100
        else:
            sl_pct_medio = 1.5
        capital_ejemplo = 1000
        riesgo_half = capital_ejemplo * kelly_half
        posicion_half = riesgo_half / (sl_pct_medio / 100) if sl_pct_medio > 0 else 0

        st.markdown(f"""
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:10px 0;">
            <div class="streak-badge"><span class="num" style="color:{C['text']}">{payoff_k:.2f}</span><span class="desc">Payoff ratio<br>(gan./pérd. media)</span></div>
            <div class="streak-badge"><span class="num" style="color:{C['accent']}">{kelly_full:.1%}</span><span class="desc">Kelly<br>completo</span></div>
            <div class="streak-badge"><span class="num" style="color:{C['win']}">{kelly_half:.1%}</span><span class="desc">Half-Kelly<br>(recomendado)</span></div>
            <div class="streak-badge"><span class="num" style="color:{C['muted']}">{kelly_quarter:.1%}</span><span class="desc">Quarter-Kelly<br>(conservador)</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-banner">
            <strong>¿Qué es el criterio de Kelly?</strong>
            Es una fórmula que calcula el porcentaje de tu capital que deberías arriesgar en cada
            operación para maximizar el crecimiento a largo plazo. Se calcula a partir del Win Rate
            ({wr_k:.1%}) y del Payoff Ratio ({payoff_k:.2f}:1).
            <br><br>
            El Kelly completo ({kelly_full:.1%}) es matemáticamente óptimo pero agresivo.
            En la práctica se usa <strong>Half-Kelly ({kelly_half:.1%})</strong>.
            <br><br>
            <strong>Ejemplo con ${capital_ejemplo:,} de capital:</strong>
            con Half-Kelly arriesgarías ${riesgo_half:.0f} por trade.
            Con un SL medio de ~{sl_pct_medio:.1f}%, tu posición sería de ~${posicion_half:,.0f} en BTC.
        </div>
        """, unsafe_allow_html=True)

        # Monte Carlo
        st.markdown('<div class="section-title">Simulación Monte Carlo — Probabilidad de ruina</div>', unsafe_allow_html=True)
        N_SIMS = 10_000
        CAPITAL_INICIAL = 1000
        RUINA_PCT = 0.50
        pnl_pct_arr = df_f["pnl_pct"].values
        np.random.seed(42)
        n_trades_mc = len(pnl_pct_arr)
        max_drawdowns = np.zeros(N_SIMS)
        final_equities = np.zeros(N_SIMS)
        ruinas = 0
        for sim in range(N_SIMS):
            shuffled = np.random.permutation(pnl_pct_arr)
            equity = CAPITAL_INICIAL
            peak = equity
            max_dd_pct = 0.0
            ruined = False
            for pnl_pct_val in shuffled:
                equity *= (1 + pnl_pct_val / 100)
                if equity > peak:
                    peak = equity
                dd_pct = (peak - equity) / peak * 100
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
                if not ruined and equity <= CAPITAL_INICIAL * (1 - RUINA_PCT):
                    ruined = True
            max_drawdowns[sim] = max_dd_pct
            final_equities[sim] = equity
            if ruined:
                ruinas += 1
        prob_ruina = ruinas / N_SIMS * 100
        dd_median = np.median(max_drawdowns)
        dd_95 = np.percentile(max_drawdowns, 95)
        equity_median = np.median(final_equities)
        equity_p10 = np.percentile(final_equities, 10)
        equity_p90 = np.percentile(final_equities, 90)
        retorno_median = (equity_median / CAPITAL_INICIAL - 1) * 100
        color_ruina = C["win"] if prob_ruina < 5 else (C["accent"] if prob_ruina < 15 else C["loss"])

        st.markdown(f"""
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:10px 0;">
            <div class="streak-badge"><span class="num" style="color:{color_ruina}">{prob_ruina:.1f}%</span><span class="desc">Prob. de ruina<br>(&gt;{RUINA_PCT:.0%} pérdida)</span></div>
            <div class="streak-badge"><span class="num" style="color:{C['loss']}">{dd_95:.1f}%</span><span class="desc">Max DD<br>(peor 5% casos)</span></div>
            <div class="streak-badge"><span class="num" style="color:{C['win']}">{retorno_median:+.1f}%</span><span class="desc">Retorno mediano<br>({n_trades_mc} trades)</span></div>
            <div class="streak-badge"><span class="num" style="color:{C['text']}">{N_SIMS:,}</span><span class="desc">Simulaciones<br>realizadas</span></div>
        </div>
        """, unsafe_allow_html=True)

        fig_mc_dd = go.Figure()
        fig_mc_dd.add_trace(go.Histogram(x=max_drawdowns, nbinsx=80, marker_color=C["loss"], opacity=0.7, hovertemplate="DD: %{x:.1f}%<br>Sims: %{y}<extra></extra>"))
        fig_mc_dd.add_vline(x=dd_median, line_dash="dash", line_color=C["accent"], annotation_text=f"Mediana: {dd_median:.1f}%")
        fig_mc_dd.add_vline(x=dd_95, line_dash="dash", line_color=C["loss"], annotation_text=f"P95: {dd_95:.1f}%")
        plotly_layout(fig_mc_dd, height=240)
        fig_mc_dd.update_layout(title=dict(text="Distribución de Max Drawdown (10.000 simulaciones)", font=dict(size=13)), xaxis_title="Max Drawdown (%)", yaxis_title="Simulaciones")
        st.plotly_chart(fig_mc_dd, use_container_width=True)

        fig_mc_eq = go.Figure()
        fig_mc_eq.add_trace(go.Histogram(x=final_equities, nbinsx=80, marker_color=C["win"], opacity=0.7, hovertemplate="Capital: $%{x:,.0f}<br>Sims: %{y}<extra></extra>"))
        fig_mc_eq.add_vline(x=equity_median, line_dash="dash", line_color=C["accent"], annotation_text=f"Mediana: ${equity_median:,.0f}")
        fig_mc_eq.add_vline(x=CAPITAL_INICIAL, line_dash="dash", line_color=C["loss"], annotation_text="Capital inicial")
        plotly_layout(fig_mc_eq, height=240)
        fig_mc_eq.update_layout(title=dict(text=f"Capital final simulado (inicio: ${CAPITAL_INICIAL:,})", font=dict(size=13)), xaxis_title="Capital final ($)", yaxis_title="Simulaciones")
        st.plotly_chart(fig_mc_eq, use_container_width=True)

        st.markdown(f"""
        <div class="info-banner">
            <strong>¿Qué es Monte Carlo?</strong>
            Cogemos los {n_trades_mc} trades del test y los barajamos al azar {N_SIMS:,} veces —
            como barajar una baraja de cartas. Cada vez el orden es diferente, simulando miles de
            "historias alternativas".
            <br><br>
            <strong>Resultados:</strong> En el <strong>{prob_ruina:.1f}%</strong> de los casos
            se perdió más del {RUINA_PCT:.0%} del capital. En el peor 5% de escenarios, el drawdown
            máximo fue {dd_95:.1f}%. La equity mediana terminó en <strong>${equity_median:,.0f}</strong>
            (rango probable: ${equity_p10:,.0f} — ${equity_p90:,.0f}).
        </div>
        """, unsafe_allow_html=True)

        # Detalle de trades (expandible)
        with st.expander(f"Ver detalle de trades test (umbral ≥ {umbral})"):
            cols = ["open_time", "direction", "rule", "entry_price", "close_reason", "pnl_abs", "r_multiple", "model_proba"]
            df_show = df_f[[c for c in cols if c in df_f.columns]].copy()
            df_show.columns = [c.replace("_", " ").title() for c in df_show.columns]
            df_show = df_show.sort_values("Open Time", ascending=False)
            st.dataframe(df_show, use_container_width=True, hide_index=True, height=350)

    # ──────────────────────────────────────────────────────
    # SECCIÓN 7 — EN PRODUCCIÓN
    # ──────────────────────────────────────────────────────
    st.markdown(section_header(7, "En Producción",
        "Cómo fluye una señal desde el mercado hasta Telegram"), unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 40px 1fr 40px 1fr 40px 1fr 40px 1fr;align-items:center;gap:0;margin:12px 0;">
        <div class="prod-step">
            <div class="ps-icon">📡</div>
            <div class="ps-name">Binance API</div>
            <div class="ps-desc">Descarga vela 1H más reciente al cierre de cada hora</div>
        </div>
        <div class="prod-arrow">→</div>
        <div class="prod-step">
            <div class="ps-icon">🔢</div>
            <div class="ps-name">Calcular features</div>
            <div class="ps-desc">Aplica EMA, RSI, ATR y las 20 features del modelo</div>
        </div>
        <div class="prod-arrow">→</div>
        <div class="prod-step">
            <div class="ps-icon">📏</div>
            <div class="ps-name">Reglas técnicas</div>
            <div class="ps-desc">¿Pullback o breakout? ¿Filtra el 4H? ¿Hay señal?</div>
        </div>
        <div class="prod-arrow">→</div>
        <div class="prod-step">
            <div class="ps-icon">🧠</div>
            <div class="ps-name">Modelo ML</div>
            <div class="ps-desc">XGBoost asigna probabilidad 0–1 a la señal</div>
        </div>
        <div class="prod-arrow">→</div>
        <div class="prod-step">
            <div class="ps-icon">📲</div>
            <div class="ps-name">Telegram</div>
            <div class="ps-desc">Señal + prob + SL + TP1 enviados al canal privado</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px;">
        <div class="callout-indigo callout">
            <strong>VPS — 24/7</strong><br>
            El bot corre en un servidor en Núremberg.
            Tres servicios activos: señal live (1H), monitor de posiciones (5 min)
            y auto-restart con watchdog.
        </div>
        <div class="callout-warn callout">
            <strong>Disclaimer importante</strong><br>
            BTC Signal Engine es un sistema informativo. <em>No ejecuta operaciones</em>.
            No es asesoramiento financiero. Los resultados pasados no garantizan
            rendimiento futuro.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────
    # SECCIÓN 8 — REENTRENAMIENTO
    # ──────────────────────────────────────────────────────
    st.markdown(section_header(8, "Reentrenamiento",
        "Cuándo y cómo se actualiza el modelo"), unsafe_allow_html=True)

    st.markdown("""
    <div class="rt-grid">
        <div class="rt-card">
            <div class="rt-num">PASO 01</div>
            <div class="rt-name">Detectar señal</div>
            <div class="rt-desc">~6 meses de nuevos datos en producción. El win rate o R-medio cae de forma sostenida.</div>
        </div>
        <div class="rt-card">
            <div class="rt-num">PASO 02</div>
            <div class="rt-name">Extender dataset</div>
            <div class="rt-desc">Descargar datos nuevos y ejecutar el pipeline completo (01→06) con la ventana ampliada.</div>
        </div>
        <div class="rt-card">
            <div class="rt-num">PASO 03</div>
            <div class="rt-name">Reentrenar y validar</div>
            <div class="rt-desc">Mismo código 07_train_model_v3.py. Mover el cutoff de test. Validar walk-forward en nuevos folds.</div>
        </div>
        <div class="rt-card">
            <div class="rt-num">PASO 04</div>
            <div class="rt-name">Desplegar</div>
            <div class="rt-desc">Subir model.joblib y model_results_v3.csv al VPS vía git. Reiniciar servicios automáticamente.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-success callout">
        <strong>El pipeline se diseñó para ser reentrenable en &lt;30 minutos.</strong>
        Todos los parámetros están centralizados. Añadir nuevos activos (ETH, SOL)
        requiere solo crear una carpeta de datos y repetir el mismo flujo.
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# FOOTER
# ==========================================================
def render_footer():
    st.markdown("""
    <div class="footer">
        BTC Signal Engine &mdash; BTC/USDT Trading Intelligence &mdash;
        Modelo XGBoost v3 &mdash; Walk-forward validated<br>
        Los resultados pasados no garantizan rendimiento futuro.
        Esto no es asesoramiento financiero.
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# MAIN
# ==========================================================
def main():
    st.set_page_config(
        page_title="BTC Signal Engine — BTC Trading",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    aplicar_css()
    st_autorefresh(interval=AUTO_REFRESH_MS, key="auto_refresh")
    render_navbar()

    tab1, tab2 = st.tabs([
        "⚡  Señales Live",
        "🧠  El Modelo",
    ])

    with tab1:
        tab_live()
    with tab2:
        tab_modelo_combined()

    render_footer()


if __name__ == "__main__":
    main()
