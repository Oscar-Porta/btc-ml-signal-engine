# BTC ML Signal Engine

An end-to-end research and production system for **BTC/USDT trading signals on 1-hour candles**: rule-based entries, a gradient-boosting filter, realistic cost modelling, walk-forward validation, and a 24/7 live engine with Telegram notifications and limit-order tracking.

> **Status: archived research project.** The system ran in production on a VPS and has since been decommissioned. This repository is published as a portfolio piece. **It is not financial advice and it does not execute trades** — it only emits informational signals.

---

## Why this project is interesting

Most retail trading backtests look great and fail live. This project's main value is the **audit that found out why**, and the honest numbers that came after it:

| Issue found | Effect on the original backtest | Fix |
|---|---|---|
| **Look-ahead bias** in multi-timeframe features: 1h signals were joined to the 4h candle *still in progress*, whose indicators include up to 3 hours of future prices (75% of signals affected) | The top feature (4h RSI) was mostly leaked information; win rates of 54–67% were inflated | Join only on 4h candles already **closed** at decision time (`close_time <= decision_time`), in both backtest and live code |
| **No trading costs** | ~0.14R per trade of friction ignored | Taker fee + slippage per side; labels computed on **net** R-multiples |
| **Overlapping trades** in the backtest, while live trading allows one position at a time | Backtest PnL unreachable by a real follower | Sequential simulation replicating the live blocking rule |
| **Test set reused for selection** (thresholds and TP/SL chosen on it) | Optimistic estimates | Decisions moved to walk-forward validation |

After the fixes, the original strategy with market orders was **unprofitable at every threshold** (gross edge ~0.09R < cost ~0.14R). The edge was recovered by changing the **execution**, not the model: placing a **limit order at the signal price** (maker fee, no slippage) fills 98.4% of the time and cuts friction to ~0.05R.

## Results (walk-forward, out-of-sample, net of costs)

4 expanding folds, 2023 → 2026, model retrained on each fold using only past data.

| Configuration | Threshold | OOS trades | Win rate | Mean net R / trade | Yearly consistency |
|---|---|---|---|---|---|
| Market orders (original) | 0.55 | 163 | 39.9% | −0.05 | negative |
| **Limit orders (final)** | **0.60** | **150** | **42.7%** | **+0.098** | **2023, 2024, 2025 positive** |
| 4h-candle variant | none | 604 | 38.4% | +0.059 | fades in 2025–26 |

Final configuration: profit factor 1.15, max drawdown −15.1R. The win rate is below 50% by design: winners are +2R and losers −1R.

**Statistical honesty:** +0.098R has a standard error of ±0.120 with n = 150, so the edge is **not statistically distinguishable from zero** on historical data alone. Live validation was the intended final judge.

### What did not work (and is documented)

- **7 model-improvement experiments** (direction-signed features, market-regime features — efficiency ratio, ADX, Bollinger width —, order-flow features, |R| sample weights, histogram GBM, class weights, 5-seed ensemble): none beat the 20-feature baseline. The bottleneck was data volume, not the algorithm.
- **Multi-asset pooling:** adding ETH improved BTC predictions at lower thresholds (+0.107R at 0.50), but adding SOL degraded everything. Not confirmed.
- **4h timeframe:** lower relative costs, but too few samples (~700) for the filter to generalise.

## How it works

```
Binance OHLCV (1h + 4h)
   └─ 02 indicators: EMA20/50, RSI14, ATR14, 20-bar swing high/low
       └─ 03 rules (4h trend filter, closed candles only)
             • Pullback to EMA20 in trend
             • Breakout of 20-bar structure
           └─ 04 backtest: SL 1.25×ATR · TP 2R · 24h time exit · worst-case intrabar · net costs
               └─ 06 dataset: 20 features (momentum, volatility regime, candle shape, 4h context…)
                   └─ 07 GradientBoostingClassifier + walk-forward
                       └─ 08 live engine (hourly): signal → limit order → Telegram
                           10 position monitor: fill / expiry / SL / TP / time exit
                           09 Streamlit dashboard
```

The live engine and the training pipeline compute features identically; `src/experiments/test_paridad_features.py` checks batch/live parity on 500 historical signals.

## Repository layout

```
src/
  01_download_data.py … 07_walk_forward.py   research pipeline
  08_live_signal.py                           hourly live engine (limit-order flow, Telegram routing)
  09_dashboard.py                             Streamlit dashboard
  10_position_monitor.py                      order/position lifecycle monitor
  experiments/                                limit-execution simulator, retraining, pooling, 4h variant, Phase C
  tools/                                      TP/SL optimiser, data download, metric export
deploy/                                       systemd units and VPS setup script
docs/research/                                research notes and decision log (Spanish)
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # integrations are optional

python src/01_download_data.py  # public Binance data, no API key needed
python src/02_indicators.py
python src/03_rules.py
python src/04_backtest.py
python src/06_prepare_dataset.py
python src/07_train_model_v3.py
python src/experiments/exp_limit_retrain.py   # final limit-execution model + walk-forward
```

`scikit-learn` is pinned to 1.5.2: serialized models must be loaded with the same version they were trained with.

## Stack

Python · pandas · NumPy · scikit-learn · python-binance · Streamlit · Telegram Bot API · Supabase · systemd on a Linux VPS. Companion repos: `signal-dashboard-web` (Next.js front end) and `vps-health-monitor` (daily health checks and model-drift monitoring).

## Notes

- Research notes in `docs/research/` are in Spanish. Documents written before July 2026 contain **pre-correction metrics** (inflated by the look-ahead bias described above); the decision log `plan_revision_2026.md` explains each correction.
- Market data and trained models are not included; the pipeline regenerates them.
