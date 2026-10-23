# Propuesta Fase C — Mejora del modelo (variante 1h + ejecución limit)

> Fecha: 4 julio 2026 · Pendiente de aprobación del autor
> Punto de partida: walk-forward 0.60 → +0.098R, 150 trades OOS en 3 años, 3/3 años positivos
> Objetivo: ensanchar la ventaja para que el umbral 0.45-0.50 sea rentable (más señales/año) y que aguante en régimen lateral

## Criterio de éxito (definido ANTES de experimentar, para no autoengañarnos)

Un experimento se adopta solo si, en el walk-forward de 4 folds con ejecución limit y PnL neto: (a) mejora el R medio en los umbrales objetivo (0.45-0.50) sin romper el 0.60, y (b) ningún fold pasa de positivo a claramente negativo. El test se mira UNA vez por lote, no por prueba (lección H4).

**Meta de la fase:** R medio ≥ +0.10R en un umbral con ≥150 trades/año OOS. Si tras C1-C6 no se alcanza, evaluamos honestamente si el producto es viable con ~50 trades/año en 0.60 o replanteamos.

## Experimentos, por orden de prioridad

**C1 — Features de régimen de mercado** (ataca la debilidad #1: lateralidad). Kaufman Efficiency Ratio a 20 y 50 velas, ADX 4h, ancho de Bandas de Bollinger normalizado, pendiente de EMA50 (velocidad de tendencia), días desde el último cruce de EMAs 4h. Coste: bajo. Expectativa: media-alta.

**C2 — Features de flujo, coste cero** (ya vienen en las klines de Binance y las tiramos): taker buy ratio (presión compradora real), number_of_trades relativo a su media. Requiere re-descargar guardando esas columnas (cambio menor en `01_download_data.py`). Coste: bajo. Expectativa: media-alta — son ortogonales a todo lo que tiene el modelo.

**C3 — Features firmadas por dirección**: roc_5, roc_20, dist_ema20/50, body_pct multiplicadas por (+1 LONG / −1 SHORT). Unifica los patrones espejo de long y short — equivale a duplicar la densidad de datos. Sustituyen a las versiones sin firmar. Coste: muy bajo. Expectativa: media.

**C4 — Poda de features muertas**: fuera `atr_14` absoluto (no estacionaria, proxy de época) y revisar `rsi_14_4h`/`ema_spread_4h_pct` tras C1 (la permutation importance ya las señala). Menos ruido = mejor generalización con 4,400 muestras. Coste: muy bajo.

**C5 — Bake-off de algoritmos**: GB actual vs XGBoost nativo vs LightGBM vs CatBoost, con early stopping y las mismas features/folds. También scale_pos_weight y ensemble de 5-10 semillas del ganador. Coste: medio. Expectativa: baja-media (mejora operativa segura: entrenos 10-50× más rápidos).

**C6 — Calibración + umbrales por valor esperado**: isotonic/Platt sobre validación temporal. Con TP 2R, SL 1R y peaje limit ~0.05R, el breakeven está en p_real ≈ 0.35 → los umbrales dejan de ser números mágicos y "Activo/Selectivo" se redefinen como niveles de esperanza de ganancia. Coste: bajo. No mejora el ranking: mejora el producto y la monitorización.

**C7 (opcional) — Etiquetado**: sample_weight = |R| en la clasificación; si no aporta, regresión sobre R winsorizado. Coste: bajo. Expectativa: incierta.

## Método de trabajo

Lotes de 1-2 experimentos → walk-forward → decisión documentada en la bitácora → siguiente lote. Nada se acumula sin validar. Al final de la fase: reentreno definitivo, tabla honesta final, y decisión de despliegue (A7) contigo.

## Qué NO haremos (y por qué)

SMOTE/datos sintéticos (rompen la estructura temporal), meta-modelo de régimen (sobreingeniería con 4,400 muestras — el régimen entra como features), redes neuronales (peores que GBM en tabular pequeño), tocar el VPS antes de tener el modelo final validado.

## Después de Fase C (recordatorio del plan general)

Fase B pendiente se integra al desplegar: módulo único de features con test de paridad batch/live (crítico: la ejecución limit exige reescribir la lógica de envío/monitor del live), versiones fijadas, monitor de drift. Luego Fase D (ETH/SOL) — que además multiplicará el dataset para reintentar la variante 4h con pooling.
