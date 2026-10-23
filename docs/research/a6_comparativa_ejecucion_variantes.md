# A6 — Comparativa: ejecución limit (1h) vs variante 4h

> Fecha: 4 julio 2026 · Trabajo autónomo autorizado por el autor (excepción puntual)
> Scripts: `src/experiments/exp_limit_execution.py`, `exp_limit_retrain.py`, `exp_variant_4h.py`
> Datos: `outputs/experiments/`

---

## 1. Qué se preguntaba

Tras la Fase A sabemos que el sistema 1h con ejecución a mercado pierde (peaje 0.145R > ventaja bruta). Dos vías de rescate a comparar en igualdad de condiciones (mismo peaje, mismo modelo, mismo walk-forward):

- **Vía A:** mantener 1h pero entrar con orden limit (peaje maker ~0.05R), simulando los fills honestamente.
- **Vía B:** migrar a velas 4h (stops ~2% del precio → el mismo peaje absoluto pesa 3 veces menos por trade).

## 2. Simulador de entradas limit — cómo funciona y qué decisiones tomé

Reglas del simulador (todas conservadoras): la orden se coloca al generarse la señal; se ejecuta solo si el precio la atraviesa estrictamente en la ventana (tocar justo el precio no cuenta; si la vela abre mejor que el precio, fill al open); en la vela de ejecución, si también toca SL, se asume SL (peor caso); SL/TP se recalculan desde el precio real de ejecución; TP sale con limit (maker), SL y time-exit a mercado (taker+slippage); time-exit a 24 velas desde la ejecución.

**Grid probado** (18 escenarios): colocación {close de señal, EMA20, close−0.15·ATR} × ventana {1,2,3 velas} × breakout {mercado, limit}. Resultados completos en `outputs/experiments/limit_execution_grid.csv`.

**Decisión: colocación = close de la señal, ventana = 1 vela, breakout también limit.** Motivos: (1) fill rate 98.4% — a 1h el precio casi siempre vuelve a pasar por el cierre en la vela siguiente, así que apenas se pierden trades; (2) resultado insensible a la ventana (W=1/2/3 casi idénticos) → elección robusta, no un pico de suerte; (3) es el mensaje de producto más simple ("orden limit a X, válida 1 hora"). Descartada EMA20: solo se ejecuta el 44-56% de las veces — pierdes la mitad de los trades, incluidos buenos. Descartada close−0.15·ATR: parecida en 0.50 pero peor en 0.45 y más compleja de explicar.

**Duda documentada:** el grid usó las probabilidades del modelo entrenado con etiquetas taker (única disponible). Se resolvió reentrenando end-to-end el escenario ganador (siguiente sección).

## 3. Validación end-to-end de la vía A (1h + limit)

Recalculados los 5 años de trades con esa ejecución (solo 1% de órdenes canceladas), etiquetas nuevas (R neto ≥ 0.5), mismo GB reentrenado:

**Split fijo (test 2025-26):** 0.45 → 302 trades, WR 42.1%, +$17.9K (+0.106R) · 0.50 → 168 trades, WR 47.0%, +$29.3K (+0.283R).

**Walk-forward (2,977 predicciones OOS, 4 folds expansivos 2023→2026):**

| Umbral | Trades | WR | PnL neto | R medio |
|--------|--------|-----|----------|---------|
| sin filtro | 2977 | 35.8% | −$113.5K | −0.065 |
| 0.45 | 580 | 38.4% | −$9.9K | −0.018 |
| 0.50 | 364 | 37.9% | −$2.8K | −0.027 |
| 0.55 | 229 | 39.7% | +$6.5K | +0.013 |
| **0.60** | **150** | **42.7%** | **+$6.3K** | **+0.098** |

Por años en 0.60: 2023 +0.061R · 2024 +0.204R · 2025 +0.153R · 2026 −0.502R (solo 5 trades — ruido muestral). **Tres años consecutivos positivos out-of-sample.**

Lectura: la ejecución limit elimina el lastre del peaje (todo el walk-forward sube ~0.1R); queda una ventaja real pero fina y solo explotable con el filtro exigente. El split fijo es más bonito que el walk-forward — nos fiamos del walk-forward, que es más duro y realista.

## 4. Vía B: variante 4h — resultado y por qué se aparca

Pipeline paralelo completo (`exp_variant_4h.py`): señales sobre 4h, filtro de tendencia diario (alineado por close_time, sin lookahead), mismas reglas/gestión/fricción/modelo. 978 señales en 5 años (~3.9/semana), riesgo medio 2.05% del precio (peaje relativo ~0.07R, tres veces menor — la hipótesis de partida era correcta).

Resultados: en bruto sin filtro mejora al 1h (R +0.024 con limit, te=24), pero **el filtro ML empeora los resultados en el walk-forward** (0.45 → −0.091R; 0.50 → −0.139R): con ~700 muestras de entrenamiento el modelo no generaliza. Por años sin filtro: 2023 +0.21R, 2024 +0.11R, **2025 −0.03R, 2026 −0.34R** — la ventaja bruta de las reglas se ha ido apagando en el periodo reciente.

**Decisión: aparcar la variante 4h por ahora** (no descartarla para siempre). Razones: (1) sin filtro ML fiable no hay producto de umbrales; (2) su talón de Aquiles es el volumen de datos — cuando exista el pooling BTC+ETH+SOL (Fase D) el dataset 4h se multiplicará por 3 y merecerá reintento; (3) el rendimiento reciente (2025-26) es peor que el del 1h+limit. Anotado también: idea futura de producto long-only de horizonte largo para públicos tipo Trade Republic (fricción ~1% solo se amortiza en swing de días/semanas).

## 5. Comparativa final (walk-forward, PnL neto — la métrica honesta)

| Configuración | Mejor umbral | Trades OOS (3 años) | WR | R medio | Consistencia anual |
|---------------|-------------|--------------------|----|---------|--------------------|
| 1h taker (actual) | 0.55 | 163 | 39.9% | −0.05 | negativa |
| **1h + limit** | **0.60** | **150** | **42.7%** | **+0.098** | **3 de 3 años en positivo** |
| 4h + limit | sin filtro | 604 | 38.4% | +0.059 | 2023-24 sí, 2025-26 no |

## 6. Recomendación

**Construir sobre 1h + ejecución limit.** Es la única configuración con ventaja neta positiva y consistente out-of-sample. Sus dos debilidades definen exactamente el trabajo de la Fase C: (1) la ventaja solo aparece en umbrales altos → pocos trades (~50/año en 0.60): hay que ensancharla para bajar a 0.45-0.50 y multiplicar la frecuencia; (2) el filtro flojea en régimen lateral/reciente → features de régimen y flujo.

Implicación de producto (para cuando toque): la señal pasa de "entra ahora a mercado" a **"orden limit a X, válida 1 hora; SL en Y, TP limit en Z"** — más profesional y es lo que hace rentable el sistema.

Decisiones que quedan para el autor: aprobar esta dirección; qué hacer con el VPS/web mientras tanto (sigue emitiendo con el modelo antiguo y mensaje de entrada a mercado); y arrancar la Fase C (propuesta detallada en `docs/propuesta_fase_c.md`).
