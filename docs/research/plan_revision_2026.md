# Plan de revisión del modelo — Guion y bitácora

> Inicio: 3 julio 2026
> Documento vivo: aquí se marca en qué punto estamos y se registra cada decisión tomada.
> Análisis de partida: `docs/revision_modelo_fable.md`

---

## 📍 DÓNDE ESTAMOS AHORA

**Fase C ejecutada (4 jul): 7 experimentos, NINGUNO supera a BASE — el modelo actual ya extrae lo que los datos dan de sí. Hallazgo clave: la ventaja (+0.098R ± 0.120) está dentro del ruido estadístico con 150 trades OOS. El cuello de botella es DATOS, no modelo. Ver `docs/fase_c_resultados.md`. Decisiones abiertas para el autor: desplegar modelo limpio+limit al VPS, mensaje público, adelantar Fase D (pooling ETH/SOL).**

---

### 4 jul 2026 (2ª sesión autónoma: SOL + drift + dashboard + web)

**Veredicto SOL (el tercer test):** añadir SOL DEGRADA el pooled en ambas variantes (con/sin feature de símbolo): BTC en 0.50 pasa de +0.107R (BTC+ETH) a −0.025/−0.089R; el conjunto de 3 monedas es negativo en todos los umbrales. Interpretación honesta: la mejora del pooled-2 era frágil (parte suerte, parte que ETH y BTC son gemelos y SOL no); la generalización multi-activo NO está confirmada. **La única configuración que ha sobrevivido a todos los tests sigue siendo BTC-only @ 0.60 (+0.098R, 3/3 años).** Consecuencias: (a) recomendación de despliegue vuelve a BTC-only 0.60; (b) ETH/SOL como productos NO tienen respaldo aún (ambos negativos a umbrales de producto); (c) la Fase D como producto se pospone — como datos, seguirá útil cuando haya más historia u otro enfoque (p.ej. normalización por régimen).

**Decisiones del autor registradas:** VPS esperaba a SOL → con este veredicto, la propuesta activa es desplegar BTC-only 0.60 (pendiente de su OK + smoke test juntos). Web: solo "Selectivo" durante la validación (hecho en landing; pendiente ocultar Activo en el configurador). SOL descargado por el autor (43,817 velas, `download_symbol.py`).

**Entregado además:** monitor de drift `health_monitor/src/model_drift.py` (3 chequeos: distribución de probas vs referencia, WR rodante vs banda binomial, fill-rate vs 98.4%; cron semanal, alertas al canal técnico) — commit `fa68984` en su repo, rama `revision-2026`. Dashboard: banner de orden limit pendiente + fill-rate. Web: calculadora solo-Selectivo (commit `5fd1078`).

### 4 jul 2026 (sesión autónoma P1+P2+P3, autorizada)

**P3 — Fase D adelantada (pooling).** SOL no descargable desde este entorno (API Binance bloqueada; NO se rodeó la restricción — queda 1 comando para el autor). Pooling BTC+ETH (8,891 trades, pipeline idéntico validado: BTC reproduce exactamente los 4,453 conocidos; velas 4h de ETH por agregación exacta de 1h, verificada contra las nativas de BTC). Resultado clave con el MISMO walk-forward:

| Config | 0.45 | 0.50 | 0.60 |
|--------|------|------|------|
| BTC-only (referencia) | −0.018R | −0.027R | **+0.098R** (150 tr, 3/3 años) |
| Pooled+símbolo, eval BTC | **+0.045R** | **+0.107R** (171 tr, 3/4 folds+) | −0.141R |

Lectura: el pooling mejora justo donde la Fase C quería (0.45-0.50, más señales) y confirma la hipótesis "el cuello de botella son los datos". Sigue dentro del ruido individualmente (+0.107 ± 0.112) pero es la segunda config independiente en ~+0.1R. ETH por sí solo aún no es operable. Modelos guardados: `model_limit_btc.joblib` (rec. 0.60) y `model_limit_pooled.joblib` (rec. 0.50), ambos sklearn 1.5.2.

**P1 — Despliegue preparado (NO desplegado).** 08: mensaje "orden LIMIT válida 1h" + orden pendiente (`open_orders.json`) + bloqueo extendido + `MODEL_FILE` explícito. 10: convierte órdenes en posición al toque o cancela al expirar (con `cancelled_orders.csv` para medir fill-rate real). Test de paridad batch↔live: 20 features idénticas en 500 señales ✅. Ciclo de órdenes probado offline ✅. Guía: `docs/guia_despliegue_limit.md` (smoke test supervisado obligatorio).

**P2 — Web sincerada y coordinada.** Retirados TODOS los números del modelo contaminado (había celdas con 96-100% WR). Nuevas métricas = walk-forward limit neto, mismas fuentes que el motor (`walkforward_1h_limit.csv`). Curva de equity del hero ahora es la REAL (+14.7R, con su drawdown). Posicionamiento: transparencia radical ("también publicamos lo incómodo"). Señal descrita como orden limit. "XGBoost"→"modelo predictivo (ML)". Pendiente del autor: los nombres "Activo/Selectivo" (Activo fulltime queda en PF 0.97 — decisión de producto).

**Decisiones tomadas en autónomo (documentadas):** pooling sin SOL; PnL de la web expresado con base $10k arriesgando 1% ($100=1R); C6 sigue pospuesta; no se tocó el VPS.

## Guion del plan

Leyenda: ✅ hecho · 🔄 en curso · ⏸ pendiente · ❌ descartado

### FASE A — Corregir la vara de medir
*Objetivo: que el backtest cuente la verdad. Todo lo demás se decide con estos números.*

| Paso | Qué | Estado |
|------|-----|--------|
| A0 | Crear rama git `revision-2026` para trabajar sin tocar producción | ✅ (desde `feature/c1-supabase-routing`) |
| A1 | Corregir alineación 4h (lookahead) en `03_rules.py`, `06_prepare_dataset.py` y `08_live_signal.py` | ✅ commit `6722317` |
| A2 | Regenerar pipeline completo (señales → backtest → dataset) y medir cuánto cambia solo por H1 | ✅ commit `18ee228` |
| A3 | Añadir comisiones y slippage al backtest (0.05% + 0.02% por lado; PnL/R/labels en NETO — decisión del autor) | ✅ |
| A4 | Replicar en el backtest el bloqueo de solapamiento que ya tiene el live | ✅ |
| A5 | Reentrenar el modelo con los datos corregidos | ✅ adelantado (3 jul, sklearn 1.5.2 = versión VPS) |
| A6 | Analizar los números honestos y decidir umbrales (Activo/Selectivo pueden cambiar) | ⏸ |
| A7 | Desplegar al VPS (con OK explícito) y actualizar documentación/web si las métricas cambian | ⏸ |

### FASE B — Robustez
| Paso | Qué | Estado |
|------|-----|--------|
| B1 | Fijar versiones de librerías (resolver aviso sklearn 1.7.2 vs 1.5.2 del VPS) | ⏸ |
| B2 | Módulo único de features compartido entre pipeline histórico y live + test de paridad | ⏸ |
| B3 | Parametrizar el símbolo (config por instrumento, sin duplicar scripts) | ⏸ |
| B4 | Calibración de probabilidades (umbrales interpretables como probabilidad real) | ⏸ |
| B5 | Monitor de drift del modelo integrado en health_monitor | ⏸ |

### FASE C — Mejora del modelo
| Paso | Qué | Estado |
|------|-----|--------|
| C1 | Features nuevas por lotes (taker buy ratio, régimen de mercado, contexto diario, features firmadas) | ⏸ |
| C2 | Bake-off de algoritmos (GB actual vs XGBoost nativo vs LightGBM vs CatBoost) | ⏸ |
| C3 | Experimentos de etiquetado (pesos por magnitud de R) | ⏸ |
| C4 | Ensemble de semillas | ⏸ |

### FASE D — Expansión
| Paso | Qué | Estado |
|------|-----|--------|
| D1 | ETH sobre la base refactorizada, con features de contexto BTC | ⏸ |
| D2 | SOL ídem | ⏸ |
| D3 | Canales Telegram, /track y health_monitor para los nuevos instrumentos | ⏸ |

---

## Registro de decisiones

| Fecha | Decisión | Motivo |
|-------|----------|--------|
| 4 jul 2026 | Simulador limit: colocación al close de señal, ventana 1 vela, breakout también limit | Fill 98.4%, robusto en W=1/2/3, mensaje de producto simple. EMA20 descartada (fill 44-56%). Detalle en `a6_comparativa_ejecucion_variantes.md` |
| 4 jul 2026 | Variante 4h APARCADA (no descartada) | Filtro ML no fiable con ~700 muestras; ventaja bruta apagándose en 2025-26. Reintentar con pooling multi-activo en Fase D |
| 4 jul 2026 | Dirección recomendada: 1h + limit, Fase C para ensanchar ventaja | Único walk-forward positivo y consistente (0.60: +0.098R, 3/3 años) |
| 4 jul 2026 | Idea aparcada: producto long-only horizonte largo para públicos tipo Trade Republic | TR: solo market orders en cripto, sin shorts, spread ~1%, sin API — incompatible con el sistema actual |
| 3 jul 2026 (noche) | El autor: disposición total a migrar a la variante 4h si resulta más rentable — sin apego a la forma actual | Criterio para A6: se decide por rentabilidad honesta, no por trabajo ya invertido |
| 3 jul 2026 | Empezar por Fase A (corregir backtest) antes que añadir features o cambiar algoritmo | Cualquier experimento se mediría contra números inflados por el lookahead 4h (75.4% de señales afectadas) |
| 3 jul 2026 | No crear carpeta/repo nuevo; trabajar en rama git dentro de `trading_project_signal` | Mantener historial git y flujo de deploy al VPS intactos (ver explicación en bitácora) |

---

## Bitácora

### 4 jul 2026 (tarde) — Fase C en autónomo: 7 experimentos, veredicto honesto
- el autor aprobó la dirección 1h+limit y autorizó continuar en autónomo con la Fase C tal como estaba propuesta.
- Ejecutados contra el criterio predefinido: L1 firmadas+poda, L2 régimen, L3 flujo, C7 pesos, C5 HistGB ×2 y ensemble 5 semillas. **Ninguno adoptado** (detalle en `fase_c_resultados.md`).
- Hallazgo central: BASE 0.60 = +0.098R con error estándar ±0.120 → ventaja no distinguible de cero con n=150. El bottleneck es volumen de datos, no algoritmo.
- Decisiones sobre la marcha documentadas: C6 calibración pospuesta (no hay ventaja demostrada que calibrar); xgboost/lightgbm no instalables en el entorno → HistGB como proxy; C2b taker_buy_ratio pendiente de re-descarga de histórico.
- Camino que puede cambiar el juego: datos live limpios + pooling ETH/SOL (Fase D) → ×3 datos, reabre también la variante 4h.

### 4 jul 2026 — A6 en autónomo (autorizado): simulador limit + variante 4h + Fase C
- Rama pusheada a GitHub por el autor (`origin/revision-2026`).
- **A6.1** Grid de 18 escenarios de ejecución limit (`exp_limit_execution.py`). Ganador robusto: close/W1/breakout-limit.
- **A6.2** Validación end-to-end del ganador (`exp_limit_retrain.py`): etiquetas recalculadas 5 años, reentreno, walk-forward. 0.60 → +0.098R OOS, positivo 2023, 2024 y 2025 (2026: 5 trades, ruido). Split fijo más optimista (0.50 +0.283R) — nos fiamos del walk-forward.
- **A6.3** Variante 4h completa (`exp_variant_4h.py`): hipótesis del peaje confirmada (0.07R) pero filtro ML no generaliza (poco dato) y reglas apagándose en 2025-26 → aparcada.
- Entregables: `a6_comparativa_ejecucion_variantes.md` (análisis + decisiones documentadas) y `propuesta_fase_c.md` (plan detallado con criterio de éxito predefinido).
- Dudas resueltas sin consultar (modo excepcional de hoy), todas documentadas en la comparativa: regla de fill conservadora, SL/TP desde precio de ejecución, probas provisionales del grid resueltas con reentreno end-to-end, folds del walk-forward propios.


### 3 jul 2026 (noche) — A4 + walk-forward + escenarios de fricción (ejecutado en autónomo con permiso del autor)

**A4 — Bloqueo de solapamiento (réplica exacta del live).** Verificado en `08_live_signal.py`: el bloqueo ocurre ANTES del filtro ML — toda señal emitida abre posición y bloquea las siguientes hasta su cierre, aunque no pase umbral. Replicado secuencialmente sobre los 4,499 trades: solo el 44% de las señales llegan a emitirse (1,961). Guardado en `outputs/trades_secuencial_live.csv`. Test 2025-26 con bloqueo (neto): 0.45 → 165 trades, WR 39.4%, −$13.0K; 0.50 → 107 trades, −$5.7K. Supuesto documentado: señales de la misma vela se emiten juntas (así lo hace live); el dataset de entrenamiento mantiene todas las señales (el bloqueo afecta a la cartera, no a la calidad individual de cada señal).

**Walk-forward con datos netos (4 folds, 2023-2026, 3,008 predicciones out-of-sample).** Negativo en TODOS los umbrales (mejor caso 0.55: −$1.1K). El diagnóstico de A3 no es un artefacto del test set: se sostiene en todas las ventanas temporales. Dato útil: reentrenar periódicamente mejora vs modelo fijo (+52% en el periodo comparable), aunque no basta para volverse rentable.

**Escenarios de fricción (para A6).** Recalculado el peaje por trade según ejecución (test 2025-26, sin bloqueo):

| Umbral | S0 actual (taker, 0.14%) | S1 entrada limit/maker | S2 maker ideal |
|--------|--------------------------|------------------------|----------------|
| 0.45 | −$11.9K (−0.03R) | **+$10.8K (+0.10R)** | +$20.7K (+0.16R) |
| 0.50 | −$0.7K (+0.05R) | **+$12.8K (+0.19R)** | +$18.2K (+0.24R) |

S1 = entrada con orden limit (maker 0.02%, sin slippage) + TP1 limit; SL/time-exit a mercado. AVISO: asume que la limit siempre se ejecuta (optimista — en la realidad se perderían algunos trades, incluidos buenos). Conclusión provisional: **el estilo de ejecución puede cambiar el signo del sistema**; es la palanca más prometedora para A6, combinable con mejora del modelo (Fase C).

### 3 jul 2026 — A3 ejecutado: fricción realista (0.14% por trade) — resultado crítico
- Decisiones del autor: perfil futuros (taker 0.05%/lado) + slippage (0.02%/lado); PnL, R-multiple y labels en NETO.
- Backtest completo (5 años, sin filtro ML): PF pasa de 0.99 a **0.78**; PnL −374K.
- Test 2025-26 con el modelo reentrenado (labels netos):

| Umbral | Trades | WR | PnL neto | R medio |
|--------|--------|-----|----------|---------|
| sin filtro | 1167 | 36.4% | −$118.7K | — |
| 0.45 | 318 | 41.2% | −$11.9K | −0.034 |
| 0.50 | 185 | 44.3% | −$0.7K | +0.049 |
| 0.60 | 43 | 30.2% | −$14.3K | −0.373 |

- **Diagnóstico:** la ventaja bruta del sistema (~+0.09R/trade sin filtro, +0.26R en 0.45) es menor que el peaje (~0.14R/trade). Con costes honestos, ningún umbral es rentable en el test; el mejor queda en breakeven. El problema no es un bug: es que el margen de la estrategia es demasiado fino para su horizonte (1h, TP 2R con riesgo ~1%).
- Vías posibles a discutir en A6: reconstruir ventaja con features limpias (Fase C), reducir fricción real (entradas limit/maker: 0.02% y sin slippage, natural en la regla pullback), horizonte mayor (velas 4h → costes relativos menores), o combinación.
- Contexto que da esperanza moderada: el live sin filtro hizo +8.65% en mayo-jun (54 trades, muestra pequeña); y el test set es un único periodo — el walk-forward sobre datos netos dirá más.

### 3 jul 2026 — A5 adelantado: reentreno con dataset limpio (decisión del autor)
- Entrenado con sklearn 1.5.2 (la versión del VPS → modelo compatible con producción, adelanta parte de B1).
- **Tabla de umbrales honesta vs la antigua (test 2025-26):**

| Umbral | Antes (contaminado) | Ahora (limpio) |
|--------|--------------------|----------------|
| 0.45 "Activo" | 364 tr · WR 54.1% · +$132K | 336 tr · WR 42.6% · +$42.6K |
| 0.50 | 261 tr · WR 57.1% · +$113K | 204 tr · WR 46.1% · +$31.7K |
| 0.60 "Selectivo" | 112 tr · WR 67.0% · +$73K | 46 tr · WR 37.0% · **−$5.2K** |

- **Conclusión dura:** gran parte de la "ventaja" del modelo era el lookahead. El modelo limpio SÍ filtra (0.45: R medio +0.256 vs +0.086 sin filtro; PnL/trade +$127 vs +$33), pero mucho menos de lo que creíamos. El umbral 0.60 deja de ser viable (46 trades, PnL negativo).
- **Permutation importance reveladora:** `rsi_14_4h` (antes feature #1) ahora resta (−0.021 en test) — su gloria era la fuga de información. Las features con señal real: `volume_sma20_ratio`, `body_pct`, `roc_5`, `rsi_14`.
- Pendiente: estos números aún NO incluyen comisiones (A3) ni bloqueo de solapamiento (A4) → bajarán más.
- Implicación producto: los nombres "Activo/Selectivo" y las métricas de la web tendrán que redefinirse en A6.
- El VPS sigue corriendo el modelo contaminado → decidir en A6/A7 cuándo sustituirlo.

### 3 jul 2026 — A2 ejecutado: impacto del fix medido
- Backup de outputs antiguos en `outputs/archive_pre_fix_2026/`.
- De regalo: optimizado `04_backtest.py` (solo pasa las 24 velas que necesita cada trade; misma lógica, de minutos a 2 segundos).
- **Señales:** 4,574 → 4,499. Solo 78 desaparecen y 3 aparecen (~1.7%). El filtro `trend_4h` es lento (EMAs de 4h) y rara vez cambia por mirar una vela antes. Buena noticia: el motor de reglas era casi inmune al lookahead.
- **Backtest sin filtro ML:** casi igual (test 2025-26: PnL +42.3K → +38.0K, WR 38.0% → 37.7%, PF 1.08 → 1.07).
- **Features del modelo:** aquí está el daño. `rsi_14_4h`, `ema_spread_4h_pct` y `atr_pct_4h` cambian de valor en el **75% de las filas** (rsi_14_4h: cambio medio 2.8 puntos, máximo 26). Los labels no cambian (0%).
- **Conclusión:** el problema del lookahead no estaba en las señales sino en lo que el modelo "veía". El modelo actual aprendió con features contaminadas → hay que reentrenar para saber si la tabla de umbrales (WR 54-67%) sobrevive.

### 3 jul 2026 — A0 + A1 ejecutados
- Rama `revision-2026` creada desde `feature/c1-supabase-routing` (el estado que corre en el VPS). Producción intacta.
- Fix del lookahead aplicado (commit `6722317`):
  - `03_rules.py` y `06_prepare_dataset.py`: el merge 4h→1h ahora empareja por `close_time` de la vela 4h ≤ momento de decisión (cierre de la vela 1h). Solo velas 4h cerradas.
  - `08_live_signal.py`: se descarta la vela 4h en formación que devuelve Binance antes de calcular indicadores.
- Verificación en memoria (sin regenerar outputs): 0 filas con futuro tras el fix (antes 3,448 de 4,574 señales). 150 filas con NaN en `trend_4h` = calentamiento inicial de 2021, sin efecto.
- Outputs (`signals.csv`, `trades.csv`, `ml_dataset.csv`, modelo) AÚN reflejan la versión antigua → siguiente paso A2.

### 3 jul 2026 — Arranque de la revisión
- Revisión completa entregada (`revision_modelo_fable.md`): 6 hallazgos (H1-H6), respuestas a las 12 preguntas, plan por fases.
- Verificado con datos reales: 3,448 de 4,574 señales (75.4%) usan una vela 4h que cierra 1-3 horas después del momento de decisión.
- el autor aprueba empezar el plan. Pendiente: aprobar A0+A1.
