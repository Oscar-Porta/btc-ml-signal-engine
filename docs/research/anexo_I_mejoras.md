# Anexo I — Mejoras iterativas post-producción

> Documento que recoge las mejoras, refinamientos y correcciones
> realizadas sobre el sistema BTC/USDT ya en producción (Fases 0-15).
> Complementa el documento base `presentation_notes.md`.

---

## Mejora 1 — Badge de operación cerrada en dashboard

**Fecha:** 23 de abril de 2026

**Problema:** Cuando una operación se cerraba, la tarjeta de señal en el
tab Live no indicaba que ya estaba cerrada ni su resultado. Creaba
confusión al entrar al dashboard después del cierre.

**Solución:** Se añadió un badge visible ("Operación cerrada · Ganadora ✓"
o "Operación cerrada · Perdedora ✗") junto con el PnL final en %,
R-multiple y motivo de cierre.

**Archivos:** `src/09_dashboard.py`

---

## Mejora 2 — SL a breakeven después de TP1

**Fecha:** 23 de abril de 2026

**Problema:** Cuando el precio alcanzaba TP1 y se hacía cierre parcial
del 50%, el Stop Loss del 50% restante se mantenía en el nivel original.
Esto significaba que una operación que ya era ganadora (tocó TP1) podía
terminar en pérdida si el precio se revertía.

**Solución:** Tras alcanzar TP1, el SL se mueve automáticamente al precio
de entrada (breakeven). El 50% restante busca TP2 sin riesgo de pérdida.
Este comportamiento ya estaba implementado en el backtest (`04_backtest.py`,
línea 170), así que las métricas históricas ya lo reflejaban. Solo faltaba
en el monitor live.

**Cambios:**
- `10_position_monitor.py`: al detectar TP1, se guarda `original_stop_loss`
  y se actualiza `stop_loss` al `entry_price`.
- `09_dashboard.py`: la tarjeta de operación en curso muestra
  "SL = Entrada (protegido)" cuando `tp1_hit` es True.
- Mensaje Telegram de cierre parcial: incluye aviso de que el SL se ha
  movido a entrada y que la operación está protegida.

---

## Mejora 3 — Bloqueo de señales con posición abierta

**Fecha:** 23 de abril de 2026

**Problema:** El sistema podía emitir múltiples señales de entrada
mientras ya había una operación abierta. Esto creaba confusión en
Telegram (¿qué señal se cerró?) y en el dashboard (varias posiciones
simultáneas).

**Solución:** Antes de emitir cualquier señal nueva, el script comprueba
si existe alguna posición abierta en `open_positions.json`. Si la hay,
la señal se ignora y se loguea. Una posición a la vez por instrumento.
Diseñado pensando en multi-instrumento: cuando se añadan ETH y SOL,
cada uno podrá tener una posición abierta simultáneamente, pero nunca
dos del mismo par.

**Archivos:** `src/08_live_signal.py` (paso 5B en `ejecutar_comprobacion`)

---

## Mejora 4 — Rediseño del mensaje de Telegram (entrada)

**Fecha:** 23 de abril de 2026

**Problema:** El mensaje original tenía mucha información mezclada sin
jerarquía clara. El riesgo se mostraba en USD absolutos calculados para
1 BTC, lo cual no era útil para el usuario. La probabilidad de la IA
mencionaba "umbral 0.55" que nadie entendía.

**Solución:** Mensaje reestructurado en bloques claros:
1. **Cabecera:** dirección + regla + hora (prominente)
2. **Niveles:** entrada, SL, TP1, TP2
3. **Riesgo:** solo en % + ejemplo con 100€
4. **IA:** "X% de confianza" sin mencionar umbrales internos
5. **Contexto:** RSI, ATR, trends
6. **Enlace al dashboard**

**Archivos:** `src/08_live_signal.py` (función `formatear_mensaje`)

---

## Mejora 5 — Rediseño del mensaje de Telegram (cierre)

**Fecha:** 23 de abril de 2026

**Problema:** El mensaje de cierre mostraba PnL en USD absolutos (1 BTC)
y no indicaba claramente la hora de apertura de la operación cerrada.

**Solución:** Mensaje con PnL en % + ejemplo 100€, hora de apertura
visible, y cuando es cierre parcial (TP1) incluye sección "Siguiente fase"
explicando que el SL se movió a entrada y que el 50% restante busca TP2.

**Archivos:** `src/10_position_monitor.py` (función `notificar_cierre`)

---

## Mejora 6 — Columna "Resultado" en historial de señales

**Fecha:** 23 de abril de 2026

**Problema:** La tabla de historial de señales en el tab Live no mostraba
si cada señal había sido ganadora o perdedora.

**Solución:** Se cruzan las señales de `live_signals_log.csv` con los
resultados de `position_results.csv`. Si hay resultado cerrado, se
muestra el % de ganancia/pérdida. Si la señal está abierta, muestra
"Abierta". Si no hay datos, muestra "—".

**Archivos:** `src/09_dashboard.py` (sección historial en `tab_live`)

---

## Mejora 7 — Eliminación de vista sin filtro IA en Equity

**Fecha:** 23 de abril de 2026

**Problema:** El tab Equity mostraba dos vistas: "Backtest completo
(sin filtro IA)" y "Solo trades filtrados por IA". La primera mostraba
señales sin filtrar (WR ~27%) que no eran representativas del sistema
real y confundían al usuario.

**Solución:** Se eliminó la vista sin filtro IA. Solo queda la vista
filtrada por el modelo, con selector de umbral, equity curve, drawdown
y rachas. Es lo que el usuario realmente ve en producción.

**Archivos:** `src/09_dashboard.py` (función `tab_equity`)

---

## Mejora 8 — Textos explicativos en cada tab del dashboard

**Fecha:** 23 de abril de 2026

**Problema:** Cada tab del dashboard mostraba datos sin contexto.
Un usuario nuevo no sabía qué estaba viendo ni por qué importaba.

**Solución:** Se añadió un banner explicativo al inicio de cada tab:
- **Live:** qué son las señales, cómo se generan, qué significa la
  operación en curso.
- **Historial:** qué es el filtro IA y por qué importa.
- **Modelo:** qué es el XGBoost, cómo se validó, qué significa R≥0.5.
- **Equity:** qué es la equity curve, el drawdown y las rachas.

Tono profesional-técnico pero accesible para novatos.

**Archivos:** `src/09_dashboard.py` (inicio de cada función tab_*)

---

## Mejora 9 — Disclaimer de comisiones en dashboard

**Fecha:** 26 de abril de 2026

**Problema:** Las métricas del backtest y equity no reflejaban que cada
broker cobra comisiones, spread y slippage diferentes.

**Solución:** Se añadió un texto informativo en el banner del tab Equity
indicando que los resultados son brutos y que las comisiones típicas
(0.1%-0.3% por operación) reducen el rendimiento real.

**Archivos:** `src/09_dashboard.py` (banner de `tab_equity`)

---

## Mejora 10 — Criterio de Kelly en dashboard

**Fecha:** 26 de abril de 2026

**Problema:** El usuario no tenía una guía cuantitativa para decidir
cuánto capital arriesgar por trade.

**Solución:** Sección en el tab Equity que calcula Kelly completo,
Half-Kelly (recomendado) y Quarter-Kelly (conservador) a partir del
Win Rate y Payoff Ratio del sistema filtrado por IA. Incluye ejemplo
práctico con 1.000€ y el SL medio del sistema.

**Archivos:** `src/09_dashboard.py` (sección Kelly en `tab_equity`)

---

## Mejora 11 — Simulación Monte Carlo

**Fecha:** 26 de abril de 2026

**Problema:** No había forma de cuantificar la probabilidad de sufrir
una pérdida severa por mala suerte en el orden de los trades.

**Solución:** 10.000 simulaciones que barajan los trades al azar.
Muestra probabilidad de ruina (perder >50% del capital), distribución
de max drawdown (mediana y percentil 95), y distribución de capital
final. Dos histogramas interactivos y texto explicativo accesible
para novatos. Todo en el tab Equity del dashboard.

**Archivos:** `src/09_dashboard.py` (sección Monte Carlo en `tab_equity`)

---

## Mejora 12 — Rendimiento live con contexto y porcentajes

**Fecha:** 27 de abril de 2026

**Problema:** La sección "Rendimiento live" mostraba PnL en USD
absolutos (1 BTC) sin fecha de inicio ni contexto. No se entendía
desde cuándo se contaban las señales.

**Solución:** Se añadió la fecha de la primera señal ("desde X"),
el PnL acumulado en %, y dos ejemplos claros: "Con $100 → $X" y
"Con 1 BTC → $X". Ahora son 4 tarjetas en vez de 3. Cada tarjeta
incluye un icono de info (ℹ) con tooltip explicativo al pasar el
ratón: qué significa el dato, cómo se calcula, y desde cuándo se
cuentan los trades. Se reutiliza el mismo patrón CSS del Win Rate.

**Archivos:** `src/09_dashboard.py` (sección Rendimiento live en `tab_live`)

---

## Mejora 13 — Historial de señales: colores y orden

**Fecha:** 27 de abril de 2026

**Problema:** La columna Resultado estaba al final de la tabla, las
señales anteriores al monitor marcaban "Abierta" (confuso), y no
había indicación visual de ganadora/perdedora.

**Solución:** Resultado ahora es la segunda columna (tras Fecha).
Las señales sin datos de resultado (pre-monitor) muestran "Sin datos"
en gris. Resultado positivo en verde, negativo en rojo, "Abierta"
en azul. Se usa `Styler` de Pandas para los colores.

**Archivos:** `src/09_dashboard.py` (historial en `tab_live`)

---

## Mejora 14 — Resumen diario con rango horario

**Fecha:** 27 de abril de 2026

**Problema:** El resumen diario de Telegram decía "Resumen diario |
27 abril" pero no indicaba qué período cubría exactamente.

**Solución:** Cambiado a "Resumen últimas 24h" con rango explícito:
"Desde 26 abr 08:00 hasta 27 abr 08:00 (Madrid)".

**Archivos:** `src/08_live_signal.py` (función `_enviar_resumen_diario`)

---

## Mejora 15 — Unificación de zonas horarias a hora Madrid

**Fecha:** 27 de abril de 2026

**Problema:** Las horas en el sistema mezclaban formatos: "UTC",
"+2:00", sin etiqueta... El sistema ya convertía internamente a
Europe/Madrid pero los textos decían "UTC", lo que era incorrecto
y confuso.

**Solución:** Todas las horas visibles al usuario ahora dicen
"(Madrid)": dashboard (heartbeat, señal live, señal cerrada),
Telegram (señal de entrada, cierre de posición, resumen diario).

**Archivos:** `src/09_dashboard.py`, `src/08_live_signal.py`,
`src/10_position_monitor.py`

---

## Mejora 16 — PnL combinado en operaciones con TP1 parcial

**Fecha:** 29 de abril de 2026

**Problema:** Cuando una operación tocaba TP1 (cierre parcial del 50%)
y luego el 50% restante tocaba SL en breakeven, el sistema reportaba
solo el resultado del segundo tramo. Esto causaba 4 bugs:

1. **GANADOR/PERDEDOR incorrecto:** El mensaje de Telegram del cierre
   final decía "PERDEDOR -0.21%" cuando la operación combinada era
   ganadora (+0.28% = 0.76%×50% + (-0.21%)×50%).
2. **R-multiple = 0:** Tras TP1 el SL se movía a breakeven
   (entry_price). El cálculo usaba `risk = abs(entry - stop_loss)` = 0.
3. **Hora UTC etiquetada como Madrid:** `open_time` se guardaba en UTC
   pero se mostraba con etiqueta "(Madrid)" sin convertir zona horaria.
   Causaba discrepancias de 2h en los mensajes.
4. **Historial incorrecto:** El dashboard mostraba solo el PnL del
   cierre final (-0.21%) en vez del resultado combinado (+0.28%).

**Solución:**

- **R-multiple:** Usa `pos.get("original_stop_loss", pos["stop_loss"])`
  para calcular el riesgo real con el SL original, no el movido a BE.
- **PnL combinado:** Al detectar TP1, se guarda `tp1_pnl_pct` en la
  posición. En el cierre final, se calcula
  `combined_pnl_pct = tp1_pnl × 50% + final_pnl × 50%`.
  GANADOR/PERDEDOR se decide con el PnL combinado.
- **Mensaje Telegram:** El cierre final tras TP1 muestra desglose:
  "TP1 (50%): +0.76%, Cierre final (50%): -0.21%, PnL total: +0.28%".
- **Hora Madrid:** `open_time` se convierte de UTC a `Europe/Madrid`
  con `ZoneInfo` antes de formatear.
- **CSV:** Nueva columna `combined_pnl_pct` en `position_results.csv`.
  Solo tiene valor cuando hubo TP1 parcial.
- **Dashboard:** Historial y rendimiento live usan `combined_pnl_pct`
  cuando existe. Win Rate y PnL acumulado reflejan el resultado real.
- **Script de corrección:** `src/fix_combined_pnl.py` recalcula
  operaciones pasadas: añade `combined_pnl_pct`, corrige `stop_loss`
  y `r_multiple` en filas donde el SL estaba en breakeven.

**Archivos:** `src/10_position_monitor.py`, `src/09_dashboard.py`,
`src/fix_combined_pnl.py` (script de corrección única)

---

## Mejora 17 — Backtest con PnL real (cierres parciales)

**Fecha:** 30 de abril de 2026

**Problema:** El backtest simulaba correctamente la mecánica de TP1
(mover SL a breakeven), pero al registrar el resultado del trade
calculaba el PnL como si toda la posición saliera al precio final.
No simulaba que el 50% ya se cerró en TP1.

Esto causaba que las métricas del backtest no reflejaran la realidad:

| Métrica          | Backtest antiguo | Backtest corregido |
|------------------|------------------|--------------------|
| Win Rate         | 27.7%            | 52.1% (todos)      |
| Avg Win          | +1.62%           | +0.86%             |
| Payoff Ratio     | 1.79             | 0.95               |
| Breakeven WR     | 35.9%            | 51.2%              |

Con filtro IA (proba >= 0.55): WR 66.2%, breakeven 52.7%, PnL +16.69%.

El dashboard mostraba 36% como breakeven (fijo), lo que daba la
impresión de que un WR del 42% era suficiente cuando en realidad
el breakeven real con parciales es ~53%.

**Solución:**

- **`_resultado()` en backtest:** Recibe `tp1_hit`. Si True,
  calcula PnL combinado: `50% × PnL_TP1 + 50% × PnL_final`.
- **`simular_trade()`:** Pasa `tp1_hit=True` en todas las llamadas
  de Fase B (breakeven, TP2) y time_exit cuando corresponde.
- **trades.csv y model_results_v3.csv:** Regenerados con los PnL
  corregidos. Nueva columna `tp1_hit`.
- **Dashboard:** `WR_BREAKEVEN` ahora se calcula dinámicamente
  desde el payoff ratio real de `model_results_v3.csv`.
  El tooltip explica la fórmula.
- **Bug calculadora:** Corregido símbolo `-` faltante en resultado
  negativo (mostraba `$4.54` en vez de `-$4.54`).

**Archivos:** `src/04_backtest.py`, `src/09_dashboard.py`,
`outputs/trades.csv`, `outputs/model_results_v3.csv`

---

## Mejora 18 — Reentrenamiento del modelo con PnL corregido

**Fecha:** 1 de mayo de 2026

**Problema:** Tras la Mejora 17 (backtest con PnL real de cierres
parciales), los `trades.csv` corregidos cambiaron el R-multiple de
los 1.034 trades de breakeven: antes tenían R ≈ 0 (label=0), ahora
tienen R = 0.5 (label=1). El modelo en producción estaba entrenado
con los labels antiguos, lo que significaba que había aprendido a
evitar patrones que en realidad eran ganadores. ~23% del dataset
tenía la etiqueta equivocada.

**Impacto del error:** El modelo antiguo, al umbral 0.55, solo
dejaba pasar 65 trades en el período de test y generaba $17,310
de PnL. Estaba filtrando el 90% de las señales válidas.

**Solución:** Re-ejecución del pipeline completo:
1. `06_prepare_dataset.py` → regeneró `ml_dataset.csv` con labels
   corregidos (47.8% ganadores vs ~25% antes).
2. `07_train_model_v3.py` → reentrenó el XGBoost con los mismos
   hiperparámetros y features.

**Métricas antes vs después (umbral 0.55, test set):**

| Métrica          | Modelo antiguo | Modelo reentrenado |
|------------------|----------------|--------------------|
| Trades           | 65             | 611                |
| Win Rate         | ~66%           | 61.9%              |
| Payoff Ratio     | 0.95           | 1.15               |
| Breakeven WR     | 52.7%          | 46.5%              |
| Margen (WR - BE) | ~13 pts        | 15.3 pts           |
| PnL total        | $17,310        | $131,699           |
| PnL total (%)    | ~17%           | 147.66%            |

El Win Rate baja ligeramente (de 66% a 62%) pero el payoff ratio
sube de 0.95 a 1.15, lo que baja el breakeven de 52.7% a 46.5%.
El margen de seguridad aumenta de ~13 a 15.3 puntos. Y el modelo
pasa muchas más señales válidas (611 vs 65).

**Comentario actualizado en `06_prepare_dataset.py`:** Se corrigió
la documentación de `crear_label()` que decía incorrectamente que
los breakeven tenían R ≈ 0 y eran label=0.

**Archivos:** `src/06_prepare_dataset.py` (comentario),
`outputs/ml_dataset.csv`, `outputs/model_v3.joblib`,
`outputs/model_results_v3.csv`, `outputs/feature_importance_v3.png`

---

*Última actualización: 1 de mayo de 2026.*
