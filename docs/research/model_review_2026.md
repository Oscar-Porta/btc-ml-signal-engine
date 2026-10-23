# Revisión completa del modelo predictivo — Chat Fable 5

> Fecha: 3 julio 2026
> Base: `traspaso_revision_completa.md` + lectura de `02_indicators.py`, `03_rules.py`, `04_backtest.py`, `06_prepare_dataset.py`, `07_train_model_v3.py`, `08_live_signal.py`
> Nada de lo aquí propuesto está implementado. Todo es análisis para discutir.

---

## 1. Resumen ejecutivo

El proyecto está muy bien construido para el nivel del que partías: pipeline ordenado, split temporal correcto, walk-forward, etiquetas coherentes con la config de trading, y una cultura sana de "medir PnL, no F1". Eso es más de lo que hacen muchos proyectos profesionales.

Dicho esto, he encontrado **un problema crítico y cuatro importantes** que afectan a la validez de los números del backtest. Mi recomendación es corregirlos ANTES de añadir features o cambiar de algoritmo, porque ahora mismo cualquier experimento se mediría contra una vara de medir torcida.

| # | Hallazgo | Gravedad | Efecto |
|---|----------|----------|--------|
| H1 | Lookahead en las features de 4h (backtest ve el futuro) | **Crítica** | Resultados de backtest inflados; el modelo aprende de información que en live no existe |
| H2 | Backtest sin comisiones ni slippage | Alta | PnL sobreestimado ~0.1–0.25R por trade |
| H3 | Backtest permite trades solapados; live los bloquea | Alta | Las métricas del backtest no representan lo que vive un suscriptor |
| H4 | Umbrales y Config A elegidos mirando el test set | Alta | Esperable degradación en live (sesgo de selección) |
| H5 | Features calculadas en dos códigos distintos (batch vs live) | Media-Alta | Riesgo permanente de divergencia; H1 es un síntoma de esto |
| H6 | `atr_14` absoluto como feature (no estacionaria) | Media | El modelo puede estar usando el "nivel de precio de la época" como pista |

---

## 2. Hallazgos en detalle

### H1 — Lookahead bias en las features de 4h (crítico)

**Qué pasa.** En el pipeline histórico, `merge_asof(direction="backward")` asigna a cada vela de 1h la vela de 4h cuyo `open_time` es anterior o igual. El problema: los indicadores de esa vela 4h (RSI, EMAs, ATR, trend) están calculados con su **cierre**, que puede ocurrir hasta 3 horas DESPUÉS de la señal.

Ejemplo concreto: señal al cierre de la vela 1h que abre a las 17:00 (decisión a las 18:00). Se le asigna la vela 4h que abre a las 17:00... y cierra a las 21:00. Su `rsi_14_4h` incluye 3 horas de precio futuro. Solo 1 de cada 4 velas de 1h queda alineada sin futuro.

**Por qué es grave aquí.** Las features afectadas son precisamente las más importantes del modelo: `rsi_14_4h` (#1), `ema_spread_4h_pct` (#2), `atr_pct_4h` (#6), y además `trend_4h`, que es el **filtro obligatorio de dirección** de las reglas. O sea: hasta las señales mismas del backtest usan información futura.

**Y en live es distinto pero también incorrecto.** `08_live_signal.py` descarga velas con `get_klines`, que incluye la vela 4h **en formación** (parcial), calcula indicadores sobre ella y hace el mismo merge. Resultado: en live el modelo recibe valores de 4h calculados sobre una vela a medias — una distribución que nunca vio en entrenamiento. Es el clásico *train/serve skew*.

**Solución conceptual** (a discutir): usar siempre la **última vela 4h CERRADA**. En el histórico, hacer shift de las columnas 4h una vela antes del merge (o mergear por `close_time` ≤ momento de decisión). En live, descartar la última vela de 4h antes de calcular indicadores. Después: regenerar señales, backtest, dataset y reentrenar. Los números cambiarán — probablemente a peor, y eso es bueno: serán los de verdad.

**Matiz honesto:** el sistema live desde mayo ya operaba con la versión "sin futuro" (vela parcial), y aun así el periodo sin filtro dio +8.65%. Eso sugiere que las reglas tienen algo real. Pero el modelo se entrenó con valores que no verá jamás, así que su rendimiento live es una incógnita hasta corregir esto.

### H2 — Sin comisiones ni slippage

`04_backtest.py` calcula el PnL como diferencia pura de precios. En Binance, la comisión taker de ida y vuelta es ~0.1–0.2% del nocional. Con un riesgo típico de ~0.6–1% del precio (SL 1.25×ATR), eso equivale a **0.1–0.25R por trade**. Sobre miles de trades es una diferencia enorme, y afecta más a los umbrales bajos (más trades). Aunque el producto sea solo informativo, las métricas que publiques en la web deberían descontar fricción, o un suscriptor no las reproducirá.

Añadido barato: la entrada real de un usuario sería el open de la vela siguiente (recibe el Telegram tras el cierre), no el close exacto de la vela de señal. En cripto 1h la diferencia es pequeña pero existe; se puede modelar con un slippage fijo.

### H3 — Solapamiento: backtest sí, live no

El backtest simula las 4,574 señales aunque se solapen (varias posiciones abiertas a la vez, hasta 24h cada una, con ~2.5 señales/día). En producción, el sistema bloquea señales solapadas (Anexo I). Consecuencias: el PnL y drawdown del backtest no son alcanzables por un usuario que sigue el canal, y el dataset de entrenamiento contiene trades que live nunca emitiría. Propuesta: re-simular con la misma política de bloqueo que live y comparar. Como mínimo, saber cuánto cambia.

### H4 — El test set ya está "gastado"

Config A salió de comparar 558 combinaciones (143 con reentrenamiento) y los umbrales 0.45/0.60 se eligieron mirando la tabla de resultados sobre el test 2025-2026. Cada vez que eliges "lo que mejor sale" en un conjunto de datos, una parte de esa ventaja es azar que no se repetirá. No significa que Config A sea mala — el margen sobre alternativas era amplio —, pero sí que el PnL del test set es una **cota optimista**, no una expectativa. La evaluación honesta que queda es la que ya tienes en marcha: los datos live desde el 24 de junio. Regla para el futuro: las decisiones se toman en una ventana de validación, y el test/live solo se mira una vez por iteración grande.

### H5 — Doble implementación de features

Los indicadores y features están implementados dos veces: en pandas vectorizado (02/06) y fila a fila en `08_live_signal.py`. Cada divergencia entre ambos es un bug silencioso que el modelo paga (H1 es exactamente eso). Antes de ETH/SOL, propondría extraer un **módulo único de features** que usen tanto el pipeline histórico como el live, con un test de paridad: misma vela → mismos 20 valores, error si difieren. Esto también resuelve de raíz el mantenimiento ×3 al añadir instrumentos.

### H6 — `atr_14` absoluto como feature

`atr_14` en USD no es comparable entre BTC a 16k y BTC a 100k. Que sea la feature #4 es sospechoso: probablemente el modelo la usa como proxy de "época/nivel de precio", un atajo que no generaliza. Ya tienes las versiones sanas (`atr_pct`, `atr_regime`, `atr_pct_4h`). Propuesta: eliminarla y medir. Lo mismo aplicaría a cualquier feature futura en unidades absolutas.

---

## 3. Respuestas a las 12 preguntas

### 3.1 Mejora del modelo

**P1. Features adicionales.** Por orden de coste/beneficio:

- *Coste cero (ya las descargas y las tiras):* las klines de Binance incluyen `taker_buy_base` y `number_of_trades`. De ahí salen dos features clásicas: **taker buy ratio** (presión compradora/vendedora real) y **trades relativos** (actividad). Muy alineadas con lo que le falta al modelo (flujo).
- *Derivadas baratas:* pendiente de las EMAs (velocidad de la tendencia, no solo separación), distancia al swing high/low en unidades de ATR, ancho de Bandas de Bollinger (squeeze → detecta lateralidad), ADX de 4h, Kaufman Efficiency Ratio (ver P6), contexto diario (tendencia 1d, distancia a EMA50 1d).
- *Externas:* **open interest de futuros** de Binance (histórico disponible con buena resolución, no sufre el problema de las 8h del funding) y **basis** spot-perp. Volatilidad implícita: existe DVOL de Deribit vía API. Liquidaciones: el histórico es difícil de conseguir; lo dejaría. On-chain: demasiado lento para horizonte de 24h; lo descartaría.

Advertencia importante: con ~3,400 muestras de train, añadir 15 features de golpe es receta para overfitting. Lotes de 2-3, validando cada lote con walk-forward, y podando lo que no aporte (como ya hicisteis con el funding — bien hecho).

**P2. Algoritmo.** Primero una aclaración: lo que hay en producción es `GradientBoostingClassifier` de scikit-learn, no XGBoost — conviene corregirlo en la documentación. ¿Migrar a XGBoost nativo / LightGBM / CatBoost? Sí, pero con expectativas modestas: en datasets tabulares pequeños las tres familias rinden parecido. Los beneficios reales son operativos: entrenan 10-50× más rápido (tus barridos de 143 reentrenos pasarían de horas a minutos), tienen *early stopping* (menos overfitting sin tunear a mano) y no dependen del pickle de sklearn (ayuda con H-sklearn del VPS). CatBoost suele ir mejor en datasets pequeños por su *ordered boosting*. Recomendación: bake-off de los tres contra el GB actual **después** de corregir H1, con walk-forward como juez. Prioridad media.

**P3. Feature engineering.** La mejora más interesante no son interacciones (los árboles ya las capturan solos) sino **features firmadas por dirección**. Ahora mismo el modelo no sabe si el trade es LONG o SHORT (se eliminó `direction_num`), y features como `roc_20` o `dist_ema20_pct` significan lo contrario según la dirección: momentum positivo es favorable en un LONG y desfavorable en un SHORT. Transformar a "relativas al trade" (ej. `roc_20_signed = roc_20 × (+1 long / −1 short)`) unifica los patrones de long y short en uno solo — en la práctica es como duplicar la densidad de datos. Menor: `hour` como entero hace que 23h y 0h parezcan lejanas; codificación sin/cos lo arregla (efecto pequeño con árboles, pero gratis).

**P4. Calibración.** Sí, pero entendiendo qué compra. La calibración (isotonic con validación temporal, o Platt) **no cambia el orden** de los trades, así que el PnL por umbral no mejora: mejora el *significado* del número. Con probabilidades reales puedes: (a) definir umbrales por valor esperado — con TP 2R y SL 1R el breakeven está en p≈0.33, todo lo que esté por encima tiene EV positivo; (b) mantener "Activo/Selectivo" estables entre reentrenamientos (hoy 0.45 significa una cosa en este modelo y otra en el siguiente); (c) monitorizar drift con Brier score. Coste bajo, recomendada en la fase de robustez.

**P5. Etiquetado.** El binario R≥0.5 está bien razonado (limpia los time-exits ruidosos). Dos experimentos baratos si quieres exprimir: (a) regresión sobre el R-multiple winsorizado a [−1, +2] y rankear por R predicho — usa la magnitud, no solo el signo; (b) mantener clasificación pero con `sample_weight = |r_multiple|`, para que equivocarse en un −1R pese más que en un −0.1R. Beneficio incierto; probaría (b) primero por simplicidad. Prioridad baja hasta cerrar H1.

**P6. Régimen de mercado.** Es tu intuición correcta ("bien en tendencia, mal en lateral") y la vía más prometedora de mejora de señal. Pero empezaría por lo simple: **régimen como features, no como meta-modelo**. Un HMM o un clasificador de régimen separado es sobreingeniería con 4.5k muestras. Candidatas: Kaufman Efficiency Ratio (|cambio neto| / suma de |cambios|; ~1 = tendencia limpia, ~0 = ruido lateral), ADX 4h, ancho de Bollinger normalizado, y el `atr_regime` que ya tienes. Para validar que funciona: parte el test en buckets por ER y comprueba que el modelo con estas features recorta pérdidas en el bucket lateral. Si algún día quieres ir más lejos, el siguiente paso sería un umbral dinámico según régimen — pero eso se discute cuando haya datos.

**P7. Reentrenamiento.** No acortaría la ventana: con tan pocas muestras, tirar años de datos duele más que el drift que evitas (y tu walk-forward mostró consistencia entre ventanas, señal de que los patrones son estables). Propuesta: **ventana expansiva** (siempre todo el histórico) con reentrenamiento **trimestral** o cada ~200 trades nuevos, lo que llegue antes. En cada reentreno: correr walk-forward como test de regresión, comparar con el modelo anterior, y desplegar solo si no empeora. Ya tienes `guia_reentrenamiento_2026.md`; esto solo le pone calendario y criterio de aceptación.

### 3.2 Arquitectura del sistema

**P8. Multi-instrumento.** Empezar con **un modelo por instrumento**, misma receta. Razones: BTC/ETH/SOL tienen microestructura distinta, y un modelo compartido mezcla regímenes de liquidez. Pero con dos matices: (a) para ETH y SOL, añadir **features de contexto BTC** (trend 4h de BTC, roc de BTC, ratio ALT/BTC) — las alts siguen a BTC y ese es probablemente el predictor más fuerte que existirá para ellas; (b) como experimento posterior, un modelo *pooled* (todos los instrumentos juntos, con el símbolo como feature) es viable porque tus features ya son porcentuales/normalizadas, y beneficiaría sobre todo a SOL (menos historia). Prerequisito de todo esto: el refactor de la sección 4 — replicar los scripts a mano ×3 sería multiplicar el problema H5.

**P9. Ensemble.** Lo único que vale su coste ahora: **promediar 5-10 modelos idénticos con semillas distintas** (o bagging). Reduce varianza gratis, típicamente estabiliza los umbrales. Stacking o combinar familias de modelos: no, con este volumen de datos añade complejidad sin retorno claro.

**P10. Monitorización de drift.** Ya logueas features y probabilidad de cada señal live (CSV + Supabase) — tienes el 80% hecho. Faltaría un job (encaja en health_monitor, frecuencia semanal/mensual) con tres chequeos: (a) **PSI** (Population Stability Index) por feature: distribución live vs distribución de train; alerta si alguna feature se desplaza; (b) **WR rolling** de los últimos 30-50 trades vs el WR esperado para su umbral, con banda de confianza binomial — es tu alarma de "el modelo ya no funciona"; (c) distribución de probabilidades emitidas vs la del backtest (si el modelo empieza a emitir todo 0.3, algo cambió). Alertas al canal técnico que ya existe.

### 3.3 Datos

**P11. Volumen.** 4,574 señales (~3,400 train) es suficiente para un GBM de 20 features — estás en la zona sana. Data augmentation sintética (SMOTE y compañía) **no**: genera trades que nunca existieron, con la estructura temporal rota, y el modelo aprende geometría del algoritmo de interpolación, no mercado. Las formas legítimas de "más datos" son: pooling multi-instrumento (P8) y features firmadas (P3), que unifican patrones espejo.

**P12. Desbalanceo.** 73/27 no es un desbalanceo serio (serio es 99/1). Y en tu caso importa aún menos: no usas la predicción binaria, usas el **ranking** de probabilidades con umbral — el desbalanceo desplaza los valores absolutos de probabilidad pero apenas cambia el orden. SMOTE descartado (ver P11). `scale_pos_weight`/class weights: se puede incluir en el bake-off de P2, expectativa neutra. Si calibras (P4), el tema queda cerrado del todo.

---

## 4. Arquitectura antes de ETH/SOL

En orden de importancia:

1. **Módulo único de features e indicadores** (resuelve H5, prerequisito de todo lo demás). Un `features.py` compartido por 02/03/06 y 08, parametrizado por símbolo, con test de paridad batch↔live.
2. **Parametrizar el símbolo** en vez de duplicar scripts: `BTCUSDT` está hardcodeado en cada script. Un archivo de config por instrumento (símbolo, canales Telegram, rutas) y el mismo código para los tres.
3. **Fijar versiones** (resuelve el warning de sklearn): mismo `requirements.txt` con versiones exactas en local y VPS, y reentrenar con la versión del VPS. Es un bug latente de reproducibilidad, no cosmético.
4. **Backtest realista**: comisiones + slippage (H2) y política de solapamiento igual que live (H3), para que backtest y experiencia del suscriptor cuenten la misma historia.
5. **Disciplina con el test set** (H4): decisiones en validación; test/live se mira una vez por iteración.

---

## 5. Plan propuesto (para discutir, no ejecutado)

- **Fase A — Corrección de la vara de medir:** fix del lookahead 4h (H1) → regenerar señales/backtest/dataset → reentrenar → añadir fees y solapamiento al backtest → números honestos nuevos. *Todo lo demás se decide con estos números.*
- **Fase B — Robustez:** módulo único de features + test de paridad, versiones fijadas, calibración, monitor de drift en health_monitor.
- **Fase C — Mejora del modelo:** features por lotes (taker buy ratio, régimen, contexto diario, firmadas), bake-off de algoritmos, experimentos de etiquetado, ensemble de semillas.
- **Fase D — Expansión:** ETH y SOL sobre la base refactorizada, con features de contexto BTC.

Una nota de gestión de expectativas: es probable que la Fase A **empeore** los números del backtest. Eso no es retroceder — es descubrir dónde estabas de verdad. Los datos live que estás recopilando desde el 24 de junio serán el árbitro final.
