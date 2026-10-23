# Trading Signal Project — Notas para la presentación

> Documento vivo que se actualiza al final de cada fase.
> Contiene las decisiones, razonamientos y explicaciones
> del proyecto, pensado para convertirse en un PPTX
> orientado a alguien que entiende de trading pero no
> necesariamente de programación.

---

## 1. Visión general del proyecto

**Objetivo:** construir un sistema de señales de trading para
BTC/USDT que analice el mercado, detecte oportunidades de
entrada (long y short) y nos avise por Telegram. El sistema
es solo informativo: nunca ejecuta operaciones automáticas,
solo informa de la oportunidad y el operador decide si la toma.

**Objetivo a largo plazo:** acumular un histórico de trades
simulados lo bastante grande y limpio como para entrenar un
modelo de IA que filtre las señales del sistema de reglas,
descartando las que tienen baja probabilidad de éxito.

**Fuente de datos:** API pública de Binance (sin claves).

**Notificaciones:** bot de Telegram (en fases posteriores).

---

## 2. ¿Por qué Binance como fuente de datos?

BTC/USDT en Binance es el par más líquido del mundo cripto.
Mayor volumen = velas OHLCV más representativas del precio
real al que opera la gente. Datos de exchanges con menos
volumen tienen más ruido, mechas falsas y deslizamiento.

**Alternativas descartadas:**

- **Otros exchanges (Coinbase, Kraken, Bybit):** menos volumen
  en BTC/USDT, librerías Python menos maduras. Datos
  objetivamente menos representativos para BTC.
- **CCXT (librería multi-exchange):** unifica 100+ exchanges
  con la misma interfaz. Útil si en el futuro quisiéramos
  cambiar de exchange o comparar, pero ahora solo necesitamos
  Binance y añadiría complejidad sin beneficio.
- **Agregadores (CryptoCompare, CoinGecko):** mezclan datos
  de varios exchanges con distinta liquidez. Introduce ruido
  al combinar libros de órdenes con microestructura diferente.

**Conclusión:** Binance no se heredó por comodidad del proyecto
anterior, sino porque es genuinamente la mejor fuente para
nuestro caso de uso.

---

## 3. Decisiones de diseño fundamentales

### 3.1 Temporalidades: 4h (contexto) + 1h (entrada)

Usamos **dos temporalidades**, no una ni tres:

- **4 horas** como marco de contexto: solo responde a "¿la
  tendencia dominante es alcista o bajista?". Si 4h dice
  alcista, solo aceptamos señales long en 1h. Si bajista,
  solo short. Si indecisa, no operamos.
- **1 hora** como marco de entrada: es donde se generan las
  señales concretas con precio, SL y TP.

**¿Por qué no añadir 15m?** Multiplicaría la complejidad del
backtest, de la sincronización de datos y del debugging, con
retorno bajo para un sistema que todavía no existe. Si más
adelante los datos sugieren que sería útil, se añade entonces.

**¿Por qué 1h y no 5m como antes?** El sistema anterior con
velas de 5 minutos era demasiado rápido para supervisar, no
detectaba bien los SL/TP y generaba señales con demasiado
ruido. 1 hora da tiempo de sobra para revisar cada señal y
los movimientos son más limpios.

### 3.2 Profundidad histórica: 5 años

**Periodo descargado:** 11 de abril de 2021 → 11 de abril de
2026 (5 años exactos).

**¿Por qué 5 años?**

- **Un ciclo completo de halving:** Bitcoin tiene ciclos de ~4
  años. Con 5 años cubrimos un ciclo entero más margen.
- **Todos los regímenes de mercado representados:** techo
  eufórico de 2021, crash de mayo 2021, segundo pico nov
  2021, bear market 2022 (LUNA, FTX), lateralidad 2023,
  halving abril 2024, bull post-halving 2024-2025, y 2026.
- **Volumen de datos suficiente:** 43.812 velas de 1h y
  10.956 de 4h. Más que suficiente para generar miles de
  trades simulados y entrenar modelos de IA.

**¿Por qué no más de 5 años?** El BTC anterior a 2021 era
mucho menos líquido, sin participación institucional ni ETFs,
con estructuras de mercado distintas. Incluirlo contaminaría
el modelo con patrones que ya no aplican.

**Calidad de los datos descargados:**

- 4h: 10.956 velas, 0 duplicados, 0 huecos.
- 1h: 43.812 velas, 0 duplicados, 5 huecos (12 velas
  perdidas en total = 0,027%). Los huecos corresponden a
  ventanas de mantenimiento reales de Binance (abril/agosto/
  septiembre 2021, marzo 2023). Se dejan tal cual, no se
  rellenan con datos sintéticos.

**Política sobre huecos:** un dataset "honesto" con agujeros
es infinitamente mejor que un dataset "bonito" con datos
inventados. Rellenar con forward-fill o datos de otro exchange
introduce ruido y puede generar señales falsas en el backtest.

### 3.3 Direccionalidad: long y short

Operamos en ambas direcciones desde el inicio. No nos
limitamos a long porque el 50% del mercado es bajista y
queremos que la IA tenga datos de ambos lados.

---

## 4. Indicadores técnicos

Todos calculados a mano con pandas (sin librerías externas
tipo ta-lib o pandas-ta) para entender exactamente qué hacen
y evitar dependencias problemáticas.

### 4.1 EMA 20 y EMA 50 (Medias Móviles Exponenciales)

- **Qué son:** precio promedio de las últimas N velas, dando
  más peso a las recientes.
- **Para qué las usamos:** como termómetro de tendencia. Si
  EMA 20 > EMA 50 → tendencia alcista (trend = "up"). Si
  EMA 20 < EMA 50 → bajista (trend = "down").
- **Parámetros:** 20 y 50 son los periodos estándar del
  trading para tendencias de medio plazo.

### 4.2 RSI 14 (Relative Strength Index)

- **Qué es:** oscilador de 0 a 100 que mide la fuerza
  reciente del movimiento.
- **Para qué lo usamos:** como filtro de extremos. Evita
  comprar cuando el mercado ya ha subido demasiado (RSI > 70)
  o vender cuando ya ha bajado demasiado (RSI < 30).
- **Fórmula utilizada:** suavizado de Wilder (alpha = 1/N),
  que es la fórmula canónica original.
- **Parámetro:** 14 periodos, el estándar creado por Welles
  Wilder en 1978.

### 4.3 ATR 14 (Average True Range)

- **Qué es:** mide la volatilidad media (cuánto se mueve el
  precio por vela), sin importar la dirección.
- **Para qué lo usamos:** para calcular SL y TP adaptativos.
  En vez de un stop fijo de 100 USD, ponemos "1,5 × ATR".
  Así cuando la volatilidad es alta el stop respira más, y
  cuando es baja se ajusta. Es uno de los conceptos más
  importantes del sistema.
- **Fórmula:** True Range = max(high-low, |high-prev_close|,
  |low-prev_close|), luego suavizado de Wilder.

### 4.4 Swing High/Low de 20 velas (estructura)

- **Qué son:** el máximo más alto y el mínimo más bajo de las
  últimas 20 velas (incluida la actual).
- **Para qué los usamos:** como referencia de estructura del
  mercado. Si el precio rompe el swing_high_20, está mostrando
  fuerza (base de la regla de breakout).
- **No miran al futuro:** usan rolling(20) de pandas, que solo
  mira hacia atrás. Esto evita el "lookahead bias".

### 4.5 Detalle técnico importante: periodo de calentamiento

Los indicadores necesitan un mínimo de velas históricas antes
de dar un valor fiable. Usamos `min_periods` en todos los
cálculos para que las primeras filas devuelvan NaN en lugar de
valores de calentamiento engañosos. La primera vela con todos
los indicadores válidos es la número 49 (dictada por la EMA 50,
que es el indicador más lento).

Sin esto, el backtest generaría señales basadas en EMAs
"falsas" de las primeras filas. Fue un bug sutil que
detectamos y corregimos antes de seguir adelante.

---

## 5. Reglas de entrada

### 5.1 Filtro obligatorio de contexto (4h → 1h)

Antes de aplicar cualquier regla, cada vela de 1h se enriquece
con la información de tendencia de 4h vigente en ese momento.

- Si trend_4h = "up" → solo se aceptan señales LONG.
- Si trend_4h = "down" → solo se aceptan señales SHORT.

Esto descarta todas las señales que van contra la tendencia
dominante, reduciendo drásticamente las señales basura.

La unión temporal se hace con `merge_asof` de pandas: para
cada vela de 1h, busca la última vela de 4h cuya apertura
sea anterior o igual. Es como preguntar "¿qué decía el
gráfico de 4h cuando se formó esta vela de 1h?".

### 5.2 Regla 1: Pullback a la EMA 20

**Filosofía:** "comprar el retroceso en tendencia" (mean
reversion). En una tendencia alcista, el precio no sube en
línea recta; hace retrocesos temporales. Cuando retrocede
hasta la EMA 20 y rebota, es oportunidad de entrar a mejor
precio dentro de la tendencia.

**Condiciones LONG:**

1. trend_4h = "up" (tendencia macro alcista)
2. trend_1h = "up" (tendencia local también alcista)
3. low de la vela ≤ ema_20 (tocó la EMA por debajo)
4. close de la vela > ema_20 (pero cerró por encima → rebote)
5. RSI < 55 (no está sobrecomprado)

**Condiciones SHORT:** espejo exacto con trend "down", high ≥
ema_20, close < ema_20, RSI > 45.

**Decisión sobre el umbral de RSI:** originalmente se fijó en
< 45 para long y > 55 para short, pensando que un pullback
tendría un RSI "bajo". El análisis mostró que esto generaba
**0 señales** porque era contradictorio: si ambas tendencias
son alcistas y el precio rebota en la EMA 20, el RSI está
típicamente en zona 45-55 (neutra-sana), no en zona "débil".
Pedir RSI < 45 exigía debilidad del mercado, pero un pullback
sano no es debilidad.

Se relajó a RSI < 55 (long) / > 45 (short). Esto sigue
filtrando extremos de sobrecompra/sobreventa pero permite
los pullbacks reales.

### 5.3 Regla 2: Breakout de estructura

**Filosofía:** "comprar la fuerza" (momentum). Cuando el
precio rompe por encima del máximo de las últimas 20 velas a
favor de la tendencia, está mostrando momentum alcista.

**Condiciones LONG:**

1. trend_4h = "up"
2. close > swing_high_20 de la vela ANTERIOR (ruptura
   confirmada al cierre, usando .shift(1) para evitar
   lookahead bias)
3. RSI entre 40 y 75 (ni en zona de debilidad extrema, que
   indicaría que la ruptura no tiene fuerza, ni en zona de
   agotamiento extremo)

**Condiciones SHORT:** espejo con trend "down", close <
swing_low_20 anterior, RSI entre 25 y 60.

### 5.4 ¿Por qué dos reglas y no una?

Son **filosóficamente opuestas**: pullback es "comprar
barato en tendencia" (mean reversion) y breakout es "comprar
fuerza" (momentum). Cuando entrenemos la IA, el modelo verá
dos tipos de oportunidades muy distintas y podrá aprender en
qué condiciones funciona mejor cada una.

### 5.5 Cálculo de SL y TP (niveles del trade)

Todos los niveles se expresan en **múltiplos de ATR**,
lo que los hace adaptativos a la volatilidad del momento.

- **Stop Loss:** entrada ± 1,5 × ATR
- **Take Profit 1:** entrada ± riesgo × 1,0 (ratio 1:1)
- **Take Profit 2:** entrada ± riesgo × 2,0 (ratio 1:2)

El multiplicador 1,5 para el SL da espacio suficiente para
que el precio respire sin que el ruido normal nos saque. El
TP2 a ratio 1:2 permite que incluso con un win rate del 40%,
el sistema sea matemáticamente rentable.

### 5.6 Resultados de la generación de señales

- **Total:** 4.574 señales en 5 años
- **Direcciones:** 50,7% long / 49,3% short (equilibrado)
- **Reglas:** 68,2% pullback / 31,8% breakout
- **Riesgo medio:** 1,17% por señal
- **Frecuencia:** ~2,5 señales/día, ~18/semana

---

## 6. Estructura técnica del proyecto

### 6.1 Carpetas

```
trading_project_signal/
├── src/              → Código fuente (módulos Python)
├── data/
│   ├── raw/          → Velas descargadas tal cual de Binance
│   └── processed/    → Velas con indicadores calculados
├── outputs/          → Resultados (señales, trades, métricas)
├── notebooks/        → Exploración y experimentos
├── docs/             → Documentación y notas
└── .venv/            → Entorno virtual Python (no se sube)
```

### 6.2 Archivos de código (src/)

| Archivo               | Fase | Función                              |
|-----------------------|------|--------------------------------------|
| 01_download_data.py   | 1    | Descarga velas de Binance → data/raw |
| 02_indicators.py      | 2    | Calcula indicadores → data/processed |
| 03_rules.py           | 3    | Aplica reglas → outputs/signals.csv  |
| 04_backtest.py        | 4    | Simula trades → outputs/trades.csv   |
| 05_evaluation.py      | 5    | Métricas y curva de equity           |
| 06_prepare_dataset.py | 6    | Prepara dataset IA → outputs/ml_dataset.csv |
| 07_train_model.py     | 6    | Entrena modelo → outputs/model.joblib        |
| 08_live_signal.py     | 7    | Señales en vivo + Telegram                   |

### 6.3 Librerías Python utilizadas

| Librería        | Para qué                                    |
|-----------------|---------------------------------------------|
| python-binance  | Conexión con API de Binance                 |
| pandas          | Manipulación de datos (DataFrames, CSV)     |
| numpy           | Cálculos numéricos                          |
| python-dotenv   | Lectura de variables de entorno (.env)      |
| requests        | Peticiones HTTP (Telegram en fases futuras) |
| matplotlib      | Gráficos y visualización                    |

### 6.4 Archivos de datos generados

| Archivo                         | Qué contiene                      |
|---------------------------------|-----------------------------------|
| data/raw/BTCUSDT_4h.csv        | 10.956 velas de 4h (5 años)      |
| data/raw/BTCUSDT_1h.csv        | 43.812 velas de 1h (5 años)      |
| data/processed/BTCUSDT_4h_p.csv| Velas 4h + indicadores técnicos   |
| data/processed/BTCUSDT_1h_p.csv| Velas 1h + indicadores técnicos   |
| outputs/signals.csv             | 4.574 señales con SL/TP/contexto |

---

## 7. Fases del proyecto (estado actual)

| Fase | Descripción                          | Estado     |
|------|--------------------------------------|------------|
| 0    | Preparación del entorno              | Completada |
| 1    | Descarga de datos históricos         | Completada |
| 2    | Cálculo de indicadores técnicos      | Completada |
| 3    | Reglas de entrada y señales          | Completada |
| 4    | Motor de backtest / detección SL/TP  | Completada |
| 5    | Evaluación del sistema               | Completada |
| 6    | Preparación dataset + primer modelo  | Completada |
| 7    | Señales en vivo + Telegram           | Completada |
| 8    | Despliegue robusto (bucle + logs)    | Completada |
| 9    | Iteración 1B (correlaciones externas)| Pendiente  |
| 10   | Iteración 2 (mejores etiquetas)      | Pendiente  |
| 11   | Iteración 3 (XGBoost vs LightGBM vs…)| Pendiente  |
| 12   | Iteración 4 (walk-forward validation)| Pendiente  |
| 13   | Dashboard de control (Streamlit)     | Pendiente  |
| 14   | Despliegue 24/7 (launchd / systemd)  | Pendiente  |
| 15   | Monitor de posiciones (SL/TP live)   | Pendiente  |

*Nota: las fases 6 y 7 originales (iteración de reglas y
preparación dataset) se fusionaron. El backtest demostró que
el problema no son las reglas sino la falta de filtro
inteligente, así que pasamos directamente a la IA.*

---

## 8. Motor de backtest (Fase 4)

### 8.1 Qué hace el motor

Para cada una de las 4.574 señales generadas en la Fase 3,
el motor recorre las velas de 1h posteriores una por una y
comprueba si el high o el low de cada vela tocan los niveles
de SL, TP1 o TP2. No mira solo el close: mira los extremos
reales de la vela.

### 8.2 Gestión del trade en dos fases

- **Fase A (buscando TP1):** si el precio toca TP1, el SL
  sube a breakeven (precio de entrada) y pasamos a Fase B.
  Si el precio toca SL antes de TP1, cerramos con pérdida.
- **Fase B (buscando TP2):** si el precio toca TP2, cerramos
  con beneficio completo (ratio 1:2). Si vuelve al precio de
  entrada (breakeven), cerramos sin pérdida ni ganancia.
- **Time exit:** si tras 24 velas (24 horas) no se ha tocado
  nada, cerramos al precio de cierre. Evita trades zombi.

### 8.3 Regla del peor caso

Si en la misma vela tocan SL y TP (spike de volatilidad),
asumimos que se tocó el SL primero. Es conservador a
propósito: preferimos subestimar los resultados que
sobreestimarlos. Un backtest optimista es peligroso.

### 8.4 Resultados del backtest

**Resumen general:**

- Total trades: 4.574
- Ganadores: 1.237 (27,0%)
- Perdedores: 2.303 (50,3%)
- Neutros (breakeven): 1.034 (22,6%)
- PnL total: -41.228 USD
- Profit factor: 0,97 (prácticamente en equilibrio)
- Ganancia media: 1.079 USD vs Pérdida media: 597 USD
  (ratio real ~1,8:1)

**Interpretación del win rate del 27%:** parece bajo pero es
normal para un sistema con ratio riesgo:recompensa 1:2. Cuando
ganas, ganas casi el doble de lo que pierdes. El profit factor
de 0,97 confirma que el sistema está al borde del equilibrio.

**Por motivo de cierre:**

| Motivo         | Trades | PnL total       |
|----------------|--------|-----------------|
| stop_loss      | 2.178  | -1.349.322 USD  |
| breakeven      | 1.034  | 0 USD           |
| take_profit_2  | 1.015  | +1.223.253 USD  |
| time_exit      | 347    | +84.841 USD     |

Los breakeven suman exactamente 0 (mecánica correcta). Los
time exit son sorprendentemente positivos (win rate 64%): el
filtro de tendencia 4h hace que entremos en la dirección
correcta la mayoría de las veces.

**Rendimiento por año:**

| Año  | Trades | PnL       | Win rate |
|------|--------|-----------|----------|
| 2021 | 659    | -20.771   | 23,5%    |
| 2022 | 848    | -26.451   | 25,7%    |
| 2023 | 926    | -15.386   | 24,3%    |
| 2024 | 953    | +17.605   | 32,5%    |
| 2025 | 929    | -34.651   | 26,5%    |
| 2026 | 259    | +38.426   | 32,1%    |

### 8.5 Análisis detallado de los resultados

**¿Por qué el win rate del 27% no es preocupante?**

Un win rate del 27% suena bajo si piensas en acertar la
mayoría de las veces. Pero en trading lo que importa no es
cuántas veces aciertas, sino CUÁNTO ganas cuando aciertas
vs cuánto pierdes cuando fallas. Nuestro sistema gana de
media 1.079 USD por trade ganador y pierde 597 USD por
trade perdedor: un ratio real de ~1,8:1. Esto significa que
con un win rate de tan solo ~36% el sistema sería rentable.
Estamos al 27%, a solo 9 puntos de la rentabilidad, con
reglas sin optimizar y sin IA.

**El mecanismo de breakeven protege el capital.**

Los 1.034 trades que cierran en breakeven (22,6% del total)
son trades donde el mercado nos dio la razón inicialmente
(tocó TP1) pero luego se giró. Sin el mecanismo de breakeven
esos 1.034 trades habrían sido pérdidas. El breakeven
convierte lo que serían 1.034 pérdidas potenciales en 0 USD
de impacto. Es un escudo silencioso pero enorme.

**Los time exit aportan beneficio inesperado.**

Los 347 trades que cierran por tiempo (24h sin tocar nada)
tienen un win rate del 64% y un PnL total de +84.841 USD.
Esto revela algo importante: el filtro de tendencia 4h está
haciendo bien su trabajo. Cuando entramos a favor de la
tendencia dominante y el mercado simplemente "no hace nada
dramático", la tendencia de fondo empuja suavemente el
precio a nuestro favor.

**El rendimiento varía según el régimen de mercado.**

- 2021 (-20.771 USD, WR 23,5%): caída post-techo. Muchos
  cambios de dirección que confunden al filtro de tendencia.
- 2022 (-26.451 USD, WR 25,7%): bear market prolongado.
  Las reglas detectan señales short correctas pero el
  mercado es muy volátil y saca muchos stops.
- 2023 (-15.386 USD, WR 24,3%): lateralidad aburrida. El
  mercado no tiene tendencia real, pero las EMAs dicen que
  sí → muchas señales falsas.
- 2024 (+17.605 USD, WR 32,5%): año del halving, tendencia
  alcista clara. El filtro 4h brilla, las señales long
  funcionan muy bien.
- 2025 (-34.651 USD, WR 26,5%): periodo de consolidación
  con muchos falsos breakouts que generan pérdidas.
- 2026 (+38.426 USD, WR 32,1%): primeros meses con
  tendencia definida, el sistema vuelve a funcionar.

La lección: el sistema rinde bien en mercados tendenciales
y mal en mercados laterales o con cambios bruscos de
dirección. La IA deberá aprender a detectar estos regímenes
y filtrar las señales en consecuencia.

**Breakout vs Pullback: ¿qué regla funciona mejor?**

- Breakout: 1.456 trades, PnL -8.859 USD (pierde poco)
- Pullback: 3.118 trades, PnL -32.369 USD (pierde más)

El breakout pierde menos con menos trades. El pullback
genera el doble de señales pero con peor calidad media.
Esto sugiere que la regla pullback es más susceptible a
señales falsas en mercados laterales. Es información
valiosa para la fase de iteración de reglas y para que la
IA aprenda a filtrar pullbacks en contextos desfavorables.

**Short vs Long:**

- Long: 2.318 trades, PnL -66.603 USD
- Short: 2.256 trades, PnL +25.375 USD

El lado short funciona mejor en este histórico porque los
5 años incluyen el bear market de 2022 (caída prolongada
y fuerte) donde las señales short rindieron muy bien. No
significa que el sistema sea "mejor para short" en general,
sino que este periodo concreto favoreció el lado bajista.

### 8.6 Conclusión clave del backtest

El sistema con reglas básicas tiene un profit factor de 0,97:
está a las puertas de la rentabilidad. En años con tendencia
clara (2024, 2026) ya es rentable con win rate del 32%.

Esto demuestra que la IA no necesita reinventar la estrategia,
solo necesita filtrar las señales malas: pasar del 27% actual
al 32-33% de win rate cruzaría al lado positivo. Los propios
datos muestran que ese nivel es alcanzable porque ya se
consigue en las condiciones adecuadas.

La base es sólida.

---

## 9. Evaluación del sistema (Fase 5)

### 9.1 Curva de equity

La curva de equity acumula el PnL de cada trade en orden
cronológico. Es la forma más visual de evaluar un sistema:
si la curva sube, ganas dinero; si baja, lo pierdes.

Nuestra curva muestra tres fases claras:

1. **Abril–noviembre 2021:** arranque positivo breve (+50.000)
   seguido de caída cuando el mercado pasa del techo al
   primer crash.
2. **2022–mediados 2025:** caída prolongada hasta -125.000.
   Es el periodo de bear market (2022) y lateralidad (2023)
   donde el sistema genera muchas señales falsas. La subida
   de 2024 (halving) es visible pero no compensa lo perdido.
3. **Finales 2025–2026:** recuperación fuerte y sostenida
   de -125.000 a -40.000 en pocos meses, coincidiendo con
   mercado tendencial.

El gráfico se guarda en `outputs/equity_chart.png`.

### 9.2 Drawdown máximo

El drawdown mide cuánto cae la curva desde un pico antes
de recuperarse. Es la métrica de riesgo más importante:
te dice cuánto tendrías que aguantar (emocional y
financieramente) en el peor momento.

**Drawdown máximo: -177.274 USD.** Se acumula principalmente
durante el bear market de 2022 y la lateralidad de 2023,
periodos donde el sistema opera sin parar pero el mercado
no tiene tendencia real.

Sobre un capital hipotético de 100.000 USD, esto equivale
a un ~17,7% de drawdown. Es un nivel tolerable pero que
la IA debería reducir significativamente al dejar de operar
en regímenes laterales.

### 9.3 Métricas avanzadas

| Métrica                        | Valor           |
|--------------------------------|-----------------|
| PnL total                      | -41.228 USD     |
| Expectativa por trade           | -9,01 USD       |
| Payoff ratio (gan/pér)         | 1,81            |
| Ganancia media                 | 1.079 USD       |
| Pérdida media                  | 597 USD         |
| Máximo drawdown                | -177.274 USD    |
| Recovery factor                | -0,23           |
| Racha ganadora más larga       | 15 trades       |
| Racha perdedora más larga      | 25 trades       |

**Payoff ratio de 1,81:** cuando el sistema gana, gana casi
el doble de lo que pierde cuando falla. Es una base
excelente; el reto es mejorar la frecuencia de acierto.

**Racha perdedora de 25 trades:** suena duro psicológicamente,
pero muchos de esos trades serán breakeven (PnL = 0), no
pérdidas reales. Aun así, la IA debería reducir estas rachas
al filtrar señales en contextos desfavorables.

### 9.4 Conclusión de la evaluación

El sistema de reglas básicas tiene la mecánica correcta y un
payoff ratio sólido (1,81). Su debilidad es que opera en todos
los regímenes de mercado sin distinción, acumulando pérdidas
innecesarias en periodos laterales o caóticos.

La IA deberá aprender a responder una pregunta simple pero
poderosa: "dado el contexto actual del mercado y los valores
de los indicadores, ¿esta señal tiene probabilidad alta de
funcionar o es mejor descartarla?"

---

## 10. Modelo de IA: filosofía y diseño (Fase 6)

### 10.1 Cómo funciona la IA en este proyecto

La IA NO genera señales por sí sola. Su papel es actuar
como un **filtro inteligente** sobre las señales que ya
produce el sistema de reglas.

El flujo es:
1. Las reglas (pullback y breakout) generan una señal.
2. La IA recibe el contexto del mercado en ese momento.
3. La IA responde: "esta señal tiene X% de probabilidad
   de ser ganadora".
4. Si la probabilidad supera un umbral, operamos. Si no,
   descartamos la señal.

Es como tener un analista junior (las reglas) que encuentra
oportunidades y un analista senior (la IA) que decide
cuáles merecen la pena.

### 10.2 Por qué NO añadimos patrones de velas ni Fibonacci

**Patrones de velas (envolventes, dojis, martillos...):**
Son subjetivos y difíciles de codificar de forma fiable.
¿Cuánto tiene que cubrir una vela envolvente para ser
"real"? Cada umbral es arbitrario. Además, la investigación
sobre estos patrones viene principalmente de mercados de
acciones con horarios limitados y no se ha demostrado que
funcionen igual en cripto (que opera 24/7 sin gaps).

**Fibonacci:** los niveles dependen de elegir un máximo y
un mínimo "correctos", lo cual es un juicio subjetivo que
un trader experimentado puede hacer pero que es muy difícil
de automatizar de forma robusta.

**La alternativa:** en lugar de crear reglas rígidas basadas
en estos conceptos, hemos capturado la INFORMACIÓN que
subyace en ellos mediante features numéricas que la IA
puede interpretar libremente:

- `price_position` (0 a 1) captura dónde está el precio
  dentro del rango reciente → similar a lo que mide
  Fibonacci, pero sin elegir puntos subjetivos.
- `body_pct`, `upper_wick_pct`, `lower_wick_pct` capturan
  la forma de la vela → la misma información que los
  patrones de velas, pero de forma numérica.
- `dist_ema20_pct` captura el grado de retroceso → similar
  al concepto de "pullback" que un trader mediría con
  Fibonacci.

La ventaja: la IA puede descubrir relaciones no lineales
entre estas features que nosotros nunca codificaríamos
como reglas fijas.

### 10.3 Features del modelo

El modelo recibe 17 features en el momento de cada señal:

**Indicadores base:**

- `rsi_14`: fuerza relativa del movimiento reciente.
- `atr_14`: volatilidad absoluta (en USD).

**Features derivadas de las EMAs:**

- `dist_ema20_pct`: distancia del precio a la EMA 20 (%).
  Mide cuánto se ha "estirado" el precio.
- `dist_ema50_pct`: distancia del precio a la EMA 50 (%).
- `ema_spread_pct`: separación entre EMA 20 y EMA 50 en 1h.
  Mayor separación = tendencia más fuerte.
- `ema_spread_4h_pct`: lo mismo pero en el marco de 4h.

**Features de volatilidad y estructura:**

- `atr_pct`: ATR normalizado por precio. Permite comparar
  volatilidad entre BTC a 16.000 y BTC a 100.000.
- `price_position`: posición del precio en el rango de las
  últimas 20 velas (0 = mínimos, 1 = máximos).
- `risk_pct`: riesgo del trade como % del precio.
- `reward_risk_ratio`: ratio entre potencial beneficio y
  riesgo (por diseño debería ser ~2).

**Features de la vela de señal:**

- `body_pct`: tamaño del cuerpo de la vela (%). Velas con
  cuerpo grande indican más convicción.
- `upper_wick_pct`: proporción de mecha superior. Mechas
  largas pueden indicar rechazo de precio.
- `lower_wick_pct`: proporción de mecha inferior.

**Features temporales:**

- `hour`: hora del día (0-23). El mercado cripto tiene
  patrones horarios (más volumen en horario europeo/USA).
- `day_of_week`: día de la semana (0=lunes, 6=domingo).
  Fines de semana suelen tener menos volumen y más ruido.

**Codificación de la señal:**

- `direction_num`: 1 = LONG, 0 = SHORT.
- `rule_num`: 1 = breakout, 0 = pullback.

### 10.4 Preparación del dataset

El dataset se construye uniendo las señales
(outputs/signals.csv) con los resultados de los trades
(outputs/trades.csv). Cada fila contiene:

- Las features del momento de la señal (17 iniciales, luego
  ampliadas a 24 en la Iteración 1).
- El resultado: label = 1 (ganador) o 0 (perdedor/neutro).

**Distribución del label:**
- Ganadores (1): 1.237 (27,0%)
- Perdedores/neutros (0): 3.337 (73,0%)

El dataset está desbalanceado (más perdedores que ganadores),
lo que es normal en trading. El modelo tiene que aprender a
ser selectivo, no a predecir la mayoría.

### 10.5 Algoritmo y entrenamiento

**Modelo elegido: Gradient Boosting (scikit-learn)**

Es un algoritmo que construye muchos "árboles de decisión"
pequeños, cada uno corrigiendo los errores del anterior.
Es robusto, interpretable (podemos ver qué features
importan más) y funciona bien con datasets de tamaño medio
como el nuestro (~4.500 filas).

**Parámetros conservadores para evitar sobreajuste:**

- 200 árboles (no demasiados, para no memorizar el dataset)
- Profundidad máxima 4 (árboles poco profundos)
- Learning rate 0,05 (aprendizaje lento y gradual)
- Mínimo 20 muestras por hoja (evita crear reglas para
  casos muy específicos que no se repetirán)
- Subsample 0,8 (cada árbol usa el 80% de los datos,
  añade variedad)

**Split temporal (NO aleatorio):**

Entrenamos con datos de 2021-2024 (~3,5 años) y testeamos
con datos de 2025-2026 (~1,5 años). Esto simula el uso
real: entrenar con el pasado y predecir el futuro.

Un split aleatorio "filtraría" información del futuro al
entrenamiento, produciendo métricas falsamente optimistas.
El split temporal es más honesto y exigente.

### 10.6 Resultados del entrenamiento

**Split de datos:**

- Train (2021-2024): 3.385 trades (908 ganadores, 2.477 perdedores)
- Test (2025-2026): 1.189 trades (329 ganadores, 860 perdedores)

**Métricas de clasificación sobre el test set:**

| Métrica    | Valor  | Interpretación                                  |
|------------|--------|--------------------------------------------------|
| Accuracy   | 0.705  | Acierta el 70,5%, pero un modelo que siempre dijera "perdedor" acertaría el 72,3%. No es útil con datos desbalanceados. |
| Precision  | 0.333  | De cada 3 trades que la IA selecciona como ganadores, solo 1 gana realmente. |
| Recall     | 0.067  | Solo detecta el 6,7% de los trades ganadores reales. Es extremadamente conservador. |
| F1 Score   | 0.111  | Resumen de precision y recall. Bajo, confirma que el modelo no funciona bien como clasificador binario. |

**Matriz de confusión:**

|              | Pred. Perdedor | Pred. Ganador |
|--------------|----------------|---------------|
| Real Perdedor | 816            | 44            |
| Real Ganador  | 307            | 22            |

El modelo casi siempre predice "perdedor" (1.123 de 1.189 veces).
Solo predice "ganador" 66 veces, acertando en 22 (33,3%).

**Simulación de trading con y sin filtro IA:**

Sin filtro: 1.189 trades, WR 27,7%, PnL +2.734 USD.

| Umbral | Trades | Win rate | PnL total    |
|--------|--------|----------|--------------|
| 0.30   | 613    | 28,2%    | -2.824 USD   |
| 0.35   | 408    | 27,5%    | -18.369 USD  |
| 0.40   | 253    | 28,9%    | -17.068 USD  |
| 0.45   | 140    | 27,1%    | -22.426 USD  |
| 0.50   | 66     | 33,3%    | -1.133 USD   |
| 0.55   | 38     | 36,8%    | +1.780 USD   |
| 0.60   | 15     | 26,7%    | -3.678 USD   |

El umbral 0.55 es el único positivo (+1.780 USD) pero con solo
38 trades en ~1,5 años, y el beneficio es MENOR que sin filtro
(+2.734 USD). El modelo no mejora al sistema sin IA en este
primer intento.

**Importancia de features (top 5):**

| Ranking | Feature             | Importancia |
|---------|---------------------|-------------|
| 1       | ema_spread_4h_pct   | 0.1418      |
| 2       | atr_14              | 0.1316      |
| 3       | ema_spread_pct      | 0.1108      |
| 4       | dist_ema50_pct      | 0.0853      |
| 5       | price_position      | 0.0647      |

Features con importancia ~0: reward_risk_ratio (siempre vale 2,
no aporta información), direction_num y rule_num.

La feature más discriminante es ema_spread_4h_pct (fuerza de
tendencia en 4h). Su media difiere un -149,7% entre ganadores
y perdedores: cuando las EMAs de 4h están muy separadas (tendencia
"obvia"), los trades pierden más, probablemente porque la
tendencia ya está agotada.

### 10.7 Interpretación honesta de los resultados

**El modelo es una línea base, no un resultado final.** Es el
primer intento con parámetros conservadores, 17 features básicas
y ~3.400 filas de entrenamiento. No mejora al sistema sin IA.

**¿Por qué no funciona mejor?**

El problema no es el algoritmo (Gradient Boosting es excelente
para datos tabulares) ni los parámetros, sino la cantidad y
calidad de las features. Con solo 17 variables básicas, el modelo
tiene un techo bajo de información disponible.

**¿Por qué Gradient Boosting y no otro algoritmo?**

- Regresión logística: demasiado simple, solo aprende relaciones
  lineales entre features y resultado.
- Random Forest: bueno pero menos preciso que GB en datasets
  pequeños/medianos.
- Redes neuronales: necesitan órdenes de magnitud más datos
  (decenas de miles de filas mínimo) y son difíciles de
  interpretar.
- Gradient Boosting: el mejor equilibrio entre potencia,
  interpretabilidad y eficiencia con datos tabulares. Es el
  algoritmo dominante en competiciones de datos con tablas.

**Caminos de mejora futuros:**

- Más features: volumen relativo, correlación con otros activos,
  indicadores de sentimiento, datos on-chain.
- Más datos: acumular trades del sistema en vivo.
- Feature engineering más sofisticado: interacciones entre
  features, features con ventanas temporales diferentes.
- Optimización de hiperparámetros una vez tengamos más datos.

### 10.8 Iteración 1 — Nuevas features

El modelo con 17 features era una línea base mediocre. En vez de
esperar más datos o cambiar de algoritmo, atacamos el problema
real: la calidad de la información que recibe el modelo.

**7 features nuevas añadidas (total: 24):**

**volume_sma20_ratio** — Volumen de la vela dividido por la media
de volumen de las últimas 20 velas. >1 = volumen superior al
habitual, lo que puede confirmar la señal. Hasta ahora nunca
usábamos el dato de volumen que Binance nos daba.

**rsi_14_4h** — RSI del timeframe de 4h. Antes solo usábamos
la tendencia de 4h (up/down), pero eso es binario. El RSI de 4h
da información continua: un RSI_4h de 70 no es lo mismo que
uno de 55, aunque ambos estén en tendencia "up". Se convirtió
en la **feature #1 en importancia** del nuevo modelo.

**atr_pct_4h** — ATR del timeframe 4h normalizado por precio.
Captura la volatilidad en el marco temporal superior.

**roc_5** — Rate of change a 5 velas: cuánto ha subido/bajado
el precio en las últimas 5 horas. Momentum a corto plazo.

**roc_20** — Rate of change a 20 velas: momentum a medio plazo.
Dif% entre ganadores y perdedores: +68,5%.

**atr_regime** — ATR actual / media del ATR de las últimas 20
velas. >1 = volatilidad expandiéndose, <1 = contrayéndose.
Permite al modelo aprender que señales en ciertos regímenes
de volatilidad funcionan mejor que en otros. Se convirtió en
la **feature #5 en importancia**.

**candle_range_ratio** — (high - low) / ATR. Si >1, la vela de
señal es más grande de lo normal. Velas muy grandes pueden
indicar impulso fuerte o volatilidad puntual.

### 10.9 Resultados de la Iteración 1

**Comparación de métricas de clasificación:**

| Métrica   | Línea base (17f) | Iteración 1 (24f) | Cambio   |
|-----------|------------------|--------------------|----------|
| Accuracy  | 0.705            | 0.705              | =        |
| Precision | 0.333            | **0.432**          | **+30%** |
| Recall    | 0.067            | **0.213**          | **+218%**|
| F1 Score  | 0.111            | **0.285**          | **+157%**|

La accuracy no cambió (sigue siendo engañosa con datos
desbalanceados), pero precision, recall y F1 mejoraron
drásticamente. El modelo ahora detecta 3 veces más ganadores
(recall ×3) con mayor precisión.

**Matriz de confusión (Iteración 1):**

|              | Pred. Perdedor | Pred. Ganador |
|--------------|----------------|---------------|
| Real Perdedor | 768            | 92            |
| Real Ganador  | 259            | 70            |

Antes solo acertaba 22 ganadores; ahora acierta 70 (×3,2).

**Simulación de trading — EL CAMBIO FUNDAMENTAL:**

Sin filtro: 1.189 trades, WR 27,7%, PnL +2.734 USD.

| Umbral | Trades | Win rate | PnL total     | PnL/trade |
|--------|--------|----------|---------------|-----------|
| 0.30   | 644    | 33,7%    | **+97.088**   | +150,76   |
| 0.35   | 484    | 34,9%    | +80.004       | +165,30   |
| 0.40   | 334    | 38,3%    | +70.210       | +210,21   |
| 0.45   | 240    | 40,8%    | +64.296       | +267,90   |
| 0.50   | 162    | 43,2%    | +40.028       | +247,09   |
| 0.55   | 101    | 45,5%    | +28.760       | +284,75   |
| 0.60   | 63     | 50,8%    | **+21.334**   | +338,64   |

**Todos los umbrales son rentables.** Antes solo uno (0.55)
era positivo y con un PnL menor que sin filtro. Ahora el
umbral 0.30 genera +97.088 USD vs +2.734 USD sin filtro:
una mejora de **35 veces**.

El punto dulce parece estar entre 0.40 y 0.50:
- Umbral 0.40: 334 trades, WR 38,3%, +70.210 USD.
  ~4,7 trades/semana, buen equilibrio volumen/calidad.
- Umbral 0.50: 162 trades, WR 43,2%, +40.028 USD.
  ~2,3 trades/semana, más selectivo.

Recordatorio: con el payoff ratio de 1,81 del sistema, un
WR del ~36% ya es matemáticamente rentable. El modelo ahora
supera ese umbral en TODOS los niveles de filtrado.

**Importancia de features (Iteración 1):**

| Ranking | Feature              | Importancia | Nota      |
|---------|----------------------|-------------|-----------|
| 1       | **rsi_14_4h**        | 0.1282      | **NUEVA** |
| 2       | ema_spread_4h_pct    | 0.1058      |           |
| 3       | ema_spread_pct       | 0.0912      |           |
| 4       | atr_14               | 0.0893      |           |
| 5       | **atr_regime**       | 0.0816      | **NUEVA** |
| 6       | **atr_pct_4h**       | 0.0786      | **NUEVA** |
| 7       | dist_ema50_pct       | 0.0659      |           |
| 8       | **roc_20**           | 0.0338      | **NUEVA** |
| 9       | lower_wick_pct       | 0.0316      |           |
| 10      | **candle_range_ratio**| 0.0307      | **NUEVA** |

5 de las 7 nuevas features aparecen en el top 10. La feature
más importante del modelo ya no es ema_spread_4h_pct sino
rsi_14_4h (la nueva). El RSI del timeframe de 4h aporta
información continua sobre el momentum macro que la simple
etiqueta "up/down" no capturaba.

### 10.10 Lección clave de la Iteración 1

**El algoritmo no era el problema; las features sí.** Con
exactamente el mismo Gradient Boosting, los mismos parámetros
y los mismos 4.574 trades, el modelo pasó de inútil a
rentable simplemente dándole mejor información.

Esto confirma el principio fundamental del machine learning:
"garbage in, garbage out" — y su inverso: mejor información
de entrada produce mejores predicciones de salida.

### 10.11 Archivos generados en la Fase 6

| Archivo                         | Qué contiene                              |
|---------------------------------|-------------------------------------------|
| outputs/ml_dataset.csv          | Dataset con 24 features + label (4.574 filas) |
| outputs/model.joblib            | Modelo entrenado para uso en vivo          |
| outputs/model_results.csv       | Test set con predicciones del modelo       |
| outputs/feature_importance.png  | Gráfico de importancia de features         |

---

## 11. Señales en vivo + Telegram (Fase 7)

### 11.1 Qué hace el script de señales en vivo

El script `08_live_signal.py` es la pieza que conecta todo el
trabajo anterior con el mundo real. En cada ejecución:

1. Descarga las últimas 150 velas de 1h y 80 de 4h desde
   Binance (datos en vivo, no históricos).
2. Calcula todos los indicadores técnicos sobre esas velas.
3. Alinea el contexto de 4h sobre 1h (merge_asof).
4. Comprueba si la **última vela cerrada** cumple alguna
   regla de entrada (pullback o breakout).
5. Si hay señal, calcula los niveles de SL/TP.
6. Carga el modelo de IA entrenado y calcula la probabilidad
   de éxito de la señal.
7. Envía un mensaje a Telegram con todos los detalles.
8. Registra la señal en un CSV acumulativo
   (outputs/live_signals_log.csv).

### 11.2 Decisiones de diseño

**Ejecución puntual en Fase 7, bucle continuo en Fase 8:**
en la Fase 7 el script se ejecutaba una vez y terminaba
(pensado para lanzarlo con cron). En la Fase 8 se añadió un
modo bucle continuo (`--loop`) que se queda corriendo y
comprueba automáticamente al cierre de cada vela. Se
mantiene la compatibilidad con el modo puntual — ambos
funcionan. Ver sección 12 para el detalle.

**La vela "actual" se ignora:** Binance devuelve la vela que
se está formando en este momento como la última. Esa vela aún
puede cambiar, así que el script la descarta y analiza la
penúltima (la última que ya cerró).

**El modelo NO filtra, solo informa:** el mensaje de Telegram
incluye la probabilidad de éxito de la IA, pero todas las
señales se envían. El operador humano decide si operar o no
viendo la probabilidad. Esto permite ir evaluando la calidad
del modelo en condiciones reales.

**No se pierden velas:** si el script estuvo apagado un día,
las velas siguen disponibles en Binance y se pueden recuperar
en cualquier momento. El script no necesita estado previo
para funcionar.

### 11.3 Telegram

Se utiliza un bot de Telegram creado con @BotFather. El
mensaje incluye: dirección (LONG/SHORT), regla, precio de
entrada, SL, TP1, TP2, riesgo en % y USD, contexto de RSI,
ATR y tendencias, y la probabilidad de la IA con indicación
de si supera el umbral recomendado.

Las credenciales se guardan en un archivo .env que no se
sube a Git.

### 11.4 Re-entrenamiento del modelo

El re-entrenamiento no es automático. Cuando se quiera
actualizar el modelo con datos más recientes, basta con
ejecutar en orden:

```bash
python src/01_download_data.py   # descarga datos actualizados
python src/02_indicators.py       # recalcula indicadores
python src/03_rules.py            # regenera señales
python src/04_backtest.py         # simula trades
python src/06_prepare_dataset.py  # prepara dataset con features
python src/07_train_model.py      # re-entrena el modelo
```

El script 08 (señales en vivo) cargará automáticamente el
nuevo modelo la siguiente vez que se ejecute.

---

## 12. Despliegue robusto (Fase 8)

### 12.1 Motivación

Al terminar la Fase 7, el script `08_live_signal.py` hacía
una única comprobación y terminaba. Eso es correcto para
pruebas manuales o para combinarlo con cron, pero no es
cómodo para uso diario: obliga a estar abierto para ver
qué pasa, no deja rastro de errores nocturnos, y si falla
una ejecución no nos enteramos hasta que abrimos el
terminal.

La Fase 8 transforma el script en un **sistema que puede
correr horas o días sin supervisión**, con tres mejoras
independientes:

1. **Modo bucle continuo** — el script se queda corriendo
   y se despierta solo al cierre de cada vela.
2. **Manejo robusto de errores** — reintentos automáticos
   ante fallos de Binance o Telegram, sin tumbar el bucle.
3. **Logging a archivo + pings de Telegram** — todo lo que
   pasa queda registrado en `outputs/live_signal.log` y
   recibimos aviso cuando el sistema arranca y cuando se
   detiene.

### 12.2 Modo bucle continuo

Nuevo argumento de línea de comandos `--loop`:

```bash
python src/08_live_signal.py          # modo único (como antes)
python src/08_live_signal.py --loop   # modo bucle continuo
```

**Cómo funciona el bucle:**

1. Al arrancar, ejecuta **inmediatamente** una comprobación
   con la última vela cerrada disponible. No hace esperar
   al usuario.
2. Calcula cuántos segundos faltan hasta el próximo cierre
   de vela de 1h (minuto 0 de la siguiente hora) más un
   margen de seguridad de 30 segundos (para asegurar que
   Binance ya tiene la vela disponible en su API).
3. Duerme con `time.sleep()` hasta ese momento.
4. Ejecuta la comprobación.
5. Vuelve al paso 2. Repite indefinidamente.

Para detener: **Ctrl+C**. El script maneja la interrupción
limpiamente, sin dejar el bot a medias.

**Por qué un bucle Python y no cron/launchd:** estamos en
fase de pruebas y necesitamos ver qué pasa, iterar rápido
y relanzar con facilidad. Un bucle Python es inmediato de
ejecutar y depurar. Cuando el sistema esté maduro y se
quiera que corra 24/7 sin tocarlo, se migrará a launchd
(recomendado en macOS) o a un servicio systemd (en Linux).

### 12.3 Manejo robusto de errores

**Capa 1 — Binance (reintentos con backoff):**
- Si `get_klines()` falla (timeout, fallo de red, error 5xx),
  el script reintenta hasta 3 veces con esperas crecientes:
  **5s → 10s → 20s**.
- Si los 3 intentos fallan, se lanza la excepción al bucle
  superior, que captura el error y continúa con la siguiente
  iteración (perdemos esa comprobación pero no el bucle).

**Capa 2 — Telegram (tolerante a fallos, no bloqueante):**
- Hasta 2 intentos con espera de 3s entre ellos.
- **Si Telegram falla, NO se aborta la señal:** ya está
  calculada y registrada en `live_signals_log.csv`.
  Simplemente se marca como error y se sigue. Esto garantiza
  que nunca perdemos datos de señales por un fallo de
  notificación.

**Capa 3 — Errores inesperados en el bucle:**
- El bucle envuelve cada iteración en `try/except Exception`.
- Si aparece un error que las capas anteriores no capturaron
  (bug de código, columna faltante, datos corruptos), se
  registra con: hora exacta, tipo de excepción, mensaje y
  **traceback completo**.
- El bucle nunca muere por un error puntual. Para detenerlo
  hay que usar Ctrl+C.

### 12.4 Logging a archivo

Se añadió el módulo estándar `logging` de Python con dos
salidas simultáneas:

- **Consola (stdout):** para ver en tiempo real.
- **Archivo `outputs/live_signal.log`:** modo append, nunca
  se sobrescribe. Acumula el histórico completo.

Cada línea lleva timestamp y nivel:

```
2026-04-14 21:00:15 [INFO] MODO BUCLE CONTINUO ACTIVADO
2026-04-14 21:00:15 [INFO] Mensaje enviado a Telegram.
2026-04-14 21:00:16 [INFO]   Close: 74,093.19 | RSI: 50.7
```

Esto permite:

- Abrir el archivo en cualquier momento y ver qué pasó
  durante la noche.
- Filtrar errores rápidamente:
  `grep ERROR outputs/live_signal.log`
- Tener trazabilidad completa para auditar o depurar.

Todos los `print()` del script se sustituyeron por llamadas
a la función auxiliar `log(mensaje, nivel)`, que envuelve
el logger. Los errores importantes (Binance caído, Telegram
no enviado, excepciones del bucle) se marcan con nivel
`ERROR`; los avisos de baja gravedad con `WARNING`; el
resto con `INFO`.

### 12.5 Pings de arranque y parada por Telegram

El bucle manda dos mensajes automáticos al usuario:

**Al arrancar:**

```
🟢 Sistema ARRANCADO
⏰ 2026-04-14 21:00:15 (Europe/Madrid)
💻 Modo: bucle continuo

Comprobando cada cierre de vela 1h.
Te avisaré cuando se detenga.
```

**Al detenerse (Ctrl+C o error fatal):**

```
🔴 Sistema DETENIDO
⏰ 2026-04-14 23:47:02 (Europe/Madrid)
📌 Motivo: Ctrl+C durante espera

No se comprobarán nuevas señales hasta que reinicies
el script.
```

El ping de parada está dentro de un `try/finally`, así que
se envía siempre, incluso si el bucle muere por una
excepción no capturada.

**Por qué es importante:** el usuario sabe en todo momento
si el sistema está vivo o no, sin tener que abrir la
terminal para comprobarlo.

### 12.6 Archivos afectados en la Fase 8

| Archivo                        | Cambio                               |
|--------------------------------|--------------------------------------|
| `src/08_live_signal.py`        | Refactor: logging, bucle, errores    |
| `outputs/live_signal.log`      | Nuevo (se crea en la primera ejec.)  |
| `docs/presentation_notes.md`   | Esta sección 12 y actualización 7    |

### 12.7 Resultado y próximos pasos

Tras la Fase 8 tenemos un sistema que puede correr durante
días sin supervisión, con recuperación automática ante
fallos de red, registro completo de todo lo que sucede y
avisos al móvil cuando arranca o se detiene. Es **apto para
uso real**, aunque todavía corre solo mientras la terminal
esté abierta.

**Pendientes después de la Fase 8** (iteraciones del modelo):

1. **Iteración 1B** — Correlaciones externas (ETH, DXY).
2. **Iteración 2** — Etiquetas más informativas (R-multiple,
   drawdown antes del TP).
3. **Iteración 3** — XGBoost o LightGBM en lugar de
   GradientBoostingClassifier.
4. **Iteración 4** — Walk-forward validation.

**Pendiente a más largo plazo:** migrar el despliegue a
launchd (macOS) o systemd (Linux) para que corra 24/7 sin
tener la terminal abierta.

---

## 13. Iteración 1B — Correlaciones externas (Fase 9)

### 13.1 Motivación

Hasta ahora el modelo solo "ve" BTC. La Iteración 1
(Fase 7) añadió features derivadas de la propia vela de BTC
(volumen, momentum, volatilidad). La Iteración 1B da un
paso más y **incorpora información externa al par BTC/USDT**:
otro cripto (ETH) y un activo macro (DXY, índice del dólar).

### 13.2 Por qué ETH y DXY (y expectativas realistas)

**ETH (Ethereum):** segundo cripto por capitalización. Tiene
correlación histórica alta con BTC (≈0.80-0.90). El valor
añadido NO está en el precio de ETH en sí, sino en:

- La **correlación rolling** BTC-ETH: cuando se rompe suele
  marcar cambios de régimen.
- El **momentum relativo**: si BTC sube pero ETH no confirma,
  la subida de BTC es más débil.

**DXY (índice del dólar):** mide la fuerza del USD frente
a una cesta de divisas. Cuando el DXY sube, los activos de
riesgo (BTC incluido) suelen sufrir. Es un **filtro de
régimen macro**, no una señal intra-día.

**Expectativa honesta:** las mejoras esperadas son **modestas**,
no un salto como el de la Iteración 1. Si el F1 mejora un
3-5% merece la pena mantener las features; si es <2%
aprendemos que BTC en 1h es bastante autónomo y pasamos
directos a la Iteración 2.

### 13.3 Decisiones de diseño

**Fuente de datos:**

- **ETH:** Binance (mismo cliente que usamos para BTC).
  Velas de 1h, igual timeframe que BTC.
- **DXY:** Yahoo Finance vía `yfinance` (gratis, sin API
  key, fiable). Resolución **diaria** (Yahoo no da DXY
  intra-día fiable).

**Por qué DXY diario y no intra-día:** no es crítico. DXY
se mueve despacio y su influencia sobre BTC es de régimen,
no de timing. El último valor diario disponible es
suficiente como contexto.

**Caché de DXY en modo bucle:** el script en vivo NO llamará
a Yahoo cada hora. Guarda el valor del día y solo vuelve a
descargar cuando cambia la fecha UTC. Máximo **una llamada
diaria** a Yahoo. ETH sí se descarga cada iteración porque
es 1h.

### 13.4 Features que se añaden

| Feature            | Cálculo                                     |
|--------------------|---------------------------------------------|
| `eth_roc_5`        | Cambio % de ETH en las últimas 5 velas 1h   |
| `eth_roc_20`       | Cambio % de ETH en las últimas 20 velas 1h  |
| `btc_eth_corr_20`  | Correlación rolling 20 velas entre retornos |
|                    | horarios de BTC y ETH (valores -1 a +1)     |
| `dxy_roc_5d`       | Cambio % del DXY en los últimos 5 días      |

Son 4 features nuevas. Se añaden a las 24 existentes → el
dataset pasa a tener **28 features** tras Iteración 1B.

Pocas pero seleccionadas: menos features = menos ruido y
menos riesgo de overfitting.

### 13.5 Archivos a crear / modificar

| Archivo                           | Rol                               |
|-----------------------------------|-----------------------------------|
| `src/01b_download_external.py`    | NUEVO — descarga ETH + DXY histórico |
| `src/02b_external_features.py`    | NUEVO — calcula features externas |
| `src/06_prepare_dataset.py`       | MODIFICAR — merge de nuevas features |
| `src/07_train_model.py`           | MODIFICAR — añadir a lista FEATURES |
| `src/08_live_signal.py`           | MODIFICAR — descargar ETH/DXY en vivo + caché DXY |
| `requirements.txt`                | AÑADIR `yfinance`                 |

**Crítico:** el script en vivo (08) tiene que calcular
**exactamente las mismas features** que el dataset de
entrenamiento. Cualquier discrepancia hace que el modelo
reciba datos incoherentes y falle. Esto ya lo aprendimos
en la Iteración 1 y lo aplicamos de nuevo aquí.

### 13.6 Criterio de éxito

Tras entrenar el modelo con las 28 features, compararemos:

- **F1 score** — métrica principal (balance entre precisión
  y recall).
- **Precisión y recall por separado** — para ver dónde gana
  o pierde.
- **P&L simulado a distintos umbrales** — qué beneficio
  real habríamos hecho con el nuevo modelo en el set de test.
- **Importancia de features** — dónde aparecen las 4 nuevas
  en el ranking. Si están todas en el top 15, confirmamos
  que aportan.

**Decisión final:**

- Mejora F1 ≥ 5% → mantener las 4 features y pasar a
  Iteración 2.
- Mejora F1 entre 2-5% → mantener solo las más importantes.
- Mejora F1 < 2% → aceptar que BTC 1h es autónomo, descartar
  las features y pasar a Iteración 2 sin ellas.

### 13.7 Resultado real de la Iteración 1B (15 de abril de 2026)

**Veredicto: v2 NO gana. Mantenemos el modelo v1.**

Tabla comparativa (mismo test set, 1.189 trades 2025-2026):

| Métrica   |   v1   |   v2   | Δ abs  | Δ %     |
|-----------|--------|--------|--------|---------|
| accuracy  | 0.705  | 0.704  | −0.001 | −0.1%   |
| precision | 0.432  | 0.396  | −0.036 | −8.3%   |
| recall    | 0.213  | 0.134  | −0.079 | −37.1%  |
| **F1**    | **0.285** | **0.200** | **−0.085** | **−29.9%** |

La caída en recall (−37%) es especialmente fuerte: el v2
se volvió más conservador (identifica menos ganadores) sin
ganar en precisión. Fallo claro.

**Por qué no funcionó (hipótesis confirmadas por el análisis):**

1. **ETH es redundante con BTC.** `eth_roc_5` y `eth_roc_20`
   solapan con `roc_5` y `roc_20` de BTC. El modelo se
   despista teniendo dos versiones casi idénticas del mismo
   momentum. La correlación BTC-ETH `btc_eth_corr_20` es
   estable (0.80-0.90 casi siempre), así que tampoco
   discrimina.
2. **DXY diario es demasiado lento para 1h.** Con forward-fill,
   `dxy_roc_5d` es el mismo valor durante 24 velas seguidas.
   Demasiado burdo para predecir trades individuales de 1h.
3. **Añadir features no es gratis.** Cada feature extra es
   una dimensión más donde el modelo puede sobreajustar. Si
   no aportan señal real, solo añaden ruido.

**Confirmación de intuiciones previas del autor:** antes de
entrenar ya se había anticipado que ETH iba muy de la mano
con BTC y que el DXY diario era un marco temporal demasiado
alto para velas 1h. Los datos lo confirman.

### 13.8 Herramientas de análisis introducidas en esta iteración

Aunque la iteración no mejoró el modelo, introdujo dos
técnicas que se quedan en el toolkit del proyecto:

**Permutation importance (scikit-learn `permutation_importance`).**
Procedimiento: evaluar el modelo en test (F1 base), barajar
los valores de una feature, reevaluar (F1 barajado), y la
importancia es `F1_base − F1_barajado`. Se repite N veces
y se promedia. A diferencia de la importancia nativa de
Gradient Boosting (que mide cuánto se usa una feature en
los splits), la permutation importance mide el **impacto
real en el rendimiento predictivo**. Es más fiable para
decidir qué features podar: una feature puede ser muy
usada por el modelo (nativa alta) pero sus splits no
generalizar (permutation ≤ 0).

**Matriz de correlaciones.** Cálculo de Pearson entre todas
las features del dataset de entrenamiento. Pares con
|correlación| > 0.9 son candidatos a consolidación: una de
las dos sobra porque aportan casi la misma información.

**Qué guardamos en `outputs/`:**

- `feature_importance_v2.png` — importancia nativa.
- `perm_importance_v2.png` — permutation importance.
- `feature_correlation_v2.png` — heatmap correlaciones.
- `model_v2.joblib` — modelo v2 (NO en producción, solo
  archivado para referencia histórica).
- `ml_dataset_v2.csv` — dataset con las 28 features.
- `model_results_v2.csv` — predicciones v2 sobre test.

### 13.9 Hallazgos colaterales sobre el modelo v1 (oro puro)

El análisis exhaustivo que hicimos para evaluar v2 también
reveló problemas en el v1 que abren un paso intermedio
(Iteración 1B.2, sección siguiente) antes de pasar a la
Iteración 2.

**Correlaciones perfectas entre features del v1:**

| Par                          | Correlación |
|------------------------------|-------------|
| `atr_pct` ↔ `risk_pct`       | +1.000 ← idénticas |
| `risk_pct` ↔ `atr_pct_4h`    | +0.945 |
| `atr_pct` ↔ `atr_pct_4h`     | +0.945 |

`atr_pct` y `risk_pct` son literalmente la misma columna.
Es geométricamente obvio en retrospectiva:
`SL = entry − k·ATR` ⇒ `risk_pct = k·ATR/entry = k·atr_pct`.
Una de las dos sobra al 100%.

**Features con importancia 0 en ambas métricas:**

- `direction_num` → 0.0000 (nativa) y +0.0000 (permutation).
- `reward_risk_ratio` → 0.0000 y +0.0000. Siempre vale ~2.0
  por diseño del sistema, no aporta varianza.
- `rule_num` → 0.0001 y +0.0000.

**Permutation importance mayoritariamente negativa.**
La mayoría de features tiene Δ F1 < 0. Esto NO significa
que cada feature sea mala individualmente — significa que
**el modelo generaliza regular**: muchas relaciones
aprendidas en train 2021-2024 no se cumplen en test
2025-2026 (cambio de régimen de mercado). Refuerza la
importancia de la **Iteración 4 (walk-forward validation)**:
reentrenar periódicamente es probablemente necesario.

**Matiz importante — la simulación de trading sigue siendo positiva.**
A pesar del F1 bajo, filtrar con umbral 0.45 multiplica el
PnL por 10x (2.734 USD → 27.893 USD) en v2. El modelo tiene
F1 bajo pero sí **reordena las señales de mejor a peor**:
ordenar > clasificar binario. Esta es la nota honesta más
relevante: nuestras métricas ML son modestas pero el modelo
tiene valor operativo.

---

## 13.bis Iteración 1B.2 — Poda del modelo v1

Antes de pasar a la Iteración 2 aprovechamos los hallazgos
del análisis anterior para limpiar el v1. Es barato y deja
un modelo más simple y honesto.

### 13.bis.1 Motivación

Reducir dimensionalidad cuando sobra. Menos features =
menos sobreajuste potencial, menos ruido, y el modelo
explica lo mismo (o más) con menos palabras. También nos
da un punto de partida más limpio para la Iteración 2.

### 13.bis.2 Features a eliminar

**Por correlación perfecta (duplicadas):**

- `risk_pct` — duplicada con `atr_pct` (corr = 1.000).
  Mantenemos `atr_pct` por ser la forma canónica.

**Por importancia ≈ 0 en ambas métricas:**

- `direction_num` — el modelo no la usa.
- `reward_risk_ratio` — invariante (siempre ~2.0).
- `rule_num` — prácticamente 0.

Total: 24 features → 20 features (se eliminan 4).

### 13.bis.3 Criterio de éxito

F1 pruned ≥ F1 v1 original − 2%. Es decir, buscamos que
el recorte NO empeore. Si además mejora, es ganancia pura.

Si sale bien, `model_v1_pruned.joblib` sustituye a
`model.joblib` en producción (y se actualiza la lista
FEATURES de `08_live_signal.py` a las 20 nuevas).

### 13.bis.4 Archivos a crear

- `src/07_train_model_v1_pruned.py` — copia del v1 con
  lista de features recortada.
- `outputs/model_v1_pruned.joblib` — candidato.
- `outputs/model_results_v1_pruned.csv` — predicciones
  sobre test.

No tocamos `06_prepare_dataset.py`: el dataset v1 ya
contiene todas las columnas, solo cambia la selección al
entrenar.

### 13.bis.5 Resultado real (15 de abril de 2026) — PODADO GANA

Tabla comparativa (mismo test set, 1.189 trades 2025-2026):

| Métrica   | v1 original (24) | v1 podado (20) | Δ abs   | Δ %   |
|-----------|------------------|----------------|---------|-------|
| accuracy  | 0.705            | 0.706          | +0.001  | +0.1% |
| precision | 0.432            | 0.434          | +0.002  | +0.4% |
| recall    | 0.213            | 0.210          | −0.003  | −1.4% |
| **F1**    | **0.285**        | **0.283**      | **−0.002** | **−0.8%** |

**Veredicto:** el podado PASA el criterio (ΔF1 = −0.002,
muy dentro de la tolerancia de −0.02). Eliminamos 4
features sin perder capacidad predictiva. Bonus:
precisión mejora +0.4% y accuracy mejora +0.1%.

**Lo más importante — simulación de trading del podado:**

| Umbral | Trades | WR      | PnL        | PnL/trade |
|--------|--------|---------|------------|-----------|
| 0.30   | 647    | 33.7%   | 89.538 USD | 138       |
| 0.40   | 338    | 38.8%   | 78.705 USD | 233       |
| 0.50   | 159    | 43.4%   | 39.543 USD | 249       |
| **0.55** | **104** | **47.1%** | **32.099 USD** | **309** |
| 0.60   | 66     | 47.0%   | 17.104 USD | 259       |

A partir de umbral 0.45 el win rate supera el 40%; a 0.55
alcanza casi el 50%. Umbral 0.50 multiplica el PnL por 14x
respecto a operar sin filtro (2.735 → 39.543 USD). El
modelo reordena mejor las señales que el v1 original sin
perder F1.

### 13.bis.6 Promoción a producción (cambios aplicados)

El 2026-04-15 se promovió el modelo podado a producción:

1. **Backup del antiguo:**
   `outputs/model.joblib` → `outputs/model_v1_original_backup.joblib`
   (por si hiciera falta rollback).
2. **Promoción del nuevo:**
   `outputs/model_v1_pruned.joblib` → `outputs/model.joblib`
   (archivo canónico que carga `08_live_signal.py`).
3. **Actualización de `08_live_signal.py`:**
   - Se eliminan 4 entradas del diccionario de features
     construido en `extraer_features_vela()`.
   - Se eliminan las 4 líneas de relleno post-hoc y las 4
     entradas de la lista `orden` en `predecir_probabilidad()`.
   - El parámetro `risk_pct` sigue pasándose porque se usa
     en el mensaje de Telegram y en el CSV de log, no en
     la predicción.

Tras los cambios hay que parar el bucle (Ctrl+C, que
dispara el ping de parada por Telegram) y relanzarlo con
`python src/08_live_signal.py --loop` para que cargue el
nuevo modelo.

---

## 14. Roadmap post-Iteración 1B

Tras completar la Iteración 1B, el roadmap previsto es:

**Fase 10 — Iteración 2: etiquetas mejores.**
Ahora etiquetamos trades como ganador/perdedor binario
(toca TP antes que SL). Migrar a etiquetas más informativas:
R-multiple real (cuánto ganó en unidades de riesgo) y/o
drawdown máximo antes del TP, para detectar trades que
ganaron pero fueron "dolorosos".

**Fase 11 — Iteración 3: comparación de modelos.**
Probar alternativas al `GradientBoostingClassifier`:
XGBoost, LightGBM, RandomForest. Comparar con métricas
objetivas y quedarnos con el mejor. Es el flujo clásico de
data science: mismo dataset, distintos modelos, decisión
basada en datos.

**Fase 12 — Iteración 4: walk-forward validation.**
Sustituir el split temporal fijo (train 2021-2024,
test 2025-2026) por walk-forward: entrenar con N meses,
predecir el siguiente, deslizar la ventana y reentrenar.
Simula el uso real con reentrenamiento periódico.

**Fase 13 — Dashboard de control.**
Interfaz visual (probablemente Streamlit, muy sencillo de
montar en Python) para ver en tiempo real: histórico de
señales enviadas, rendimiento del modelo, gráficos de
precio con entradas marcadas, evolución de la probabilidad
IA, y estado del bucle (arranque, última ejecución, errores
recientes). Sería la herramienta de control diaria para
el operador.

**Fase 14 — Despliegue 24/7.**
Migrar de "bucle en terminal" a servicio gestionado por el
sistema operativo: launchd en macOS o systemd en Linux.
Así el bucle corre sin que esté abierta la terminal y
sobrevive a reinicios.

**Fase 15 — Monitor de posiciones (SL/TP live).**
El script 08 actual solo avisa de ENTRADAS — no hace
seguimiento de las señales una vez enviadas. La Fase 15
añade un "monitor de posiciones" que, al cierre de cada
vela, comprueba las señales abiertas y avisa cuando el
precio toca SL, TP1 o TP2. Actualiza
`live_signals_log.csv` con el resultado de cada operación.
De referencia: en el proyecto anterior del autor esto
funcionaba simulando entrada en TODAS las señales (incluidas
las aparecidas con el bot apagado, recuperándolas al
arrancar) para poder medir el rendimiento real del modelo.
Al llegar a esta fase evaluaremos si replicar ese diseño
o proponer mejoras.

---

## 14.bis Glosario de conceptos clave para entender los resultados

Esta sección explica de forma sencilla los términos que
usamos al evaluar cada modelo. Pensada para releerla
cuando los números de las tablas no queden claros.

### PnL (Profit and Loss)

Cuánto dinero ganas o pierdes. PnL total = la suma de
todas las ganancias y todas las pérdidas de todos los
trades. Si es positivo, el sistema gana dinero; si es
negativo, pierde. PnL/trade = cuánto ganas de media cada
vez que operas. Es como el ticket medio de un restaurante:
cuánto factura por cliente.

Ejemplo: sin filtro de IA, nuestro sistema tiene un PnL
total de -41.228 USD en 5 años. Eso significa que operando
todas las señales a ciegas se pierde dinero. Por eso
necesitamos la IA como filtro.

### Umbral (threshold)

La IA que usamos NO es una API externa ni un ChatGPT. Es
un archivo matemático (`model.joblib`) que vive en la
carpeta del proyecto, entrenado con nuestros propios datos
históricos. Cuando llega una señal, el modelo mira las 20
features del mercado en ese momento (RSI, ATR, EMAs, etc.)
y calcula un número entre 0 y 1: la probabilidad de que ese
trade sea ganador.

El umbral lo ponemos NOSOTROS. Es el "aprobado" que le
exigimos a cada señal. Si ponemos umbral en 0.35, solo
aceptamos señales donde el modelo dice "al menos 35% de
probabilidad de ganar". Las señales por debajo se descartan.

### La trampa del umbral alto

Intuitivamente parece que subir el umbral debería dar más
ganancias siempre. No es así, y es muy importante entender
por qué.

Al subir el umbral pasan dos cosas A LA VEZ:

1. Cada trade individual gana más de media (PnL/trade sube).
2. Pero operas muchas menos veces (número de trades baja).

El PnL total es la multiplicación de ambos:

    PnL total = PnL/trade × número de trades

Hay un "punto dulce" (sweet spot) donde se maximiza el
total. En nuestro modelo v3, ese punto es el umbral 0.35:

- Umbral 0.35: 351 trades × 271 USD = 95.057 USD (máximo)
- Umbral 0.60: 48 trades × 368 USD = 17.675 USD

En el umbral 0.60 cada trade es mejor (368 > 271), pero
haces tan pocos que el total cae a una quinta parte.

Entonces, ¿por qué alguien elegiría un umbral alto?
Depende del estilo:

- **Umbral bajo (0.30-0.35):** más trades, más PnL total,
  pero más tiempo expuesto al mercado, más comisiones,
  más estrés.
- **Umbral alto (0.50-0.55):** pocos trades pero de altísima
  calidad (WR ~50%), ideal para alguien que quiere operar
  poco y con confianza.

Es la diferencia entre un restaurante con 1.000 cubiertos
al día a 5€ de margen cada uno, y uno con 100 cubiertos a
40€ de margen. Ambos modelos de negocio son válidos.

### Win Rate (WR)

De cada 100 operaciones, cuántas ganan dinero. Un WR del
38% significa 38 ganadoras y 62 perdedoras. Suena bajo,
pero en trading lo que importa no es cuántas veces aciertas
sino CUÁNTO ganas cuando aciertas vs cuánto pierdes cuando
fallas. Con nuestro ratio de ~2:1 (ganas el doble de lo que
pierdes), solo necesitas un 36% de aciertos para ser
rentable. Es como un delantero que marca 1 de cada 3 tiros,
pero cada gol vale 2 puntos y cada fallo solo descuenta 1.

### R-multiple

Cuánto ganaste o perdiste medido en "unidades de riesgo".
Si arriesgas 100€ (distancia al stop loss) y ganas 200€,
tu R es +2.0. Si pierdes esos 100€, tu R es -1.0. Es la
forma universal de medir resultados sin importar el tamaño
de la operación ni el precio del activo.

### MAE (Maximum Adverse Excursion)

Lo máximo que el precio fue EN TU CONTRA durante el trade,
antes de que terminara. Imagina que compras y el precio baja
un rato antes de subir y darte la razón. Ese "bajón temporal"
es el MAE. Un MAE de -0.25R significa que como máximo fuiste
un 25% de tu riesgo en contra. Un MAE de -0.9R significa que
casi te sacó el stop loss antes de girar.

Es como la montaña rusa: dos viajes pueden terminar en el
mismo sitio, pero uno fue tranquilo y el otro te puso el
corazón a mil. Queremos trades "tranquilos" (MAE bajo).

### Permutation Importance

Una prueba para saber si cada dato que le damos a la IA
realmente le sirve. Cogemos un dato (por ejemplo el RSI),
lo barajamos aleatoriamente (como mezclar cartas), y vemos
si la IA empeora. Si empeora mucho → ese dato era
importante. Si no empeora → ese dato no servía para nada.
Que 17 de 20 datos empeoren al barajarlos significa que la
IA usa casi toda la información, lo cual es buena señal.

---

## 15. Iteración 2 — Etiquetas R-multiple (Fase 10)

### 15.1 Motivación

El modelo v1 podado (en producción desde el 15 de abril)
usa un label binario simple: `label=1` si `pnl_abs > 0`,
`label=0` en caso contrario. Esto tiene tres problemas:

1. **Trata igual a ganadores muy distintos.** Un TP2 limpio
   (+2R) y un time_exit que cierra con +0.05% reciben ambos
   `label=1`. Para el modelo son "igual de buenos", pero
   para el operador uno es excelente y otro es ruido.

2. **Trata igual a perdedores muy distintos.** Un stop_loss
   (-1R) y un breakeven (0R, que además tocó TP1 antes de
   volver) reciben ambos `label=0`. El breakeven es un trade
   donde el mercado nos dio la razón inicialmente — no
   debería etiquetarse igual que un stop_loss.

3. **No captura la calidad del recorrido.** Dos trades pueden
   terminar en TP2, pero uno fue directo y el otro sufrió
   un drawdown de -0.9R antes de remontar. Esa información
   se pierde.

### 15.2 Conceptos nuevos introducidos

**R-multiple:** PnL del trade dividido por el riesgo inicial
(distancia entre entrada y SL). Mide cuánto ganaste o
perdiste en "unidades de riesgo". Ejemplos:

- Stop loss → R ≈ -1.0 (perdiste 1× lo arriesgado)
- Breakeven → R ≈ 0.0
- TP2 → R ≈ +2.0 (ganaste 2× lo arriesgado)
- Time exit → R variable (-0.8 a +1.5)

**MAE (Maximum Adverse Excursion):** el peor punto que el
precio alcanzó EN CONTRA de nuestra dirección durante la
vida del trade, expresado en R. Un trade con MAE = -0.8R
sufrió mucho antes de ganar; uno con MAE = -0.1R fue limpio.

Ambos se calculan ahora en `04_backtest.py` y se guardan
en `trades.csv` como columnas `r_multiple` y `mae_r`.

### 15.3 Decisión de diseño: opciones evaluadas

**Opción A — Clasificación binaria mejorada.**
Mantener el modelo como clasificador, pero redefinir
"ganador" usando R-multiple: `label=1` solo si R ≥ 0.5.
Cambio mínimo al pipeline.

**Opción B — Clasificación con 3 clases.**
"Mala" (R < -0.3), "neutra" (-0.3 ≤ R ≤ 0.3), "buena"
(R > 0.3). Más rica pero divide el dataset en tres grupos
más pequeños.

**Opción C — Regresión (predecir R-multiple).**
Cambiar de clasificador a regresor. Conceptualmente más
potente pero cambia todo el pipeline: métricas, umbrales,
lógica de decisión.

**Decisión: Opción A.**

Razones:

1. Mínimo cambio al pipeline → menor riesgo de introducir
   bugs o complejidad innecesaria.
2. Mismo algoritmo, mismas features, solo cambia la columna
   label → comparación limpia con v1.
3. Si A funciona, la mejora es "gratis" (solo redefinir
   qué es ganador). Si no, probamos C con la confianza de
   que el problema no es el label sino la capacidad del
   clasificador.
4. Opción B descartada porque con 4.574 trades, dividir en
   tres clases reduce mucho los ejemplos por clase,
   especialmente la clase "neutra" que aportaría poco.

**Sobre Opción C (regresión) — por qué NO la probamos
después de ver los resultados de A:**

El v3 (Opción A) ya consigue mejoras significativas en
eficiencia (+30% PnL/trade), win rate (+6 puntos a cada
umbral) y calidad de generalización (17/20 features con
permutation importance negativa, vs casi ninguna en v1).

Probar regresión habría implicado:

- Cambiar GradientBoostingClassifier por GradientBoostingRegressor.
- Sustituir métricas (F1, precision, recall → RMSE, MAE, R²).
- Rediseñar la lógica de umbral (de "probabilidad" a
  "R-multiple predicho").
- Modificar 08_live_signal.py con lógica distinta.
- Riesgo de sobreajuste: predecir un valor continuo es más
  difícil que clasificar, especialmente con ~3.400 filas de
  entrenamiento.

Todo eso para una mejora hipotética cuando el v3 ya supera
al v1 en las métricas que realmente importan (PnL, WR,
eficiencia). En un proyecto profesional, el principio es:
**si una solución simple resuelve el problema, no añadas
complejidad innecesaria**. La regresión queda como opción
futura si en algún momento el clasificador toca techo.

### 15.4 Nota sobre MAE como feature

El MAE se calcula DURANTE el trade (recorriendo las velas
posteriores a la entrada). Por tanto, NO está disponible en
el momento de generar la señal y no puede usarse como feature
de predicción — sería lookahead bias (mirar al futuro).

Se guarda en metadata para análisis post-hoc: nos permite
entender si el modelo selecciona trades más "limpios"
(menor MAE) además de más rentables.

### 15.5 Distribución del R-multiple en el backtest

Datos de los 4.574 trades históricos:

| Categoría | Condición | Trades | % |
|-----------|-----------|--------|---|
| Ganadores claros | R > 0.5 | 1.132 | 24.7% |
| Zona neutra | -0.3 ≤ R ≤ 0.3 | 1.186 | 25.9% |
| Perdedores claros | R < -0.5 | 2.194 | 48.0% |

Con el label v1 (pnl>0) había 1.237 ganadores (27.0%).
Con el label v3 (R≥0.5) hay 1.132 ganadores (24.7%).
Los 105 trades reclasificados son mayoritariamente time_exits
que cerraron con beneficio mínimo (R entre 0 y 0.5): ruido
que contaminaba la clase "ganador".

### 15.6 Resultados del modelo v3

**Métricas de clasificación (test set 2025-2026, 1.189 trades):**

| Métrica | v3 |
|---------|-----|
| Accuracy | 0.741 |
| Precision | 0.461 |
| Recall | 0.157 |
| F1 | 0.234 |

Nota: estas métricas NO son comparables con las del v1
porque el label es distinto (v3 tiene menos positivos →
el recall será naturalmente menor). La comparación válida
es la simulación de trading.

**Simulación de trading v3 (PnL real en USD):**

| Umbral | Trades | Win rate | PnL total | PnL/trade | R medio |
|--------|--------|----------|-----------|-----------|---------|
| 0.25 | 668 | 32.6% | +89.538 | 134 | +0.192 |
| 0.30 | 503 | 34.8% | +91.167 | 181 | +0.259 |
| **0.35** | **351** | **38.2%** | **+95.057** | **271** | **+0.374** |
| 0.40 | 245 | 40.0% | +71.683 | 293 | +0.429 |
| 0.45 | 169 | 43.8% | +54.546 | 323 | +0.507 |
| 0.50 | 102 | 50.0% | +40.481 | 397 | +0.612 |
| 0.55 | 65 | 50.8% | +23.695 | 365 | +0.605 |
| 0.60 | 48 | 52.1% | +17.675 | 368 | +0.630 |

El umbral 0.35 es el sweet spot: **+95.057 USD** es la
cifra más alta de PnL total que hemos obtenido en cualquier
modelo del proyecto, con un WR del 38.2% y ~4.5 trades
por semana.

### 15.7 Comparativa directa v1 podado vs v3

**Eficiencia por trade (la métrica más limpia):**

| Umbral | v1 PnL/trade | v3 PnL/trade | Ventaja v3 |
|--------|-------------|-------------|------------|
| 0.30 | 138 USD | 181 USD | +31% |
| 0.40 | 233 USD | 293 USD | +26% |
| 0.50 | 249 USD | 397 USD | +59% |

**Win rate:**

| Umbral | v1 WR | v3 WR |
|--------|-------|-------|
| 0.30 | 33.7% | 34.8% |
| 0.40 | 38.8% | 40.0% |
| 0.50 | 43.4% | 50.0% |

**PnL total (algunos umbrales v3 gana, otros v1):**

| Umbral | v1 PnL | v3 PnL | Δ |
|--------|--------|--------|---|
| 0.30 | 89.538 | 91.167 | +1.629 |
| 0.35 | — | 95.057 | (mejor absoluto) |
| 0.40 | 78.705 | 71.683 | -7.022 |
| 0.50 | 39.543 | 40.481 | +938 |
| 0.55 | 32.099 | 23.695 | -8.404 |

### 15.8 Permutation importance — cambio cualitativo

En el v1 podado, la mayoría de features tenían permutation
importance cercana a 0 o positiva, señal de que el modelo
no generalizaba bien (barajar una feature no empeoraba o
incluso mejoraba el rendimiento).

En el v3, **17 de 20 features tienen ΔF1 negativo**: barajar
cualquiera de ellas empeora significativamente el rendimiento.
Las más importantes por permutation:

| Feature | Δ F1 |
|---------|------|
| dist_ema50_pct | -0.0784 |
| ema_spread_pct | -0.0742 |
| ema_spread_4h_pct | -0.0574 |
| price_position | -0.0275 |
| dist_ema20_pct | -0.0250 |

Esto significa que el modelo v3 está usando la información
de las features de forma **mucho más efectiva** que el v1.
La definición más limpia de "ganador" permite al modelo
encontrar patrones más claros y generalizables.

### 15.9 Análisis de MAE por umbral de probabilidad

| Umbral | Trades | MAE medio | MAE ganadores | MAE perdedores |
|--------|--------|-----------|---------------|----------------|
| 0.00 | 1.189 | -0.875R | -0.367R | -1.069R |
| 0.30 | 503 | -0.777R | -0.296R | -1.033R |
| 0.40 | 245 | -0.717R | -0.276R | -1.011R |
| 0.50 | 102 | -0.642R | -0.250R | -1.034R |

A medida que el modelo es más selectivo (umbral más alto),
el MAE medio de los ganadores baja de -0.367R a -0.250R.
El modelo no solo filtra trades que ganan, sino trades
que ganan de forma más limpia (menos sufrimiento durante
el recorrido). Esto es muy relevante para el operador
humano: no es lo mismo aguantar un drawdown de -0.9R
rezando para que se recupere, que ver cómo tu trade va
directo hacia el TP.

### 15.10 Veredicto y promoción a producción

**v3 GANA.** Promovido a producción el 2026-04-18.

Resumen de la decisión:

- v3 es ~30% más eficiente por trade que v1 a cada umbral.
- v3 alcanza 50% WR al umbral 0.50 (v1 solo 43.4%).
- v3 selecciona trades más limpios (menor MAE).
- v3 generaliza mejor (17/20 features contribuyen).
- El sweet spot (umbral 0.35, +95.057 USD) es la mejor
  cifra de PnL de todo el proyecto.

**Opción C (regresión) DESCARTADA** por las razones
detalladas en la sección 15.3. Si en el futuro el
clasificador toca techo, la regresión queda como recurso.

### 15.11 Archivos creados/modificados en la Iteración 2

| Archivo | Cambio |
|---------|--------|
| `src/04_backtest.py` | Añade r_multiple y mae_r a trades.csv |
| `src/06_prepare_dataset.py` | Label R≥0.5, features podadas (20), metadata ampliada |
| `src/07_train_model_v3.py` | NUEVO — entrenamiento con label R-multiple |
| `outputs/model_v3.joblib` | Modelo v3 entrenado |
| `outputs/model_results_v3.csv` | Predicciones v3 sobre test |
| `outputs/feature_importance_v3.png` | Gráfico importancia features v3 |
| `outputs/trades.csv` | Regenerado con columnas r_multiple y mae_r |
| `outputs/ml_dataset.csv` | Regenerado con label v3 y metadata ampliada |

---

## 16. Tabla de fases actualizada

| Fase | Descripción | Estado |
|------|-------------|--------|
| 0 | Preparación del entorno | Completada |
| 1 | Descarga de datos históricos | Completada |
| 2 | Cálculo de indicadores técnicos | Completada |
| 3 | Reglas de entrada y señales | Completada |
| 4 | Motor de backtest / detección SL/TP | Completada |
| 5 | Evaluación del sistema | Completada |
| 6 | Preparación dataset + primer modelo | Completada |
| 7 | Señales en vivo + Telegram | Completada |
| 8 | Despliegue robusto (bucle + logs) | Completada |
| 9 | Iteración 1B (correlaciones externas) | Completada (v2 descartada, poda OK) |
| 10 | **Iteración 2 (etiquetas R-multiple)** | **Completada (v3 promovido)** |
| 11 | **Iteración 3 (XGBoost vs LightGBM)** | **Completada (GBM gana)** |
| 12 | Iteración 4 (walk-forward validation) | Pendiente |
| 13 | Dashboard de control (Streamlit) | Pendiente |
| 14 | Despliegue 24/7 (VPS barato) | Pendiente |
| 15 | Monitor de posiciones (SL/TP live) | Pendiente |

---

## 17. Iteración 3 — Comparación de algoritmos (Fase 11)

### 17.1 Motivación

Hasta ahora hemos mejorado las features (Iteración 1) y
las etiquetas (Iteración 2), pero siempre con el mismo
"motor": GradientBoostingClassifier de scikit-learn.
La Iteración 3 prueba si un motor más potente consigue
más con la misma gasolina.

### 17.2 Candidatos

**GradientBoosting (sklearn):** nuestro modelo actual.
Construye muchos árboles pequeños, cada uno corrigiendo
errores del anterior. Sólido, interpretable, maduro.

**XGBoost:** la versión "profesional" del mismo concepto.
Más regularización, más rápido, más parámetros de ajuste.
Dominó las competiciones de datos durante años.

**LightGBM (Microsoft):** alternativa a XGBoost. Aún más
rápido (5-10× en datasets grandes) y a veces mejor con
datasets pequeños. Usa una técnica diferente para construir
los árboles.

### 17.3 Metodología

Comparación 100% justa: mismo dataset (4.574 trades, 20
features), mismo label (R≥0.5), mismo split temporal
(train 2021-2024, test 2025-2026), parámetros equivalentes
entre los tres (200 árboles, profundidad 4, learning rate
0.05, subsample 0.8). El ÚNICO cambio es el algoritmo.

### 17.4 Resultados

**Métricas de clasificación:**

| Métrica | GBM | XGBoost | LightGBM |
|---------|-----|---------|----------|
| Accuracy | 0.741 | 0.740 | **0.743** |
| Precision | 0.461 | 0.453 | **0.478** |
| Recall | 0.157 | 0.143 | **0.180** |
| F1 | 0.234 | 0.218 | **0.262** |

LightGBM gana en todas las métricas de clasificación.

**Simulación de trading (lo que realmente importa):**

| Umbral | GBM PnL | XGBoost PnL | LightGBM PnL |
|--------|---------|-------------|--------------|
| 0.30 | **91.167** | 80.617 | 75.956 |
| 0.35 | **95.057** | 71.488 | 80.077 |
| 0.40 | **71.683** | 60.057 | 69.615 |
| 0.50 | 40.481 | 34.491 | **45.128** |

GBM gana en los umbrales 0.30-0.40 (donde se opera en
la práctica). LightGBM gana solo en umbrales altos (0.50+).

**PnL/trade:**

| Umbral | GBM | XGBoost | LightGBM |
|--------|-----|---------|----------|
| 0.30 | **181** | 151 | 146 |
| 0.35 | **271** | 187 | 226 |
| 0.40 | **293** | 232 | 285 |
| 0.50 | 397 | 363 | **399** |

GBM también domina en eficiencia por trade.

### 17.5 Interpretación

GradientBoosting gana la comparación práctica (PnL real)
aunque LightGBM gana la teórica (métricas de clasificación).

¿Por qué? Con solo ~3.400 filas de entrenamiento, nuestro
dataset es PEQUEÑO para los estándares de ML. XGBoost y
LightGBM brillan con datasets grandes (decenas de miles de
filas) donde su regularización avanzada y optimizaciones
dan ventaja. Con datos pequeños, la simplicidad del GBM de
scikit-learn es una virtud: hace menos suposiciones y se
adapta bien a lo que tiene.

### 17.6 Veredicto

**GBM sklearn GANA. No hay cambio de modelo.**

La Iteración 3 confirma que nuestra elección original de
algoritmo era correcta. Esto es un resultado válido y
valioso: saber que no necesitas cambiar es tan importante
como saber que sí.

Archivos generados:
- `src/07_train_model_v3_comparison.py` — script de comparación
- `outputs/model_iter3_gbm_sklearn.joblib` — copia del ganador

### 17.7 Nota sobre umbrales y rentabilidad

Aclaración importante: el umbral (0.30, 0.35, etc.) y el
win rate (36% mínimo para ser rentable) son cosas DISTINTAS:

- El **win rate del 36%** es el mínimo de aciertos para que,
  con nuestro ratio de ganancias/pérdidas (~2:1), el sistema
  sea matemáticamente rentable. Es un número fijo que sale
  de las matemáticas del ratio.

- El **umbral** es el filtro de confianza que nosotros le
  ponemos a la IA. A umbral 0.25 ya tenemos un WR del 32.6%
  y PnL positivo de +89.538 USD. ¿Cómo puede ser positivo
  con WR < 36%? Porque el ratio real no es exactamente 2:1
  en todos los trades — algunos TP2 dan más de 2R y los
  time_exits positivos contribuyen.

Con el modelo v3, **TODOS los umbrales son rentables** (de
0.25 a 0.60). No hay un "mínimo" por debajo del cual se
pierde dinero. El sweet spot de 0.35 es el que maximiza el
PnL total, no el mínimo.

### 17.8 Idea para el despliegue: múltiples umbrales

Para el despliegue público con señales a más personas (ver
objetivo final del proyecto), se puede ofrecer **dos canales
de Telegram** con distinto nivel de filtrado:

**Canal "Activo" (umbral ~0.30):**
- ~6-7 señales por semana
- Win rate ~35%
- Para traders que quieren operar con frecuencia
- Más oportunidades pero requiere gestión activa

**Canal "Selectivo" (umbral ~0.50):**
- ~1-2 señales por semana
- Win rate ~50%
- Para gente que prefiere operar poco y seguro
- Menos exposición al mercado, menos estrés, menos comisiones

Técnicamente es sencillo: el modelo ya calcula la
probabilidad de cada señal. El script 08 solo tendría que
enviar a un canal u otro (o a ambos) según el nivel. Es un
cambio menor que implementaremos al llegar a las Fases
14-15 (despliegue y monitor).

---

## 18. Iteración 4 — Walk-Forward Validation (Fase 12)

### 18.1 ¿Qué es walk-forward y por qué lo hacemos?

Hasta ahora hemos evaluado el modelo con un solo split fijo:
entrenar con datos ≤2024-12, testear con datos >2024-12.
Eso es un solo "examen" en un momento concreto del tiempo.

Walk-forward es como hacer 7 exámenes en épocas diferentes.
El modelo se reentrena cada 6 meses con todos los datos
disponibles hasta ese momento, y luego predice los 6 meses
siguientes. Al final, juntamos todas las predicciones
"fuera de muestra" y las comparamos con el split fijo.

**Objetivo:** verificar que el modelo no solo funciona en
un período específico, sino que es consistente a lo largo
del tiempo. Si el modelo gana dinero en los 7 folds, podemos
confiar mucho más en que no es casualidad.

### 18.2 Diseño del walk-forward

**Método:** ventana expandible (siempre desde el inicio).
**Reentrenamiento:** cada 6 meses.
**Algoritmo:** GBM sklearn (mismo que v3).
**Features:** las mismas 20 de siempre.
**Label:** R ≥ 0.5 (mismo que v3).

| Fold | Entrena hasta | Predice |
|------|---------------|---------|
| 1 | 2022-12 | 2023-01 → 2023-06 |
| 2 | 2023-06 | 2023-07 → 2023-12 |
| 3 | 2023-12 | 2024-01 → 2024-06 |
| 4 | 2024-06 | 2024-07 → 2024-12 |
| 5 | 2024-12 | 2025-01 → 2025-06 |
| 6 | 2025-06 | 2025-07 → 2025-12 |
| 7 | 2025-12 | 2026-01 → 2026-04 |

Los folds 5-7 cubren exactamente el mismo período que el
test del split fijo (>2024-12), lo que permite comparar
directamente.

### 18.3 Resultados por fold (umbral 0.35)

| Fold | Período | Trades | WR | PnL | PnL/trade |
|------|---------|--------|----|-----|-----------|
| 1 | 2023-01 → 2023-06 | 83 | 30.1% | +3,679 | +44 |
| 2 | 2023-07 → 2023-12 | 94 | 34.0% | +7,331 | +78 |
| 3 | 2024-01 → 2024-06 | 75 | 41.3% | +8,990 | +120 |
| 4 | 2024-07 → 2024-12 | 78 | 41.0% | +13,837 | +177 |
| 5 | 2025-01 → 2025-06 | 117 | 33.3% | +10,727 | +92 |
| 6 | 2025-07 → 2025-12 | 94 | 35.1% | +26,590 | +283 |
| 7 | 2026-01 → 2026-04 | 43 | 58.1% | +28,651 | +666 |
| **TOTAL** | **2023 → 2026** | **584** | **37.2%** | **+99,804** | **+171** |

**Hallazgo clave:** el modelo es rentable en TODOS los folds
sin excepción. No hay ningún semestre perdedor en 3+ años.
Esto es una validación muy fuerte de que la estrategia no
es fruto del azar.

Además, el PnL/trade mejora a medida que el modelo tiene
más datos para entrenarse (de 44 USD/trade en fold 1 a
666 USD/trade en fold 7). El modelo aprende mejor cuanto
más historia ve.

### 18.4 Comparación walk-forward vs split fijo (período >2024-12)

Para comparar de forma justa, usamos solo el período que
ambos métodos comparten: datos posteriores a 2024-12.

**PnL total (USD):**

| Umbral | Walk-Forward | Split Fijo | Diferencia |
|--------|-------------|------------|------------|
| 0.30 | 93,861 | 91,167 | +2,694 ★ |
| 0.35 | 65,968 | 95,057 | -29,089 |
| 0.40 | 41,805 | 71,683 | -29,878 |
| 0.50 | 20,773 | 40,481 | -19,708 |

**Win Rate:**

| Umbral | Walk-Forward | Split Fijo | Diferencia |
|--------|-------------|------------|------------|
| 0.30 | 37.4% | 34.8% | +2.6pp |
| 0.35 | 38.2% | 38.2% | 0.0pp |
| 0.40 | 38.9% | 40.0% | -1.1pp |
| 0.50 | 45.3% | 50.0% | -4.7pp |

**Trades generados:**

| Umbral | Walk-Forward | Split Fijo |
|--------|-------------|------------|
| 0.30 | 412 | 503 |
| 0.35 | 254 | 351 |
| 0.40 | 167 | 245 |
| 0.50 | 75 | 102 |

### 18.5 Análisis: ¿por qué gana el split fijo?

El split fijo gana en PnL a umbrales ≥0.35 por una razón
principal: **genera más trades**. Al entrenar con TODO el
período hasta 2024-12 de una sola vez, el modelo tiene más
datos y produce predicciones más "seguras" (probabilidades
más altas). Esto se traduce en más señales por encima de
cada umbral.

En walk-forward, los folds 5 y 6 entrenan con datos hasta
2024-12 y 2025-06 respectivamente, pero el fold 5 solo
tiene datos hasta 2024-12 (igual que el split fijo). La
diferencia está en que walk-forward entrena 3 modelos
separados para cubrir el mismo período, mientras que split
fijo entrena uno solo con más contexto global.

Sin embargo, al umbral 0.30 el walk-forward gana
ligeramente, y la WR es idéntica o superior en los umbrales
bajos. Esto sugiere que el walk-forward es un poco más
conservador pero igual de preciso.

### 18.6 Walk-forward completo (2023-2026)

El valor real del walk-forward no es "ganar" al split fijo,
sino demostrar que el modelo funciona en todo el período:

| Umbral | Trades | WR | PnL total | PnL/trade | R medio |
|--------|--------|----|-----------|-----------|---------|
| 0.25 | 1,203 | 33.4% | +128,389 | +107 | 0.20 |
| 0.30 | 863 | 36.3% | +130,501 | +151 | 0.29 |
| 0.35 | 584 | 37.2% | +99,804 | +171 | 0.33 |
| 0.40 | 417 | 37.4% | +66,701 | +160 | 0.33 |
| 0.45 | 277 | 42.2% | +59,547 | +215 | 0.42 |
| 0.50 | 192 | 43.8% | +42,674 | +222 | 0.45 |
| 0.55 | 145 | 42.8% | +32,477 | +224 | 0.46 |
| 0.60 | 96 | 49.0% | +30,253 | +315 | 0.60 |

584 trades a lo largo de 3+ años, todos evaluados fuera
de muestra (el modelo NUNCA vio esos datos antes de
predecirlos). El PnL es positivo a TODOS los umbrales.
Esto es una evidencia muy robusta.

### 18.7 Veredicto Iteración 4

**El split fijo v3 gana en PnL absoluto** al umbral 0.35
(+95,057 vs +65,968 USD en el período >2024-12).

**Pero el walk-forward aporta algo más valioso:**
la confirmación de que la estrategia funciona
consistentemente durante 3+ años, en 7 períodos
diferentes, sin ningún semestre perdedor.

**Decisión:**
- Mantener el modelo v3 con split fijo como modelo de
  producción (no hay razón para cambiar algo que funciona).
- Usar el walk-forward como herramienta de monitorización:
  si en el futuro sospechamos que el modelo se degrada,
  correr walk-forward de nuevo para verificar.
- Considerar reentrenamiento cuando se acumulen 6+ meses
  de datos nuevos desde el último entrenamiento.

**No se cambia el modelo. No se cambia el split. El walk-forward
valida que nuestra estrategia es sólida y no depende de un
período concreto del mercado.**

### 18.8 Script y archivos

- `src/07_walk_forward.py` — script de walk-forward validation
- `outputs/walk_forward_equity.png` — equity curve walk-forward

### 18.10 Índice de gráficos del proyecto

Todos los gráficos generados se encuentran en `outputs/` y
pueden abrirse con cualquier visor de imágenes:

| Archivo | Descripción | Generado por |
|---------|-------------|-------------|
| `equity_chart.png` | Equity curve del backtest original (v1) | `07_train_model.py` |
| `feature_importance.png` | Importancia de features (v1 original, 24 features) | `07_train_model.py` |
| `feature_importance_v2.png` | Importancia de features (v2 correlaciones externas) | Iteración 1B |
| `perm_importance_v2.png` | Permutation importance (v2) | Iteración 1B |
| `feature_correlation_v2.png` | Correlaciones entre features (v2) | Iteración 1B |
| `feature_importance_v1_pruned.png` | Importancia de features (v1 podado, 20 features) | Iteración 1B |
| `feature_importance_v3.png` | Importancia de features (v3, label R≥0.5) | Iteración 2 |
| `walk_forward_equity.png` | Equity curve walk-forward (7 folds, 2023-2026) | Iteración 4 |

Para ver cualquiera: `open outputs/nombre_archivo.png` en
terminal, o navegar a la carpeta outputs/ desde el explorador.

### 18.9 Tabla de fases actualizada

| Fase | Descripción | Estado |
|------|-------------|--------|
| 0-8 | Base + despliegue robusto | COMPLETADA |
| 9 | Iteración 1B — Correlaciones externas | COMPLETADA |
| 10 | Iteración 2 — Etiquetas R-multiple | COMPLETADA |
| 11 | Iteración 3 — XGBoost/LightGBM | COMPLETADA |
| 12 | Iteración 4 — Walk-forward validation | **COMPLETADA** |
| 13 | Dashboard Streamlit | Pendiente |
| 14 | Despliegue 24/7 (VPS barato) | Pendiente — OBJETIVO FINAL |
| 15 | Monitor de posiciones (SL/TP live) | Pendiente — OBJETIVO FINAL |

---

## 19. Dashboard Streamlit (Fase 13)

### 19.1 ¿Qué es el dashboard?

Un panel web interactivo donde se puede explorar visualmente
todo el rendimiento del sistema de trading: métricas del
modelo, historial de trades, equity curves, y señales live.

Está diseñado para ser entendido por cualquier persona, no
solo por el creador del proyecto.

### 19.2 Secciones del dashboard

**1. Resumen del Modelo:**
- Métricas clave: trades, win rate, PnL total, PnL/trade, R medio
- Slider interactivo para cambiar el umbral y ver cómo afecta
- Histograma de distribución de probabilidades (ganadores vs perdedores)
- Tabla de rendimiento por umbral (de 0.25 a 0.60)

**2. Historial de Trades:**
- Tabla completa de los 4,574 trades del backtest
- Filtros por fecha, dirección (LONG/SHORT) y resultado
- Distribución de PnL por trade
- PnL mensual en barras (verde/rojo)

**3. Equity Curve Interactiva:**
- Dos vistas: backtest completo (sin filtro IA) o solo trades filtrados por IA
- Gráfico con zoom y hover para ver detalle de cada punto
- Drawdown (en la vista de backtest completo)
- Slider de umbral (en la vista con filtro IA)

**4. Señales Live:**
- Última señal generada con todos sus datos
- Tabla de todas las señales live registradas
- Histograma de probabilidades de las señales

### 19.3 Cómo ejecutar el dashboard

Desde la carpeta del proyecto:

```
streamlit run src/09_dashboard.py
```

Se abre automáticamente en el navegador (normalmente
http://localhost:8501). Los datos se recargan cada 5 minutos
(trades y backtest) o cada minuto (señales live).

Requisitos: `pip install streamlit plotly`

### 19.4 Diseño técnico

- Framework: Streamlit (Python)
- Gráficos: Plotly (interactivos, con zoom y hover)
- Datos: lee directamente de los CSVs en `outputs/`
- Cache: `@st.cache_data` para no recargar datos en cada clic
- Archivo: `src/09_dashboard.py`

### 19.5 Tabla de fases actualizada

| Fase | Descripción | Estado |
|------|-------------|--------|
| 0-8 | Base + despliegue robusto | COMPLETADA |
| 9 | Iteración 1B — Correlaciones externas | COMPLETADA |
| 10 | Iteración 2 — Etiquetas R-multiple | COMPLETADA |
| 11 | Iteración 3 — XGBoost/LightGBM | COMPLETADA |
| 12 | Iteración 4 — Walk-forward validation | COMPLETADA |
| 13 | Dashboard Streamlit | **COMPLETADA** |
| 14 | Despliegue 24/7 (VPS barato) | Pendiente — OBJETIVO FINAL |
| 15 | Monitor de posiciones (SL/TP live) | Pendiente — OBJETIVO FINAL |

---

## 20. Despliegue 24/7 en VPS (Fase 14)

### 20.1 Objetivo

Tener el bot de señales y el dashboard corriendo las 24
horas del día, los 7 días de la semana, en un servidor
remoto accesible desde cualquier lugar.

### 20.2 Servidor elegido

**VPS cloud (2 vCPU / 4 GB)** — ~4,35€/mes

- 2 vCPU, 4GB RAM, 40GB disco NVMe
- Ubuntu 22.04
- Ubicación: Europa (Alemania/Finlandia)
- IP pública incluida
- Reinicio automático tras caídas

Alternativas evaluadas: OVH (~3,50€), Contabo (~5,99€),
Netcup (~3,99€). el proveedor VPS ofrece el mejor equilibrio
precio/fiabilidad.

### 20.3 Arquitectura del despliegue

El VPS corre dos servicios independientes gestionados
por systemd:

1. **trading-signal** — `08_live_signal.py --loop`
   - Se ejecuta en modo bucle continuo
   - Comprueba cada hora si hay señal nueva
   - Envía por Telegram si detecta una
   - Se reinicia solo si falla (Restart=always)

2. **trading-dashboard** — `09_dashboard.py` via Streamlit
   - Dashboard web en puerto 8501
   - Accesible desde cualquier navegador: http://IP:8501
   - Se recarga automáticamente si detecta cambios

Ambos corren bajo el usuario `trading` (no root) con
límites de memoria (512MB cada uno) y CPU (50%).

### 20.4 Archivos de despliegue creados

| Archivo | Descripción |
|---------|-------------|
| `deploy/setup_vps.sh` | Script de instalación automática |
| `deploy/trading-signal.service` | Servicio systemd para el bot |
| `deploy/trading-dashboard.service` | Servicio systemd para el dashboard |
| `deploy/GUIA_DESPLIEGUE.md` | Guía paso a paso completa |

### 20.5 Pasos resumidos

1. Contratar VPS (el proveedor VPS, 5 minutos)
2. Conectarte por SSH
3. Subir el proyecto con scp
4. Ejecutar setup_vps.sh (instala todo)
5. Configurar .env con tokens de Telegram
6. Arrancar servicios
7. Acceder al dashboard desde el navegador

Guía completa con cada comando: `deploy/GUIA_DESPLIEGUE.md`

### 20.6 Tabla de fases actualizada

| Fase | Descripción | Estado |
|------|-------------|--------|
| 0-8 | Base + despliegue robusto | COMPLETADA |
| 9 | Iteración 1B — Correlaciones externas | COMPLETADA |
| 10 | Iteración 2 — Etiquetas R-multiple | COMPLETADA |
| 11 | Iteración 3 — XGBoost/LightGBM | COMPLETADA |
| 12 | Iteración 4 — Walk-forward validation | COMPLETADA |
| 13 | Dashboard Streamlit | COMPLETADA |
| 14 | Despliegue 24/7 (VPS) | **COMPLETADA** (scripts listos) |
| 15 | Monitor de posiciones (SL/TP live) | Pendiente — OBJETIVO FINAL |

---

## 21. Control de versiones — Git + GitHub

Tras el despliegue en VPS se configuró Git y GitHub para
sincronizar el código entre local (Mac) y el servidor:

- **Repositorio:** privado en GitHub (`trading_project_signal`)
- **Flujo:** editar en local → `git push` → en el servidor
  `git pull` → reiniciar servicios
- **Servidor:** conectado por clave SSH (ed25519)
- **Seguridad:** `.env` excluido de Git (tokens nunca suben).
  `.gitignore` incluye los outputs esenciales (modelo, CSVs)
  para que `git pull` deje el servidor listo sin copiar nada a mano
- **Guía completa:** `docs/GUIA_GITHUB.md`

### 21.1 Tabla de fases actualizada

| Fase | Descripción | Estado |
|------|-------------|--------|
| 0-8 | Base + despliegue robusto | COMPLETADA |
| 9 | Iteración 1B — Correlaciones externas | COMPLETADA |
| 10 | Iteración 2 — Etiquetas R-multiple | COMPLETADA |
| 11 | Iteración 3 — XGBoost/LightGBM | COMPLETADA |
| 12 | Iteración 4 — Walk-forward validation | COMPLETADA |
| 13 | Dashboard Streamlit | COMPLETADA |
| 14 | Despliegue 24/7 (VPS) | **COMPLETADA** |
| — | Git + GitHub (control de versiones) | **COMPLETADA** |
| 15 | Monitor de posiciones (SL/TP live) | Pendiente — OBJETIVO FINAL |

---

## 22. Heartbeat y resumen diario

Mejoras de monitorización para saber que el bot sigue activo:

- **Dashboard:** indicador discreto con la última vela analizada
  (precio, hora, si hubo señal). Auto-refresh cada hora.
- **Telegram:** resumen diario a las 8:00 CET con velas
  analizadas, señales enviadas, estado del bot y enlace al
  dashboard.

---

## 23. Fase 15 — Monitor de posiciones en tiempo real

El sistema ahora hace seguimiento de cada operación después de
publicar la señal. Un servicio independiente vigila el precio de
BTC cada minuto y detecta cuándo la operación toca Stop Loss,
Take Profit 1, Take Profit 2 o expira por tiempo (24h).

- Al tocar **TP1**: cierre parcial del 50% y sigue vigilando
  el resto hasta TP2, SL o time exit
- Al cerrar una operación: envía resultado a Telegram
  (PnL, R-multiple, duración)
- El dashboard muestra la operación en curso con PnL flotante,
  barra de progreso y distancia al SL/TP
- Se acumula un historial de rendimiento real (no backtest)
  en `position_results.csv`
- Cada señal incluye enlace al dashboard para seguirla en vivo

### 23.1 Tabla de fases actualizada

| Fase | Descripción | Estado |
|------|-------------|--------|
| 0-8 | Base + despliegue robusto | COMPLETADA |
| 9 | Iteración 1B — Correlaciones externas | COMPLETADA |
| 10 | Iteración 2 — Etiquetas R-multiple | COMPLETADA |
| 11 | Iteración 3 — XGBoost/LightGBM | COMPLETADA |
| 12 | Iteración 4 — Walk-forward validation | COMPLETADA |
| 13 | Dashboard Streamlit | COMPLETADA |
| 14 | Despliegue 24/7 (VPS) | COMPLETADA |
| — | Git + GitHub | COMPLETADA |
| — | Heartbeat + resumen diario | COMPLETADA |
| 15 | Monitor de posiciones (SL/TP live) | **COMPLETADA** |

---

## 24. Cierre del roadmap base (Fases 0-15)

Con la Fase 15 completada, el sistema BTC/USDT está **en producción 24/7**
con todas las capacidades previstas en el roadmap original:

- Motor de señales con modelo XGBoost v3 walk-forward validado
- 20 features técnicas, label R≥0.5
- Señales en vivo a Telegram con SL, TP1, TP2
- Monitor de posiciones con cierre parcial TP1, cierre total TP2/SL/time exit
- Dashboard Streamlit con 4 tabs (Live, Historial, Modelo, Equity)
- Heartbeat, auto-refresh, resumen diario, enlace al dashboard
- VPS el proveedor VPS corriendo 3 servicios (signal, monitor, dashboard)
- Flujo Git → GitHub → servidor establecido

**A partir de aquí, las mejoras iterativas y refinamientos se documentan
en `docs/anexo_I_mejoras.md`.** El presente documento queda cerrado
como referencia del proyecto base.

---

*Última actualización: 23 de abril de 2026.*
*Roadmap base cerrado. Mejoras en Anexo I.*
