# Anexo III — Funding Rate como feature del modelo

> Integrar el funding rate de Binance Futures como nueva
> feature del modelo predictivo ML para mejorar la calidad
> de las señales.

---

## 1. Qué es el funding rate

El funding rate es una tasa que se cobra cada 8 horas en
los contratos perpetuos de Binance Futures (00:00, 08:00
y 16:00 UTC). Funciona como mecanismo para mantener el
precio del futuro alineado con el precio spot.

- **Funding rate positivo:** los longs pagan a los shorts.
  Indica que el mercado está muy alcista / sobrecomprado.
  Muchos traders apalancados están largos.
- **Funding rate negativo:** los shorts pagan a los longs.
  Indica que el mercado está bajista / sobrevendido.
  Muchos traders apalancados están cortos.
- **Funding rate neutro (~0.01%):** equilibrio normal.

Valores típicos: entre -0.01% y +0.03%. Valores extremos
(>0.05% o <-0.03%) suelen preceder correcciones.

Si la tasa alcanza el cap/floor (+0.3% / -0.3%), Binance
cambia la frecuencia de 8h a 1h hasta que se normalice.

---

## 2. Hipótesis

El funding rate puede mejorar el modelo de dos formas:

**Hipótesis A — Filtro contrarian:** cuando el funding rate
es muy alto y generamos una señal LONG, el mercado está
sobrecomprado y el trade tiene menos probabilidad de
funcionar. Lo mismo al revés para SHORT con funding muy
negativo. El modelo debería aprender a bajar la probabilidad
en estos casos.

**Hipótesis B — Confirmación de tendencia:** un funding rate
moderadamente positivo en tendencia alcista confirma que
hay demanda real (no solo spot). Podría mejorar la calidad
de señales LONG en tendencia sana.

No asumimos cuál es correcta. Dejamos que el modelo lo
aprenda de los datos.

---

## 3. API de Binance

**Endpoint:** `GET /fapi/v1/fundingRate`

**Librería:** `python-binance` (ya instalada en el proyecto).
Método: `client.futures_funding_rate(symbol='BTCUSDT')`

**Parámetros:**
- `symbol`: BTCUSDT (obligatorio)
- `startTime`, `endTime`: rango de fechas (ms)
- `limit`: máx 1000 registros por llamada

**Respuesta:**
```json
{
  "symbol": "BTCUSDT",
  "fundingRate": "0.00010000",
  "fundingTime": 1698768000000,
  "markPrice": "34271.29000000"
}
```

**Frecuencia:** cada 8h (3 registros por día).
**Histórico:** disponible desde 2019 para BTCUSDT.

Nuestras señales son de velas 1h, así que para cada señal
necesitamos el último funding rate conocido en ese momento
(el de la liquidación de 8h más reciente anterior a la
señal).

---

## 4. Features candidatas

A partir del funding rate bruto, proponemos estas features:

| Feature              | Descripción                                    |
|----------------------|------------------------------------------------|
| `funding_rate`       | Último funding rate conocido (valor bruto)      |
| `funding_rate_ma8`   | Media de los últimos 8 fundings (~2.7 días)    |
| `funding_rate_ma24`  | Media de los últimos 24 fundings (~8 días)     |
| `funding_extreme`    | 1 si abs(funding) > percentil 90 histórico     |

Empezaremos con las 4 y tras el primer entrenamiento
evaluaremos si alguna tiene importancia baja y debe
eliminarse.

---

## 5. Plan de implementación

### Paso 1 — Descargar histórico de funding rate
Crear `src/01c_download_funding.py` que descargue todo
el histórico de funding rate de BTCUSDT desde 2020 y lo
guarde en `data/raw/BTCUSDT_funding_rate.csv`.

### Paso 2 — Integrar funding rate en el dataset ML
Modificar `src/06_prepare_dataset.py` para que cargue el
CSV de funding rate, calcule las 4 features derivadas
(ma8, ma24, extreme) y las alinee con cada señal mediante
merge_asof (dirección backward: cada señal recibe el
último funding liquidado antes de esa hora).

No se modifica `02_indicators.py` porque el funding rate
no es un indicador técnico derivado del precio, sino un
dato externo. Encaja mejor en `06_prepare_dataset.py`,
donde ya se integran otros datos externos (RSI y ATR de
4h) con el mismo patrón de merge_asof.

### Paso 4 — Reentrenar modelo y comparar
Ejecutar `src/07_train_model_v3.py` con las nuevas features
y comparar métricas contra el modelo actual (Config A sin
funding rate). Mismo esquema: train/test split, múltiples
umbrales.

### Paso 5 — Integrar en señales live
Si el modelo mejora, modificar `src/08_live_signal.py`
para que consulte el funding rate actual de Binance al
generar cada señal y lo pase al modelo.

### Paso 6 — Documentar resultados
Actualizar este anexo con los resultados del entrenamiento
y la decisión de desplegar o no.

---

## 6. Riesgos y consideraciones

- **Lookahead bias:** el funding rate se liquida cada 8h.
  Solo podemos usar el último valor liquidado, nunca el
  "próximo" funding rate (que aún no se ha fijado).
- **Frecuencia diferente:** funding = cada 8h, señales =
  cada 1h. Habrá muchas señales con el mismo valor de
  funding. Esto es correcto y esperado.
- **Datos faltantes:** si Binance no tiene dato para algún
  periodo, usamos forward-fill (el último conocido).
- **No es una señal por sí sola:** el funding rate se añade
  como feature del modelo, no como regla de entrada
  independiente.

---

## 7. Resultados

### 7.1 Datos descargados

Se descargaron 6.668 registros de funding rate desde abril
2020 hasta mayo 2026. Media de 3 registros por día, sin
duplicados. Funding rate medio: 0.0107% (ligeramente
positivo, normal en mercado alcista). El 4.2% de los
registros tienen funding extremo positivo (>0.05%).

### 7.2 Prueba A — 4 features (24 features total)

Se añadieron las 4 features candidatas al modelo (24 en
total) y se reentrenó con el mismo pipeline y split
temporal.

Comparativa con el modelo actual (20 features, Config A):

| Umbral | PnL 20 feat | PnL 24 feat | Diferencia |
|--------|-------------|-------------|------------|
| 0.35   | +$132.763   | +$140.681   | +$7.918    |
| 0.40   | +$123.216   | +$133.359   | +$10.143   |
| 0.45   | +$131.791   | +$102.554   | -$29.237   |
| 0.50   | +$112.687   | +$74.944    | -$37.743   |
| 0.55   | +$96.876    | +$60.559    | -$36.317   |

Mejora en umbrales bajos (0.35-0.40), pero empeora en los
umbrales altos (0.45+) que son los que dan mayor margen de
seguridad. Las features de funding tienen importancia nativa
alta (funding_rate_ma24 2a, funding_rate_ma8 4a), pero esa
importancia no se traduce en mejores trades en el test set.

### 7.3 Prueba B — Solo funding_rate_ma8 (21 features)

Se probó con solo la feature más significativa en permutation
importance (funding_rate_ma8) para reducir ruido.

| Umbral | PnL 20 feat | PnL 21 feat | Diferencia |
|--------|-------------|-------------|------------|
| 0.35   | +$132.763   | +$127.381   | -$5.382    |
| 0.40   | +$123.216   | +$127.035   | +$3.819    |
| 0.45   | +$131.791   | +$122.932   | -$8.859    |
| 0.50   | +$112.687   | +$102.113   | -$10.574   |
| 0.55   | +$96.876    | +$82.833    | -$14.043   |
| 0.60   | +$73.181    | +$78.453    | +$5.272    |

Menos degradación que con las 4 features, pero sigue sin
superar al modelo de 20 features en los umbrales clave.

### 7.4 Análisis

El funding rate no mejora el modelo por una limitación
estructural: se actualiza cada 8 horas pero las señales
son cada hora. Muchas señales consecutivas reciben el
mismo valor de funding, lo que limita cuánta información
nueva aporta. El modelo detecta correlaciones en el
periodo de train que no se mantienen en el test.

### 7.5 Decisión

Se descarta el funding rate como feature del modelo. Se
mantiene el modelo actual de 20 features (Config A).

El script de descarga (`src/01c_download_funding.py`) y los
datos (`data/raw/BTCUSDT_funding_rate.csv`) quedan disponibles
en la rama por si se quiere revisitar en el futuro con otro
enfoque (por ejemplo, como filtro de reglas en vez de feature
del modelo).

---

**Rama:** `feature/funding-rate`

*Fecha: 2 de mayo de 2026.*
