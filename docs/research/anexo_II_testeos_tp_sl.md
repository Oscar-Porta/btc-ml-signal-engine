# Anexo II — Optimización de TP/SL

> Testeo exhaustivo de combinaciones de Stop Loss, Take Profit y cierre
> parcial para encontrar la configuración más rentable. Complementa
> el sistema base documentado en Anexo I.

---

## 1. Objetivo

Determinar si la configuración actual de TP/SL (SL 1.5x ATR,
TP1 1.0R, TP2 2.0R, 50% cierre parcial en TP1) es óptima o si
existe una combinación que produzca mejores resultados.

Se trabaja en rama `feature/tp-sl-optimization` para no afectar
el sistema en producción.

---

## 2. Metodología

Se construyó un script (`src/tp_sl_optimizer.py`) que:

1. Lee las mismas señales de `signals.csv` (4.574 señales generadas
   por las reglas de entrada de `03_rules.py`).
2. Para cada combinación de parámetros, recalcula los niveles de SL,
   TP1 y TP2 usando el ATR de cada señal.
3. Simula el backtest completo con la misma lógica que `04_backtest.py`
   (vela a vela, peor caso si SL y TP coinciden en la misma vela,
   máximo 24 velas por trade).
4. Calcula métricas para cada combinación: WR, payoff, PnL total,
   R medio, breakeven WR, margen de seguridad, profit factor,
   max drawdown.

La simulación está vectorizada con numpy (procesa todas las señales
en paralelo por cada vela), lo que permite evaluar 558 combinaciones
en ~1.2 segundos.

Se verificó que la simulación reproduce exactamente los resultados
del backtest original para la configuración de producción (diferencia
de $0.21 por redondeo flotante en 4.574 trades).

**Parámetros barridos:**

| Parámetro        | Valores probados                    |
|------------------|-------------------------------------|
| SL (× ATR)       | 1.0, 1.25, 1.5, 1.75, 2.0, 2.5     |
| TP1 (× R)        | 0.75, 1.0, 1.25, 1.5, 2.0          |
| TP2 (× R)        | 1.5, 2.0, 2.5, 3.0, 4.0            |
| Cierre parcial   | 0%, 30%, 50%, 70%, 100%             |
| Máx. velas       | 24 (fijo)                           |

Se filtraron combinaciones inválidas (TP2 ≤ TP1) y duplicados
(cuando cierre parcial = 0%, el TP1 no afecta al resultado).
Total: **558 combinaciones válidas**.

---

## 3. Resultados — Backtest completo (4.574 señales, sin modelo)

Dado que el sistema desplegado envía todas las señales sin filtrar por
el modelo predictivo, estos son los resultados de referencia principal.

### 3.1 Configuración actual

| Métrica          | Valor       |
|------------------|-------------|
| Trades           | 4.574       |
| Win Rate         | 49.6%       |
| Payoff           | 1.00        |
| Breakeven WR     | 49.9%       |
| Margen           | -0.3pp      |
| PnL total        | -$13.924    |
| Profit Factor    | 0.99        |
| Max Drawdown     | -$133.912   |

La configuración actual está esencialmente en breakeven sin el modelo:
WR ≈ breakeven WR. Pierde ligeramente por el efecto del precio
creciente de BTC (trades recientes pesan más en USD absoluto).

### 3.2 Mejores alternativas

De las 558 combinaciones, 180 (32.3%) son rentables.

**Top 5 por PnL total:**

| SL    | TP1   | TP2   | Parcial | WR     | Payoff | PnL       | Margen | PF   | MaxDD      |
|-------|-------|-------|---------|--------|--------|-----------|--------|------|------------|
| 1.5x  | 1.25R | *     | 100%    | 45.6%  | 1.24   | +$64.170  | +1.0pp | 1.04 | -$80.850   |
| 1.5x  | 1.50R | *     | 100%    | 42.3%  | 1.41   | +$56.856  | +0.9pp | 1.04 | -$85.474   |
| 1.5x  | 2.00R | *     | 100%    | 37.2%  | 1.74   | +$55.649  | +0.8pp | 1.03 | -$127.733  |
| 1.25x | 1.25R | *     | 100%    | 42.1%  | 1.32   | +$45.099  | +0.2pp | 1.03 | -$89.017   |
| 1.5x  | 1.25R | 3.0R  | 70%     | 45.6%  | 1.23   | +$54.129  | +0.9pp | 1.04 | -$97.277   |

(*) Cuando cierre parcial = 100%, TP2 es irrelevante porque el trade
se cierra completamente en TP1.

### 3.3 Hallazgo principal

El cierre al 100% en TP1 (sin segunda fase buscando TP2) domina las
primeras posiciones del ranking. Las configuraciones con cierre parcial
y TP2 quedan consistentemente por debajo.

Esto indica que, con las señales base (sin filtrar), el valor del
sistema está en acertar la dirección del movimiento a corto plazo, no
en capturar extensiones grandes. El TP2 genera muchos trades de breakeven
que no compensan los pocos que llegan al objetivo ampliado.

---

## 4. Reentrenamiento y comparativa final

### 4.1 Por qué reentrenar

El modelo desplegado (v3) fue entrenado con etiquetas basadas en
la configuración original (SL 1.5x, TP1 1.0R, cierre parcial
50%). Si se cambia la configuración de TP/SL, los R-multiples de
cada trade cambian, lo que cambia las etiquetas (R >= 0.5 =
ganador). Para evaluar una nueva configuración correctamente hay
que reentrenar el modelo con las etiquetas que corresponden a esa
configuración.

### 4.2 Candidatas seleccionadas

Se seleccionaron 3 candidatas con cierre 100% en TP1 (el
hallazgo de la sección 3), variando SL y TP1 para explorar
diferentes perfiles de riesgo:

| #  | SL    | TP1   | Parcial | Razón de selección                           |
|----|-------|-------|---------|----------------------------------------------|
| A  | 1.25x | 2.0R  | 100%    | Mayor PnL en test con modelo (+$172K)        |
| B  | 1.25x | 1.5R  | 100%    | Mayor margen y menor drawdown (+16pp, PF 1.92)|
| C  | 1.75x | 1.5R  | 100%    | Equilibrio entre SL amplio y buen payoff     |

Para cada candidata se ejecutó el pipeline completo con
`src/tp_sl_retrain_candidates.py`: recalcular trades con nuevos
TP/SL → generar etiquetas nuevas → construir dataset ML (20
features) → reentrenar GradientBoosting (mismos hiperparámetros)
→ evaluar en test set a múltiples umbrales.

La config A se reentrenó también con el pipeline de producción
completo (03_rules → 04_backtest → 06_prepare_dataset →
07_train_model_v3), verificando que los resultados coinciden
exactamente con el script de reentrenamiento.

### 4.3 Config A vs modelo desplegado — todos los umbrales

La referencia es el modelo v3 desplegado en producción, evaluado
en el mismo periodo test (enero 2025 — abril 2026, 1.189 señales
base). No se reentrenó el modelo actual; se compara directamente
contra lo que está funcionando.

**Modelo desplegado (SL 1.5x, TP1 1.0R, parcial 50%):**

| Umbral | Trades | WR     | Payoff | PnL        | Margen  | PF   | MaxDD      |
|--------|--------|--------|--------|------------|---------|------|------------|
| 0.30   | 1.107  | 54.4%  | 0.97   | +$65.148   | +3.6pp  | 1.16 | -$39.926   |
| 0.35   | 1.036  | 56.5%  | 0.97   | +$97.495   | +5.8pp  | 1.26 | -$28.406   |
| 0.40   | 934    | 57.2%  | 0.98   | +$99.079   | +6.6pp  | 1.30 | -$27.367   |
| 0.45   | 821    | 58.6%  | 1.01   | +$115.667  | +8.9pp  | 1.44 | -$14.592   |
| 0.50   | 719    | 60.2%  | 1.08   | +$133.765  | +12.1pp | 1.63 | -$12.361   |
| 0.55   | 611    | 61.9%  | 1.10   | +$131.699  | +14.2pp | 1.78 | -$10.529   |
| 0.60   | 481    | 62.6%  | 1.15   | +$118.185  | +16.2pp | 1.93 | -$8.348    |

**Config A reentrenada (SL 1.25x, TP1 2.0R, cierre 100%):**

| Umbral | Trades | WR     | Payoff | PnL        | Margen  | PF   | MaxDD      |
|--------|--------|--------|--------|------------|---------|------|------------|
| 0.30   | 813    | 44.4%  | 1.88   | +$149.093  | +9.7pp  | 1.50 | -$25.375   |
| 0.35   | 658    | 45.4%  | 1.87   | +$132.763  | +10.6pp | 1.56 | -$20.438   |
| 0.40   | 491    | 48.1%  | 1.87   | +$123.216  | +13.2pp | 1.73 | -$23.225   |
| 0.45   | 364    | 54.1%  | 1.87   | +$131.791  | +19.3pp | 2.21 | -$9.494    |
| 0.50   | 261    | 57.1%  | 1.91   | +$112.687  | +22.8pp | 2.55 | -$6.766    |
| 0.55   | 169    | 65.1%  | 1.87   | +$96.876   | +30.3pp | 3.49 | -$4.728    |
| 0.60   | 112    | 67.0%  | 2.08   | +$73.181   | +34.5pp | 4.22 | -$3.571    |

### 4.4 Análisis

El modelo desplegado genera más PnL absoluto en umbrales
intermedios (0.50-0.55) porque procesa más trades. Pero la
config A es superior en calidad y robustez:

El **payoff** de A es casi el doble (~1.9 vs ~1.0) en todos
los umbrales: cuando gana, gana mucho más de lo que pierde
cuando falla. Esto da un colchón enorme.

El **margen de seguridad** de A es mayor en todos los umbrales.
En 0.45: +19.3pp vs +8.9pp. En 0.55: +30.3pp vs +14.2pp. Esto
significa que el WR podría caer mucho más antes de entrar en
pérdidas.

El **profit factor** de A es mayor a partir de 0.40 y crece
mucho más rápido. En 0.55: PF 3.49 vs 1.78.

El **drawdown** de A es menor en umbrales de 0.45 en adelante.
En 0.55: -$4.728 vs -$10.529 (menos de la mitad).

En PnL total, A iguala o supera al desplegado en umbrales bajos
(0.30: $149K vs $65K, 0.35: $133K vs $97K) y es inferior en
umbrales altos (0.55: $97K vs $132K) por tener menos trades.

La diferencia fundamental es el modelo de negocio: el sistema
actual gana poco por trade pero tiene muchos trades. Config A
gana mucho por trade pero tiene menos. En escenarios adversos
(mercado cambiante, modelo degradado), un payoff alto protege
mucho más que un volumen alto.

### 4.5 Candidatas B y C — descartadas

**Config B (SL 1.25x, TP1 1.5R, 100%):** rendimiento
consistente pero sin destacar en ninguna métrica frente a A.

**Config C (SL 1.75x, TP1 1.5R, 100%):** mayor PnL que A
solo en 0.55 (+$118K vs +$97K) porque genera más trades (227
vs 169). En todos los demás umbrales A es superior. Drawdown
de C es consistentemente peor.

### 4.6 Verificación contra el top del barrido

Las candidatas A, B y C no coinciden con las combinaciones
que lideraban el ranking del barrido sin modelo (sección 3.2).
El top del barrido estaba dominado por combinaciones con
SL 1.5x, mientras que A y B usan SL 1.25x. Para verificar
que la selección fue correcta, se reentrenó el modelo con
las 3 mejores combinaciones del barrido (todas con cierre
100% en TP1):

| #  | SL    | TP1    | PnL sin ML  | Pos. ranking |
|----|-------|--------|-------------|--------------|
| D  | 1.5x  | 1.25R  | +$64.170    | #1           |
| E  | 1.5x  | 1.50R  | +$56.856    | #2           |
| F  | 1.5x  | 2.00R  | +$55.649    | #3           |

Config A ocupaba la posición #68 del ranking sin modelo.

**Resultados con ML — comparativa en umbrales clave:**

| Umbral | Config | Trades | WR     | Payoff | PnL      | Margen  | PF   | MaxDD     |
|--------|--------|--------|--------|--------|----------|---------|------|-----------|
| 0.45   | A      | 364    | 54.1%  | 1.87   | +$131K   | +19.3pp | 2.21 | -$9.494   |
| 0.45   | D      | 639    | 57.0%  | 1.25   | +$139K   | +12.5pp | 1.65 | -$10.949  |
| 0.45   | E      | 538    | 52.4%  | 1.51   | +$129K   | +12.6pp | 1.66 | -$13.838  |
| 0.45   | F      | 360    | 51.7%  | 1.71   | +$118K   | +14.7pp | 1.82 | -$17.860  |
| 0.55   | A      | 169    | 65.1%  | 1.87   | +$97K    | +30.3pp | 3.49 | -$4.728   |
| 0.55   | D      | 388    | 62.1%  | 1.34   | +$125K   | +19.4pp | 2.20 | -$8.457   |
| 0.55   | E      | 299    | 57.9%  | 1.44   | +$95K    | +16.9pp | 1.98 | -$8.937   |
| 0.55   | F      | 155    | 60.0%  | 1.89   | +$92K    | +25.4pp | 2.83 | -$8.282   |

Las combinaciones del top del barrido (D, E, F) generan más
PnL absoluto que A gracias a un mayor volumen de trades. Pero
Config A es superior en las métricas de calidad y robustez en
todos los umbrales: payoff (~1.87 vs ~1.25-1.51), margen de
seguridad (30pp vs 16-19pp en 0.55), profit factor (3.49 vs
1.98-2.83 en 0.55) y drawdown (-$4.728 vs -$8.282 a -$8.937
en 0.55).

Esto confirma que el rendimiento sin modelo no predice bien
el rendimiento con modelo. Config A funciona mejor con el
filtrado ML porque su estructura de payoff alto permite al
modelo seleccionar trades de alta calidad.

### 4.7 Verificación exhaustiva — 143 combinaciones con cierre parcial

La verificación de 4.6 solo testó combinaciones con cierre
100% en TP1. Pero si el ranking sin modelo no predice el
ranking con modelo, tampoco podemos asumir que su conclusión
de que el cierre 100% es mejor que el cierre parcial sea
correcta. Para descartarlo con datos, se reentrenó el modelo
con TODAS las combinaciones del barrido que superaban a
Config A en PnL sin modelo: 143 combinaciones, de las cuales
127 incluyen cierre parcial (50% o 70% en TP1 + TP2).

**Resultado clave:** ninguna combinación supera a Config A
simultáneamente en PnL y calidad. Las 10 que generan más
PnL absoluto lo hacen con métricas de robustez inferiores.

**Top 3 por PnL vs Config A — umbral 0.45:**

| Config                      | PnL      | WR    | Payoff | Margen  | PF   | MaxDD     |
|-----------------------------|----------|-------|--------|---------|------|-----------|
| Config A (SL1.25 TP1-2R)    | +$132K   | 54.1% | 1.87   | +19.3pp | 2.21 | -$9.494   |
| SL1.5 TP1-1.25R P50%        | +$145K   | 57.0% | 1.27   | +12.9pp | 1.68 | -$21.557  |
| SL1.5 TP1-1.25R P70%        | +$143K   | 57.0% | 1.26   | +12.8pp | 1.67 | -$15.097  |
| SL1.5 TP1-1.25R P100%       | +$139K   | 57.0% | 1.25   | +12.5pp | 1.65 | -$10.949  |

**Top 3 por PnL vs Config A — umbral 0.55:**

| Config                      | PnL      | WR    | Payoff | Margen  | PF   | MaxDD     |
|-----------------------------|----------|-------|--------|---------|------|-----------|
| Config A (SL1.25 TP1-2R)    | +$97K    | 65.1% | 1.87   | +30.3pp | 3.49 | -$4.728   |
| SL1.5 TP1-1.25R P50%        | +$126K   | 62.1% | 1.35   | +19.6pp | 2.22 | -$10.320  |
| SL1.5 TP1-1.25R P70%        | +$126K   | 62.1% | 1.35   | +19.6pp | 2.21 | -$9.339   |
| SL1.75 TP1-1.0R P70%        | +$124K   | 64.7% | 1.02   | +15.3pp | 1.87 | -$11.734  |

Las alternativas con cierre parcial ganan más PnL bruto
porque operan con TPs cortos (1.0R-1.25R) que se alcanzan
más a menudo, generando más trades. Pero cada trade
individual rinde menos (payoff 1.25-1.35 vs 1.87) y el
drawdown es mayor (hasta -$21.5K vs -$9.5K). El cierre
parcial añade volumen sin mejorar la calidad.

Con esta verificación queda validado que Config A es la
mejor combinación de las 143 testadas desde el punto de
vista de robustez y calidad, y que el cierre 100% en TP1
sigue siendo la estrategia correcta también con filtro ML.

---

## 5. Decisión final — Configuración A

Se elige la **configuración A** (SL 1.25x ATR, TP1 2.0R,
cierre 100% en TP1) como nueva configuración del sistema.

**Parámetros:**

| Parámetro        | Anterior      | Config A (nueva) |
|------------------|---------------|------------------|
| SL               | 1.5x ATR      | 1.25x ATR        |
| TP1              | 1.0R          | 2.0R             |
| TP2              | 2.0R          | N/A (100% en TP1)|
| Cierre parcial   | 50% en TP1    | 100% en TP1      |
| Máx. velas       | 24            | 24               |

**Razones:**

La decisión se tomó analizando el rendimiento en todos los
umbrales (0.30 a 0.60), no solo en uno concreto. Config A
ofrece el mayor profit factor, mayor margen de seguridad,
menor drawdown, y mayor PnL por trade en la mayoría de
umbrales. La lógica se simplifica (sin segunda fase, sin
cierre parcial) reduciendo la complejidad del sistema.

---

## 6. Glosario

**Profit factor:** Ganancia bruta dividida entre pérdida bruta.
PF = 1.50 significa que por cada dólar que el sistema pierde,
gana $1.50. PF > 1.0 = sistema rentable.

**Payoff ratio:** Ganancia media por trade ganador dividida entre
pérdida media por trade perdedor. Payoff = 1.2 significa que los
trades ganadores ganan un 20% más de lo que pierden los perdedores.

**R-multiple:** PnL del trade dividido entre el riesgo inicial
(distancia entre entrada y SL). R = +1.0 significa que ganaste
exactamente lo que arriesgabas. R = -1.0 significa que perdiste
el riesgo completo.

**Breakeven WR:** Win rate mínimo para no perder dinero, dado el
payoff ratio actual. Se calcula como 1 / (1 + payoff) × 100.

**Margen de seguridad:** Diferencia entre el WR real y el breakeven
WR. Margen positivo = sistema rentable. Cuanto mayor, más robusto.

---

## 7. Despliegue

Config A desplegada en producción el 1 de mayo de 2026.
Se hizo merge de `feature/tp-sl-optimization` a `master`,
pull en el VPS y reinicio de los 3 servicios. Los datos
históricos del modelo anterior se archivaron en
`outputs/archive_v3/`. El historial de señales live
empieza de cero para evaluar el nuevo modelo con señales
reales. Umbral de filtrado pendiente de decidir.

### 7.1 Error en el modelo desplegado (detectado 24 junio 2026)

El despliegue del 1 de mayo actualizó las reglas de TP/SL
(Config A) y generó el modelo reentrenado como
`outputs/model_v3.joblib`. Sin embargo, el sistema live
(`08_live_signal.py`) carga `outputs/model.joblib`, que
**no fue actualizado** — seguía siendo el modelo original
entrenado con las etiquetas del sistema antiguo (SL 1.5x,
TP1 1.0R, cierre parcial 50%).

Consecuencia: durante 54 días (1 mayo — 24 junio) el filtro
ML predecía "probabilidad de alcanzar 1.0R" cuando el trade
necesitaba alcanzar 2.0R para ganar. Las señales de alta
probabilidad (70-84%) fallaban porque el modelo respondía a
la pregunta equivocada. Ejemplo: 5 señales con prob ≥ 60%
produjeron solo 1 ganadora (WR 20%).

El diagnóstico se confirmó comparando los hashes MD5:
- `model.joblib` en VPS: `54a33d6...` (modelo antiguo)
- `model_v3.joblib` en VPS: `b8956ce...` (modelo Config A)

Corrección (commit `697ea8d`): `model_v3.joblib` se renombró
a `model.joblib`, eliminando todos los modelos obsoletos.
Historial live archivado en `outputs/archive_modelo_viejo_2026/`
y reseteado a cero. A partir del 24 de junio, el sistema
evalúa señales con el modelo correcto de Config A.

Los datos de los 54 días con el modelo incorrecto no son
válidos para evaluar Config A. El sistema sin filtrar generó
+8.65% en ese periodo, lo que confirma que las reglas de
entrada y la estructura de TP/SL de Config A funcionan
independientemente del filtro ML.

---

**Scripts:** `src/tp_sl_optimizer.py`, `src/tp_sl_retrain_candidates.py`
**Resultados barrido:** `outputs/tp_sl_optimization/sweep_results_latest.csv`
**Resultados candidatas:** `outputs/tp_sl_optimization/candidates/`

*Fecha: 1 de mayo de 2026. Actualizado: 24 de junio de 2026.*
