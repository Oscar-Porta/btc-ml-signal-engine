# Fase C — Resultados (4 julio 2026)

> Trabajo autónomo autorizado por el autor. Script: `src/experiments/fase_c.py`
> Referencia (BASE): 1h + ejecución limit, 20 features, GB, walk-forward 4 folds — 0.60: +0.098R, 150 trades OOS

## Qué se probó y qué salió

Todos los experimentos evaluados con el mismo walk-forward y PnL neto, contra el criterio predefinido en `propuesta_fase_c.md` (mejorar 0.45-0.50 sin romper 0.60 ni volver negativo ningún fold):

| Experimento | Resultado | ¿Adoptado? |
|-------------|-----------|------------|
| L1 — features firmadas + sin atr_14 (C3+C4) | Mejora leve 0.45-0.55 (0.50: +0.030R) pero rompe 0.60 (−0.072R) | No |
| L2 — L1 + régimen: ER, ADX 4h, BB width, pendiente EMA, velas desde cruce (C1) | Empeora en general (0.45: −0.049R) | No |
| L3 — L2 + n_trades relativo (C2a) | Mixto, sin mejora clara | No |
| C7 — sample_weight=\|R\| | Aplana todo hacia cero | No |
| C5 — HistGB (proxy de LightGBM, early stopping) | Claramente peor (0.55: −0.142R) | No |
| C5 — HistGB + class_weight=balanced | Sin mejora | No |
| C5 — Ensemble 5 semillas del GB | No supera a BASE en 0.60 | No |

Notas operativas: xgboost/lightgbm no se pudieron instalar en el entorno de trabajo (timeout) — HistGB de sklearn hizo de proxy (mismo diseño que LightGBM); el bake-off literal con XGB/CatBoost puede hacerse en el Mac del autor si algún día se quiere, expectativa baja visto lo visto. El taker_buy_ratio (C2b) requiere re-descargar el histórico guardando esa columna — pendiente, único experimento de la lista original sin probar.

## El hallazgo que manda: la ventaja está dentro del ruido

BASE en 0.60: R medio **+0.098**, desviación típica 1.47, n=150 → error estándar **±0.120**. Es decir: el intervalo razonable va de −0.02R a +0.22R. **Con los datos actuales no se puede afirmar que la ventaja sea real** — podría serlo (tres años seguidos en positivo es alentador), pero también podría ser ruido con suerte.

Esto reencuadra todo: el cuello de botella del proyecto **no es el modelo** (7 intentos serios de mejorarlo, ninguno supera al actual — las 20 features + GB ya extraen lo que hay), **es la cantidad de datos y el tamaño de la ventaja**. Con 4.4k trades históricos y ~50 señales/año en el umbral útil, ningún algoritmo puede demostrar más.

## Decisión tomada sobre la marcha (documentada)

C6 (calibración + umbrales por valor esperado) **se pospone**: calibrar probabilidades tiene sentido cuando hay una ventaja demostrada que calibrar; hoy sería pulir la mirilla de un rifle que no sabemos si dispara. La matemática queda lista para entonces: con TP 2R, SL 1R y peaje limit ~0.05R, el breakeven está en p_real ≈ 0.35.

## Qué puede cambiar el juego (por orden de impacto esperado)

1. **Datos live con ejecución limit** (gratis, ya en marcha conceptualmente): cada mes de producción son ~30-50 trades reales del régimen actual. Es el único juez definitivo.
2. **Pooling multi-activo** (Fase D adelantable): replicar señales para ETH y SOL multiplica el dataset ×3 → el modelo puede aprender patrones con soporte estadístico real, y la variante 4h (aparcada por falta de datos) se vuelve a abrir.
3. **taker_buy_ratio** (C2b): pendiente de re-descarga, única feature nueva sin probar.
4. Revisar las reglas de entrada mismas (generan 36% WR bruto — mejorar la materia prima, no solo el filtro). Territorio nuevo, para discutir.

## Estado del producto (para decisión del autor)

Con lo que sabemos hoy: el sistema 1h+limit en 0.60 es la mejor configuración conocida, plausiblemente rentable pero no demostrada. Los números de la web/marketing actuales (WR 54-67%) provienen del modelo con lookahead y no son defendibles. Decisiones abiertas: mensaje público durante la validación, si se despliega el modelo limpio al VPS ya (recomendado: sí, con mensaje limit — al menos los datos live que se recopilen serán del sistema bueno), y si se adelanta la Fase D para el pooling.
