"""
==========================================================
FASE 5 - Evaluación del sistema y curva de equity
==========================================================
Este script lee el histórico de trades (outputs/trades.csv)
producido por el backtest y genera:

  1. Curva de equity acumulada (cómo evoluciona el capital).
  2. Drawdown máximo (peor caída desde un pico antes de
     recuperarse; mide el riesgo real del sistema).
  3. Gráfico .png con ambas visualizaciones.
  4. Resumen de métricas avanzadas por consola.

La curva de equity es la herramienta de evaluación más
importante en trading cuantitativo: te dice de un vistazo
si el sistema tiene potencial o no.
==========================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sin ventana (para generar PNG)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
OUTPUT_DIR = "outputs"
TRADES_FILE = os.path.join(OUTPUT_DIR, "trades.csv")
EQUITY_FILE = os.path.join(OUTPUT_DIR, "equity_curve.csv")
CHART_FILE = os.path.join(OUTPUT_DIR, "equity_chart.png")


# ==========================================================
# FUNCIÓN 1 - Cargar trades
# ==========================================================
def cargar_trades():
    """Lee el CSV de trades y convierte fechas."""
    df = pd.read_csv(TRADES_FILE)

    for col in ["signal_time", "exit_time"]:
        df[col] = pd.to_datetime(
            df[col], format="ISO8601", utc=True
        )
        df[col] = df[col].dt.tz_convert("Europe/Madrid")

    df = df.sort_values("signal_time").reset_index(drop=True)

    print(f"Trades cargados: {len(df)}")
    return df


# ==========================================================
# FUNCIÓN 2 - Calcular curva de equity y drawdown
# ==========================================================
def calcular_equity_y_drawdown(df):
    """
    Construye la curva de equity acumulando el PnL absoluto
    de cada trade en orden cronológico.

    Drawdown = diferencia entre la equity actual y el máximo
    histórico hasta ese punto. Representa cuánto estás
    "por debajo de tu mejor momento".

    Drawdown máximo = la mayor caída que sufre la curva antes
    de recuperarse. Es la métrica de riesgo más importante:
    si tu drawdown máximo es de 50.000 USD, necesitas tener
    el estómago y el capital para aguantar esa caída.
    """
    df = df.copy()

    # Equity acumulada
    df["equity"] = df["pnl_abs"].cumsum()

    # Pico máximo hasta cada trade
    df["equity_peak"] = df["equity"].cummax()

    # Drawdown en cada punto
    df["drawdown"] = df["equity"] - df["equity_peak"]

    # Drawdown en porcentaje del pico (más intuitivo)
    # Evitamos dividir por 0 si el pico es 0 o negativo.
    df["drawdown_pct"] = np.where(
        df["equity_peak"] > 0,
        (df["drawdown"] / df["equity_peak"]) * 100,
        0
    )

    max_dd = df["drawdown"].min()
    max_dd_pct = df["drawdown_pct"].min()

    return df, max_dd, max_dd_pct


# ==========================================================
# FUNCIÓN 3 - Métricas avanzadas
# ==========================================================
def calcular_metricas_avanzadas(df, max_dd):
    """
    Calcula métricas complementarias que dan una imagen
    más completa del rendimiento.
    """
    total = len(df)
    ganadores = df[df["pnl_abs"] > 0]
    perdedores = df[df["pnl_abs"] < 0]

    pnl_total = df["pnl_abs"].sum()

    # Racha máxima ganadora y perdedora
    # (número consecutivo de trades ganadores/perdedores)
    es_ganador = (df["pnl_abs"] > 0).astype(int)
    es_perdedor = (df["pnl_abs"] < 0).astype(int)

    racha_ganadora = _racha_maxima(es_ganador)
    racha_perdedora = _racha_maxima(es_perdedor)

    # Expectativa por trade (lo que esperas ganar de media)
    expectativa = pnl_total / total if total > 0 else 0

    # Recovery factor: PnL total / |drawdown máximo|
    # Mide cuánto gana el sistema por unidad de sufrimiento.
    recovery_factor = (
        pnl_total / abs(max_dd) if max_dd != 0 else 0
    )

    # Ratio ganancia media / pérdida media
    avg_win = ganadores["pnl_abs"].mean() if len(ganadores) > 0 else 0
    avg_loss = abs(perdedores["pnl_abs"].mean()) if len(perdedores) > 0 else 0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    return {
        "pnl_total": pnl_total,
        "expectativa_por_trade": expectativa,
        "max_drawdown_abs": max_dd,
        "recovery_factor": recovery_factor,
        "payoff_ratio": payoff_ratio,
        "racha_ganadora_max": racha_ganadora,
        "racha_perdedora_max": racha_perdedora,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def _racha_maxima(serie_binaria):
    """
    Calcula la racha máxima de 1s consecutivos en una serie.
    Ej: [0,1,1,1,0,1,1] → 3
    """
    if serie_binaria.sum() == 0:
        return 0

    grupos = serie_binaria.ne(serie_binaria.shift()).cumsum()
    rachas = serie_binaria.groupby(grupos).sum()
    return int(rachas.max())


# ==========================================================
# FUNCIÓN 4 - Generar gráfico
# ==========================================================
def generar_grafico(df, max_dd):
    """
    Genera un gráfico con dos paneles:
      - Arriba: curva de equity con línea de pico.
      - Abajo: drawdown en USD.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    fechas = df["signal_time"]

    # --- Panel superior: equity ---
    ax1.plot(fechas, df["equity"], color="#2196F3",
             linewidth=1, label="Equity")
    ax1.plot(fechas, df["equity_peak"], color="#90CAF9",
             linewidth=0.7, linestyle="--", alpha=0.7,
             label="Pico máximo")
    ax1.fill_between(fechas, df["equity"], 0,
                     where=(df["equity"] >= 0),
                     color="#C8E6C9", alpha=0.3)
    ax1.fill_between(fechas, df["equity"], 0,
                     where=(df["equity"] < 0),
                     color="#FFCDD2", alpha=0.3)
    ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
    ax1.set_ylabel("PnL acumulado (USD)")
    ax1.set_title(
        "Curva de Equity — BTC/USDT 1h — "
        f"{len(df)} trades (5 años)",
        fontsize=13, fontweight="bold"
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel inferior: drawdown ---
    ax2.fill_between(fechas, df["drawdown"], 0,
                     color="#EF5350", alpha=0.4)
    ax2.plot(fechas, df["drawdown"], color="#EF5350",
             linewidth=0.8)
    ax2.axhline(y=max_dd, color="#B71C1C", linewidth=0.8,
                linestyle="--", alpha=0.7,
                label=f"Max DD: {max_dd:,.0f} USD")
    ax2.set_ylabel("Drawdown (USD)")
    ax2.set_xlabel("Fecha")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Formato de fechas
    ax2.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %Y")
    )
    ax2.xaxis.set_major_locator(
        mdates.MonthLocator(interval=4)
    )
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Gráfico guardado: {CHART_FILE}")


# ==========================================================
# FUNCIÓN 5 - Mostrar métricas
# ==========================================================
def mostrar_metricas(metricas, max_dd_pct):
    """Imprime las métricas avanzadas por consola."""
    print(f"\n{'='*60}")
    print("EVALUACIÓN AVANZADA DEL SISTEMA")
    print(f"{'='*60}")

    print(f"\nPnL total: {metricas['pnl_total']:,.2f} USD")
    print(f"Expectativa por trade: {metricas['expectativa_por_trade']:,.2f} USD")
    print(f"\nPayoff ratio (ganancia media / pérdida media): "
          f"{metricas['payoff_ratio']:.2f}")
    print(f"Ganancia media: {metricas['avg_win']:,.2f} USD")
    print(f"Pérdida media: {metricas['avg_loss']:,.2f} USD")

    print(f"\nMáximo drawdown: {metricas['max_drawdown_abs']:,.2f} USD")
    print(f"Máximo drawdown (%): {max_dd_pct:.2f}%")
    print(f"Recovery factor: {metricas['recovery_factor']:.2f}")

    print(f"\nRacha ganadora más larga: "
          f"{metricas['racha_ganadora_max']} trades seguidos")
    print(f"Racha perdedora más larga: "
          f"{metricas['racha_perdedora_max']} trades seguidos")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("Evaluando sistema...\n")

    # 1. Cargar
    df = cargar_trades()

    # 2. Equity y drawdown
    df_equity, max_dd, max_dd_pct = calcular_equity_y_drawdown(df)

    # 3. Métricas avanzadas
    metricas = calcular_metricas_avanzadas(df_equity, max_dd)

    # 4. Mostrar
    mostrar_metricas(metricas, max_dd_pct)

    # 5. Gráfico
    generar_grafico(df_equity, max_dd)

    # 6. Guardar curva de equity como CSV
    cols_equity = [
        "signal_time", "direction", "rule", "close_reason",
        "pnl_abs", "equity", "equity_peak", "drawdown"
    ]
    df_equity[cols_equity].to_csv(EQUITY_FILE, index=False)
    print(f"Curva de equity guardada: {EQUITY_FILE}")


if __name__ == "__main__":
    main()
