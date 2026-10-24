"""
Simulación de Montecarlo para estimar valores futuros de una inversión para el retiro.
=====================================================================================

Idea general del método de Montecarlo
-------------------------------------
No sabemos qué rendimiento tendrá el mercado cada mes. En lugar de suponer un único
rendimiento fijo, generamos miles de "futuros posibles" (trayectorias) sorteando
rendimientos mensuales aleatorios con una distribución de probabilidad. Al final
miramos la distribución de los saldos obtenidos: su media, sus percentiles y qué
fracción de las trayectorias alcanza la meta. Esa fracción es una estimación de la
probabilidad de alcanzarla (ley de los grandes números: con más trayectorias, la
estimación se vuelve más estable).

Modelo que se usa en este programa (resumen; el README tiene las fórmulas completas)
-----------------------------------------------------------------------------------
1. Rendimientos: cada mes el portafolio se multiplica por un factor aleatorio
       R_t = exp(X_t),   X_t ~ Normal(m, s^2)
   es decir, el rendimiento logarítmico mensual es normal y el factor de crecimiento
   es log-normal. Así el saldo nunca se vuelve negativo por rendimientos (un -100 %
   o peor es imposible), algo que sí puede pasar si se sortean rendimientos simples
   con una normal.

2. Conversión de tasas anuales a mensuales:
       s = sigma / sqrt(12)                    (la varianza crece con el tiempo)
       m = ln(1 + mu) / 12 - s^2 / 2           (corrección de convexidad)
   Con esto E[R_t] = (1 + mu)^(1/12) y el valor esperado de un año completo es
   exactamente 1 + mu. Aquí mu es el rendimiento anual esperado (nominal) y sigma
   la volatilidad anual de los rendimientos logarítmicos.

3. Comisión anual sobre saldo c: cada mes se cobra el factor (1 - c)^(1/12).

4. Inflación anual pi: tasa mensual equivalente (1 + pi)^(1/12) - 1. Se supone
   constante (determinista) para separar el riesgo de mercado del efecto de precios.

5. Orden de operaciones en cada mes t = 1, ..., n:
       - si las aportaciones son al INICIO: saldo += aportación
       - saldo *= R_t                 (rendimiento del mercado)
       - saldo *= (1 - c)^(1/12)      (comisión)
       - si las aportaciones son al FINAL: saldo += aportación

6. Pesos nominales vs. pesos de hoy:
       valor_real_t = valor_nominal_t / (1 + pi)^(t / 12)
   Los pesos nominales son los que aparecerán en el estado de cuenta; los pesos de
   hoy (pesos reales) descuentan la inflación y dicen cuánto poder de compra
   representa ese saldo con los precios actuales. La meta se expresa en pesos de hoy.

AVISO: los parámetros por defecto son ilustrativos. Esto es un ejercicio académico,
no un pronóstico ni una asesoría financiera.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Número mínimo de trayectorias que exige la actividad para la ejecución desde la
# línea de comandos. La función de simulación acepta menos (útil en pruebas).
MIN_SIMULACIONES_CLI = 10_000

# Percentiles que se reportan en el resumen.
PERCENTILES = (5, 25, 50, 75, 95)


# ---------------------------------------------------------------------------
# 1. Parámetros
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Parametros:
    """Todos los supuestos de la simulación en un solo lugar.

    Las tasas se expresan como DECIMALES (0.08 = 8 %). En la línea de comandos se
    escriben como porcentaje (8 = 8 %) y se convierten antes de llegar aquí.
    """

    saldo_inicial: float = 50_000.0        # pesos que ya se tienen ahorrados hoy
    aportacion_mensual: float = 3_000.0    # pesos de hoy que se aportan cada mes
    anios: int = 30                        # horizonte de inversión en años
    rendimiento_anual: float = 0.08        # mu: rendimiento anual esperado (nominal)
    volatilidad_anual: float = 0.15        # sigma: volatilidad anual (desv. estándar)
    inflacion_anual: float = 0.04          # pi: inflación anual supuesta
    comision_anual: float = 0.0055         # c: comisión anual sobre el saldo
    meta_retiro: float = 2_000_000.0      # meta en PESOS DE HOY
    semilla: int | None = 2026             # semilla del generador aleatorio
    n_simulaciones: int = 10_000           # número de trayectorias
    aportacion_al_inicio: bool = True      # True: inicio de mes; False: fin de mes
    aportacion_indexada: bool = True       # True: la aportación sube con la inflación

    def validar(self) -> None:
        """Revisa que los supuestos tengan sentido antes de simular."""
        if self.saldo_inicial < 0:
            raise ValueError("El saldo inicial no puede ser negativo.")
        if self.aportacion_mensual < 0:
            raise ValueError("La aportación mensual no puede ser negativa.")
        if int(self.anios) != self.anios or self.anios < 1:
            raise ValueError("Los años de inversión deben ser un entero mayor o igual a 1.")
        if self.rendimiento_anual <= -1:
            raise ValueError("El rendimiento anual debe ser mayor que -100 %.")
        if self.volatilidad_anual < 0:
            raise ValueError("La volatilidad no puede ser negativa.")
        if self.inflacion_anual <= -1:
            raise ValueError("La inflación anual debe ser mayor que -100 %.")
        if not 0 <= self.comision_anual < 1:
            raise ValueError("La comisión anual debe estar entre 0 % y menos de 100 %.")
        if self.meta_retiro < 0:
            raise ValueError("La meta de retiro no puede ser negativa.")
        if int(self.n_simulaciones) != self.n_simulaciones or self.n_simulaciones < 1:
            raise ValueError("El número de simulaciones debe ser un entero positivo.")

    @property
    def meses(self) -> int:
        return int(self.anios) * 12


# ---------------------------------------------------------------------------
# 2. Conversión de tasas anuales a mensuales
# ---------------------------------------------------------------------------
def parametros_mensuales_lognormales(rendimiento_anual: float,
                                     volatilidad_anual: float) -> tuple[float, float]:
    """Devuelve (m, s): media y desviación estándar del rendimiento LOGARÍTMICO mensual.

    - s = sigma / sqrt(12): si los meses son independientes, las varianzas se suman,
      así que la varianza anual es 12 * s^2 y la desviación se escala con sqrt(12).
    - m = ln(1 + mu) / 12 - s^2 / 2: para una log-normal, E[exp(X)] = exp(m + s^2/2).
      Restar s^2/2 hace que el factor mensual esperado sea (1 + mu)^(1/12) y que el
      factor anual esperado sea exactamente (1 + mu). Sin esta corrección la
      simulación sobreestimaría el rendimiento promedio.
    """
    s = volatilidad_anual / math.sqrt(12)
    m = math.log1p(rendimiento_anual) / 12 - 0.5 * s ** 2
    return m, s


def tasa_mensual_equivalente(tasa_anual: float) -> float:
    """Tasa mensual compuesta equivalente: (1 + r)^(1/12) - 1.

    Se usa compuesta (y no r / 12) para que 12 meses compongan exactamente la tasa
    anual: (1 + r_mensual)^12 = 1 + r_anual.
    """
    return (1 + tasa_anual) ** (1 / 12) - 1


def factores_inflacion(inflacion_anual: float, meses: int) -> np.ndarray:
    """Índice de precios acumulado (1 + pi)^(t/12) para t = 0, 1, ..., meses."""
    t = np.arange(meses + 1)
    return (1 + inflacion_anual) ** (t / 12)


# ---------------------------------------------------------------------------
# 3. Simulación
# ---------------------------------------------------------------------------
def simular_trayectorias(p: Parametros) -> np.ndarray:
    """Simula las trayectorias mensuales del saldo en PESOS NOMINALES.

    Devuelve una matriz de forma (n_simulaciones, meses + 1). La columna 0 es el
    saldo inicial y la columna t es el saldo al cierre del mes t.
    """
    p.validar()
    n, meses = int(p.n_simulaciones), p.meses

    # Generador moderno de NumPy. La misma semilla produce exactamente los mismos
    # números aleatorios, así que los resultados son reproducibles.
    rng = np.random.default_rng(p.semilla)

    # Sorteo de TODOS los rendimientos logarítmicos de una vez: una fila por
    # trayectoria y una columna por mes. Los meses son independientes entre sí.
    m, s = parametros_mensuales_lognormales(p.rendimiento_anual, p.volatilidad_anual)
    rend_log = rng.normal(loc=m, scale=s, size=(n, meses))
    factores_rend = np.exp(rend_log)          # factor de crecimiento R_t > 0

    # Factor mensual de comisión: cobrar (1 - c)^(1/12) doce veces = (1 - c) al año.
    factor_comision = (1 - p.comision_anual) ** (1 / 12)

    # Aportaciones nominales de cada mes. Si están indexadas, la aportación del mes t
    # (t = 1..n) es A * (1 + pi_mensual)^(t - 1): la primera es A y luego crece con la
    # inflación, de modo que su poder de compra se mantiene aproximadamente constante.
    pi_m = tasa_mensual_equivalente(p.inflacion_anual)
    t = np.arange(meses)                      # 0, 1, ..., meses - 1
    if p.aportacion_indexada:
        aportaciones = p.aportacion_mensual * (1 + pi_m) ** t
    else:
        aportaciones = np.full(meses, float(p.aportacion_mensual))

    # Matriz de resultados; la columna 0 es el saldo inicial en todas las trayectorias.
    saldos = np.empty((n, meses + 1))
    saldos[:, 0] = p.saldo_inicial

    # Recorremos los meses; cada paso actualiza las n trayectorias a la vez
    # (operación vectorizada), lo que es mucho más rápido que un ciclo por trayectoria.
    saldo = saldos[:, 0].copy()
    for mes in range(meses):
        if p.aportacion_al_inicio:
            saldo += aportaciones[mes]        # el dinero entra y ya gana el mes
        saldo *= factores_rend[:, mes]        # rendimiento aleatorio del mercado
        saldo *= factor_comision              # comisión de la administradora
        if not p.aportacion_al_inicio:
            saldo += aportaciones[mes]        # el dinero entra al cierre del mes
        saldos[:, mes + 1] = saldo

    return saldos


def a_pesos_de_hoy(saldos_nominales: np.ndarray, inflacion_anual: float) -> np.ndarray:
    """Convierte trayectorias nominales a pesos de hoy dividiendo entre (1 + pi)^(t/12)."""
    meses = saldos_nominales.shape[-1] - 1
    return saldos_nominales / factores_inflacion(inflacion_anual, meses)


def total_aportado(p: Parametros) -> tuple[float, float]:
    """Total aportado (saldo inicial + aportaciones) en pesos nominales y en pesos de hoy."""
    meses = p.meses
    pi_m = tasa_mensual_equivalente(p.inflacion_anual)
    t = np.arange(meses)
    if p.aportacion_indexada:
        aport = p.aportacion_mensual * (1 + pi_m) ** t
    else:
        aport = np.full(meses, float(p.aportacion_mensual))
    # La aportación del mes t se deflacta con el índice del momento en que entra.
    momento = t if p.aportacion_al_inicio else t + 1
    nominal = p.saldo_inicial + aport.sum()
    real = p.saldo_inicial + (aport / (1 + pi_m) ** momento).sum()
    return float(nominal), float(real)


# ---------------------------------------------------------------------------
# 4. Estadísticas
# ---------------------------------------------------------------------------
def resumir(valores_finales: np.ndarray, meta: float | None = None) -> dict:
    """Media, desviación estándar, percentiles y (opcional) probabilidad de alcanzar la meta."""
    v = np.asarray(valores_finales, dtype=float)
    resumen = {
        "media": float(v.mean()),
        "desviacion_estandar": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "minimo": float(v.min()),
        "maximo": float(v.max()),
    }
    for q, valor in zip(PERCENTILES, np.percentile(v, PERCENTILES)):
        resumen[f"p{q}"] = float(valor)
    if meta is not None:
        # Probabilidad estimada = fracción de trayectorias con saldo >= meta.
        exito = v >= meta
        prob = float(exito.mean())
        resumen["probabilidad_meta"] = prob
        # Error estándar de una proporción: sqrt(p (1 - p) / N). Indica cuánto puede
        # variar la estimación por usar un número finito de trayectorias.
        resumen["error_estandar_prob"] = math.sqrt(prob * (1 - prob) / v.size)
    return resumen


@dataclass
class Resultado:
    """Todo lo que produce una corrida: trayectorias y resúmenes."""

    parametros: Parametros
    nominal: np.ndarray            # trayectorias en pesos nominales
    real: np.ndarray               # trayectorias en pesos de hoy
    resumen_nominal: dict
    resumen_real: dict
    aportado_nominal: float
    aportado_real: float
    determinista_real: float       # valor final si el rendimiento fuera siempre el esperado


def ejecutar(p: Parametros) -> Resultado:
    """Corre la simulación completa y calcula los resúmenes."""
    nominal = simular_trayectorias(p)
    real = a_pesos_de_hoy(nominal, p.inflacion_anual)

    # La meta está en pesos de hoy, así que la probabilidad se calcula en pesos de hoy.
    # (Es equivalente a comparar en nominal contra la meta inflada al año de retiro.)
    resumen_real = resumir(real[:, -1], p.meta_retiro)
    meta_nominal = p.meta_retiro * (1 + p.inflacion_anual) ** p.anios
    resumen_nominal = resumir(nominal[:, -1], meta_nominal)

    # Escenario de referencia sin volatilidad: crecer cada mes con el rendimiento
    # esperado. Sirve para ver el "arrastre de volatilidad": la mediana simulada suele
    # quedar por debajo de este valor aunque la media sea parecida.
    p_det = Parametros(**{**asdict(p), "volatilidad_anual": 0.0, "n_simulaciones": 1})
    determinista = a_pesos_de_hoy(simular_trayectorias(p_det), p.inflacion_anual)[0, -1]

    aportado_nominal, aportado_real = total_aportado(p)
    return Resultado(p, nominal, real, resumen_nominal, resumen_real,
                     aportado_nominal, aportado_real, float(determinista))


# ---------------------------------------------------------------------------
# 5. Reporte en texto
# ---------------------------------------------------------------------------
def _pesos(x: float) -> str:
    return f"${x:,.0f}"


def _pct(x: float) -> str:
    return f"{x * 100:.2f} %"


def reporte_texto(r: Resultado) -> str:
    """Arma el reporte legible que se imprime en pantalla y se guarda en archivo."""
    p = r.parametros
    m, s = parametros_mensuales_lognormales(p.rendimiento_anual, p.volatilidad_anual)
    lineas = [
        "=" * 72,
        "SIMULACIÓN DE MONTECARLO - INVERSIÓN PARA EL RETIRO",
        "=" * 72,
        "Supuestos (ilustrativos; no son pronóstico ni asesoría financiera):",
        f"  Saldo inicial ..................... {_pesos(p.saldo_inicial)}",
        f"  Aportación mensual ................ {_pesos(p.aportacion_mensual)}"
        + (" (pesos de hoy, sube con la inflación)" if p.aportacion_indexada
           else " (monto nominal fijo)"),
        f"  Momento de la aportación .......... "
        + ("inicio de cada mes" if p.aportacion_al_inicio else "fin de cada mes"),
        f"  Horizonte ......................... {p.anios} años ({p.meses} meses)",
        f"  Rendimiento anual esperado ........ {_pct(p.rendimiento_anual)} nominal",
        f"  Volatilidad anual ................. {_pct(p.volatilidad_anual)}",
        f"  Inflación anual ................... {_pct(p.inflacion_anual)}",
        f"  Comisión anual sobre saldo ........ {_pct(p.comision_anual)}",
        f"  Meta de retiro .................... {_pesos(p.meta_retiro)} pesos de hoy"
        f" (= {_pesos(p.meta_retiro * (1 + p.inflacion_anual) ** p.anios)} nominales)",
        f"  Trayectorias simuladas ............ {p.n_simulaciones:,}",
        f"  Semilla aleatoria ................. {p.semilla}",
        "",
        "Conversión a parámetros mensuales (rendimiento logarítmico ~ Normal(m, s^2)):",
        f"  m = ln(1 + mu)/12 - s^2/2 = {m:.6f}   s = sigma/sqrt(12) = {s:.6f}",
        f"  Inflación mensual = (1 + pi)^(1/12) - 1 = "
        f"{tasa_mensual_equivalente(p.inflacion_anual) * 100:.4f} %",
        "",
        f"Total aportado (incluye saldo inicial): {_pesos(r.aportado_nominal)} nominales"
        f" = {_pesos(r.aportado_real)} pesos de hoy",
        f"Valor final sin volatilidad (rendimiento constante): "
        f"{_pesos(r.determinista_real)} pesos de hoy",
        "",
        f"Valor final al cabo de {p.anios} años:",
        f"  {'Estadístico':<22}{'Pesos nominales':>22}{'Pesos de hoy':>22}",
        f"  {'-' * 66}",
    ]
    filas = [("Media", "media")] + [(f"Percentil {q}", f"p{q}") for q in PERCENTILES]
    for etiqueta, clave in filas:
        lineas.append(f"  {etiqueta:<22}{_pesos(r.resumen_nominal[clave]):>22}"
                      f"{_pesos(r.resumen_real[clave]):>22}")
    prob = r.resumen_real["probabilidad_meta"]
    ee = r.resumen_real["error_estandar_prob"]
    lineas += [
        "",
        f"Probabilidad de alcanzar la meta: {_pct(prob)}"
        f"  (error estándar de Montecarlo: +/- {ee * 100:.2f} puntos)",
        "",
        "Lectura: el percentil 5 es un escenario desfavorable (solo 5 % de los futuros",
        "simulados terminan peor) y el percentil 95 uno favorable. La mediana (P50) es",
        "el resultado 'típico'; la media queda arriba porque la distribución tiene cola",
        "larga hacia la derecha.",
        "=" * 72,
    ]
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# 6. Gráfica
# ---------------------------------------------------------------------------
def graficar(r: Resultado, ruta: str | Path | None = None, mostrar: bool = False,
             n_muestra: int = 40) -> None:
    """Dos paneles en pesos de hoy: abanico de trayectorias y distribución del valor final."""
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")               # dibuja sin abrir ventana (sirve en servidores)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    p = r.parametros
    real = r.real
    anios = np.arange(real.shape[1]) / 12

    # Colores: un solo tono azul (claro -> oscuro) para magnitud y un naranja para la meta.
    azul_claro, azul_medio, azul_oscuro = "#c6dbf5", "#7fb0ea", "#1d5fb0"
    naranja, tinta, tinta_suave = "#d9531e", "#0b0b0b", "#52514e"

    millones = FuncFormatter(lambda x, _: f"${x / 1e6:,.1f} M")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor("white")

    # --- Panel izquierdo: abanico de percentiles mes a mes ---
    bandas = np.percentile(real, [5, 25, 50, 75, 95], axis=0)
    ax1.fill_between(anios, bandas[0], bandas[4], color=azul_claro, lw=0,
                     label="Rango P5 a P95 (90 % de los casos)")
    ax1.fill_between(anios, bandas[1], bandas[3], color=azul_medio, lw=0,
                     label="Rango P25 a P75 (50 % de los casos)")
    # Algunas trayectorias individuales para mostrar que cada futuro es irregular.
    k = min(n_muestra, real.shape[0])
    for fila in real[:k]:
        ax1.plot(anios, fila, color=azul_oscuro, alpha=0.08, lw=0.8)
    ax1.plot(anios, bandas[2], color=azul_oscuro, lw=2.2, label="Mediana (P50)")
    ax1.axhline(p.meta_retiro, color=naranja, lw=2, ls="--",
                label=f"Meta: {_pesos(p.meta_retiro)}")
    ax1.set_title("Evolución del saldo (pesos de hoy)", loc="left", fontsize=13, color=tinta)
    ax1.set_xlabel("Años desde hoy", color=tinta_suave)
    ax1.set_ylabel("Saldo en pesos de hoy", color=tinta_suave)
    ax1.yaxis.set_major_formatter(millones)
    ax1.set_xlim(0, p.anios)
    ax1.set_ylim(0, bandas[4, -1] * 1.08)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    # --- Panel derecho: histograma del valor final ---
    finales = real[:, -1]
    tope = np.percentile(finales, 99.5)      # recorta la cola extrema para que se lea
    ax2.hist(np.clip(finales, None, tope), bins=60, color=azul_medio,
             edgecolor="white", linewidth=0.6)
    # Las líneas de percentiles se identifican en la leyenda (con su valor) para que
    # las etiquetas no se encimen cuando dos percentiles quedan cerca.
    for q, estilo in ((5, ":"), (50, "-"), (95, "-.")):
        x = r.resumen_real[f"p{q}"]
        ax2.axvline(x, color=azul_oscuro, lw=1.6, ls=estilo, label=f"P{q}: {_pesos(x)}")
    prob = r.resumen_real["probabilidad_meta"]
    ax2.axvline(p.meta_retiro, color=naranja, lw=2, ls="--",
                label=f"Meta: {_pesos(p.meta_retiro)}")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="white",
               framealpha=1, fontsize=9,
               title=f"Probabilidad de alcanzar la meta: {prob * 100:.1f} %",
               title_fontproperties={"weight": "bold", "size": 9.5})
    ax2.set_title(f"Distribución del saldo final a {p.anios} años", loc="left",
                  fontsize=13, color=tinta)
    ax2.set_xlabel("Saldo final en pesos de hoy (cola > P99.5 agrupada)", color=tinta_suave)
    ax2.set_ylabel("Número de trayectorias", color=tinta_suave)
    ax2.xaxis.set_major_formatter(millones)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#b5b4ad")
        ax.tick_params(colors=tinta_suave)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(f"Montecarlo de retiro: {p.n_simulaciones:,} trayectorias, semilla "
                 f"{p.semilla} (supuestos ilustrativos)", fontsize=11, color=tinta_suave,
                 x=0.01, ha="left")
    fig.tight_layout()
    if ruta:
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=150)
    if mostrar:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Línea de comandos
# ---------------------------------------------------------------------------
def _entero_min_simulaciones(texto: str) -> int:
    valor = int(texto)
    if valor < MIN_SIMULACIONES_CLI:
        raise argparse.ArgumentTypeError(
            f"se requieren al menos {MIN_SIMULACIONES_CLI:,} trayectorias")
    return valor


def construir_parser() -> argparse.ArgumentParser:
    d = Parametros()   # valores por defecto
    ap = argparse.ArgumentParser(
        description="Simulación de Montecarlo para el ahorro para el retiro. "
                    "Las tasas se escriben en PORCENTAJE (8 = 8 %%).")
    ap.add_argument("--saldo-inicial", type=float, default=d.saldo_inicial,
                    help="saldo ahorrado hoy, en pesos (por defecto %(default).0f)")
    ap.add_argument("--aportacion-mensual", type=float, default=d.aportacion_mensual,
                    help="aportación mensual en pesos de hoy (por defecto %(default).0f)")
    ap.add_argument("--anios", type=int, default=d.anios,
                    help="años de inversión (por defecto %(default)s)")
    ap.add_argument("--rendimiento", type=float, default=d.rendimiento_anual * 100,
                    help="rendimiento anual esperado nominal en %% (por defecto %(default)s)")
    ap.add_argument("--volatilidad", type=float, default=d.volatilidad_anual * 100,
                    help="volatilidad anual en %% (por defecto %(default)s)")
    ap.add_argument("--inflacion", type=float, default=d.inflacion_anual * 100,
                    help="inflación anual en %% (por defecto %(default)s)")
    ap.add_argument("--comision", type=float, default=d.comision_anual * 100,
                    help="comisión anual sobre saldo en %% (por defecto %(default)s)")
    ap.add_argument("--meta", type=float, default=d.meta_retiro,
                    help="meta de retiro en pesos de hoy (por defecto %(default).0f)")
    ap.add_argument("--semilla", type=int, default=d.semilla,
                    help="semilla aleatoria para reproducir resultados (por defecto %(default)s)")
    ap.add_argument("--simulaciones", type=_entero_min_simulaciones, default=d.n_simulaciones,
                    help=f"número de trayectorias, mínimo {MIN_SIMULACIONES_CLI:,} "
                         "(por defecto %(default)s)")
    ap.add_argument("--aportacion-al-final", action="store_true",
                    help="aportar al final de cada mes (por defecto se aporta al inicio)")
    ap.add_argument("--sin-indexar", action="store_true",
                    help="mantener la aportación nominal fija (por defecto sube con la inflación)")
    ap.add_argument("--grafica", default="resultados/grafica.png",
                    help="ruta del PNG de la gráfica (por defecto %(default)s)")
    ap.add_argument("--salida", default=None,
                    help="ruta opcional para guardar el reporte en texto")
    ap.add_argument("--mostrar", action="store_true",
                    help="además de guardar la gráfica, abrirla en una ventana")
    return ap


def parametros_desde_args(a: argparse.Namespace) -> Parametros:
    """Convierte los argumentos (porcentajes) a un objeto Parametros (decimales)."""
    return Parametros(
        saldo_inicial=a.saldo_inicial,
        aportacion_mensual=a.aportacion_mensual,
        anios=a.anios,
        rendimiento_anual=a.rendimiento / 100,
        volatilidad_anual=a.volatilidad / 100,
        inflacion_anual=a.inflacion / 100,
        comision_anual=a.comision / 100,
        meta_retiro=a.meta,
        semilla=a.semilla,
        n_simulaciones=a.simulaciones,
        aportacion_al_inicio=not a.aportacion_al_final,
        aportacion_indexada=not a.sin_indexar,
    )


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    try:
        p = parametros_desde_args(args)
        resultado = ejecutar(p)
    except ValueError as e:
        print(f"Error en los parámetros: {e}", file=sys.stderr)
        return 2

    texto = reporte_texto(resultado)
    print(texto)
    if args.salida:
        Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
        Path(args.salida).write_text(texto + "\n", encoding="utf-8")
        print(f"\nReporte guardado en: {args.salida}")
    if args.grafica:
        graficar(resultado, args.grafica, mostrar=args.mostrar)
        print(f"Gráfica guardada en: {args.grafica}")
    return 0


if __name__ == "__main__":
    # La consola de Windows a veces no usa UTF-8; esto evita errores con acentos.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
