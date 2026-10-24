"""
Pruebas automáticas de la simulación de Montecarlo.

Hay tres tipos de pruebas:
  1. Casos DETERMINISTAS: con volatilidad 0 la simulación no tiene azar, así que el
     resultado debe coincidir con las fórmulas cerradas de interés compuesto y de
     anualidades. Si coinciden, la lógica mes a mes (aportación, rendimiento,
     comisión, inflación) está bien implementada.
  2. REPRODUCIBILIDAD: la misma semilla debe dar exactamente los mismos números y
     semillas distintas deben dar resultados distintos.
  3. Validaciones y propiedades estadísticas (la media simulada se acerca al valor
     teórico, los percentiles están ordenados, etc.).

Ejecutar con:  python -m pytest -v
"""

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

# Permite importar el módulo desde la carpeta raíz del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import montecarlo_retiro as mc  # noqa: E402

# Parámetros base sin azar ni costos; cada prueba cambia solo lo que necesita.
BASE = mc.Parametros(
    saldo_inicial=100_000.0,
    aportacion_mensual=0.0,
    anios=10,
    rendimiento_anual=0.0,
    volatilidad_anual=0.0,
    inflacion_anual=0.0,
    comision_anual=0.0,
    meta_retiro=0.0,
    semilla=123,
    n_simulaciones=50,
    aportacion_al_inicio=True,
    aportacion_indexada=False,
)


def final(p: mc.Parametros) -> np.ndarray:
    """Saldos nominales finales de todas las trayectorias."""
    return mc.simular_trayectorias(p)[:, -1]


# ---------------------------------------------------------------------------
# 1. Conversión de tasas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mu, sigma", [(0.08, 0.15), (0.05, 0.0), (-0.02, 0.30)])
def test_conversion_lognormal_reproduce_rendimiento_anual_esperado(mu, sigma):
    m, s = mc.parametros_mensuales_lognormales(mu, sigma)
    assert s == pytest.approx(sigma / math.sqrt(12))
    # E[factor mensual] = exp(m + s^2/2); doce meses deben dar 1 + mu.
    assert math.exp(m + s ** 2 / 2) ** 12 == pytest.approx(1 + mu)


@pytest.mark.parametrize("r", [0.0, 0.04, 0.12])
def test_tasa_mensual_equivalente_compone_la_anual(r):
    assert (1 + mc.tasa_mensual_equivalente(r)) ** 12 == pytest.approx(1 + r)


# ---------------------------------------------------------------------------
# 2. Casos deterministas (volatilidad 0)
# ---------------------------------------------------------------------------
def test_sin_rendimiento_solo_suma_aportaciones():
    p = replace(BASE, aportacion_mensual=1_000.0)
    esperado = 100_000 + 1_000 * 120
    np.testing.assert_allclose(final(p), esperado)


def test_interes_compuesto_sin_aportaciones():
    p = replace(BASE, rendimiento_anual=0.07)
    np.testing.assert_allclose(final(p), 100_000 * 1.07 ** 10, rtol=1e-12)


def test_todas_las_trayectorias_son_iguales_sin_volatilidad():
    p = replace(BASE, rendimiento_anual=0.07, aportacion_mensual=500.0)
    trayectorias = mc.simular_trayectorias(p)
    assert np.all(trayectorias == trayectorias[0])


@pytest.mark.parametrize("al_inicio", [True, False])
def test_anualidad_coincide_con_formula_cerrada(al_inicio):
    r, A, S0, n = 0.06, 2_000.0, 100_000.0, 120
    i = mc.tasa_mensual_equivalente(r)
    # Valor futuro de una anualidad vencida (fin de mes): A * ((1+i)^n - 1) / i.
    # Anualidad anticipada (inicio de mes): la misma multiplicada por (1 + i).
    anualidad = A * ((1 + i) ** n - 1) / i
    if al_inicio:
        anualidad *= 1 + i
    esperado = S0 * (1 + i) ** n + anualidad
    p = replace(BASE, rendimiento_anual=r, aportacion_mensual=A, aportacion_al_inicio=al_inicio)
    np.testing.assert_allclose(final(p), esperado, rtol=1e-12)


def test_aportar_al_inicio_rinde_mas_que_al_final():
    p = replace(BASE, rendimiento_anual=0.06, aportacion_mensual=2_000.0)
    assert final(p)[0] > final(replace(p, aportacion_al_inicio=False))[0]


def test_comision_reduce_el_saldo_como_factor_anual():
    p = replace(BASE, rendimiento_anual=0.08, comision_anual=0.01)
    esperado = 100_000 * (1.08 * 0.99) ** 10
    np.testing.assert_allclose(final(p), esperado, rtol=1e-12)


def test_pesos_de_hoy_descuentan_la_inflacion():
    # Si el rendimiento nominal es igual a la inflación, el poder de compra no cambia.
    p = replace(BASE, rendimiento_anual=0.04, inflacion_anual=0.04)
    nominal = mc.simular_trayectorias(p)
    real = mc.a_pesos_de_hoy(nominal, p.inflacion_anual)
    np.testing.assert_allclose(nominal[:, -1], 100_000 * 1.04 ** 10, rtol=1e-12)
    np.testing.assert_allclose(real, 100_000, rtol=1e-12)


def test_aportacion_indexada_conserva_su_valor_real():
    # Rendimiento = inflación, aportación indexada al inicio de mes: en pesos de hoy
    # cada aportación vale A, así que el saldo real final es S0 + A * n.
    p = replace(BASE, rendimiento_anual=0.05, inflacion_anual=0.05,
                aportacion_mensual=1_500.0, aportacion_indexada=True)
    real = mc.a_pesos_de_hoy(mc.simular_trayectorias(p), p.inflacion_anual)
    np.testing.assert_allclose(real[:, -1], 100_000 + 1_500 * 120, rtol=1e-10)


def test_total_aportado_nominal_y_real():
    p = replace(BASE, aportacion_mensual=1_000.0, inflacion_anual=0.04)
    nominal, real = mc.total_aportado(p)
    assert nominal == pytest.approx(100_000 + 1_000 * 120)
    assert real < nominal                     # aportación fija pierde valor real
    nominal_idx, real_idx = mc.total_aportado(replace(p, aportacion_indexada=True))
    assert real_idx == pytest.approx(100_000 + 1_000 * 120)
    assert nominal_idx > real_idx


def test_escenario_determinista_en_resultado():
    p = replace(BASE, rendimiento_anual=0.07, volatilidad_anual=0.2, n_simulaciones=200)
    r = mc.ejecutar(p)
    assert r.determinista_real == pytest.approx(100_000 * 1.07 ** 10)


# ---------------------------------------------------------------------------
# 3. Reproducibilidad
# ---------------------------------------------------------------------------
def test_misma_semilla_mismos_resultados():
    p = replace(BASE, rendimiento_anual=0.08, volatilidad_anual=0.15,
                aportacion_mensual=1_000.0, n_simulaciones=10_000, semilla=2026)
    a = mc.simular_trayectorias(p)
    b = mc.simular_trayectorias(p)
    np.testing.assert_array_equal(a, b)
    assert mc.resumir(a[:, -1], 300_000) == mc.resumir(b[:, -1], 300_000)


def test_semillas_distintas_dan_resultados_distintos():
    p = replace(BASE, rendimiento_anual=0.08, volatilidad_anual=0.15, n_simulaciones=1_000)
    a = final(replace(p, semilla=1))
    b = final(replace(p, semilla=2))
    assert not np.array_equal(a, b)


def test_reporte_reproducible_con_parametros_por_defecto():
    p = mc.Parametros()
    assert p.n_simulaciones >= 10_000
    assert mc.reporte_texto(mc.ejecutar(p)) == mc.reporte_texto(mc.ejecutar(p))


def test_semilla_none_funciona():
    p = replace(BASE, volatilidad_anual=0.1, semilla=None)
    assert final(p).shape == (50,)


# ---------------------------------------------------------------------------
# 4. Propiedades estadísticas y forma de los resultados
# ---------------------------------------------------------------------------
def test_forma_de_la_matriz_y_saldo_inicial():
    p = replace(BASE, volatilidad_anual=0.15, n_simulaciones=10_000, anios=5)
    t = mc.simular_trayectorias(p)
    assert t.shape == (10_000, 61)
    assert np.all(t[:, 0] == 100_000)
    assert np.all(t > 0)                      # la log-normal nunca da saldo negativo


def test_media_simulada_cercana_al_valor_teorico():
    # Sin aportaciones ni comisiones, E[saldo final] = S0 * (1 + mu)^anios.
    p = replace(BASE, rendimiento_anual=0.08, volatilidad_anual=0.15,
                n_simulaciones=20_000, semilla=7)
    v = final(p)
    teorico = 100_000 * 1.08 ** 10
    error_estandar = v.std(ddof=1) / math.sqrt(v.size)
    assert abs(v.mean() - teorico) < 4 * error_estandar


def test_percentiles_ordenados_y_media_mayor_que_mediana():
    p = replace(BASE, rendimiento_anual=0.08, volatilidad_anual=0.2, n_simulaciones=10_000)
    res = mc.resumir(final(p))
    assert res["p5"] < res["p25"] < res["p50"] < res["p75"] < res["p95"]
    assert res["media"] > res["p50"]          # asimetría positiva de la log-normal


def test_resumir_con_valores_conocidos():
    v = np.arange(1, 101, dtype=float)        # 1, 2, ..., 100
    res = mc.resumir(v, meta=91)
    assert res["media"] == pytest.approx(50.5)
    assert res["p50"] == pytest.approx(50.5)
    assert res["p5"] == pytest.approx(5.95)   # interpolación lineal de NumPy
    assert res["probabilidad_meta"] == pytest.approx(0.10)   # 91..100


def test_probabilidad_meta_extremos():
    p = replace(BASE, rendimiento_anual=0.08, volatilidad_anual=0.15, n_simulaciones=10_000)
    r = mc.ejecutar(replace(p, meta_retiro=0.0))
    assert r.resumen_real["probabilidad_meta"] == 1.0
    r = mc.ejecutar(replace(p, meta_retiro=1e15))
    assert r.resumen_real["probabilidad_meta"] == 0.0


def test_probabilidad_real_y_nominal_coinciden():
    # Comparar en pesos de hoy contra la meta es lo mismo que comparar en nominal
    # contra la meta inflada al año de retiro.
    p = replace(BASE, rendimiento_anual=0.08, volatilidad_anual=0.15, inflacion_anual=0.04,
                aportacion_mensual=1_000.0, meta_retiro=200_000.0, n_simulaciones=10_000)
    r = mc.ejecutar(p)
    assert r.resumen_real["probabilidad_meta"] == pytest.approx(
        r.resumen_nominal["probabilidad_meta"])


# ---------------------------------------------------------------------------
# 5. Validaciones y línea de comandos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cambio", [
    {"saldo_inicial": -1}, {"aportacion_mensual": -1}, {"anios": 0}, {"anios": 2.5},
    {"rendimiento_anual": -1}, {"volatilidad_anual": -0.1}, {"inflacion_anual": -1},
    {"comision_anual": 1}, {"meta_retiro": -1}, {"n_simulaciones": 0},
])
def test_parametros_invalidos(cambio):
    with pytest.raises(ValueError):
        mc.simular_trayectorias(replace(BASE, **cambio))


def test_cli_exige_al_menos_10000_trayectorias(capsys):
    with pytest.raises(SystemExit):
        mc.main(["--simulaciones", "500"])


def test_cli_convierte_porcentajes_y_guarda_archivos(tmp_path):
    salida, grafica = tmp_path / "salida.txt", tmp_path / "grafica.png"
    codigo = mc.main(["--rendimiento", "6", "--volatilidad", "12", "--inflacion", "3.5",
                      "--comision", "0.5", "--anios", "5", "--semilla", "1",
                      "--salida", str(salida), "--grafica", str(grafica)])
    assert codigo == 0
    texto = salida.read_text(encoding="utf-8")
    assert "6.00 % nominal" in texto and "Probabilidad de alcanzar la meta" in texto
    assert grafica.stat().st_size > 10_000


def test_cli_reporta_parametros_invalidos(capsys):
    assert mc.main(["--anios", "0", "--grafica", ""]) == 2
    assert "Error en los parámetros" in capsys.readouterr().err
