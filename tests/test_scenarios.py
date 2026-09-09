"""Pruebas de L4: inmutabilidad, linaje y composicion."""

from __future__ import annotations

import pandas as pd
import pytest

from xdt.twin.scenarios import (
    apagon,
    baseline,
    compose,
    desde_historico_eaglei,
    enfriamiento,
    ola_de_calor,
)
from xdt.twin.state import StateStore


@pytest.fixture
def estado(con, panel_state, t0):
    StateStore(con).put_facts(panel_state, source="gridmet", known_at=t0)
    return StateStore(con).snapshot(
        geo_level="state", valid_from="2023-07-01", valid_to="2023-07-10"
    )


def test_escenario_no_muta_el_estado_original(estado):
    """La firma de §9 devuelve un estado nuevo. Si mutara, componer escenarios
    contaminaria el baseline y las comparaciones dejarian de ser validas."""
    antes = estado.frame.copy(deep=True)
    nuevo = ola_de_calor(dias=3, delta_c=5.0)(estado)

    pd.testing.assert_frame_equal(estado.frame, antes)
    assert nuevo is not estado
    assert not nuevo.frame.equals(estado.frame)


def test_ola_de_calor_solo_perturba_la_cola(estado):
    """La ventana de retardos previa debe quedar intacta: el encoder temporal
    tiene que ver una ola que empieza, no un nivel plano."""
    nuevo = ola_de_calor(dias=3, delta_c=5.0)(estado)

    az_antes = estado.frame.loc["04", "tmmx"]
    az_despues = nuevo.frame.loc["04", "tmmx"]

    assert (az_despues.iloc[:-3] == az_antes.iloc[:-3]).all()
    assert (az_despues.iloc[-3:] == az_antes.iloc[-3:] + 5.0).all()
    assert nuevo.frame["scn_heatwave_days"].iloc[-1] == 3.0


def test_ola_por_percentil_local_respeta_la_aclimatacion(estado):
    """El umbral de dano es relativo al clima local (§7 M2). Un percentil 95
    debe elevar cada geografia a *su* propio p95, no a un valor nacional."""
    nuevo = ola_de_calor(dias=2, percentil_local=95.0)(estado)

    for geoid in ("04", "36"):
        p95 = estado.frame.loc[geoid, "tmmx"].quantile(0.95)
        assert nuevo.frame.loc[geoid, "tmmx"].iloc[-2:].min() >= p95 - 1e-9

    # Arizona parte de una base mas alta y debe seguir por encima de NY.
    assert (
        nuevo.frame.loc["04", "tmmx"].iloc[-1] > nuevo.frame.loc["36", "tmmx"].iloc[-1]
    )


def test_ola_exige_exactamente_un_modo_de_intensidad():
    with pytest.raises(ValueError, match="exactamente uno"):
        ola_de_calor(dias=3, delta_c=5.0, percentil_local=95.0)
    with pytest.raises(ValueError, match="exactamente uno"):
        ola_de_calor(dias=3, delta_c=None, percentil_local=None)


def test_apagon_crea_las_columnas_si_no_existen(estado):
    nuevo = apagon(fraccion_clientes=0.35, horas=12.0, dias=2)(estado)
    assert nuevo.frame["outage_customers_frac"].iloc[-1] == pytest.approx(0.35)
    assert nuevo.frame["outage_max_hours"].iloc[-1] == pytest.approx(12.0)
    # Los dias previos quedan sin corte.
    assert nuevo.frame.loc["04", "outage_customers_frac"].iloc[0] == 0.0


def test_apagon_valida_la_fraccion():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apagon(fraccion_clientes=1.5, horas=4.0)


def test_composicion_es_el_evento_compuesto(estado):
    """§9.2, el componente central: calor y apagon aplicados juntos."""
    compuesto = ola_de_calor(dias=3, delta_c=6.0) | apagon(
        fraccion_clientes=0.25, horas=18.0, dias=3
    )
    nuevo = compuesto(estado)

    assert nuevo.frame["tmmx"].iloc[-1] == estado.frame["tmmx"].iloc[-1] + 6.0
    assert nuevo.frame["outage_customers_frac"].iloc[-1] == pytest.approx(0.25)

    # El linaje registra ambas transformaciones, en orden.
    assert len(nuevo.lineage) == 2
    assert "ola_de_calor" in nuevo.lineage[0]
    assert "apagon" in nuevo.lineage[1]
    assert not nuevo.is_baseline


def test_composicion_es_asociativa_en_el_resultado(estado):
    a = ola_de_calor(dias=2, delta_c=3.0)
    b = apagon(fraccion_clientes=0.1, horas=6.0, dias=2)
    c = enfriamiento(epsilon=0.1)

    r1 = compose(compose(a, b), c)(estado)
    r2 = compose(a, compose(b, c))(estado)
    pd.testing.assert_frame_equal(r1.frame, r2.frame)


def test_scenario_id_es_determinista_y_sensible_a_los_parametros():
    s1 = ola_de_calor(dias=3, delta_c=5.0)
    s2 = ola_de_calor(dias=3, delta_c=5.0)
    s3 = ola_de_calor(dias=7, delta_c=5.0)

    assert s1.scenario_id() == s2.scenario_id()
    assert s1.scenario_id() != s3.scenario_id()


def test_baseline_es_identidad(estado):
    igual = baseline()(estado)
    pd.testing.assert_frame_equal(igual.frame, estado.frame)


def test_enfriamiento_recorta_la_anomalia_no_el_clima_base(estado):
    """Un centro de enfriamiento recorta el exceso de calor; no reescala el
    clima base. Con epsilon=1 la serie debe colapsar a la mediana local."""
    nuevo = enfriamiento(epsilon=1.0)(estado)

    for geoid in ("04", "36"):
        mediana = estado.frame.loc[geoid, "tmmx"].median()
        assert nuevo.frame.loc[geoid, "tmmx"].std() == pytest.approx(0.0, abs=1e-9)
        assert nuevo.frame.loc[geoid, "tmmx"].iloc[0] == pytest.approx(mediana)


def test_enfriamiento_marca_el_efecto_como_asumido():
    """§9.3: el efecto es asumido, no estimado. La spec debe decirlo, para que
    quede en el registro de procedencia de todo contrafactual que lo use."""
    s = enfriamiento(epsilon=0.2)
    assert s.spec["efecto"] == "asumido_no_estimado"


def test_apagon_empirico_usa_la_distribucion_de_cada_geografia(estado):
    """§9.2: un corte del p90 en Arizona no es el mismo evento que en NY."""
    hist = pd.DataFrame(
        [{"geoid": "04", "outage_customers_frac": f, "outage_max_hours": f * 100}
         for f in (0.05, 0.10, 0.40)]
        + [{"geoid": "36", "outage_customers_frac": f, "outage_max_hours": f * 100}
           for f in (0.01, 0.02, 0.03)]
    )
    nuevo = desde_historico_eaglei(hist, percentil=90.0, dias=1)(estado)

    az = nuevo.frame.loc["04", "outage_customers_frac"].iloc[-1]
    ny = nuevo.frame.loc["36", "outage_customers_frac"].iloc[-1]
    assert az > ny
    assert az == pytest.approx(0.34, abs=0.02)


def test_apagon_empirico_no_imputa_geografias_sin_historico(estado):
    """Sin historico, no hay evento. Imputar una media nacional inventaria un
    corte que nunca ocurrio en esa geografia."""
    hist = pd.DataFrame(
        [{"geoid": "04", "outage_customers_frac": 0.4, "outage_max_hours": 40.0}]
    )
    nuevo = desde_historico_eaglei(hist, dias=1)(estado)
    assert nuevo.frame.loc["36", "outage_customers_frac"].iloc[-1] == 0.0


def test_require_falla_con_mensaje_util(estado):
    with pytest.raises(KeyError, match="no_existe"):
        estado.require("no_existe")
