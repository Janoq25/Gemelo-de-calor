"""L4 · Motor de escenarios (§9 del plan).

"Un escenario es una funcion que transforma el estado del gemelo, no un
conjunto de datos aparte."

    def escenario(estado: EstadoGemelo, **params) -> EstadoGemelo: ...

De esa firma uniforme sale lo que hace de L4 un motor y no una coleccion de
scripts: los escenarios se **componen** (`calor_7d | apagon`). Cada
composicion arrastra su linaje, asi que un contrafactual siempre puede
explicar como se construyo.

Regla de disciplina que el plan exige (§15.6): los escenarios son
contrafactuales *del modelo*, no del mundo. Este modulo perturba el estado;
nunca afirma causalidad.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from xdt.hashing import hash_json
from xdt.twin.state import EstadoGemelo

TransformFn = Callable[[EstadoGemelo], EstadoGemelo]


@dataclass(frozen=True)
class Scenario:
    """Escenario nombrado y componible."""

    name: str
    fn: TransformFn
    spec: dict[str, Any] = field(default_factory=dict)

    def __call__(self, estado: EstadoGemelo) -> EstadoGemelo:
        return self.fn(estado)

    def __or__(self, other: Scenario) -> Scenario:
        """`a | b` aplica primero `a`, despues `b`."""
        return compose(self, other)

    def scenario_id(self) -> str:
        return hash_json({"name": self.name, "spec": self.spec})[:32]

    def spec_json(self) -> str:
        return json.dumps({"name": self.name, "spec": self.spec}, sort_keys=True, default=str)


def compose(*scenarios: Scenario) -> Scenario:
    """Encadena escenarios en orden de aplicacion."""
    if not scenarios:
        raise ValueError("compose() requiere al menos un escenario")

    def _apply(estado: EstadoGemelo) -> EstadoGemelo:
        for s in scenarios:
            estado = s(estado)
        return estado

    return Scenario(
        name=" | ".join(s.name for s in scenarios),
        fn=_apply,
        spec={"compose": [{"name": s.name, "spec": s.spec} for s in scenarios]},
    )


def baseline() -> Scenario:
    """Escenario identidad. Util como control explicito en comparaciones."""
    return Scenario(name="baseline", fn=lambda e: e, spec={})


# --------------------------------------------------------------------------
# 9.1 · Olas de calor sinteticas
# --------------------------------------------------------------------------
def ola_de_calor(
    *,
    dias: int,
    delta_c: float | None = None,
    percentil_local: float | None = None,
    temporada: str = "late",
    var_tmax: str = "tmmx",
    var_tmin: str = "tmmn",
    delta_tmin_c: float | None = None,
) -> Scenario:
    """Inyecta una ola de calor de `dias` dias.

    Dos formas de fijar la intensidad, excluyentes:
      - `delta_c`: sumar un incremento fijo en grados sobre la serie observada.
      - `percentil_local`: elevar tmax hasta el percentil local de cada
        geografia. Es la variante que pide §9.1, y la correcta: el umbral de
        dano es relativo a la aclimatacion regional, no absoluto.

    `temporada='early'` marca el escenario de poblacion no aclimatada. Aqui
    solo se anota como bandera en el estado; el efecto lo estima el modelo a
    partir de la feature de "primera ola de la temporada" (§7 M2). Este modulo
    no inventa multiplicadores.
    """
    if (delta_c is None) == (percentil_local is None):
        raise ValueError("especifica exactamente uno: delta_c o percentil_local")
    if dias < 1:
        raise ValueError("dias debe ser >= 1")

    spec = {
        "dias": dias, "delta_c": delta_c, "percentil_local": percentil_local,
        "temporada": temporada, "var_tmax": var_tmax,
    }

    def _apply(estado: EstadoGemelo) -> EstadoGemelo:
        estado.require(var_tmax)
        df = estado.frame.copy()

        # Los `dias` finales de cada geografia son los perturbados: la ventana
        # de retardos previa se conserva intacta para que el encoder temporal
        # vea una ola que *empieza*, no un nivel plano.
        tail = df.groupby(level="geoid").tail(dias).index

        if delta_c is not None:
            df.loc[tail, var_tmax] = df.loc[tail, var_tmax] + delta_c
            if var_tmin in df.columns:
                df.loc[tail, var_tmin] = df.loc[tail, var_tmin] + (delta_tmin_c or delta_c)
        else:
            target = df.groupby(level="geoid")[var_tmax].quantile(percentil_local / 100.0)
            tail_geoids = tail.get_level_values("geoid")
            new_vals = np.maximum(
                df.loc[tail, var_tmax].to_numpy(), target.reindex(tail_geoids).to_numpy()
            )
            df.loc[tail, var_tmax] = new_vals
            if var_tmin in df.columns and delta_tmin_c is not None:
                df.loc[tail, var_tmin] = df.loc[tail, var_tmin] + delta_tmin_c

        df["scn_heatwave_days"] = 0.0
        df.loc[tail, "scn_heatwave_days"] = float(dias)
        df["scn_early_season"] = 1.0 if temporada == "early" else 0.0

        note = (
            f"ola_de_calor(dias={dias}, "
            + (f"delta_c={delta_c}" if delta_c is not None else f"p{percentil_local}")
            + f", temporada={temporada})"
        )
        return estado.derive(df, note, scenario_spec=spec)

    return Scenario(name="ola_de_calor", fn=_apply, spec=spec)


# --------------------------------------------------------------------------
# 9.2 · Evento compuesto calor + apagon  — componente central del trabajo
# --------------------------------------------------------------------------
def apagon(
    *,
    fraccion_clientes: float,
    horas: float,
    dias: int = 1,
    var_frac: str = "outage_customers_frac",
    var_horas: str = "outage_max_hours",
) -> Scenario:
    """Inyecta un corte de suministro sobre los ultimos `dias` dias.

    Para simulacion realista, `desde_historico_eaglei()` parametriza estos
    valores con la distribucion empirica del propio condado (§9.2, doble uso
    de EAGLE-I) en lugar de con numeros elegidos a mano.
    """
    if not 0.0 <= fraccion_clientes <= 1.0:
        raise ValueError("fraccion_clientes debe estar en [0, 1]")

    spec = {"fraccion_clientes": fraccion_clientes, "horas": horas, "dias": dias}

    def _apply(estado: EstadoGemelo) -> EstadoGemelo:
        df = estado.frame.copy()
        for col, default in ((var_frac, 0.0), (var_horas, 0.0)):
            if col not in df.columns:
                df[col] = default

        tail = df.groupby(level="geoid").tail(dias).index
        df.loc[tail, var_frac] = np.maximum(df.loc[tail, var_frac].to_numpy(), fraccion_clientes)
        df.loc[tail, var_horas] = np.maximum(df.loc[tail, var_horas].to_numpy(), horas)

        return estado.derive(
            df,
            f"apagon(frac={fraccion_clientes:.3f}, horas={horas}, dias={dias})",
            scenario_spec=spec,
        )

    return Scenario(name="apagon", fn=_apply, spec=spec)


def desde_historico_eaglei(
    historico: pd.DataFrame,
    *,
    percentil: float = 90.0,
    dias: int = 1,
    col_geoid: str = "geoid",
    col_frac: str = "outage_customers_frac",
    col_horas: str = "outage_max_hours",
) -> Scenario:
    """Apagon parametrizado con la distribucion empirica observada por geografia.

    §9.2: "los escenarios son realistas porque provienen de la historia del
    propio condado". Un corte del percentil 90 en Phoenix no es el mismo
    evento que en Boston, y usar una constante nacional borraria justo la
    heterogeneidad que el trabajo quiere caracterizar.
    """
    q = percentil / 100.0
    agg = historico.groupby(col_geoid).agg(
        frac=(col_frac, lambda s: s.quantile(q)),
        horas=(col_horas, lambda s: s.quantile(q)),
    )
    spec = {"percentil": percentil, "dias": dias, "n_geoids": int(len(agg))}

    def _apply(estado: EstadoGemelo) -> EstadoGemelo:
        df = estado.frame.copy()
        for col in (col_frac, col_horas):
            if col not in df.columns:
                df[col] = 0.0

        tail = df.groupby(level="geoid").tail(dias).index
        geoids = tail.get_level_values("geoid")
        # Geografias sin historico quedan sin perturbar (0.0), nunca imputadas
        # con una media nacional que inventaria un evento que no ocurrio.
        f = agg["frac"].reindex(geoids).fillna(0.0).to_numpy()
        h = agg["horas"].reindex(geoids).fillna(0.0).to_numpy()
        df.loc[tail, col_frac] = np.maximum(df.loc[tail, col_frac].to_numpy(), f)
        df.loc[tail, col_horas] = np.maximum(df.loc[tail, col_horas].to_numpy(), h)

        return estado.derive(
            df, f"apagon_empirico(p{percentil}, dias={dias})", scenario_spec=spec
        )

    return Scenario(name="apagon_empirico", fn=_apply, spec=spec)


# --------------------------------------------------------------------------
# 9.3 · Intervencion de enfriamiento comunitario
# --------------------------------------------------------------------------
def enfriamiento(
    *,
    epsilon: float,
    var_tmax: str = "tmmx",
    var_marca: str = "scn_cooling_epsilon",
) -> Scenario:
    """Reduce la exposicion efectiva en una fraccion `epsilon`.

    ATENCION — §9.3 del plan. El efecto es **asumido, no estimado**. No existe
    registro nacional de centros de enfriamiento y el efecto causal de abrir
    uno no es identificable con estos datos. El uso legitimo de este escenario
    es el analisis de umbral: barrer epsilon en [0.05, 0.40] y preguntar que
    reduccion minima haria falta para mover el riesgo de forma detectable.

    Nunca reportar como "abrir N centros evita M hospitalizaciones".
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon debe estar en [0, 1]")

    spec = {"epsilon": epsilon, "efecto": "asumido_no_estimado"}

    def _apply(estado: EstadoGemelo) -> EstadoGemelo:
        estado.require(var_tmax)
        df = estado.frame.copy()

        # La reduccion se aplica sobre la *anomalia* respecto a la mediana
        # local, no sobre la temperatura absoluta: un centro de enfriamiento
        # recorta el exceso de calor, no reescala el clima base.
        base = df.groupby(level="geoid")[var_tmax].transform("median")
        anomalia = df[var_tmax] - base
        df[var_tmax] = base + anomalia * (1.0 - epsilon)
        df[var_marca] = epsilon

        return estado.derive(
            df, f"enfriamiento(epsilon={epsilon:.3f}) [efecto asumido]", scenario_spec=spec
        )

    return Scenario(name="enfriamiento", fn=_apply, spec=spec)


def barrido_epsilon(
    valores: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40),
) -> list[Scenario]:
    """Barrido de sensibilidad sobre epsilon, tal como lo pide §9.3."""
    return [enfriamiento(epsilon=e) for e in valores]
