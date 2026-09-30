"""M2 · Tabla canonica estado-dia para la Etapa A (§5.1).

Une el desenlace diario de EPHT (medida 1385) con la exposicion termica de
gridMET agregada al estado, y deriva las caracteristicas del modelo.

Tres decisiones declaradas, que el reporte CRISP-DM repite:

1. **Exposicion ponderada por poblacion sobre los condados descargados.**
   Para CONUS completo se descargan los condados mas poblados hasta cubrir
   ~80 % de la poblacion de cada estado (`xdt ingest gridmet --pop-coverage`).
   El promedio se renormaliza sobre esos condados y la cobertura queda en la
   columna `pop_coverage` para que ningun estado con cobertura pobre entre
   sin que se vea.
2. **Temporada calida (mayo-septiembre).** Fuera de ella el desenlace es casi
   siempre cero y un modelo aprenderia "verano vs invierno", lo que infla el
   AUC sin medir riesgo por calor.
3. **Umbrales locales sin fuga.** El p95 de cada estado sale de un periodo de
   referencia (`ref_years`) que nunca se usa como conjunto de prueba.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from xdt.features.thermal import (
    above_threshold,
    first_heatwave_of_season,
    heat_index_c,
    local_percentile,
    streak,
)
from xdt.harmonize.centroids import NON_CONUS_STATES, parse_county_centroids
from xdt.twin.state import now_utc

#: Desenlace: tasa diaria de urgencias por HRI, poblacion VA (medida 1385).
OUTCOME_VAR = "epht_m1_rate"
WARM_MONTHS = (5, 6, 7, 8, 9)
GRIDMET_VARS = ("tmmx", "tmmn", "rmax", "rmin")

# Regiones HHS (10). Agrupan estados para la validacion cruzada espacial.
HHS_REGION: dict[str, int] = {
    **dict.fromkeys(["09", "23", "25", "33", "44", "50"], 1),
    **dict.fromkeys(["34", "36"], 2),
    **dict.fromkeys(["10", "11", "24", "42", "51", "54"], 3),
    **dict.fromkeys(["01", "12", "13", "21", "28", "37", "45", "47"], 4),
    **dict.fromkeys(["17", "18", "26", "27", "39", "55"], 5),
    **dict.fromkeys(["05", "22", "35", "40", "48"], 6),
    **dict.fromkeys(["19", "20", "29", "31"], 7),
    **dict.fromkeys(["08", "30", "38", "46", "49", "56"], 8),
    **dict.fromkeys(["04", "06", "32"], 9),
    **dict.fromkeys(["16", "41", "53"], 10),
}

#: Caracteristicas del modelo, con su descripcion para el reporte.
FEATURES: dict[str, str] = {
    "heat_index": "Indice de calor maximo del dia (°C, NWS con tmax y HR minima)",
    "tmax": "Temperatura maxima (°C)",
    "tmin": "Temperatura minima (°C); calor nocturno sin alivio",
    "rh_min": "Humedad relativa minima (%)",
    "hi_excess_p95": "Indice de calor menos el p95 local del estado (°C)",
    "hi_mean_3d": "Media movil de 3 dias del indice de calor (°C)",
    "hi_lag1": "Indice de calor del dia anterior (°C)",
    "tmin_mean_3d": "Media movil de 3 dias de la minima (°C)",
    "hot_streak": "Dias consecutivos sobre el p95 local, hasta hoy",
    "first_heatwave": "1 si el dia pertenece a la primera ola de la temporada",
    "hi_clim_ref": "Indice de calor medio de verano del estado en la referencia (aclimatacion)",
    "doy_sin": "Estacionalidad: seno del dia del ano",
    "doy_cos": "Estacionalidad: coseno del dia del ano",
    "weekend": "1 si es sabado o domingo",
}


@dataclass(frozen=True)
class Panel:
    frame: pd.DataFrame  # una fila por estado-dia de temporada calida
    ref_years: tuple[int, ...]
    known_at: pd.Timestamp

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802 - notacion estandar de ML
        return self.frame[list(FEATURES)]

    @property
    def y(self) -> pd.Series:
        return self.frame["y"]


def _county_weights(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Poblacion por condado desde el artefacto de centroides registrado en L0."""
    from xdt.harmonize.centroids import fetch_county_centroids
    from xdt.ingest.http import CachedClient

    client = CachedClient(source="census")
    try:
        cen = parse_county_centroids(fetch_county_centroids(client))
    finally:
        client.close()
    cen = cen[~cen["state_fips"].isin(NON_CONUS_STATES)]
    return cen[["geoid", "state_fips", "population", "lat", "lon"]]


def state_exposure(
    con: duckdb.DuckDBPyConnection,
    *,
    start: str,
    end: str,
    known_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """gridMET condado -> estado, ponderado por poblacion y renormalizado.

    Devuelve una fila por (estado, dia) con tmmx, tmmn, rmax, rmin y
    `pop_coverage` (fraccion de la poblacion del estado con dato ese dia).
    Respeta el corte bitemporal `known_at`, como el resto del gemelo.
    """
    weights = _county_weights(con)
    cutoff = (known_at or pd.Timestamp(now_utc())).to_pydatetime()
    con.register("_w", weights)
    try:
        df = con.execute(
            f"""
            WITH facts AS (
                SELECT geoid, valid_date, variable, value FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY fact_key ORDER BY revision DESC) AS rn
                    FROM twin_state
                    WHERE geo_level = 'county' AND source = 'gridmet'
                      AND variable IN ({",".join(f"'{v}'" for v in GRIDMET_VARS)})
                      AND valid_date BETWEEN ? AND ? AND known_at <= ?
                ) WHERE rn = 1 AND value IS NOT NULL
            ),
            tot AS (SELECT state_fips, sum(population) AS pop_total FROM _w GROUP BY 1)
            SELECT w.state_fips AS geoid, f.valid_date, f.variable,
                   sum(f.value * w.population) / sum(w.population) AS value,
                   sum(w.population) / any_value(t.pop_total) AS pop_coverage
            FROM facts f
            JOIN _w w USING (geoid)
            JOIN tot t ON t.state_fips = w.state_fips
            GROUP BY 1, 2, 3
            """,
            [start, end, cutoff],
        ).df()
    finally:
        con.unregister("_w")
    if df.empty:
        raise ValueError("no hay gridMET por condado en la ventana pedida")
    wide = df.pivot_table(index=["geoid", "valid_date"], columns="variable",
                          values="value", aggfunc="first")
    cov = df.groupby(["geoid", "valid_date"])["pop_coverage"].min()
    wide["pop_coverage"] = cov
    wide.columns.name = None
    wide.index = wide.index.set_levels(pd.to_datetime(wide.index.levels[1]), level=1)
    return wide.sort_index()


def outcome(
    con: duckdb.DuckDBPyConnection, *, start: str, end: str,
    known_at: pd.Timestamp | None = None,
) -> pd.Series:
    cutoff = (known_at or pd.Timestamp(now_utc())).to_pydatetime()
    df = con.execute(
        """
        SELECT geoid, valid_date, value FROM (
            SELECT *, row_number() OVER (PARTITION BY fact_key ORDER BY revision DESC) rn
            FROM twin_state
            WHERE geo_level = 'state' AND source = 'epht' AND variable = ?
              AND valid_date BETWEEN ? AND ? AND known_at <= ?
        ) WHERE rn = 1
        """,
        [OUTCOME_VAR, start, end, cutoff],
    ).df()
    if df.empty:
        raise ValueError(f"sin desenlace {OUTCOME_VAR}: ingesta la medida 1385 primero")
    df["valid_date"] = pd.to_datetime(df["valid_date"])
    return df.set_index(["geoid", "valid_date"])["value"].sort_index()


def build_panel(
    con: duckdb.DuckDBPyConnection,
    *,
    years: Sequence[int],
    ref_years: Sequence[int],
    min_pop_coverage: float = 0.7,
    known_at: pd.Timestamp | None = None,
) -> Panel:
    """Panel estado-dia listo para modelar.

    `ref_years` define los umbrales locales (p95) y la climatologia; esos anos
    entran al entrenamiento pero nunca a una particion de prueba.
    """
    years = sorted(set(years) | set(ref_years))
    start, end = f"{years[0]}-01-01", f"{years[-1]}-12-31"
    expo = state_exposure(con, start=start, end=end, known_at=known_at)
    expo = expo[expo["pop_coverage"] >= min_pop_coverage]

    f = pd.DataFrame(index=expo.index)
    f["tmax"] = expo["tmmx"]
    f["tmin"] = expo["tmmn"]
    f["rh_min"] = expo["rmin"]
    f["heat_index"] = heat_index_c(expo["tmmx"], expo["rmin"])
    f["pop_coverage"] = expo["pop_coverage"]

    ref0, ref1 = f"{min(ref_years)}-01-01", f"{max(ref_years)}-12-31"
    p95 = local_percentile(f, "heat_index", 95.0, ref_start=ref0, ref_end=ref1)
    f["hi_excess_p95"] = f["heat_index"] - p95.reindex(
        f.index.get_level_values("geoid")).to_numpy()
    hot = above_threshold(f, "heat_index", p95)
    f["hot_streak"] = streak(hot.rename("hot"))
    f["first_heatwave"] = first_heatwave_of_season(f["hot_streak"])

    # Retardos y medias moviles: solo pasado y presente, nunca futuro.
    g = f.groupby(level="geoid")
    f["hi_lag1"] = g["heat_index"].shift(1)
    f["hi_mean_3d"] = g["heat_index"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    f["tmin_mean_3d"] = g["tmin"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    dates = f.index.get_level_values("valid_date")
    ref_mask = (dates >= pd.Timestamp(ref0)) & (dates <= pd.Timestamp(ref1)) & \
        dates.month.isin(WARM_MONTHS)
    clim = f.loc[ref_mask, "heat_index"].groupby(level="geoid").mean()
    f["hi_clim_ref"] = clim.reindex(f.index.get_level_values("geoid")).to_numpy()

    doy = dates.dayofyear.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["weekend"] = (dates.dayofweek >= 5).astype(float)

    y = outcome(con, start=start, end=end, known_at=known_at)
    f = f.join(y.rename("rate"), how="inner")
    f = f[f.index.get_level_values("valid_date").month.isin(WARM_MONTHS)]
    f = f.dropna(subset=list(FEATURES))

    geoid = f.index.get_level_values("geoid")
    f["y"] = (f["rate"] > 0).astype(int)
    f["year"] = f.index.get_level_values("valid_date").year
    f["hhs_region"] = geoid.map(HHS_REGION).astype("Int64")
    f["week"] = f.index.get_level_values("valid_date").isocalendar().week.to_numpy()
    f = f[f["hhs_region"].notna()]
    return Panel(frame=f, ref_years=tuple(sorted(ref_years)),
                 known_at=known_at or pd.Timestamp(now_utc()))


def with_warming(panel: Panel, delta_c: float) -> pd.DataFrame:
    """Variables del modelo si cada dia hubiera sido `delta_c` °C mas caluroso.

    Escenario de L4 traducido al espacio de caracteristicas: suma `delta_c` a
    la maxima y la minima, recalcula el indice de calor con la MISMA humedad
    relativa minima (formula NWS, no lineal) y propaga el cambio a las
    variables derivadas. Los umbrales locales (p95) y la climatologia de
    referencia NO cambian: el escenario pregunta que pasa si este verano es
    mas caluroso que el clima al que la poblacion esta acostumbrada.

    Aproximacion declarada: en el primer dia de la temporada, el retardo usa el
    cambio del mismo dia (el panel no guarda abril).
    """
    f = panel.X.copy()
    if delta_c == 0:
        return f
    f["tmax"] = f["tmax"] + delta_c
    f["tmin"] = f["tmin"] + delta_c
    hi_new = heat_index_c(f["tmax"], f["rh_min"])
    dhi = hi_new - f["heat_index"]
    f["heat_index"] = hi_new
    f["hi_excess_p95"] = f["hi_excess_p95"] + dhi
    by_geo = dhi.groupby(level="geoid")
    f["hi_lag1"] = f["hi_lag1"] + by_geo.shift(1).fillna(dhi)
    f["hi_mean_3d"] = f["hi_mean_3d"] + by_geo.transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    f["tmin_mean_3d"] = f["tmin_mean_3d"] + delta_c
    hot = (f["hi_excess_p95"] > 0).astype(float).rename("hot")
    f["hot_streak"] = streak(hot)
    f["first_heatwave"] = first_heatwave_of_season(f["hot_streak"])
    return f[list(FEATURES)]
