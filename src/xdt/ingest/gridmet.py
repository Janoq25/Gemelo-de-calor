"""Conector de gridMET (Climatology Lab), fuente climatica primaria (§4.2).

Meteorologia diaria de superficie, malla de 4 km sobre CONUS, 1979-presente,
con ~2 dias de rezago.

Acceso verificado en vivo el 23/9/2026
--------------------------------------
Servidor THREDDS: http://thredds.northwestknowledge.net:8080/thredds
Un archivo agregado por variable: agg_met_{var}_1979_CurrentYear_CONUS.nc

Se usa **OPeNDAP en ASCII** (`.ascii?var[t0:1:t1][i:1:i][j:1:j]`) y no el
NetCDF Subset Service (NCSS), por rendimiento medido:

    NCSS, un punto x un ano     ~14 s
    OPeNDAP, una celda x un ano ~0.8 s

A escala de miles de condados la diferencia es entre horas y dias. Ademas,
OPeNDAP ASCII no necesita netCDF4: el conector corre con el nucleo, sin el
extra `geo`.

Tres trampas, todas silenciosas, documentadas para no repetirlas:

1. **Los valores llegan empaquetados.** Ni NCSS ni OPeNDAP ASCII aplican
   `scale_factor`/`add_offset`: Phoenix el 1/7/2023 llega como "971.0" con
   unidad "K". El valor real es 971 x 0.1 + 220 = 317.1 K = 43.9 °C.

2. **El empaquetado NO es uniforme entre variables.** tmmx usa
   add_offset=220 y tmmn usa add_offset=210. Una constante compartida
   desplazaria todas las minimas 10 °C sin que nada fallara. Por eso el
   empaquetado se lee del DAS de cada variable, nunca se asume.

3. **El oceano devuelve HTTP 200 con el valor de relleno 32767.** Un
   centroide costero que caiga en una celda enmascarada produce una serie
   plausible en forma y absurda en valor (32767 x 0.1 + 220 = 3496 K). Se
   descarta como dato ausente. Fuera de la caja CONUS, en cambio, el servidor
   responde 400: Alaska, Hawai y Puerto Rico se excluyen antes de pedir.

Muestreo espacial: una celda por condado, en su centroide de POBLACION
(`harmonize.centroids`), y agregacion condado -> estado ponderada por
poblacion (`harmonize.aggregate`). Es la exposicion por residencia que el plan
asume (§15.7). El promedio areal sobre el poligono queda para M0 completo.

Unidades en el estado del gemelo: tmmx/tmmn en °C (los escenarios de L4
suman `delta_c` en grados Celsius), rmax/rmin en %, srad en W/m², vs en m/s.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from xdt.harmonize.aggregate import population_weighted
from xdt.harmonize.centroids import (
    NON_CONUS_STATES,
    fetch_county_centroids,
    parse_county_centroids,
    select_counties,
)
from xdt.ingest.base import Connector
from xdt.ingest.http import CachedClient, InvalidBody, RawArtifact
from xdt.twin.state import now_utc

THREDDS_DODS = "http://thredds.northwestknowledge.net:8080/thredds/dodsC"

# Eje temporal: "days since 1900-01-01"; el indice 0 es 1979-01-01.
DAY_EPOCH = dt.date(1900, 1, 1)
FIRST_DAY = dt.date(1979, 1, 1)

# Malla verificada el 23/9/2026 contra /ncss/.../dataset.xml. Cada respuesta
# trae las coordenadas reales de la celda y se contrastan con estas: si la
# malla cambia, la ingesta falla en vez de asignar clima al lugar equivocado.
LAT0, DLAT, NLAT = 49.400000000000006, -0.041666666666671404, 585
LON0, DLON, NLON = -124.76666663333334, 0.041666666666671404, 1386
_GRID_TOL = 1e-4

KELVIN = 273.15

# Variables por defecto: las que M2 necesita para el indice de calor.
DEFAULT_VARIABLES = ("tmmx", "tmmn", "rmax", "rmin")


@dataclass(frozen=True)
class GridmetVar:
    short: str  # nombre en el archivo y en el estado del gemelo
    ncvar: str  # nombre de la variable dentro del NetCDF
    units: str  # unidad que declara el DAS; se verifica en cada ingesta
    kelvin_to_c: bool = False

    @property
    def file(self) -> str:
        return f"agg_met_{self.short}_1979_CurrentYear_CONUS.nc"


VARIABLES: dict[str, GridmetVar] = {
    v.short: v
    for v in (
        GridmetVar("tmmx", "daily_maximum_temperature", "K", kelvin_to_c=True),
        GridmetVar("tmmn", "daily_minimum_temperature", "K", kelvin_to_c=True),
        GridmetVar("rmax", "daily_maximum_relative_humidity", "%"),
        GridmetVar("rmin", "daily_minimum_relative_humidity", "%"),
        GridmetVar("srad", "daily_mean_shortwave_radiation_at_surface", "W m-2"),
        GridmetVar("vs", "daily_mean_wind_speed", "m/s"),
    )
}


@dataclass(frozen=True)
class Packing:
    scale: float
    offset: float
    fill: float
    units: str


class GridmetConnector(Connector):
    name = "gridmet"
    cadence = "diaria, rezago ~2 dias"
    base_url = THREDDS_DODS

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.client.body_check = _check_opendap_body
        # El archivo de centroides vive en otro servidor y otra carpeta de
        # data/raw: es un insumo del Census, no de gridMET.
        self.census = CachedClient(source="census", settings=self.settings)

    # --------------------------------------------------------------- fetch
    def fetch(
        self,
        *,
        years: Sequence[int],
        variables: Sequence[str] = DEFAULT_VARIABLES,
        states: Sequence[str] | None = None,
        counties: Sequence[str] | None = None,
        refresh: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        **_: Any,
    ) -> list[RawArtifact]:
        """Descarga una serie por (variable, celda, ano).

        Devuelve tambien los insumos auxiliares (centroides y DAS de cada
        variable) como artefactos: `normalize` es asi una funcion pura de lo
        descargado, y todo lo que influyo en un hecho queda en raw_artifact.

        La cache es por celda, no por condado: dos condados cuyo centroide
        cae en la misma celda comparten descarga.
        """
        unknown = [v for v in variables if v not in VARIABLES]
        if unknown:
            raise ValueError(f"variables desconocidas {unknown}; opciones {list(VARIABLES)}")
        if not states and not counties:
            raise ValueError("indica states y/o counties: CONUS completo son ~12.000 series/ano")

        centroids_art = fetch_county_centroids(self.census)
        sel = select_counties(
            parse_county_centroids(centroids_art),
            states=list(states or []),
            counties=list(counties or []),
        )
        sel = on_grid(sel)
        if sel.empty:
            raise ValueError("la seleccion no contiene condados dentro de la malla CONUS")
        cells = sorted(set(sel["cell"]))

        artifacts: list[RawArtifact] = [centroids_art]
        for short in variables:
            v = VARIABLES[short]
            art = self.client.get(f"/{v.file}.das", resource=f"gridmet:das:{short}")
            parse_das(art.text(), v.ncvar)  # falla aqui, antes de descargar, si cambio
            artifacts.append(art)

        spans = [self._year_span(int(y)) for y in years]
        total = len(variables) * len(cells) * len(spans)
        done = 0
        for short in variables:
            v = VARIABLES[short]
            for t0, t1 in spans:
                for i, j in cells:
                    ce = f"{v.ncvar}[{t0}:1:{t1}][{i}:1:{i}][{j}:1:{j}]"
                    artifacts.append(
                        self.client.get(
                            f"/{v.file}.ascii?{ce}",
                            # El recurso fija todo lo que determina el contenido
                            # (incluido t1: el ano en curso crece cada dia).
                            resource=f"gridmet:{short}:{i}:{j}:{t0}-{t1}",
                            refresh=refresh,
                        )
                    )
                    done += 1
                    if on_progress:
                        on_progress(done, total)
        return artifacts

    def _year_span(self, year: int) -> tuple[int, int]:
        """Indices [t0, t1] del ano, recortados al ultimo dia publicado."""
        if year < FIRST_DAY.year:
            raise ValueError(f"gridMET empieza en {FIRST_DAY.year}; pediste {year}")
        t0 = (dt.date(year, 1, 1) - FIRST_DAY).days
        t1 = (dt.date(year, 12, 31) - FIRST_DAY).days
        # Solo el ano en curso (o el recien cerrado, por el rezago) necesita
        # saber cuantos dias hay publicados.
        if dt.date(year, 12, 31) >= now_utc().date() - dt.timedelta(days=30):
            t1 = min(t1, self.n_days_available() - 1)
        if t1 < t0:
            raise ValueError(f"gridMET aun no publica datos de {year}")
        return t0, t1

    def n_days_available(self) -> int:
        """Longitud actual del eje temporal, leida del DDS.

        Se refresca en cada ejecucion en linea (el archivo crece a diario); en
        modo offline se usa la ultima copia cacheada, que es la que casa con
        los crudos de esa ejecucion.
        """
        art = self.client.get(
            f"/{VARIABLES['tmmx'].file}.dds",
            resource="gridmet:dds:tmmx",
            refresh=not self.settings.offline,
        )
        m = re.search(r"\[day = (\d+)\]", art.text())
        if not m:
            raise InvalidBody("DDS de gridMET sin dimension 'day'")
        return int(m.group(1))

    # ----------------------------------------------------------- normalize
    def normalize(self, artifacts: Iterable[RawArtifact]) -> pd.DataFrame:
        arts = list(artifacts)
        centroids = next(
            (parse_county_centroids(a) for a in arts if a.resource.startswith("census:")),
            None,
        )
        if centroids is None:
            raise ValueError("normalize() necesita el artefacto de centroides del Census")
        # El universo de pesos incluye a los condados fuera de la malla (Cayos
        # de Florida): cuentan como hueco de cobertura, no desaparecen.
        universe = centroids[~centroids["state_fips"].isin(NON_CONUS_STATES)]
        by_cell = on_grid(universe).groupby("cell")["geoid"].apply(list).to_dict()

        packings: dict[str, Packing] = {}
        for a in arts:
            if a.resource.startswith("gridmet:das:"):
                short = a.resource.rsplit(":", 1)[1]
                packings[short] = parse_das(a.text(), VARIABLES[short].ncvar)

        frames: list[pd.DataFrame] = []
        for a in arts:
            m = _DATA_RESOURCE.match(a.resource)
            if not m:
                continue
            short, i, j = m.group("var"), int(m.group("i")), int(m.group("j"))
            v = VARIABLES[short]
            if short not in packings:
                raise ValueError(f"falta el DAS de {short}: no se puede desempaquetar")
            series = parse_ascii(a.text(), v.ncvar, expect_cell=(i, j))
            values = unpack(series["raw"].to_numpy(), packings[short], v)
            ok = ~np.isnan(values)
            for geoid in by_cell.get((i, j), []):
                frames.append(
                    pd.DataFrame(
                        {
                            "geo_level": "county",
                            "geoid": geoid,
                            "valid_date": series["valid_date"].to_numpy()[ok],
                            "variable": short,
                            "value": values[ok],
                        }
                    )
                )

        if not frames:
            return pd.DataFrame(columns=["geo_level", "geoid", "valid_date", "variable", "value"])

        county = pd.concat(frames, ignore_index=True)
        # Una misma celda puede llegar en dos artefactos del ano en curso
        # (t1 distinto); el mas reciente gana.
        county = county.drop_duplicates(
            subset=["geoid", "valid_date", "variable"], keep="last"
        )
        state = population_weighted(
            county,
            universe.loc[universe["state_fips"].isin(county["geoid"].str[:2].unique()),
                         ["geoid", "population", "state_fips"]],
            parent_col="state_fips",
            parent_level="state",
        )
        return pd.concat(
            [county, state.drop(columns="weight_coverage")], ignore_index=True
        )

    def source_version(self, artifacts: Iterable[RawArtifact]) -> str | None:
        stamps = [a.fetched_at for a in artifacts]
        return f"gridmet-{max(stamps):%Y%m%d}" if stamps else None

    def close(self) -> None:
        super().close()
        self.census.close()


# ------------------------------------------------------------------ helpers
_DATA_RESOURCE = re.compile(r"^gridmet:(?P<var>[a-z]+):(?P<i>\d+):(?P<j>\d+):\d+-\d+$")


def grid_cell(lat: float, lon: float) -> tuple[int, int]:
    """Celda (i, j) de la malla gridMET que contiene el punto."""
    i = int(round((lat - LAT0) / DLAT))
    j = int(round((lon - LON0) / DLON))
    if not (0 <= i < NLAT and 0 <= j < NLON):
        raise ValueError(f"({lat}, {lon}) fuera de la malla CONUS de gridMET")
    return i, j


def on_grid(centroids: pd.DataFrame) -> pd.DataFrame:
    """Condados cuyo centroide cae dentro de la malla, con su celda asignada.

    Excluye Alaska, Hawai y Puerto Rico, y tambien condados de estados CONUS
    que quedan fuera de la caja: Monroe (FL), los Cayos, tiene su centroide
    en 24.75 N y la malla empieza en 25.06 N.
    """
    df = centroids[~centroids["state_fips"].isin(NON_CONUS_STATES)].copy()
    cells = []
    for lat, lon in zip(df["lat"], df["lon"], strict=True):
        try:
            cells.append(grid_cell(lat, lon))
        except ValueError:
            cells.append(None)
    df["cell"] = cells
    return df[df["cell"].notna()].reset_index(drop=True)


def cell_center(i: int, j: int) -> tuple[float, float]:
    return LAT0 + i * DLAT, LON0 + j * DLON


def parse_das(text: str, ncvar: str) -> Packing:
    """Empaquetado de una variable a partir del DAS de OPeNDAP."""
    m = re.search(rf"^\s*{re.escape(ncvar)} \{{(.*?)^\s*\}}", text, re.S | re.M)
    if not m:
        raise InvalidBody(f"el DAS no describe la variable {ncvar}")
    block = m.group(1)

    def attr(name: str, default: str | None = None) -> str:
        am = re.search(rf"\b{name}\s+\"?([^\";]+)\"?;", block)
        if am is None:
            if default is None:
                raise InvalidBody(f"el DAS de {ncvar} no trae {name}")
            return default
        return am.group(1).strip()

    return Packing(
        scale=float(attr("scale_factor", "1.0")),
        offset=float(attr("add_offset", "0.0")),
        fill=float(attr("_FillValue")),
        units=attr("units"),
    )


def unpack(raw: np.ndarray, packing: Packing, var: GridmetVar) -> np.ndarray:
    """Enteros empaquetados -> unidades fisicas del gemelo. Relleno -> NaN."""
    if packing.units != var.units:
        raise InvalidBody(
            f"{var.short}: el DAS declara unidad {packing.units!r}, se esperaba "
            f"{var.units!r}. Revisa la conversion antes de ingerir."
        )
    raw = raw.astype(float)
    out = raw * packing.scale + packing.offset
    out[raw == packing.fill] = np.nan
    if var.kelvin_to_c:
        out = out - KELVIN
    return out


def parse_ascii(
    text: str, ncvar: str, *, expect_cell: tuple[int, int] | None = None
) -> pd.DataFrame:
    """Serie de una celda desde la respuesta `.ascii` de OPeNDAP.

    Verifica que el numero de valores coincide con la dimension declarada
    (una respuesta truncada por la red no debe pasar por una serie corta) y
    que las coordenadas devueltas son las de la celda pedida.
    """
    sections: dict[str, tuple[int, list[str]]] = {}
    current: str | None = None
    header = re.compile(rf"^{re.escape(ncvar)}\.(\w+)\[(\d+)\]")
    for line in text.splitlines():
        hm = header.match(line.strip())
        if hm:
            current = hm.group(1)
            sections[current] = (int(hm.group(2)), [])
            continue
        if current is not None and line.strip():
            sections[current][1].append(line.strip())

    if ncvar not in sections or "day" not in sections:
        raise InvalidBody(f"respuesta OPeNDAP sin la variable {ncvar} o sin el eje 'day'")

    n, rows = sections[ncvar]
    raw = np.array([float(r.rsplit(",", 1)[1]) for r in rows])
    days = np.array([float(x) for x in ",".join(sections["day"][1]).split(",") if x.strip()])
    if len(raw) != n or len(days) != n:
        raise InvalidBody(f"respuesta truncada: se declararon {n} dias y llegaron {len(raw)}")

    if expect_cell is not None:
        lat = float(sections["lat"][1][0])
        lon = float(sections["lon"][1][0])
        elat, elon = cell_center(*expect_cell)
        if abs(lat - elat) > _GRID_TOL or abs(lon - elon) > _GRID_TOL:
            raise InvalidBody(
                f"la celda {expect_cell} devolvio ({lat}, {lon}), se esperaba "
                f"({elat:.5f}, {elon:.5f}). La malla de gridMET cambio."
            )

    dates = pd.to_datetime(DAY_EPOCH) + pd.to_timedelta(days, unit="D")
    return pd.DataFrame({"valid_date": dates.date, "raw": raw})


def _check_opendap_body(content: bytes) -> None:
    """THREDDS puede devolver el documento `Error { ... }` con codigo de exito."""
    head = content[:200].lstrip()
    if head.startswith(b"Error {"):
        msg = re.search(rb'message = "([^"]*)"', content)
        raise InvalidBody(
            "OPeNDAP devolvio un error: "
            + (msg.group(1).decode("utf-8", "replace") if msg else head.decode("utf-8", "replace"))
        )
    if not (head.startswith(b"Dataset {") or head.startswith(b"Attributes {")):
        raise InvalidBody(f"respuesta OPeNDAP inesperada: {head[:60]!r}")
