"""Conector de EAGLE-I (ORNL): apagones por condado cada 15 minutos (§4.3).

Nucleo del evento compuesto calor-apagon (§9.2), la contribucion
diferenciadora del trabajo.

Fuente verificada el 23/9/2026
------------------------------
figshare, articulo 24237376, **version 4 fijada** (2014-2025). Un CSV por ano
de 0.6-1.4 GB, mas MCC.csv (clientes modelados por condado) y
coverage_history.csv (cobertura por estado y ano). Articulo de datos:
Brelsford et al. 2024, Scientific Data, doi:10.1038/s41597-024-03095-5.

Cinco trampas, documentadas para no repetirlas:

1. **Formato disperso.** Solo existen filas con `customers_out > 0`. El
   articulo: "Missing entries are either a customer outage value of zero or
   a gap in data collection; we do not distinguish between these cases."
   Aqui la ausencia se rellena con cero -- la unica lectura posible para
   tener una serie densa --, y por eso se ingiere tambien la cobertura por
   estado y ano: un "cero" en un ano de cobertura del 22 % (Alabama 2018)
   no significa lo mismo que en uno del 95 %.

2. **UTC, no hora local.** "All data in EAGLE-I is presented in Coordinated
   Universal Time (UTC)." Agregar por dia UTC pondria un apagon nocturno de
   Phoenix en el dia de calor equivocado. Se convierte al dia civil local
   con el huso predominante del estado (`harmonize.fips`).

3. **El articulo documenta mal el formato de fecha.** Dice "MM/DD/YY 00:00";
   los archivos reales usan ISO "2017-01-01 00:00:00". Se aceptan ambos y
   cualquier fila que no case con ninguno detiene la ingesta.

4. **MCC.csv pierde los ceros a la izquierda** ("1001", no "01001"). Sin
   normalizar, todos los condados de AL, AK, AZ, AR y CA quedarian sin
   denominador y fuera del join sin ningun error.

5. **Tamano.** ~1.2 GB por ano. Se descarga en streaming, se verifica el
   MD5 que publica figshare, y el agregado lo hace DuckDB leyendo el CSV
   directamente del disco: nada pasa entero por memoria de Python.

Variables que produce, por condado-dia y estado-dia (nombres alineados con
los escenarios de L4 para que `apagon()` perturbe exactamente lo observado):

    outage_customers_frac        pico diario de la fraccion de clientes sin luz
    outage_max_hours             racha continua mas larga (h) con >= 1 % sin luz
    outage_customer_hours_frac   clientes-hora sin luz por cliente (0-24)

y por estado-ano: eaglei_cov_min_pct, eaglei_cov_max_pct.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import duckdb
import pandas as pd

from xdt.harmonize.fips import ABBR_TO_FIPS, STATES
from xdt.ingest.base import Connector
from xdt.ingest.http import InvalidBody, RawArtifact

FIGSHARE_ARTICLE = 24237376
FIGSHARE_VERSION = 4
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"

SNAPSHOT_HOURS = 0.25  # cadencia de 15 minutos
# Umbral de la racha: fraccion de clientes que cuenta como "en apagon". El
# plan (§9.2) define el evento compuesto como >X % sin suministro >= Y horas;
# X queda fijado aqui para que la variable sea reproducible.
OUTAGE_THRESHOLD = 0.01

OUTAGE_VARIABLES = ("outage_customers_frac", "outage_max_hours", "outage_customer_hours_frac")


class EagleiConnector(Connector):
    name = "eaglei"
    cadence = "15 min, publicacion anual"
    base_url = ""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.states: list[str] = []
        #: diagnostico de la ultima normalizacion (condados sin MCC, recortes)
        self.report: dict[str, Any] = {}

    # --------------------------------------------------------------- fetch
    def fetch(
        self,
        *,
        years: Sequence[int],
        states: Sequence[str],
        refresh: bool = False,
        on_bytes: Callable[[int, int | None], None] | None = None,
        **_: Any,
    ) -> list[RawArtifact]:
        if not states:
            raise ValueError("indica states: la serie densa nacional son ~3.2M filas/ano")
        unknown = [s for s in states if s.zfill(2) not in STATES]
        if unknown:
            raise ValueError(f"FIPS de estado desconocidos: {unknown}")
        self.states = sorted(s.zfill(2) for s in states)

        meta = self.client.get(
            f"{FIGSHARE_API}/versions/{FIGSHARE_VERSION}",
            resource=f"figshare:{FIGSHARE_ARTICLE}:v{FIGSHARE_VERSION}",
        )
        files = {f["name"]: f for f in meta.json()["files"]}
        arts = [meta]
        for name in ("MCC.csv", "coverage_history.csv"):
            arts.append(self.client.get(files[name]["download_url"], resource=f"eaglei:{name}"))

        for y in years:
            fname = f"eaglei_outages_{int(y)}.csv"
            if fname not in files:
                raise ValueError(f"la version {FIGSHARE_VERSION} no publica {fname}")
            f = files[fname]
            art = self.client.download(
                f["download_url"],
                resource=f"eaglei:outages:{int(y)}:v{FIGSHARE_VERSION}",
                refresh=refresh,
                on_bytes=on_bytes,
            )
            if not art.from_cache:
                _verify_md5(art, f["computed_md5"])
            arts.append(art)
        return arts

    # ----------------------------------------------------------- normalize
    def normalize(self, artifacts: Iterable[RawArtifact]) -> pd.DataFrame:
        arts = list(artifacts)
        by_res = {a.resource: a for a in arts}
        if "eaglei:MCC.csv" not in by_res:
            raise ValueError("normalize() necesita MCC.csv para calcular fracciones")
        states = self.states or sorted(STATES)
        years = sorted(
            int(r.split(":")[2]) for r in by_res if r.startswith("eaglei:outages:")
        )
        outage_paths = [
            str(by_res[f"eaglei:outages:{y}:v{FIGSHARE_VERSION}"].path) for y in years
        ]

        mcc = parse_mcc(by_res["eaglei:MCC.csv"])
        mcc = mcc[mcc["geoid"].str[:2].isin(states)]
        tz = pd.DataFrame({"state_fips": states, "tz": [STATES[s][2] for s in states]})

        frames: list[pd.DataFrame] = []
        if outage_paths:
            frames.append(self._daily(outage_paths, mcc, tz, years))
        if "eaglei:coverage_history.csv" in by_res:
            cov = parse_coverage(by_res["eaglei:coverage_history.csv"])
            frames.append(cov[cov["geoid"].isin(states)])
        if not frames:
            return pd.DataFrame(columns=["geo_level", "geoid", "valid_date", "variable", "value"])
        return pd.concat(frames, ignore_index=True)

    def _daily(
        self, paths: list[str], mcc: pd.DataFrame, tz: pd.DataFrame, years: list[int]
    ) -> pd.DataFrame:
        con = duckdb.connect()
        try:
            con.execute("SET enable_progress_bar = false")  # ensucia la salida de la CLI
            con.register("mcc_df", mcc)
            con.register("tz_df", tz)
            con.execute(
                """
                CREATE TEMP TABLE raw AS
                SELECT lpad(fips_code, 5, '0') AS geoid,
                       TRY_CAST(customers_out AS DOUBLE) AS customers_out,
                       run_start_time AS ts_text,
                       COALESCE(try_strptime(run_start_time, '%Y-%m-%d %H:%M:%S'),
                                try_strptime(run_start_time, '%m/%d/%y %H:%M')) AS ts_utc
                FROM read_csv(?, header = true, union_by_name = true,
                              columns = {'fips_code': 'VARCHAR', 'county': 'VARCHAR',
                                         'state': 'VARCHAR', 'customers_out': 'VARCHAR',
                                         'run_start_time': 'VARCHAR'})
                WHERE substr(lpad(fips_code, 5, '0'), 1, 2) IN (SELECT state_fips FROM tz_df)
                """,
                [paths],
            )
            bad = con.execute(
                "SELECT count(*), any_value(ts_text) FROM raw WHERE ts_utc IS NULL"
            ).fetchone()
            if bad[0]:
                raise InvalidBody(
                    f"{bad[0]} filas de EAGLE-I con fecha en formato desconocido "
                    f"(ej. {bad[1]!r}). Ver trampa 3 del modulo."
                )
            # Extremos del archivo COMPLETO, no de los estados filtrados: un
            # estado pequeno sin apagones el 1 de enero perderia dias validos.
            (bounds_lo, bounds_hi) = con.execute(
                """
                SELECT min(t), max(t) FROM (
                    SELECT COALESCE(try_strptime(run_start_time, '%Y-%m-%d %H:%M:%S'),
                                    try_strptime(run_start_time, '%m/%d/%y %H:%M')) AS t
                    FROM read_csv(?, header = true, union_by_name = true,
                                  columns = {'fips_code': 'VARCHAR', 'county': 'VARCHAR',
                                             'state': 'VARCHAR', 'customers_out': 'VARCHAR',
                                             'run_start_time': 'VARCHAR'})
                )
                """,
                [paths],
            ).fetchone()
            (n_sel,) = con.execute("SELECT count(*) FROM raw").fetchone()
            if bounds_lo is None:
                return pd.DataFrame(
                    columns=["geo_level", "geoid", "valid_date", "variable", "value"]
                )

            no_mcc = con.execute(
                "SELECT DISTINCT geoid FROM raw WHERE geoid NOT IN (SELECT geoid FROM mcc_df)"
            ).df()["geoid"].tolist()

            # Instantanea -> fraccion, en hora local. Se recorta a 1: MCC es un
            # modelo y hay instantes con mas clientes sin luz que modelados.
            con.execute(
                """
                CREATE TEMP TABLE snap AS
                SELECT r.geoid, substr(r.geoid, 1, 2) AS state_fips,
                       timezone(t.tz, r.ts_utc AT TIME ZONE 'UTC') AS ts_local,
                       max(r.customers_out) AS customers_out,
                       any_value(m.customers) AS mcc
                FROM raw r
                JOIN mcc_df m USING (geoid)
                JOIN tz_df t ON t.state_fips = substr(r.geoid, 1, 2)
                GROUP BY 1, 2, 3
                """
            )
            (n_clip,) = con.execute(
                "SELECT count(*) FROM snap WHERE customers_out > mcc"
            ).fetchone()

            county = con.execute(_DAILY_SQL.format(src="""
                SELECT geoid, ts_local, least(customers_out / mcc, 1.0) AS frac FROM snap
            """), [OUTAGE_THRESHOLD, SNAPSHOT_HOURS, SNAPSHOT_HOURS]).df()
            state = con.execute(_DAILY_SQL.format(src="""
                SELECT s.state_fips AS geoid, s.ts_local,
                       least(sum(s.customers_out) / any_value(tot.mcc), 1.0) AS frac
                FROM snap s
                JOIN (SELECT substr(geoid, 1, 2) AS state_fips, sum(customers) AS mcc
                      FROM mcc_df GROUP BY 1) tot USING (state_fips)
                GROUP BY s.state_fips, s.ts_local
            """), [OUTAGE_THRESHOLD, SNAPSHOT_HOURS, SNAPSHOT_HOURS]).df()
        finally:
            con.close()

        days = _complete_local_days(years, bounds_lo, bounds_hi, tz)
        county_units = mcc[["geoid"]].assign(state_fips=mcc["geoid"].str[:2])
        state_units = pd.DataFrame({"geoid": tz["state_fips"], "state_fips": tz["state_fips"]})
        out = pd.concat(
            [
                _densify(county, county_units, days, "county"),
                _densify(state, state_units, days, "state"),
            ],
            ignore_index=True,
        )
        self.report = {
            "years": years,
            "counties_without_mcc": no_mcc,
            "snapshots_clipped_to_mcc": int(n_clip),
            "outage_rows_selected": int(n_sel),
            "local_days": {k: (str(a), str(b)) for k, (a, b) in days.items()},
        }
        return out

    def source_version(self, artifacts: Iterable[RawArtifact]) -> str | None:
        return f"eaglei-figshare-v{FIGSHARE_VERSION}"


# Agregado diario comun a condado y estado. `{src}` produce (geoid, ts_local,
# frac). La racha mas larga es un problema de islas: instantes consecutivos de
# 15 min sobre el umbral comparten (epoch/900 - row_number).
_DAILY_SQL = """
WITH s AS ({src}),
d AS (
    SELECT geoid, ts_local, CAST(ts_local AS DATE) AS day, frac FROM s
),
over AS (
    SELECT geoid, day,
           CAST(epoch(ts_local) / 900 AS BIGINT)
             - row_number() OVER (PARTITION BY geoid, day ORDER BY ts_local) AS grp
    FROM d WHERE frac >= ?
),
runs AS (
    SELECT geoid, day, max(n) * ? AS max_hours FROM (
        SELECT geoid, day, grp, count(*) AS n FROM over GROUP BY 1, 2, 3
    ) GROUP BY 1, 2
)
SELECT d.geoid, d.day AS valid_date,
       max(d.frac)            AS outage_customers_frac,
       coalesce(any_value(r.max_hours), 0.0) AS outage_max_hours,
       sum(d.frac) * ?        AS outage_customer_hours_frac
FROM d LEFT JOIN runs r USING (geoid, day)
GROUP BY d.geoid, d.day
"""


def _complete_local_days(
    years: list[int], lo: dt.datetime, hi: dt.datetime, tz: pd.DataFrame
) -> dict[str, tuple[dt.date, dt.date]]:
    """Dias locales cuya ventana de 24 h cae entera dentro de los datos UTC.

    El 31/12 local de Phoenix termina a las 07:00 UTC del 1/1 siguiente, que
    vive en el archivo del ano siguiente. Rellenar con cero ese dia inventaria
    siete horas sin apagones. Si el archivo contiguo no esta cargado, el dia
    se descarta en vez de completarse.
    """
    end_utc = pd.Timestamp(hi) + pd.Timedelta(minutes=15)
    out: dict[str, tuple[dt.date, dt.date]] = {}
    for fips, zone in zip(tz["state_fips"], tz["tz"], strict=True):
        first = dt.date(min(years), 1, 1)
        last = dt.date(max(years), 12, 31)
        # recorta por los extremos reales de los datos
        while pd.Timestamp(first, tz=zone).tz_convert("UTC").tz_localize(None) < pd.Timestamp(lo):
            first += dt.timedelta(days=1)
        while (pd.Timestamp(last + dt.timedelta(days=1), tz=zone)
               .tz_convert("UTC").tz_localize(None)) > end_utc:
            last -= dt.timedelta(days=1)
        if first <= last:
            out[fips] = (first, last)
    return out


def _densify(
    daily: pd.DataFrame,
    units: pd.DataFrame,
    days: dict[str, tuple[dt.date, dt.date]],
    geo_level: str,
) -> pd.DataFrame:
    """Serie densa: cada unidad x cada dia completo; la ausencia es cero."""
    grids = []
    for fips, (first, last) in days.items():
        g = units.loc[units["state_fips"] == fips, ["geoid"]]
        if g.empty:
            continue
        dates = pd.DataFrame({"valid_date": pd.date_range(first, last, freq="D").date})
        grids.append(g.merge(dates, how="cross"))
    if not grids:
        return pd.DataFrame(columns=["geo_level", "geoid", "valid_date", "variable", "value"])
    grid = pd.concat(grids, ignore_index=True)

    daily = daily.assign(valid_date=pd.to_datetime(daily["valid_date"]).dt.date)
    dense = grid.merge(daily, on=["geoid", "valid_date"], how="left")
    dense[list(OUTAGE_VARIABLES)] = dense[list(OUTAGE_VARIABLES)].fillna(0.0)
    long = dense.melt(
        id_vars=["geoid", "valid_date"], value_vars=list(OUTAGE_VARIABLES),
        var_name="variable", value_name="value",
    )
    long["geo_level"] = geo_level
    return long[["geo_level", "geoid", "valid_date", "variable", "value"]]


def parse_mcc(art: RawArtifact) -> pd.DataFrame:
    """MCC.csv -> geoid (5 digitos, trampa 4), customers."""
    df = pd.read_csv(io.StringIO(art.bytes().decode("utf-8-sig")), dtype={"County_FIPS": str})
    out = pd.DataFrame({
        "geoid": df["County_FIPS"].str.strip().str.zfill(5),
        "customers": pd.to_numeric(df["Customers"], errors="coerce"),
    })
    return out[out["customers"] > 0].drop_duplicates("geoid").reset_index(drop=True)


def parse_coverage(art: RawArtifact) -> pd.DataFrame:
    """coverage_history.csv -> hechos estado-ano de cobertura (%)."""
    df = pd.read_csv(io.StringIO(art.bytes().decode("utf-8-sig")))
    df["geoid"] = df["state"].map(ABBR_TO_FIPS)
    df["valid_date"] = pd.to_datetime(df["year"], format="%m/%d/%y").dt.date
    df = df.dropna(subset=["geoid"])
    long = df.melt(
        id_vars=["geoid", "valid_date"],
        value_vars=["min_pct_covered", "max_pct_covered"],
        var_name="variable", value_name="value",
    )
    long["variable"] = long["variable"].map(
        {"min_pct_covered": "eaglei_cov_min_pct", "max_pct_covered": "eaglei_cov_max_pct"}
    )
    long["value"] = long["value"] * 100.0
    long["geo_level"] = "state"
    return long[["geo_level", "geoid", "valid_date", "variable", "value"]]


def _verify_md5(art: RawArtifact, expected: str) -> None:
    h = hashlib.md5()  # noqa: S324 - verificacion de integridad, no seguridad
    with open(art.path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    if h.hexdigest() != expected:
        meta = art.path.with_suffix(".meta.json")
        art.path.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        raise InvalidBody(
            f"MD5 de {art.resource} no coincide con figshare "
            f"({h.hexdigest()} != {expected}); descarga corrupta, eliminada de la cache"
        )
