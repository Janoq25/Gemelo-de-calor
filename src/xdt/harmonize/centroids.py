"""Centroides de poblacion por condado (Census 2020).

Esqueleto geografico minimo de M0 que no necesita el stack geoespacial: un
punto y una poblacion por condado. Sirve para dos cosas:

1. **Muestrear rasters en el punto donde vive la gente**, no en el centroide
   geometrico. En condados grandes y despoblados del oeste (p.ej. Maricopa,
   San Bernardino) el centroide geometrico cae en desierto y describe un
   clima que casi nadie experimenta. El centroide de poblacion es coherente
   con la exposicion por residencia que el plan asume (§15.7).
2. **Ponderar por poblacion** la agregacion condado -> estado.

Es una aproximacion: una celda de 4 km por condado, no el promedio sobre el
poligono. El promedio areal ponderado es el entregable completo de M0 y
requiere el extra `geo`; este modulo no lo sustituye, lo adelanta.
"""

from __future__ import annotations

import io

import pandas as pd

from xdt.ingest.http import CachedClient, RawArtifact

CENPOP_COUNTY_URL = (
    "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt"
)

# Fuera de la malla CONUS de gridMET: Alaska, Hawai y Puerto Rico.
NON_CONUS_STATES = frozenset({"02", "15", "72"})


def fetch_county_centroids(client: CachedClient) -> RawArtifact:
    """Descarga (o lee de cache) el archivo de centroides. Es estatico."""
    return client.get(CENPOP_COUNTY_URL, resource="census:cenpop2020:county")


def parse_county_centroids(art: RawArtifact) -> pd.DataFrame:
    """Columnas: geoid, state_fips, county, state, population, lat, lon.

    El archivo trae BOM UTF-8 y FIPS con ceros a la izquierda; ambos rompen
    en silencio si se leen con los valores por defecto de pandas (la primera
    columna se llamaria '\\ufeffSTATEFP' y '01' se volveria 1).
    """
    df = pd.read_csv(
        io.StringIO(art.bytes().decode("utf-8-sig")),
        dtype={"STATEFP": str, "COUNTYFP": str},
    )
    out = pd.DataFrame(
        {
            "state_fips": df["STATEFP"].str.zfill(2),
            "geoid": df["STATEFP"].str.zfill(2) + df["COUNTYFP"].str.zfill(3),
            "county": df["COUNAME"],
            "state": df["STNAME"],
            "population": pd.to_numeric(df["POPULATION"], errors="raise"),
            "lat": pd.to_numeric(df["LATITUDE"], errors="raise"),
            "lon": pd.to_numeric(df["LONGITUDE"], errors="raise"),
        }
    )
    if out["geoid"].duplicated().any():
        raise ValueError("centroides de condado con geoid duplicado")
    return out.sort_values("geoid").reset_index(drop=True)


def select_counties(
    centroids: pd.DataFrame,
    *,
    states: list[str] | None = None,
    counties: list[str] | None = None,
) -> pd.DataFrame:
    """Filtra por estados (FIPS de 2 digitos) y/o condados (FIPS de 5)."""
    sel = centroids
    if states:
        sel = sel[sel["state_fips"].isin([s.zfill(2) for s in states])]
    if counties:
        sel = sel[sel["geoid"].isin([c.zfill(5) for c in counties])]
    return sel.reset_index(drop=True)


def top_counties_by_population(
    centroids: pd.DataFrame, *, states: list[str], coverage: float
) -> list[str]:
    """Condados mas poblados de cada estado hasta cubrir `coverage` de su poblacion.

    Recorta la descarga de gridMET para CONUS completo: ~20 % de los condados
    concentran ~80 % de la poblacion. El agregado estatal que resulta es un
    promedio ponderado sobre esos condados, no sobre el estado entero, y el
    consumidor debe declararlo (ver `features.panel`).
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage debe estar en (0, 1]")
    sel = select_counties(centroids, states=states)
    out: list[str] = []
    for _, g in sel.groupby("state_fips"):
        g = g.sort_values("population", ascending=False)
        share = g["population"].cumsum() / g["population"].sum()
        # Se incluye el condado que cruza el umbral: cobertura >= coverage.
        n = int((share < coverage).sum()) + 1
        out.extend(g["geoid"].iloc[:n])
    return sorted(out)
