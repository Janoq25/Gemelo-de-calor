"""L1 · Estado del gemelo (§6 del plan).

Snapshot versionado por (geografia, fecha), consultable en cualquier instante
pasado. La implementacion es *bitemporal*, con dos ejes de tiempo:

    valid_date : la fecha del mundo real que el hecho describe
    known_at   : el instante en que el gemelo se entero del hecho

Por que dos y no uno. Las fuentes del plan se revisan: EPHT republica anos
anteriores, gridMET llega con rezago y luego se corrige, PLACES/SVI cambian de
anada. Si el estado guardara solo "el ultimo valor conocido", evaluar en 2026
una prediccion emitida en 2023 usaria datos que en 2023 no existian. Eso es
fuga de informacion, y arruinaria tanto la validacion temporal (§10.2) como la
calibracion retrospectiva de L5.

Con `known_at` la pregunta correcta es contestable: reconstruir el estado tal
como el gemelo lo conocia en una fecha dada.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import duckdb
import pandas as pd

from xdt.hashing import hash_frame, stable_id

# Niveles geograficos admitidos. Coinciden con el inventario del plan (§4).
GEO_LEVELS = ("nation", "hhs_region", "state", "county", "tract", "zcta")

REQUIRED_FACT_COLUMNS = ("geo_level", "geoid", "valid_date", "variable")


def now_utc() -> dt.datetime:
    """Instante actual en UTC sin tzinfo.

    `known_at` se almacena SIEMPRE como UTC naive. Pasar `datetime.now()`
    (hora local) mezcla husos y produce cortes bitemporales incoherentes: en
    husos al oeste de UTC, un known_at "posterior" resulta anterior y la
    guarda de revisiones antedatadas lo rechaza. Usar esta funcion.
    """
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class EstadoGemelo:
    """Snapshot inmutable del estado, en formato ancho.

    `frame` tiene indice (geoid, valid_date) y una columna por variable.
    Es el objeto que los escenarios de L4 transforman (§9): las funciones de
    escenario reciben un EstadoGemelo y devuelven otro, nunca mutan.
    """

    geo_level: str
    frame: pd.DataFrame
    known_at: dt.datetime
    valid_from: dt.date
    valid_to: dt.date
    lineage: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def geoids(self) -> list[str]:
        return sorted(self.frame.index.get_level_values("geoid").unique().tolist())

    @property
    def variables(self) -> list[str]:
        return sorted(map(str, self.frame.columns))

    @property
    def is_baseline(self) -> bool:
        """True si el estado no ha sido perturbado por ningun escenario."""
        return not self.lineage

    def input_hash(self) -> str:
        """Hash estable del contenido; alimenta `input_hash` de L0."""
        return hash_frame(self.frame.reset_index())

    def state_id(self) -> str:
        return stable_id(
            self.geo_level, self.valid_from, self.valid_to, self.known_at, self.input_hash()
        )

    def derive(self, frame: pd.DataFrame, note: str, **meta: Any) -> EstadoGemelo:
        """Devuelve un estado nuevo, registrando la transformacion en el linaje.

        Es el unico camino admitido para producir un estado modificado: asi el
        linaje de un contrafactual queda siempre completo y auditable.
        """
        return replace(
            self,
            frame=frame,
            lineage=(*self.lineage, note),
            meta={**self.meta, **meta},
        )

    def require(self, *variables: str) -> None:
        """Falla ruidosamente si falta una variable que el llamador necesita."""
        missing = [v for v in variables if v not in self.frame.columns]
        if missing:
            raise KeyError(f"El estado no contiene {missing}. Disponibles: {self.variables}")

    def __repr__(self) -> str:  # pragma: no cover - conveniencia
        n_geo = self.frame.index.get_level_values("geoid").nunique()
        return (
            f"EstadoGemelo({self.geo_level}, {n_geo} geografias, "
            f"{len(self.variables)} variables, {self.valid_from}..{self.valid_to}, "
            f"known_at={self.known_at:%Y-%m-%d}, linaje={len(self.lineage)})"
        )


class StateStore:
    """Lectura y escritura del estado bitemporal."""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    # ------------------------------------------------------------------ write
    def put_facts(
        self,
        facts: pd.DataFrame,
        *,
        source: str,
        source_version: str | None = None,
        known_at: dt.datetime | None = None,
        raw_hash: str | None = None,
    ) -> dict[str, int]:
        """Inserta hechos con semantica SCD tipo 2 e idempotencia.

        - clave natural: (geo_level, geoid, valid_date, variable, source)
        - valor identico al ultimo conocido -> no se escribe nada
        - valor distinto                    -> nueva revision, no sobrescritura

        La idempotencia es requisito de §7 M1 ("un conector por fuente,
        idempotente"): re-ejecutar la ingesta no debe duplicar ni ensuciar el
        historial.
        """
        missing = [c for c in REQUIRED_FACT_COLUMNS if c not in facts.columns]
        if missing:
            raise ValueError(f"faltan columnas obligatorias en `facts`: {missing}")
        if "value" not in facts.columns and "value_text" not in facts.columns:
            raise ValueError("`facts` debe traer al menos una de: value, value_text")

        bad = set(facts["geo_level"].unique()) - set(GEO_LEVELS)
        if bad:
            raise ValueError(f"geo_level no reconocido: {sorted(bad)}; admitidos {GEO_LEVELS}")

        df = facts.copy()
        if "value" not in df.columns:
            df["value"] = pd.NA
        if "value_text" not in df.columns:
            df["value_text"] = pd.NA

        df["valid_date"] = pd.to_datetime(df["valid_date"]).dt.date
        df["geoid"] = df["geoid"].astype(str)
        df["variable"] = df["variable"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["value_text"] = df["value_text"].astype("object").where(df["value_text"].notna(), None)

        # `known_at` en el futuro deja el hecho invisible para toda consulta
        # por defecto (que corta en "ahora"): se escribiria sin error y no
        # aparecerian en ningun snapshot. Fallo silencioso, y el gemelo no
        # puede haberse enterado de algo que aun no ha pasado.
        stamp = known_at or now_utc()
        if stamp > now_utc() + dt.timedelta(seconds=1):
            raise ValueError(
                f"known_at={stamp} esta en el futuro. El estado quedaria invisible "
                "para las consultas por defecto. Usa xdt.twin.state.now_utc()."
            )

        df["source"] = source
        df["source_version"] = source_version
        df["raw_hash"] = raw_hash
        df["known_at"] = stamp
        df["fact_key"] = [
            stable_id(gl, gi, vd, var, source)
            for gl, gi, vd, var in zip(
                df["geo_level"], df["geoid"], df["valid_date"], df["variable"], strict=True
            )
        ]

        # Colapsa duplicados dentro del propio lote (el ultimo gana).
        n_in = len(df)
        df = df.drop_duplicates(subset=["fact_key"], keep="last")
        dupes = n_in - len(df)

        self.con.register("_incoming", df)
        try:
            self.con.execute(
                """
                CREATE OR REPLACE TEMP VIEW _latest AS
                SELECT fact_key, value, value_text, revision, known_at
                FROM (
                    SELECT fact_key, value, value_text, revision, known_at,
                           row_number() OVER (PARTITION BY fact_key ORDER BY revision DESC) AS rn
                    FROM twin_state
                    WHERE fact_key IN (SELECT fact_key FROM _incoming)
                ) t
                WHERE rn = 1
                """
            )
            # El eje `known_at` debe ser monotono por hecho. Una revision
            # antedatada respecto a la que reemplaza haria que `as_of` con un
            # corte intermedio devolviera el valor *nuevo* para un instante en
            # que el gemelo aun no lo conocia: exactamente la fuga de
            # informacion que este diseno existe para impedir.
            (n_backdated,) = self.con.execute(
                """
                SELECT count(*) FROM _incoming i
                JOIN _latest l USING (fact_key)
                WHERE i.known_at < l.known_at
                  AND (l.value IS DISTINCT FROM i.value
                       OR l.value_text IS DISTINCT FROM i.value_text)
                """
            ).fetchone()
            if n_backdated:
                (worst,) = self.con.execute(
                    """
                    SELECT max(l.known_at) FROM _incoming i
                    JOIN _latest l USING (fact_key)
                    WHERE i.known_at < l.known_at
                    """
                ).fetchone()
                raise ValueError(
                    f"{n_backdated} hecho(s) traen known_at anterior al de la revision "
                    f"que reemplazan (la mas reciente es {worst}). Antedatar una revision "
                    "corrompe el eje bitemporal. Usa un known_at >= al existente, o "
                    "carga la correccion como una fuente distinta."
                )
            # Solo se escribe lo nuevo o lo que cambio de valor.
            self.con.execute(
                """
                CREATE OR REPLACE TEMP VIEW _to_write AS
                SELECT i.*, COALESCE(l.revision + 1, 0) AS new_revision
                FROM _incoming i
                LEFT JOIN _latest l USING (fact_key)
                WHERE l.fact_key IS NULL
                   OR l.value IS DISTINCT FROM i.value
                   OR l.value_text IS DISTINCT FROM i.value_text
                """
            )
            (n_write,) = self.con.execute("SELECT count(*) FROM _to_write").fetchone()
            (n_rev,) = self.con.execute(
                "SELECT count(*) FROM _to_write WHERE new_revision > 0"
            ).fetchone()
            self.con.execute(
                """
                INSERT INTO twin_state
                    (fact_key, geo_level, geoid, valid_date, variable, value, value_text,
                     source, source_version, known_at, revision, raw_hash)
                SELECT fact_key, geo_level, geoid, valid_date, variable, value, value_text,
                       source, source_version, known_at, new_revision, raw_hash
                FROM _to_write
                """
            )
        finally:
            self.con.unregister("_incoming")

        return {
            "received": n_in,
            "duplicates_collapsed": dupes,
            "written": int(n_write),
            "revisions": int(n_rev),
            "unchanged": len(df) - int(n_write),
        }

    # ------------------------------------------------------------------- read
    def as_of(
        self,
        *,
        geo_level: str,
        valid_from: dt.date | str,
        valid_to: dt.date | str | None = None,
        variables: Sequence[str] | None = None,
        geoids: Sequence[str] | None = None,
        known_at: dt.datetime | None = None,
    ) -> pd.DataFrame:
        """Estado en formato largo, tal como se conocia en `known_at`.

        `known_at=None` significa "todo lo que sabemos hoy". Para reproducir
        una prediccion pasada hay que pasar el `known_at` que se registro con
        ella en L0.
        """
        vf = pd.to_datetime(valid_from).date()
        vt = pd.to_datetime(valid_to).date() if valid_to is not None else vf
        cutoff = known_at or now_utc()

        clauses = ["geo_level = ?", "valid_date BETWEEN ? AND ?", "known_at <= ?"]
        params: list[Any] = [geo_level, vf, vt, cutoff]
        if variables:
            placeholders = ",".join(["?"] * len(variables))
            clauses.append(f"variable IN ({placeholders})")
            params.extend(variables)
        if geoids:
            placeholders = ",".join(["?"] * len(geoids))
            clauses.append(f"geoid IN ({placeholders})")
            params.extend(str(g) for g in geoids)

        where = " AND ".join(clauses)
        sql = f"""
            SELECT geo_level, geoid, valid_date, variable, value, value_text,
                   source, source_version, known_at, revision
            FROM (
                SELECT *, row_number() OVER (
                           PARTITION BY fact_key ORDER BY revision DESC
                       ) AS rn
                FROM twin_state
                WHERE {where}
            ) t
            WHERE rn = 1
            ORDER BY geoid, valid_date, variable
        """
        return self.con.execute(sql, params).df()

    def snapshot(
        self,
        *,
        geo_level: str,
        valid_from: dt.date | str,
        valid_to: dt.date | str | None = None,
        variables: Sequence[str] | None = None,
        geoids: Sequence[str] | None = None,
        known_at: dt.datetime | None = None,
    ) -> EstadoGemelo:
        """Igual que `as_of`, pero devuelve el EstadoGemelo en formato ancho."""
        long = self.as_of(
            geo_level=geo_level,
            valid_from=valid_from,
            valid_to=valid_to,
            variables=variables,
            geoids=geoids,
            known_at=known_at,
        )
        vf = pd.to_datetime(valid_from).date()
        vt = pd.to_datetime(valid_to).date() if valid_to is not None else vf
        cutoff = known_at or now_utc()

        if long.empty:
            wide = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [pd.Index([], dtype=str), pd.DatetimeIndex([])],
                    names=["geoid", "valid_date"],
                )
            )
        else:
            # El eje temporal se normaliza a datetime64 (no a `datetime.date`)
            # porque las features de M2 -- ventana de retardos de 21 dias,
            # rachas sobre percentil, grados-dia acumulados -- se construyen con
            # operaciones rolling de pandas, que requieren un indice temporal.
            long = long.assign(valid_date=pd.to_datetime(long["valid_date"]))
            wide = long.pivot_table(
                index=["geoid", "valid_date"],
                columns="variable",
                values="value",
                aggfunc="first",
            )
            wide.columns.name = None
            wide = wide.sort_index()

        sources = sorted(long["source"].unique().tolist()) if not long.empty else []
        return EstadoGemelo(
            geo_level=geo_level,
            frame=wide,
            known_at=cutoff,
            valid_from=vf,
            valid_to=vt,
            meta={"sources": sources, "n_facts": int(len(long))},
        )

    # ---------------------------------------------------------------- utility
    def coverage(self) -> pd.DataFrame:
        """Cobertura y huecos por (fuente, nivel, variable). Entregable de M0."""
        return self.con.execute(
            """
            SELECT source, geo_level, variable,
                   count(DISTINCT geoid)     AS n_geoids,
                   min(valid_date)           AS first_date,
                   max(valid_date)           AS last_date,
                   count(*)                  AS n_rows,
                   sum(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS n_null,
                   max(revision)             AS max_revision
            FROM twin_state
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3
            """
        ).df()

    def history(
        self, *, geo_level: str, geoid: str, valid_date: dt.date | str, variable: str
    ) -> pd.DataFrame:
        """Todas las revisiones de un hecho. Util para auditar una correccion."""
        return self.con.execute(
            """
            SELECT revision, value, value_text, source, source_version, known_at, raw_hash
            FROM twin_state
            WHERE geo_level = ? AND geoid = ? AND valid_date = ? AND variable = ?
            ORDER BY revision
            """,
            [geo_level, str(geoid), pd.to_datetime(valid_date).date(), variable],
        ).df()
