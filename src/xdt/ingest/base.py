"""Contrato base de los conectores (§7 M1: "un conector por fuente, idempotente").

Todo conector recorre tres etapas explicitas y separables:

    fetch()     red -> data/raw          (crudo, inmutable, direccionado por contenido)
    normalize() raw -> DataFrame largo   (esquema canonico de hechos)
    load()      DataFrame -> L1          (estado bitemporal)

La separacion importa por una razon practica: `normalize` puede reejecutarse
sobre el crudo cacheado sin tocar la red. Cuando un endpoint cambie de forma
(R6) o cuando se descubra un error de parseo, se corrige el parser y se
reprocesa el historico completo sin volver a descargar nada.
"""

from __future__ import annotations

import datetime as dt
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import duckdb
import pandas as pd

from xdt.config import Settings, get_settings
from xdt.ingest.http import CachedClient, RawArtifact
from xdt.twin.state import StateStore

# Esquema canonico que `normalize` debe producir.
FACT_SCHEMA = ("geo_level", "geoid", "valid_date", "variable", "value")


class Connector(ABC):
    """Base de todo conector de fuente."""

    #: nombre corto, usado como carpeta en data/raw y como `source` en L1
    name: str
    #: cadencia nominal de la fuente; documenta L2 (§6)
    cadence: str = "unknown"
    #: URL base del servicio
    base_url: str = ""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.client = CachedClient(
            source=self.name, settings=self.settings, base_url=self.base_url
        )

    # ------------------------------------------------------------- interfaz
    @abstractmethod
    def fetch(self, **params: Any) -> list[RawArtifact]:
        """Descarga (o lee de cache) los artefactos crudos que pide `params`."""

    @abstractmethod
    def normalize(self, artifacts: Iterable[RawArtifact]) -> pd.DataFrame:
        """Convierte crudos en hechos con el esquema FACT_SCHEMA."""

    def source_version(self, artifacts: Iterable[RawArtifact]) -> str | None:
        """Version/anada del dataset. Se guarda con cada hecho en L1."""
        return None

    # -------------------------------------------------------------- pipeline
    def run(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        known_at: dt.datetime | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """fetch -> normalize -> registrar crudos -> cargar en L1."""
        artifacts = self.fetch(**params)
        facts = self.normalize(artifacts)
        self.validate_facts(facts)

        self.register_artifacts(con, artifacts)

        store = StateStore(con)
        raw_hash = artifacts[0].raw_hash if len(artifacts) == 1 else None
        stats = store.put_facts(
            facts,
            source=self.name,
            source_version=self.source_version(artifacts),
            known_at=known_at,
            raw_hash=raw_hash,
        )
        return {
            "connector": self.name,
            "artifacts": len(artifacts),
            "from_cache": sum(a.from_cache for a in artifacts),
            "bytes": sum(a.n_bytes for a in artifacts),
            **stats,
        }

    # ------------------------------------------------------------- utilidades
    def register_artifacts(
        self, con: duckdb.DuckDBPyConnection, artifacts: Iterable[RawArtifact]
    ) -> int:
        """Indexa en DuckDB los crudos descargados. Idempotente por raw_hash."""
        rows = [
            {
                "raw_hash": a.raw_hash,
                "source": a.source,
                "resource": a.resource,
                "params_json": json.dumps(a.params, sort_keys=True, default=str),
                "path": str(a.path),
                "n_bytes": a.n_bytes,
                "content_type": a.content_type,
                "fetched_at": a.fetched_at,
                "tool_version": self.__class__.__name__,
            }
            for a in artifacts
        ]
        if not rows:
            return 0
        df = pd.DataFrame(rows).drop_duplicates(subset=["raw_hash"], keep="last")
        con.register("_arts", df)
        try:
            con.execute(
                """
                INSERT INTO raw_artifact
                SELECT a.raw_hash, a.source, a.resource, a.params_json, a.path,
                       a.n_bytes, a.content_type, a.fetched_at, a.tool_version
                FROM _arts a
                WHERE NOT EXISTS (
                    SELECT 1 FROM raw_artifact r WHERE r.raw_hash = a.raw_hash
                )
                """
            )
        finally:
            con.unregister("_arts")
        return len(df)

    @staticmethod
    def validate_facts(facts: pd.DataFrame) -> None:
        """Verifica el esquema canonico antes de tocar el estado.

        Un conector que devuelve basura debe fallar aqui, no tres capas mas
        abajo cuando el modelo produzca un numero raro.
        """
        missing = [c for c in FACT_SCHEMA if c not in facts.columns]
        if missing:
            raise ValueError(f"normalize() no produjo las columnas {missing}")
        if facts.empty:
            return
        if facts["geoid"].isna().any():
            raise ValueError("hay geoid nulos tras normalize()")
        if facts["valid_date"].isna().any():
            raise ValueError("hay valid_date nulos tras normalize()")
        dup = facts.duplicated(subset=["geo_level", "geoid", "valid_date", "variable"]).sum()
        if dup:
            raise ValueError(
                f"normalize() produjo {dup} hechos duplicados para la misma "
                "clave (geo_level, geoid, valid_date, variable)"
            )

    def close(self) -> None:
        self.client.close()
