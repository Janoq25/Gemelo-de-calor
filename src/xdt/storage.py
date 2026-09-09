"""Capa de almacenamiento: DuckDB + Parquet.

§12.2 del plan: a esta escala DuckDB es mas rapido que PostGIS, no requiere
servidor y cabe en un archivo. Este modulo posee el esquema y nada mas; la
logica de dominio vive en `xdt.twin`.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import duckdb

from xdt.config import get_settings

SCHEMA_VERSION = 1

# --- DDL -------------------------------------------------------------------
# Se ejecuta en cada apertura; todo es CREATE ... IF NOT EXISTS.
_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    schema_version INTEGER NOT NULL,
    applied_at     TIMESTAMP NOT NULL
);

-- L2 · trazabilidad de ingesta: toda descarga cruda queda registrada.
-- data/raw es inmutable; esta tabla es su indice.
CREATE TABLE IF NOT EXISTS raw_artifact (
    raw_hash     VARCHAR PRIMARY KEY,
    source       VARCHAR   NOT NULL,
    resource     VARCHAR   NOT NULL,
    params_json  VARCHAR,
    path         VARCHAR   NOT NULL,
    n_bytes      BIGINT,
    content_type VARCHAR,
    fetched_at   TIMESTAMP NOT NULL,
    tool_version VARCHAR
);

-- L1 · ESTADO DEL GEMELO, bitemporal.
--   valid_date : la fecha del mundo real que el hecho describe
--   known_at   : el instante en que el gemelo se entero del hecho
-- Dos ejes, no uno. Es lo que permite preguntar "que creia el gemelo el
-- 1 de julio de 2023?" y no solo "que sabemos hoy sobre el 1 de julio".
-- Sin esto, evaluar una prediccion pasada contra datos revisados despues
-- seria fuga de informacion.
CREATE TABLE IF NOT EXISTS twin_state (
    fact_key       VARCHAR   NOT NULL,
    geo_level      VARCHAR   NOT NULL,
    geoid          VARCHAR   NOT NULL,
    valid_date     DATE      NOT NULL,
    variable       VARCHAR   NOT NULL,
    value          DOUBLE,
    value_text     VARCHAR,
    source         VARCHAR   NOT NULL,
    source_version VARCHAR,
    known_at       TIMESTAMP NOT NULL,
    revision       INTEGER   NOT NULL,
    raw_hash       VARCHAR,
    PRIMARY KEY (fact_key, revision)
);

-- L0 · REGISTRO DE PREDICCIONES.
-- Cada prediccion con timestamp, version de modelo, hash de entrada y semilla.
CREATE TABLE IF NOT EXISTS prediction_registry (
    prediction_id  VARCHAR   PRIMARY KEY,
    issued_at      TIMESTAMP NOT NULL,
    geo_level      VARCHAR   NOT NULL,
    geoid          VARCHAR   NOT NULL,
    target_date    DATE      NOT NULL,
    horizon_days   INTEGER,
    quantity       VARCHAR   NOT NULL,
    value          DOUBLE    NOT NULL,
    value_lower    DOUBLE,
    value_upper    DOUBLE,
    model_name     VARCHAR   NOT NULL,
    model_version  VARCHAR   NOT NULL,
    code_version   VARCHAR,
    input_hash     VARCHAR   NOT NULL,
    state_known_at TIMESTAMP,
    scenario_id    VARCHAR,
    seed           BIGINT,
    extras_json    VARCHAR
);

-- L4 · escenarios ejecutados (procedencia de cada contrafactual).
CREATE TABLE IF NOT EXISTS scenario_run (
    scenario_id   VARCHAR   PRIMARY KEY,
    name          VARCHAR   NOT NULL,
    spec_json     VARCHAR   NOT NULL,
    base_state_id VARCHAR,
    created_at    TIMESTAMP NOT NULL
);

-- L5 · observaciones que llegan despues; insumo de la calibracion retrospectiva.
CREATE TABLE IF NOT EXISTS observation (
    obs_key     VARCHAR   NOT NULL,
    geo_level   VARCHAR   NOT NULL,
    geoid       VARCHAR   NOT NULL,
    target_date DATE      NOT NULL,
    quantity    VARCHAR   NOT NULL,
    value       DOUBLE,
    censored    BOOLEAN   DEFAULT FALSE,  -- supresion de celdas (§14 R2)
    censor_lo   DOUBLE,
    censor_hi   DOUBLE,
    source      VARCHAR   NOT NULL,
    known_at    TIMESTAMP NOT NULL,
    revision    INTEGER   NOT NULL,
    PRIMARY KEY (obs_key, revision)
);

CREATE TABLE IF NOT EXISTS calibration_run (
    run_id        VARCHAR   PRIMARY KEY,
    ran_at        TIMESTAMP NOT NULL,
    model_name    VARCHAR   NOT NULL,
    model_version VARCHAR   NOT NULL,
    window_start  DATE,
    window_end    DATE,
    n_pairs       INTEGER,
    metrics_json  VARCHAR,
    drift_flag    BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_state_lookup
    ON twin_state (geo_level, variable, valid_date);
CREATE INDEX IF NOT EXISTS idx_state_geoid
    ON twin_state (geoid, valid_date);
CREATE INDEX IF NOT EXISTS idx_pred_target
    ON prediction_registry (geo_level, geoid, target_date);
CREATE INDEX IF NOT EXISTS idx_obs_target
    ON observation (geo_level, geoid, target_date, quantity);
"""


def _apply_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_DDL)
    (n,) = con.execute("SELECT count(*) FROM schema_meta").fetchone()
    if n == 0:
        con.execute(
            "INSERT INTO schema_meta VALUES (?, current_timestamp)", [SCHEMA_VERSION]
        )


def connect(path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Abre (y migra) la base del gemelo. `:memory:` es valido para tests."""
    settings = get_settings()
    target = path if path is not None else settings.twin_db_path
    if str(target) != ":memory:":
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target = str(target)
    con = duckdb.connect(target, read_only=read_only)
    if not read_only:
        _apply_schema(con)
    return con


@contextmanager
def session(
    path: str | Path | None = None, read_only: bool = False
) -> Iterator[duckdb.DuckDBPyConnection]:
    con = connect(path, read_only=read_only)
    try:
        yield con
    finally:
        con.close()


@lru_cache(maxsize=1)
def code_version() -> str:
    """SHA del commit actual, para el campo `code_version` de L0.

    Si el repo no esta inicializado o el arbol esta sucio, se marca para que
    una prediccion nunca aparente ser mas reproducible de lo que es.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unversioned"
