"""L0 · Registro de predicciones y observaciones (§6 del plan).

"Cada prediccion se persiste con timestamp, version de modelo, hash de datos de
entrada y semilla. Auditoria retrospectiva total."

Este modulo es el que hace posible L5: la calibracion retrospectiva no es mas
que reunir predicciones emitidas hace tiempo con las observaciones que
llegaron despues, y esa reunion solo puede hacerse si en su momento se guardo
lo suficiente para reconstruir el contexto de la prediccion.

La regla no negociable: una prediccion se registra con el `state_known_at` que
uso. Sin ese campo no se puede demostrar que no hubo fuga de informacion.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb
import pandas as pd

from xdt.hashing import stable_id
from xdt.storage import code_version
from xdt.twin.state import now_utc


@dataclass(frozen=True)
class Prediction:
    """Una prediccion emitida, con todo lo necesario para reproducirla."""

    geo_level: str
    geoid: str
    target_date: dt.date
    quantity: str
    value: float
    model_name: str
    model_version: str
    input_hash: str
    state_known_at: dt.datetime | None = None
    issued_at: dt.datetime | None = None
    horizon_days: int | None = None
    value_lower: float | None = None
    value_upper: float | None = None
    scenario_id: str | None = None
    seed: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def prediction_id(self) -> str:
        """Determinista: la misma prediccion emitida dos veces no se duplica."""
        return stable_id(
            self.geo_level,
            self.geoid,
            self.target_date,
            self.quantity,
            self.model_name,
            self.model_version,
            self.input_hash,
            self.scenario_id,
            self.seed,
        )


class PredictionRegistry:
    """Escritura y consulta del registro de predicciones."""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    # ------------------------------------------------------------------ write
    def log(self, predictions: Sequence[Prediction]) -> int:
        """Registra predicciones. Idempotente por `prediction_id`."""
        if not predictions:
            return 0

        now = now_utc()
        cv = code_version()
        rows = []
        for p in predictions:
            d = asdict(p)
            rows.append(
                {
                    "prediction_id": p.prediction_id(),
                    "issued_at": p.issued_at or now,
                    "geo_level": p.geo_level,
                    "geoid": str(p.geoid),
                    "target_date": pd.to_datetime(p.target_date).date(),
                    "horizon_days": p.horizon_days,
                    "quantity": p.quantity,
                    "value": float(p.value),
                    "value_lower": p.value_lower,
                    "value_upper": p.value_upper,
                    "model_name": p.model_name,
                    "model_version": p.model_version,
                    "code_version": cv,
                    "input_hash": p.input_hash,
                    "state_known_at": p.state_known_at,
                    "scenario_id": p.scenario_id,
                    "seed": p.seed,
                    "extras_json": json.dumps(d["extras"], sort_keys=True, default=str),
                }
            )

        df = pd.DataFrame(rows).drop_duplicates(subset=["prediction_id"], keep="last")
        self.con.register("_preds", df)
        try:
            (n,) = self.con.execute(
                """
                SELECT count(*) FROM _preds p
                WHERE NOT EXISTS (
                    SELECT 1 FROM prediction_registry r
                    WHERE r.prediction_id = p.prediction_id
                )
                """
            ).fetchone()
            self.con.execute(
                """
                INSERT INTO prediction_registry
                SELECT p.prediction_id, p.issued_at, p.geo_level, p.geoid, p.target_date,
                       p.horizon_days, p.quantity, p.value, p.value_lower, p.value_upper,
                       p.model_name, p.model_version, p.code_version, p.input_hash,
                       p.state_known_at, p.scenario_id, p.seed, p.extras_json
                FROM _preds p
                WHERE NOT EXISTS (
                    SELECT 1 FROM prediction_registry r
                    WHERE r.prediction_id = p.prediction_id
                )
                """
            )
        finally:
            self.con.unregister("_preds")
        return int(n)

    def log_frame(
        self,
        df: pd.DataFrame,
        *,
        model_name: str,
        model_version: str,
        quantity: str,
        input_hash: str,
        state_known_at: dt.datetime | None = None,
        scenario_id: str | None = None,
        seed: int | None = None,
        geo_level: str | None = None,
    ) -> int:
        """Atajo: registra un DataFrame con columnas geoid/target_date/value."""
        needed = {"geoid", "target_date", "value"}
        if not needed.issubset(df.columns):
            raise ValueError(f"faltan columnas {needed - set(df.columns)}")
        level = geo_level or df.get("geo_level", pd.Series(dtype=str)).iloc[0]
        preds = [
            Prediction(
                geo_level=r.get("geo_level", level),
                geoid=str(r["geoid"]),
                target_date=pd.to_datetime(r["target_date"]).date(),
                quantity=quantity,
                value=float(r["value"]),
                model_name=model_name,
                model_version=model_version,
                input_hash=input_hash,
                state_known_at=state_known_at,
                scenario_id=scenario_id,
                seed=seed,
                value_lower=r.get("value_lower"),
                value_upper=r.get("value_upper"),
            )
            for _, r in df.iterrows()
        ]
        return self.log(preds)

    # ------------------------------------------------------------------- read
    def get(
        self,
        *,
        model_name: str | None = None,
        quantity: str | None = None,
        target_from: dt.date | str | None = None,
        target_to: dt.date | str | None = None,
        issued_before: dt.datetime | None = None,
        scenario_id: str | None = None,
        baseline_only: bool = False,
    ) -> pd.DataFrame:
        """Recupera predicciones. `issued_before` permite auditar un corte pasado."""
        clauses: list[str] = []
        params: list[Any] = []
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        if quantity:
            clauses.append("quantity = ?")
            params.append(quantity)
        if target_from:
            clauses.append("target_date >= ?")
            params.append(pd.to_datetime(target_from).date())
        if target_to:
            clauses.append("target_date <= ?")
            params.append(pd.to_datetime(target_to).date())
        if issued_before:
            clauses.append("issued_at < ?")
            params.append(issued_before)
        if scenario_id:
            clauses.append("scenario_id = ?")
            params.append(scenario_id)
        if baseline_only:
            clauses.append("scenario_id IS NULL")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.con.execute(
            f"SELECT * FROM prediction_registry {where} ORDER BY target_date, geoid", params
        ).df()

    def audit(self, prediction_id: str) -> dict[str, Any] | None:
        """Devuelve todo lo registrado sobre una prediccion concreta."""
        df = self.con.execute(
            "SELECT * FROM prediction_registry WHERE prediction_id = ?", [prediction_id]
        ).df()
        if df.empty:
            return None
        rec = df.iloc[0].to_dict()
        rec["extras"] = json.loads(rec.pop("extras_json") or "{}")
        return rec


class ObservationStore:
    """Observaciones reales que llegan despues; insumo de L5.

    Trata la supresion de celdas como *censura por intervalo*, no como dato
    faltante (§14 R2 del plan). Una celda suprimida por tener <10 casos no es
    un hueco: es la informacion "el valor esta entre 0 y 10", y descartarla
    sesgaria la calibracion hacia las zonas de mayor incidencia.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def put(
        self,
        obs: pd.DataFrame,
        *,
        source: str,
        known_at: dt.datetime | None = None,
    ) -> dict[str, int]:
        """Inserta observaciones con versionado por revision (SCD tipo 2).

        Columnas requeridas: geo_level, geoid, target_date, quantity, value.
        Opcionales: censored, censor_lo, censor_hi.
        """
        required = {"geo_level", "geoid", "target_date", "quantity"}
        missing = required - set(obs.columns)
        if missing:
            raise ValueError(f"faltan columnas obligatorias: {sorted(missing)}")

        df = obs.copy()
        for col, default in (("value", pd.NA), ("censored", False),
                             ("censor_lo", pd.NA), ("censor_hi", pd.NA)):
            if col not in df.columns:
                df[col] = default

        df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
        df["geoid"] = df["geoid"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["censor_lo"] = pd.to_numeric(df["censor_lo"], errors="coerce")
        df["censor_hi"] = pd.to_numeric(df["censor_hi"], errors="coerce")
        df["censored"] = df["censored"].fillna(False).astype(bool)
        df["source"] = source
        df["known_at"] = known_at or now_utc()
        df["obs_key"] = [
            stable_id(gl, gi, td, q, source)
            for gl, gi, td, q in zip(
                df["geo_level"], df["geoid"], df["target_date"], df["quantity"], strict=True
            )
        ]
        n_in = len(df)
        df = df.drop_duplicates(subset=["obs_key"], keep="last")

        self.con.register("_obs_in", df)
        try:
            self.con.execute(
                """
                CREATE OR REPLACE TEMP VIEW _obs_latest AS
                SELECT obs_key, value, censored, revision FROM (
                    SELECT obs_key, value, censored, revision,
                           row_number() OVER (PARTITION BY obs_key ORDER BY revision DESC) AS rn
                    FROM observation
                    WHERE obs_key IN (SELECT obs_key FROM _obs_in)
                ) t WHERE rn = 1
                """
            )
            self.con.execute(
                """
                CREATE OR REPLACE TEMP VIEW _obs_write AS
                SELECT i.*, COALESCE(l.revision + 1, 0) AS new_revision
                FROM _obs_in i
                LEFT JOIN _obs_latest l USING (obs_key)
                WHERE l.obs_key IS NULL
                   OR l.value IS DISTINCT FROM i.value
                   OR l.censored IS DISTINCT FROM i.censored
                """
            )
            (n_write,) = self.con.execute("SELECT count(*) FROM _obs_write").fetchone()
            self.con.execute(
                """
                INSERT INTO observation
                    (obs_key, geo_level, geoid, target_date, quantity, value,
                     censored, censor_lo, censor_hi, source, known_at, revision)
                SELECT obs_key, geo_level, geoid, target_date, quantity, value,
                       censored, censor_lo, censor_hi, source, known_at, new_revision
                FROM _obs_write
                """
            )
        finally:
            self.con.unregister("_obs_in")

        return {"received": n_in, "written": int(n_write), "unchanged": len(df) - int(n_write)}

    def latest(
        self,
        *,
        quantity: str | None = None,
        geo_level: str | None = None,
        known_at: dt.datetime | None = None,
    ) -> pd.DataFrame:
        clauses = ["known_at <= ?"]
        params: list[Any] = [known_at or now_utc()]
        if quantity:
            clauses.append("quantity = ?")
            params.append(quantity)
        if geo_level:
            clauses.append("geo_level = ?")
            params.append(geo_level)
        where = " AND ".join(clauses)
        return self.con.execute(
            f"""
            SELECT geo_level, geoid, target_date, quantity, value,
                   censored, censor_lo, censor_hi, source, known_at, revision
            FROM (
                SELECT *, row_number() OVER (
                           PARTITION BY obs_key ORDER BY revision DESC) AS rn
                FROM observation WHERE {where}
            ) t WHERE rn = 1
            ORDER BY geoid, target_date
            """,
            params,
        ).df()
