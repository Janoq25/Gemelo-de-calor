"""Exportacion de artefactos para la interfaz.

La frontera entre el motor y cualquier visor es un archivo JSON, no un
servidor: Python escribe, el front lee. Evita montar FastAPI (que §12.2 del
plan desaconseja explicitamente por cronograma) y mantiene el visor utilizable
sin backend vivo.

Todo lo que sale de aqui lleva su procedencia: version de codigo, hashes de
los crudos y el corte bitemporal usado. Un numero sin procedencia no deberia
llegar nunca a una figura.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from xdt.storage import code_version
from xdt.twin.state import StateStore, now_utc

# FIPS de estado -> (codigo postal, nombre). Referencia estandar del Census.
STATE_FIPS: dict[str, tuple[str, str]] = {
    "01": ("AL", "Alabama"), "02": ("AK", "Alaska"), "04": ("AZ", "Arizona"),
    "05": ("AR", "Arkansas"), "06": ("CA", "California"), "08": ("CO", "Colorado"),
    "09": ("CT", "Connecticut"), "10": ("DE", "Delaware"),
    "11": ("DC", "District of Columbia"), "12": ("FL", "Florida"),
    "13": ("GA", "Georgia"), "15": ("HI", "Hawaii"), "16": ("ID", "Idaho"),
    "17": ("IL", "Illinois"), "18": ("IN", "Indiana"), "19": ("IA", "Iowa"),
    "20": ("KS", "Kansas"), "21": ("KY", "Kentucky"), "22": ("LA", "Louisiana"),
    "23": ("ME", "Maine"), "24": ("MD", "Maryland"), "25": ("MA", "Massachusetts"),
    "26": ("MI", "Michigan"), "27": ("MN", "Minnesota"), "28": ("MS", "Mississippi"),
    "29": ("MO", "Missouri"), "30": ("MT", "Montana"), "31": ("NE", "Nebraska"),
    "32": ("NV", "Nevada"), "33": ("NH", "New Hampshire"), "34": ("NJ", "New Jersey"),
    "35": ("NM", "New Mexico"), "36": ("NY", "New York"),
    "37": ("NC", "North Carolina"), "38": ("ND", "North Dakota"), "39": ("OH", "Ohio"),
    "40": ("OK", "Oklahoma"), "41": ("OR", "Oregon"), "42": ("PA", "Pennsylvania"),
    "44": ("RI", "Rhode Island"), "45": ("SC", "South Carolina"),
    "46": ("SD", "South Dakota"), "47": ("TN", "Tennessee"), "48": ("TX", "Texas"),
    "49": ("UT", "Utah"), "50": ("VT", "Vermont"), "51": ("VA", "Virginia"),
    "53": ("WA", "Washington"), "54": ("WV", "West Virginia"),
    "55": ("WI", "Wisconsin"), "56": ("WY", "Wyoming"),
}

# Estado de implementacion de las capas (§6). Se declara aqui para que la
# interfaz no pueda presentar como terminado algo que no lo esta.
LAYERS: list[dict[str, str]] = [
    {
        "id": "L0", "nombre": "Trazabilidad",
        "estado": "implementado",
        "detalle": "Cada prediccion se persiste con timestamp, version de modelo, "
                   "hash de entrada, semilla y el corte bitemporal usado.",
        "modulo": "twin/registry.py",
    },
    {
        "id": "L1", "nombre": "Estado del gemelo",
        "estado": "implementado",
        "detalle": "Snapshot bitemporal por (geografia, fecha). Consultable en "
                   "cualquier instante pasado. Las revisiones no sobrescriben.",
        "modulo": "twin/state.py",
    },
    {
        "id": "L2", "nombre": "Sincronizacion",
        "estado": "parcial",
        "detalle": "Conector EPHT operativo con cache inmutable y reintentos. "
                   "Faltan gridMET, EAGLE-I, SVI, PLACES, LACE, Landsat, NWS.",
        "modulo": "ingest/",
    },
    {
        "id": "L3", "nombre": "Emulador predictivo + XAI",
        "estado": "pendiente",
        "detalle": "Escalera de 5 modelos y SHAP. Ningun numero de riesgo se "
                   "muestra en esta interfaz porque aun no existe modelo.",
        "modulo": "models/, explain/",
    },
    {
        "id": "L4", "nombre": "Motor de escenarios",
        "estado": "parcial",
        "detalle": "Escenarios componibles con la firma escenario(estado)->estado. "
                   "Perturban el estado; el efecto sobre el riesgo requiere L3.",
        "modulo": "twin/scenarios.py",
    },
    {
        "id": "L5", "nombre": "Calibracion y aprendizaje",
        "estado": "pendiente",
        "detalle": "Compara predicciones pasadas con observaciones posteriores. "
                   "El esquema (observation, calibration_run) ya lo soporta.",
        "modulo": "twin/calibrate.py",
    },
]


def build_payload(
    con: duckdb.DuckDBPyConnection,
    *,
    geo_level: str = "state",
    known_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """Reune todo lo que la interfaz necesita, con su procedencia."""
    store = StateStore(con)
    long = store.as_of(
        geo_level=geo_level, valid_from="1900-01-01", valid_to="2100-01-01",
        known_at=known_at,
    )

    variables = sorted(long["variable"].unique().tolist()) if not long.empty else []
    years = sorted({pd.Timestamp(d).year for d in long["valid_date"]}) if not long.empty else []

    # --- valores por estado y ano -------------------------------------------
    states: list[dict[str, Any]] = []
    for fips, (postal, name) in sorted(STATE_FIPS.items()):
        sub = long[long["geoid"] == fips]
        by_year = {
            str(pd.Timestamp(r["valid_date"]).year): (
                None if pd.isna(r["value"]) else float(r["value"])
            )
            for _, r in sub.iterrows()
        }
        states.append({
            "fips": fips, "postal": postal, "name": name,
            "valores": by_year,
            "reporta": {str(y): (str(y) in by_year) for y in years},
        })

    # --- cobertura por ano ---------------------------------------------------
    cobertura = []
    for y in years:
        vals = [
            float(r["value"])
            for _, r in long.iterrows()
            if pd.Timestamp(r["valid_date"]).year == y and not pd.isna(r["value"])
        ]
        cobertura.append({
            "anio": y,
            "n_estados": len(vals),
            "min": round(min(vals), 1) if vals else None,
            "max": round(max(vals), 1) if vals else None,
            "mediana": round(float(pd.Series(vals).median()), 1) if vals else None,
            "razon_max_min": round(max(vals) / min(vals), 1) if vals and min(vals) else None,
        })

    # --- historial de revisiones (evidencia de bitemporalidad) ---------------
    revisiones = con.execute(
        """
        SELECT geoid, valid_date, variable, revision, value, known_at, source
        FROM twin_state
        WHERE fact_key IN (SELECT fact_key FROM twin_state GROUP BY fact_key
                           HAVING max(revision) > 0)
        ORDER BY geoid, valid_date, revision
        """
    ).df()
    revisiones_out = [
        {
            "fips": r["geoid"],
            "postal": STATE_FIPS.get(r["geoid"], ("??", "?"))[0],
            "anio": pd.Timestamp(r["valid_date"]).year,
            "revision": int(r["revision"]),
            "valor": None if pd.isna(r["value"]) else float(r["value"]),
            "known_at": str(r["known_at"]),
        }
        for _, r in revisiones.iterrows()
    ]

    # --- procedencia ---------------------------------------------------------
    artifacts = con.execute(
        "SELECT raw_hash, source, resource, n_bytes, fetched_at FROM raw_artifact "
        "ORDER BY fetched_at"
    ).df()

    return {
        "generado_en": now_utc().isoformat(timespec="seconds"),
        "version_codigo": code_version(),
        "corte_bitemporal": (known_at or now_utc()).isoformat(timespec="seconds"),
        "geo_level": geo_level,
        "variables": variables,
        "anios": [str(y) for y in years],
        "estados": states,
        "cobertura": cobertura,
        "revisiones": revisiones_out,
        "capas": LAYERS,
        "procedencia": {
            "n_hechos": int(len(long)),
            "fuentes": sorted(long["source"].unique().tolist()) if not long.empty else [],
            "artefactos": [
                {
                    "hash": r["raw_hash"][:16],
                    "fuente": r["source"],
                    "recurso": r["resource"],
                    "bytes": int(r["n_bytes"]),
                    "descargado": str(r["fetched_at"])[:19],
                }
                for _, r in artifacts.iterrows()
            ],
        },
    }


def write_payload(
    con: duckdb.DuckDBPyConnection, out: str | Path, **kwargs: Any
) -> Path:
    payload = build_payload(con, **kwargs)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
