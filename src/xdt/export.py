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
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from xdt.config import get_settings
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

#: Medida anual que alimenta el mapa "quien reporta" (EPHT 440, ajustada por edad).
ANNUAL_VAR = "epht_m1_age_adj_rate"
#: Conectores que el plan exige en L2 (§4). La capa no esta completa sin todos.
PLANNED_SOURCES = ("epht", "gridmet", "eaglei", "svi", "places", "lace", "landsat", "nws")
N_RECENT_ARTIFACTS = 12


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def load_latest(models_dir: Path) -> dict[str, Any] | None:
    """Metadatos del ultimo modelo desplegado (`xdt crisp`). Sin dependencias de ML."""
    p = Path(models_dir) / "latest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _layers(con: duckdb.DuckDBPyConnection, model: dict[str, Any] | None,
            l5: dict[str, int] | None) -> list[dict[str, str]]:
    """Estado de las capas CALCULADO desde la base, no declarado a mano.

    Un texto fijo se desactualiza en cuanto el motor avanza (ya paso: la
    consola decia que faltaban conectores que existian). Aqui cada frase sale
    de contar lo que hay.
    """
    n_pred, n_scen = con.execute(
        "SELECT count(*) FILTER (WHERE scenario_id IS NULL), "
        "count(*) FILTER (WHERE scenario_id IS NOT NULL) FROM prediction_registry"
    ).fetchone()
    (n_obs,) = con.execute("SELECT count(*) FROM observation").fetchone()
    n_facts, n_rev = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE revision > 0) FROM twin_state").fetchone()
    present = [r[0] for r in con.execute(
        "SELECT DISTINCT source FROM twin_state ORDER BY 1").fetchall()]
    missing = [x for x in PLANNED_SOURCES if x not in present]
    (n_cal,) = con.execute("SELECT count(*) FROM calibration_run").fetchone()

    out = [
        {"id": "L0", "nombre": "Trazabilidad", "modulo": "twin/registry.py",
         "estado": "implementado",
         "detalle": f"{_fmt(n_pred)} predicciones y {_fmt(n_scen)} predicciones de escenario "
                    f"registradas, {_fmt(n_obs)} observaciones. Cada una con versión de "
                    f"modelo, hash de entrada y corte bitemporal."},
        {"id": "L1", "nombre": "Estado del gemelo", "modulo": "twin/state.py",
         "estado": "implementado",
         "detalle": f"{_fmt(n_facts)} hechos bitemporales de {len(present)} fuentes; "
                    f"{_fmt(n_rev)} revisiones conservadas sin sobrescribir."},
        {"id": "L2", "nombre": "Sincronización", "modulo": "ingest/",
         "estado": "implementado" if not missing else "parcial",
         "detalle": f"Conectores con datos: {', '.join(present) or 'ninguno'}. "
                    + (f"Faltan: {', '.join(missing)}." if missing else "")},
    ]
    if model:
        m = model["metricas"][model["modelo"]]
        recal = " Requiere recalibración." if model["seleccion"]["requiere_recalibracion"] else ""
        out.append({
            "id": "L3", "nombre": "Emulador predictivo + XAI", "modulo": "models/, explain/",
            "estado": "parcial",
            "detalle": f"Modelo {model['modelo']} v{model['version']}, elegido entre "
                       f"{len(model['metricas'])} por validación cruzada anidada "
                       f"(AUC global {m['auc_global']:.3f}, dentro de región "
                       f"{m['auc_region_media']:.3f}). SHAP por predicción. Parcial: un solo "
                       f"año de desenlace y sin el nivel híbrido del plan.{recal}"})
    else:
        out.append({"id": "L3", "nombre": "Emulador predictivo + XAI",
                    "modulo": "models/, explain/", "estado": "pendiente",
                    "detalle": "Sin modelo desplegado: ejecutar xdt crisp."})
    out.append({
        "id": "L4", "nombre": "Motor de escenarios", "modulo": "twin/scenarios.py",
        "estado": "parcial",
        "detalle": (f"{_fmt(n_scen)} predicciones de escenario (calentamiento +1/+2/+3 °C) "
                    f"traducidas a riesgo por L3. El apagón aún no entra al modelo: "
                    f"EAGLE-I solo cubre AZ y LA.") if n_scen else
                   "Los escenarios perturban el estado; sin modelo desplegado no hay riesgo."})
    genuine = (l5 or {}).get("genuinas", 0)
    post = (l5 or {}).get("post_dicciones", 0)
    out.append({
        "id": "L5", "nombre": "Calibración y aprendizaje", "modulo": "twin/calibrate.py",
        "estado": "implementado" if genuine else ("parcial" if post else "pendiente"),
        "detalle": f"{_fmt(genuine)} parejas genuinas predicción-observación, {_fmt(post)} "
                   f"post-dicciones excluidas (emitidas después de conocer el desenlace), "
                   f"{n_cal} corridas de calibración. La calibración genuina empieza "
                   f"cuando el gemelo prediga antes de que EPHT publique."})
    return out


def _model_section(con: duckdb.DuckDBPyConnection, meta: dict[str, Any]) -> dict[str, Any]:
    """Riesgo, escenarios y explicacion por estado-dia, leidos de L0."""
    from xdt.features.panel import FEATURES  # solo pandas/numpy: no arrastra ML

    reg = con.execute(
        """
        SELECT geoid, target_date, scenario_id, value, extras_json
        FROM prediction_registry
        WHERE model_version = ? AND quantity = ?
        """,
        [meta["version"], meta["cantidad"]],
    ).df()
    obs = con.execute(
        """
        SELECT geoid, target_date, value FROM (
            SELECT *, row_number() OVER (PARTITION BY obs_key ORDER BY revision DESC) rn
            FROM observation WHERE quantity = ?
        ) WHERE rn = 1
        """,
        [meta["cantidad"]],
    ).df()
    dates = sorted({pd.Timestamp(d).date().isoformat() for d in reg["target_date"]})
    pos = {d: i for i, d in enumerate(dates)}
    scen_keys = {None: "p", **{s: f"s{i + 1}" for i, s in enumerate(meta["escenarios"])}}

    riesgo: dict[str, dict[str, list]] = {}
    porque: dict[str, list] = {}
    for r in reg.itertuples(index=False):
        postal = STATE_FIPS.get(r.geoid, (r.geoid,))[0]
        st = riesgo.setdefault(postal, {k: [None] * len(dates)
                                        for k in [*scen_keys.values(), "y"]})
        i = pos[pd.Timestamp(r.target_date).date().isoformat()]
        sid = None if pd.isna(r.scenario_id) else r.scenario_id
        st[scen_keys[sid]][i] = round(float(r.value), 4)
        if sid is None:
            ex = json.loads(r.extras_json or "{}")
            if "shap_top" in ex:
                porque.setdefault(postal, [None] * len(dates))[i] = ex["shap_top"]
    for r in obs.itertuples(index=False):
        postal = STATE_FIPS.get(r.geoid, (r.geoid,))[0]
        d = pd.Timestamp(r.target_date).date().isoformat()
        if postal in riesgo and d in pos:
            riesgo[postal]["y"][pos[d]] = int(r.value)
    return {**meta, "fechas": dates, "riesgo": riesgo, "porque": porque,
            "etiquetas": FEATURES}


def build_payload(
    con: duckdb.DuckDBPyConnection,
    *,
    geo_level: str = "state",
    known_at: dt.datetime | None = None,
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Reune todo lo que la interfaz necesita, con su procedencia."""
    store = StateStore(con)
    long = store.as_of(
        geo_level=geo_level, valid_from="1900-01-01", valid_to="2100-01-01",
        variables=[ANNUAL_VAR], known_at=known_at,
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
        WHERE variable = ? AND fact_key IN (
            SELECT fact_key FROM twin_state GROUP BY fact_key HAVING max(revision) > 0)
        ORDER BY geoid, valid_date, revision
        """,
        [ANNUAL_VAR],
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

    # --- procedencia: resumen por fuente + ultimas descargas -----------------
    by_source = con.execute(
        """
        SELECT source, count(*) AS n, sum(n_bytes) AS bytes, max(fetched_at) AS ultima
        FROM raw_artifact GROUP BY 1 ORDER BY 1
        """
    ).df()
    artifacts = con.execute(
        "SELECT raw_hash, source, resource, n_bytes, fetched_at FROM raw_artifact "
        f"ORDER BY fetched_at DESC LIMIT {N_RECENT_ARTIFACTS}"
    ).df()
    (n_facts,) = con.execute("SELECT count(*) FROM twin_state").fetchone()
    all_sources = [r[0] for r in con.execute(
        "SELECT DISTINCT source FROM twin_state ORDER BY 1").fetchall()]

    # --- modelo desplegado (L3) y calibracion (L5) ---------------------------
    meta = load_latest(Path(models_dir) if models_dir else get_settings().models_dir)
    modelo, l5 = None, None
    if meta:
        from xdt.twin.calibrate import Calibrator

        modelo = _model_section(con, meta)
        genuine, n_post = Calibrator(con).pairs(
            model_name=meta["modelo"], quantity=meta["cantidad"],
            model_version=meta["version"])
        l5 = {"genuinas": int(len(genuine)), "post_dicciones": int(n_post)}
        modelo["l5"] = l5

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
        "capas": _layers(con, meta, l5),
        "modelo": modelo,
        "procedencia": {
            "n_hechos": int(n_facts),
            "fuentes": all_sources,
            "por_fuente": [
                {"fuente": r["source"], "artefactos": int(r["n"]), "bytes": int(r["bytes"]),
                 "ultima": str(r["ultima"])[:19]}
                for _, r in by_source.iterrows()
            ],
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
    con: duckdb.DuckDBPyConnection, out: str | Path, *, html: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    payload = build_payload(con, **kwargs)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if html is not None:
        inject_html(payload, Path(html))
    return path


_PAYLOAD_LINE = re.compile(r"^const PAYLOAD = .*;$", re.MULTILINE)


def inject_html(payload: dict[str, Any], html: Path) -> None:
    """Reescribe la linea `const PAYLOAD = ...;` de la consola con datos frescos.

    La consola se abre como archivo local (sin servidor), asi que no puede
    hacer fetch del JSON: los datos viajan dentro del HTML. Antes se copiaban
    a mano y la consola quedaba atrasada respecto del motor.
    """
    text = html.read_text(encoding="utf-8")
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    body = body.replace("</", "<\\/")  # un "</script>" en el JSON cerraria el bloque
    new, n = _PAYLOAD_LINE.subn(lambda _: f"const PAYLOAD = {body};", text, count=1)
    if n != 1:
        raise ValueError(f"{html} no tiene una linea 'const PAYLOAD = ...;'")
    html.write_text(new, encoding="utf-8")
