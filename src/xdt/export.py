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
    "01": ("AL", "Alabama"),
    "02": ("AK", "Alaska"),
    "04": ("AZ", "Arizona"),
    "05": ("AR", "Arkansas"),
    "06": ("CA", "California"),
    "08": ("CO", "Colorado"),
    "09": ("CT", "Connecticut"),
    "10": ("DE", "Delaware"),
    "11": ("DC", "District of Columbia"),
    "12": ("FL", "Florida"),
    "13": ("GA", "Georgia"),
    "15": ("HI", "Hawaii"),
    "16": ("ID", "Idaho"),
    "17": ("IL", "Illinois"),
    "18": ("IN", "Indiana"),
    "19": ("IA", "Iowa"),
    "20": ("KS", "Kansas"),
    "21": ("KY", "Kentucky"),
    "22": ("LA", "Louisiana"),
    "23": ("ME", "Maine"),
    "24": ("MD", "Maryland"),
    "25": ("MA", "Massachusetts"),
    "26": ("MI", "Michigan"),
    "27": ("MN", "Minnesota"),
    "28": ("MS", "Mississippi"),
    "29": ("MO", "Missouri"),
    "30": ("MT", "Montana"),
    "31": ("NE", "Nebraska"),
    "32": ("NV", "Nevada"),
    "33": ("NH", "New Hampshire"),
    "34": ("NJ", "New Jersey"),
    "35": ("NM", "New Mexico"),
    "36": ("NY", "New York"),
    "37": ("NC", "North Carolina"),
    "38": ("ND", "North Dakota"),
    "39": ("OH", "Ohio"),
    "40": ("OK", "Oklahoma"),
    "41": ("OR", "Oregon"),
    "42": ("PA", "Pennsylvania"),
    "44": ("RI", "Rhode Island"),
    "45": ("SC", "South Carolina"),
    "46": ("SD", "South Dakota"),
    "47": ("TN", "Tennessee"),
    "48": ("TX", "Texas"),
    "49": ("UT", "Utah"),
    "50": ("VT", "Vermont"),
    "51": ("VA", "Virginia"),
    "53": ("WA", "Washington"),
    "54": ("WV", "West Virginia"),
    "55": ("WI", "Wisconsin"),
    "56": ("WY", "Wyoming"),
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


def _layers(
    con: duckdb.DuckDBPyConnection, model: dict[str, Any] | None, l5: dict[str, int] | None
) -> list[dict[str, str]]:
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
        "SELECT count(*), count(*) FILTER (WHERE revision > 0) FROM twin_state"
    ).fetchone()
    present = [
        r[0] for r in con.execute("SELECT DISTINCT source FROM twin_state ORDER BY 1").fetchall()
    ]
    missing = [x for x in PLANNED_SOURCES if x not in present]
    (n_cal,) = con.execute("SELECT count(*) FROM calibration_run").fetchone()

    out = [
        {
            "id": "L0",
            "nombre": "Trazabilidad",
            "modulo": "twin/registry.py",
            "estado": "implementado",
            "detalle": f"{_fmt(n_pred)} predicciones y {_fmt(n_scen)} predicciones de escenario "
            f"registradas, {_fmt(n_obs)} observaciones. Cada una con versión de "
            f"modelo, hash de entrada y corte bitemporal.",
        },
        {
            "id": "L1",
            "nombre": "Estado del gemelo",
            "modulo": "twin/state.py",
            "estado": "implementado",
            "detalle": f"{_fmt(n_facts)} hechos bitemporales de {len(present)} fuentes; "
            f"{_fmt(n_rev)} revisiones conservadas sin sobrescribir.",
        },
        {
            "id": "L2",
            "nombre": "Sincronización",
            "modulo": "ingest/",
            "estado": "implementado" if not missing else "parcial",
            "detalle": f"Conectores con datos: {', '.join(present) or 'ninguno'}. "
            + (f"Faltan: {', '.join(missing)}." if missing else ""),
        },
    ]
    if model:
        m = model["metricas"][model["modelo"]]
        recal = " Requiere recalibración." if model["seleccion"]["requiere_recalibracion"] else ""
        out.append(
            {
                "id": "L3",
                "nombre": "Emulador predictivo + XAI",
                "modulo": "models/, explain/",
                "estado": "parcial",
                "detalle": f"Modelo {model['modelo']} v{model['version']}, elegido entre "
                f"{len(model['metricas'])} por validación cruzada anidada "
                f"(AUC global {m['auc_global']:.3f}, dentro de región "
                f"{m['auc_region_media']:.3f}). SHAP por predicción. Parcial: un solo "
                f"año de desenlace y sin el nivel híbrido del plan.{recal}",
            }
        )
    else:
        out.append(
            {
                "id": "L3",
                "nombre": "Emulador predictivo + XAI",
                "modulo": "models/, explain/",
                "estado": "pendiente",
                "detalle": "Sin modelo desplegado: ejecutar xdt crisp.",
            }
        )
    out.append(
        {
            "id": "L4",
            "nombre": "Motor de escenarios",
            "modulo": "twin/scenarios.py",
            "estado": "parcial",
            "detalle": (
                f"{_fmt(n_scen)} predicciones de escenario (calentamiento +1/+2/+3 °C) "
                f"traducidas a riesgo por L3. El apagón aún no entra al modelo: "
                f"EAGLE-I solo cubre AZ y LA."
            )
            if n_scen
            else "Los escenarios perturban el estado; sin modelo desplegado no hay riesgo.",
        }
    )
    genuine = (l5 or {}).get("genuinas", 0)
    post = (l5 or {}).get("post_dicciones", 0)
    out.append(
        {
            "id": "L5",
            "nombre": "Calibración y aprendizaje",
            "modulo": "twin/calibrate.py",
            "estado": "implementado" if genuine else ("parcial" if post else "pendiente"),
            "detalle": f"{_fmt(genuine)} parejas genuinas predicción-observación, {_fmt(post)} "
            f"post-dicciones excluidas (emitidas después de conocer el desenlace), "
            f"{n_cal} corridas de calibración. La calibración genuina empieza "
            f"cuando el gemelo prediga antes de que EPHT publique.",
        }
    )
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
        st = riesgo.setdefault(postal, {k: [None] * len(dates) for k in [*scen_keys.values(), "y"]})
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
    return {**meta, "fechas": dates, "riesgo": riesgo, "porque": porque, "etiquetas": FEATURES}


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
        geo_level=geo_level,
        valid_from="1900-01-01",
        valid_to="2100-01-01",
        variables=[ANNUAL_VAR],
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
        states.append(
            {
                "fips": fips,
                "postal": postal,
                "name": name,
                "valores": by_year,
                "reporta": {str(y): (str(y) in by_year) for y in years},
            }
        )

    # --- cobertura por ano ---------------------------------------------------
    cobertura = []
    for y in years:
        vals = [
            float(r["value"])
            for _, r in long.iterrows()
            if pd.Timestamp(r["valid_date"]).year == y and not pd.isna(r["value"])
        ]
        cobertura.append(
            {
                "anio": y,
                "n_estados": len(vals),
                "min": round(min(vals), 1) if vals else None,
                "max": round(max(vals), 1) if vals else None,
                "mediana": round(float(pd.Series(vals).median()), 1) if vals else None,
                "razon_max_min": round(max(vals) / min(vals), 1) if vals and min(vals) else None,
            }
        )

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
    all_sources = [
        r[0] for r in con.execute("SELECT DISTINCT source FROM twin_state ORDER BY 1").fetchall()
    ]

    # --- modelo desplegado (L3) y calibracion (L5) ---------------------------
    meta = load_latest(Path(models_dir) if models_dir else get_settings().models_dir)
    modelo, l5 = None, None
    if meta:
        from xdt.twin.calibrate import Calibrator

        modelo = _model_section(con, meta)
        genuine, n_post = Calibrator(con).pairs(
            model_name=meta["modelo"], quantity=meta["cantidad"], model_version=meta["version"]
        )
        l5 = {"genuinas": int(len(genuine)), "post_dicciones": int(n_post)}
        modelo["l5"] = l5

    # --- simulaciones precomputadas del Copiloto LangGraph (L4/L3/XAI) -------
    from xdt.agent.langflow_export import get_rag_flow_definition
    from xdt.agent.tools import execute_scenario_simulation

    presets = [
        {
            "id": "az_compound",
            "title": "Arizona · Ola de calor (p95) + 20% apagón",
            "params": {
                "geo_level": "state",
                "geoids": ["04"],
                "heatwave_days": 5,
                "heatwave_percentile": 95.0,
                "blackout_frac": 0.20,
                "blackout_hours": 12.0,
            },
        },
        {
            "id": "la_compound",
            "title": "Louisiana · Ola de calor (+3°C) + 25% apagón",
            "params": {
                "geo_level": "state",
                "geoids": ["22"],
                "heatwave_days": 7,
                "heatwave_delta_c": 3.0,
                "blackout_frac": 0.25,
                "blackout_hours": 18.0,
            },
        },
        {
            "id": "az_cooling",
            "title": "Arizona · Mitigación con Centro de Enfriamiento (15%)",
            "params": {
                "geo_level": "state",
                "geoids": ["04"],
                "heatwave_days": 5,
                "heatwave_percentile": 95.0,
                "cooling_epsilon": 0.15,
            },
        },
        {
            "id": "tx_extreme",
            "title": "Texas · Evento Compuesto Extremo (+4°C + 35% corte)",
            "params": {
                "geo_level": "state",
                "geoids": ["48"],
                "heatwave_days": 7,
                "heatwave_delta_c": 4.0,
                "blackout_frac": 0.35,
                "blackout_hours": 24.0,
            },
        },
    ]
    copilot_sims = []
    for p in presets:
        sim = execute_scenario_simulation(p["params"], con)
        sim["id"] = p["id"]
        sim["title"] = p["title"]
        copilot_sims.append(sim)

    # --- Base de conocimiento documental para el chatbot RAG (LangFlow) -----
    rag_kb = [
        {
            "q": "¿De qué trata el modelo presentado y cuál es su objetivo general?",
            "tags": [
                "modelo", "trata", "presentado", "objetivo", "resumen", "proposito",
                "que es", "introduccion", "xai-dt-heat", "salud", "urgencias", "riesgo", "hri",
            ],
            "a": (
                "xai-dt-heat es un Gemelo Digital Explicable con Enfoque de Equidad "
                "Algorítmica diseñado para modelar, predecir y simular el riesgo de morbilidad "
                "por calor extremo en eventos compuestos (olas de calor concurrentes con "
                "fallas en la red eléctrica) en comunidades urbanas de EE.UU. Integra datos "
                "meteorológicos (gridMET), cortes de energía (EAGLE-I) y registros de salud "
                "pública (CDC EPHT) para anticipar picos de urgencias por golpe de calor (HRI) "
                "y evaluar intervenciones."
            ),
            "seccion": "Plan §1 y README.md",
        },
        {
            "q": "¿Cuál es el problema central que aborda el Gemelo Digital?",
            "tags": [
                "problema", "representa", "brecha", "evento compuesto", "apagon", "calor",
                "aire acondicionado", "justificacion", "falla", "red electrica", "hri",
            ],
            "a": (
                "El problema central radica en los eventos compuestos de calor extremo y "
                "apagones eléctricos. Los sistemas operativos actuales (CDC Heat & Health "
                "Index, CalHeatScore) son estáticos o ignoran la infraestructura eléctrica. "
                "Durante una ola de calor, un apagón anula el principal factor protector "
                "urbano: el aire acondicionado (AC), multiplicando exponencialmente el estrés "
                "térmico domiciliario y saturando los servicios hospitalarios de emergencias."
            ),
            "seccion": "Plan §2 (Planteamiento del Problema)",
        },
        {
            "q": "¿Por qué es un Gemelo Digital (L0–L5) y no un modelo tradicional de ML?",
            "tags": [
                "gemelo digital", "arquitectura", "l0", "l1", "l2", "l3", "l4", "l5",
                "diferencia", "ml tradicional", "capas", "por que",
            ],
            "a": (
                "A diferencia de un modelo offline estático, un Gemelo Digital mantiene "
                "sincronización viva y bidireccional con el territorio mediante 6 capas formales: "
                "L0 (Trazabilidad inmutable de predicciones y SHAP), L1 (Almacén de estado "
                "bitemporal sin fuga), L2 (Conectores de ingesta continua), L3 (Escalera de "
                "emuladores ML con XAI local), L4 (Motor de escenarios contrafactuales "
                "componibles) y L5 (Calibración retrospectiva continua frente a datos reales "
                "observados)."
            ),
            "seccion": "Plan §4 y AGENTS.md §2",
        },
        {
            "q": "¿Por qué el proyecto usa un diseño de Fusión en Dos Escalas (Etapas A, B y C)?",
            "tags": [
                "dos escalas", "fusion", "etapa a", "etapa b", "etapa c", "temporal",
                "espacial", "desagregacion", "zcta", "condado", "estado-dia",
            ],
            "a": (
                "Porque ninguna base de datos pública federal en EE.UU. contiene simultáneamente "
                "alta resolución espacial (código postal) y temporal (diaria) de morbilidad por "
                "calor. La Etapa A aprende la función temporal y retardos en panel estado-día "
                "(datos EPHT en población VA), la Etapa B estima modificadores de vulnerabilidad "
                "espacial (SVI, PLACES, LACE) a nivel de condado, y la Etapa C desagrega el "
                "riesgo a nivel ZCTA contrastándolo contra CDC HHI y CalHeatScore."
            ),
            "seccion": "Plan §5 (Unidad de Análisis)",
        },
        {
            "q": "¿Cuáles son las fuentes de datos del sistema y cómo se ingieren?",
            "tags": [
                "fuentes", "datos", "epht", "gridmet", "eagle-i", "svi", "places", "lace",
                "landsat", "ingesta", "conectores",
            ],
            "a": (
                "El gemelo integra múltiples fuentes heterogéneas procesadas por conectores L2: "
                "CDC EPHT (tasas de urgencia y mortalidad por calor extremo, medidas 440, 1385, "
                "1238), gridMET (temperatura y humedad diaria a 4 km vía OPeNDAP), EAGLE-I "
                "(clientes sin luz cada 15 min por condado), CDC SVI (vulnerabilidad social), "
                "CDC PLACES (comorbilidades de salud), Census LACE (aire acondicionado) y "
                "Landsat (temperatura superficial)."
            ),
            "seccion": "Plan §7 (Fuentes de Datos)",
        },
        {
            "q": "¿Qué pruebas estadísticas robustas validan la superioridad del modelo?",
            "tags": [
                "pruebas estadisticas", "estadistica", "delong", "bootstrap", "spiegelhalter",
                "nadeau-bengio", "diebold-mariano", "validacion", "significancia", "auc",
            ],
            "a": (
                "El módulo xdt.validate.stats implementa: DeLong con corrección Holm para "
                "contrastar curvas ROC/AUC entre modelos no anidados; Block Bootstrap para "
                "generar intervalos de confianza preservando autocorrelación temporal; "
                "Spiegelhalter Z-test para calibración de probabilidades predichas; "
                "Nadeau-Bengio para corregir la varianza en validación cruzada repetida; y "
                "Diebold-Mariano con varianza HAC para series temporales."
            ),
            "seccion": "Plan §10 y src/xdt/validate/stats.py",
        },
        {
            "q": (
                "¿Cómo se modela la interacción supra-aditiva entre calor y apagones "
                "(Hipótesis H2)?"
            ),
            "tags": [
                "interaccion", "calor", "apagon", "corte de luz", "h2", "reri",
                "supra-aditiva", "riesgo relativo", "sinergia",
            ],
            "a": (
                "Se evalúa si la concurrencia de calor extremo y corte eléctrico genera un "
                "riesgo mayor a la suma de sus riesgos individuales. Se cuantifica formalmente "
                "con el exceso relativo de riesgo por interacción: "
                "RERI = RR_11 - RR_10 - RR_01 + 1. Cuando RERI > 0, se confirma que el colapso "
                "de la red eléctrica anula la amortiguación del aire acondicionado, disparando "
                "la morbilidad comunitaria."
            ),
            "seccion": "Plan §3 (H2) y §9.2",
        },
        {
            "q": "¿Cómo se implementa la Explicabilidad Algorítmica (XAI) con SHAP?",
            "tags": [
                "xai", "explicabilidad", "shap", "shapley", "atribucion", "interpretacion",
                "importancia", "factores",
            ],
            "a": (
                "Para cada predicción individual generada en L3, el emulador calcula valores "
                "SHAP (SHapley Additive exPlanations) que descomponen aditivamente la "
                "contribución de cada factor (carga térmica acumulada, duración de la ola de "
                "calor, corte de luz y mitigación). Los vectores SHAP se persisten en L0 "
                "(prediction_registry) para auditoría clínica y explicaciones en la consola."
            ),
            "seccion": "Plan §11 y src/xdt/explain/",
        },
        {
            "q": "¿Cómo se realiza la Auditoría de Equidad Algorítmica (Fairness)?",
            "tags": [
                "equidad", "fairness", "svi", "justicia", "paridad", "equalized odds",
                "sesgo", "vulnerabilidad", "h3",
            ],
            "a": (
                "La hipótesis H3 audita que el gemelo digital no perjudique a comunidades "
                "vulnerables. Se evalúan métricas de equidad (Equalized Odds, paridad demográfica "
                "y balance de errores) desagregadas por quintiles de vulnerabilidad social "
                "(CDC SVI) con remuestreo Bootstrap, garantizando alta sensibilidad y exactitud "
                "en zonas históricamente desfavorecidas."
            ),
            "seccion": "Plan §3 (H3) y AGENTS.md §4",
        },
        {
            "q": "¿Qué significa que el estado sea Bitemporal (valid_date y known_at)?",
            "tags": [
                "bitemporal", "valid_date", "known_at", "viaje en el tiempo", "fuga",
                "leakage", "revisiones", "duckdb", "tiempo",
            ],
            "a": (
                "Cada registro en el almacén L1 cuenta con dos marcas de tiempo: valid_date "
                "(cuándo ocurrió el hecho en el mundo real) y known_at (cuándo fue registrado en "
                "la base del gemelo). Esto permite viajar en el tiempo mediante consultas "
                "'as_of(known_at=...)', evaluando modelos únicamente con la información "
                "disponible en el momento histórico y eliminando la fuga retrospectiva "
                "(data leakage)."
            ),
            "seccion": "Plan §6 y src/xdt/twin/state.py",
        },
        {
            "q": "¿Qué es el Motor de Escenarios Contrafactuales (L4) y cómo funciona?",
            "tags": [
                "escenarios", "contrafactual", "l4", "operadores", "simulacion", "what if",
                "calentamiento", "enfriamiento", "composicion",
            ],
            "a": (
                "En L4, los escenarios son transformaciones funcionales e inmutables del estado "
                "que se componen mediante el operador pipe (|). Por ejemplo: "
                "'ola_de_calor(delta_c=3.0) | apagon(frac=0.25) | enfriamiento(eps=0.15)'. Esto "
                "permite simular escenarios climáticos extremos y evaluar el impacto protector "
                "de políticas públicas preservando el linaje de auditoría."
            ),
            "seccion": "Plan §4.4 y src/xdt/twin/scenarios.py",
        },
        {
            "q": "¿Por qué las predicciones del sistema aplican una 'Salvaguarda Causal'?",
            "tags": [
                "salvaguarda", "causal", "etica", "rigor", "responsabilidad", "causalidad",
                "epidemiologia", "limitacion",
            ],
            "a": (
                "Siguiendo las directrices éticas del plan (§15.6), las salidas del gemelo son "
                "proyecciones contrafactuales del modelo estadístico para soporte a la toma de "
                "decisiones en salud pública y emergencias. El sistema no afirma causalidad "
                "determinista individual ni sustituye el criterio epidemiológico y clínico experto."
            ),
            "seccion": "Plan §15.6 (Salvaguarda Causal)",
        },
        {
            "q": "¿Cuál es la Escalera de Modelos (Model Ladder) en L3?",
            "tags": [
                "escalera", "model ladder", "modelos", "lightgbm", "random forest",
                "regresion logistica", "umbral", "benchmark", "cv",
            ],
            "a": (
                "Para garantizar rigor metodológico, L3 implementa una escalera de complejidad "
                "creciente: Nivel 0 (Umbral meteorológico NWS), Nivel 1 (Regresión Logística con "
                "splines), Nivel 2 (Random Forest) y Nivel 3 (LightGBM GBDT). Los modelos se "
                "contrastan con validación cruzada anidada por bloques espacio-temporales y el "
                "mejor modelo se despliega en producción."
            ),
            "seccion": "Plan §9 y src/xdt/models/ladder.py",
        },
        {
            "q": "¿Cómo funcionan los agentes LangGraph y LangFlow integrados?",
            "tags": [
                "agentes", "langgraph", "langflow", "copiloto", "rag", "chat", "grafo",
                "nodos", "inteligencia artificial",
            ],
            "a": (
                "El gemelo integra dos componentes agénticos: 1) LangGraph Copilot "
                "(xdt.agent.graph), un grafo con 4 nodos (parse -> simulate -> explain -> "
                "synthesize) que interpreta lenguaje natural, ejecuta simulaciones L4, extrae "
                "XAI y genera reportes; y 2) LangFlow RAG Assistant (gemelo_rag_flow.json), un "
                "flujo documental exportable que indexa la documentación del plan para responder "
                "consultas técnicas con citas."
            ),
            "seccion": "AGENTS.md §2 y src/xdt/agent/",
        },
        {
            "q": "¿Cómo se ejecuta y prueba el Gemelo Digital desde la terminal (CLI)?",
            "tags": [
                "ejecutar", "comandos", "cli", "instrucciones", "como usar", "pasos",
                "terminal", "consola", "ingest", "crisp",
            ],
            "a": (
                "Las operaciones principales se realizan con uv run xdt: 1) Ingesta: "
                "'xdt ingest epht', 'xdt ingest gridmet', 'xdt ingest eaglei'. 2) Pipeline "
                "CRISP-DM y entrenamiento: 'xdt crisp'. 3) Copiloto interactivo: "
                "'xdt agent chat' o 'xdt agent copilot <consulta>'. 4) Exportar a la consola web: "
                "'xdt export' (y abrir app/consola.html)."
            ),
            "seccion": "README.md y src/xdt/cli.py",
        },
        {
            "q": "¿Cómo se modelan los Centros de Enfriamiento y su impacto mitigador?",
            "tags": [
                "enfriamiento", "centros de enfriamiento", "mitigacion", "refugio climatico",
                "intervencion", "resiliencia", "cooling",
            ],
            "a": (
                "El operador contrafactual 'enfriamiento(epsilon=0.15)' simula la apertura de "
                "centros comunitarios de enfriamiento con respaldo eléctrico durante olas de calor "
                "y apagones. Modela la absorción de población vulnerable y cuantifica la reducción "
                "neta en la probabilidad de urgencia médica por golpe de calor."
            ),
            "seccion": "Plan §4.4 y src/xdt/twin/scenarios.py",
        },
    ]

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
        "agentes": {
            "langgraph_simulations": copilot_sims,
            "langflow_flow": get_rag_flow_definition(),
            "rag_kb": rag_kb,
        },
        "procedencia": {
            "n_hechos": int(n_facts),
            "fuentes": all_sources,
            "por_fuente": [
                {
                    "fuente": r["source"],
                    "artefactos": int(r["n"]),
                    "bytes": int(r["bytes"]),
                    "ultima": str(r["ultima"])[:19],
                }
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
    con: duckdb.DuckDBPyConnection,
    out: str | Path,
    *,
    html: str | Path | None = None,
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
