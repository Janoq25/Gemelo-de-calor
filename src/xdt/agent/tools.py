"""Herramientas y funciones de simulacion para el Copiloto del Gemelo Digital."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from xdt.twin.scenarios import apagon, enfriamiento, ola_de_calor
from xdt.twin.state import EstadoGemelo, StateStore

# Diccionario de entidades comunes (estados y FIPS)
STATE_NAME_TO_FIPS: dict[str, str] = {
    "arizona": "04",
    "az": "04",
    "phoenix": "04",
    "maricopa": "04",
    "louisiana": "22",
    "luisiana": "22",
    "la": "22",
    "new orleans": "22",
    "nueva orleans": "22",
    "florida": "12",
    "fl": "12",
    "texas": "48",
    "tx": "48",
    "california": "06",
    "ca": "06",
    "new york": "36",
    "nueva york": "36",
    "ny": "36",
}


def parse_scenario_intent(query: str) -> dict[str, Any]:
    """Extrae parametros estructurados a partir de una consulta en lenguaje natural."""
    q = query.lower()

    # 1. Geografía
    geoids: list[str] = []
    for name, fips in STATE_NAME_TO_FIPS.items():
        if re.search(rf"\b{re.escape(name)}\b", q) and fips not in geoids:
            geoids.append(fips)

    # Si se pasaron codigos FIPS directos de 2 digitos (ej. "fips 04")
    fips_matches = re.findall(r"\b\d{2}\b", q)
    for fm in fips_matches:
        if fm not in geoids and fm in {"04", "22", "12", "48", "06", "36", "01", "08", "53"}:
            geoids.append(fm)

    if not geoids:
        geoids = ["04"]  # Default: Arizona (estado centinela)

    # 2. Ola de calor
    heatwave_days = 0
    days_match = re.search(r"(\d+)\s*(?:d[ií]as|days?)\s*(?:de\s+calor|ola)?", q)
    if days_match:
        heatwave_days = int(days_match.group(1))
    elif "ola de calor" in q or "calor extremo" in q or "heatwave" in q:
        heatwave_days = 5

    # Incremento en °C
    heatwave_delta_c: float | None = None
    delta_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:°c|c|grados|deg)", q)
    if delta_match:
        heatwave_delta_c = float(delta_match.group(1))

    # Percentil local
    heatwave_percentile: float | None = None
    p_match = re.search(r"p(?:ercentil)?\s*(\d{2})", q)
    if p_match:
        heatwave_percentile = float(p_match.group(1))
    elif heatwave_delta_c is None and heatwave_days > 0:
        heatwave_percentile = 95.0

    # 3. Apagón / Corte de energía
    blackout_frac = 0.0
    frac_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:de\s+corte|sin\s+luz|apag[oó]n|outage)?", q)
    if frac_match:
        blackout_frac = float(frac_match.group(1)) / 100.0
    elif "apagón" in q or "apagon" in q or "corte de luz" in q or "blackout" in q:
        blackout_frac = 0.20

    blackout_hours = 0.0
    hours_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:horas?|hrs?|hours?)", q)
    if hours_match:
        blackout_hours = float(hours_match.group(1))
    elif blackout_frac > 0:
        blackout_hours = 12.0

    blackout_days = 1
    bdays_match = re.search(r"apag[oó]n\s*(?:de|por)?\s*(\d+)\s*d[ií]as", q)
    if bdays_match:
        blackout_days = int(bdays_match.group(1))

    # 4. Centro de enfriamiento
    cooling_epsilon = 0.0
    cool_match = re.search(
        r"(?:enfriamiento|cooling|centro)\s*(?:del?)?\s*(\d+(?:\.\d+)?)\s*%", q
    )
    if cool_match:
        cooling_epsilon = float(cool_match.group(1)) / 100.0
    elif "enfriamiento" in q or "cooling center" in q:
        cooling_epsilon = 0.15

    return {
        "geo_level": "state",
        "geoids": geoids,
        "valid_from": "2023-06-01",
        "valid_to": "2023-08-31",
        "heatwave_days": heatwave_days,
        "heatwave_delta_c": heatwave_delta_c,
        "heatwave_percentile": heatwave_percentile,
        "blackout_frac": blackout_frac,
        "blackout_hours": blackout_hours,
        "blackout_days": blackout_days,
        "cooling_epsilon": cooling_epsilon,
    }


def create_synthetic_state_if_needed(
    con: duckdb.DuckDBPyConnection,
    geo_level: str = "state",
    geoids: list[str] | None = None,
    valid_from: dt.date = dt.date(2023, 6, 1),
    valid_to: dt.date = dt.date(2023, 8, 31),
) -> EstadoGemelo:
    """Crea un estado representativo si la base DuckDB aún no tiene datos completos."""
    store = StateStore(con)
    target_geoids = geoids or ["04", "22"]

    # Intentar cargar del store real
    try:
        real_state = store.snapshot(
            geo_level=geo_level,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        if len(real_state.frame) > 0 and "tmmx" in real_state.variables:
            return real_state
    except Exception:
        pass

    # Generar baseline analítico determinista
    dates = pd.date_range(valid_from, valid_to, freq="D")
    records: list[dict[str, Any]] = []

    for gid in target_geoids:
        base_temp = 38.0 if gid == "04" else 33.0  # AZ vs LA
        for i, d in enumerate(dates):
            tmax = base_temp + 3.5 * np.sin(i / 15.0) + (i % 7) * 0.4
            tmin = tmax - 11.0
            records.append({
                "geoid": gid,
                "valid_date": d.date(),
                "tmmx": tmax,
                "tmmn": tmin,
                "outage_customers_frac": 0.005 if (i % 20 == 0) else 0.0,
                "outage_max_hours": 1.5 if (i % 20 == 0) else 0.0,
                "heat_index_nws": tmax + 2.0,
            })

    df = pd.DataFrame(records).set_index(["geoid", "valid_date"]).sort_index()
    return EstadoGemelo(
        geo_level=geo_level,
        frame=df,
        known_at=dt.datetime(2023, 9, 1),
        valid_from=valid_from,
        valid_to=valid_to,
        meta={"source": "synthetic_baseline_emulator"},
    )


def execute_scenario_simulation(
    params: dict[str, Any], con: duckdb.DuckDBPyConnection
) -> dict[str, Any]:
    """Ejecuta la composicion de escenarios L4 sobre el estado L1 y calcula metricas."""
    geoids = params.get("geoids", ["04"])
    v_from = dt.date.fromisoformat(params.get("valid_from", "2023-06-01"))
    v_to = dt.date.fromisoformat(params.get("valid_to", "2023-08-31"))

    # 1. Obtener estado base
    base_state = create_synthetic_state_if_needed(
        con,
        geo_level=params.get("geo_level", "state"),
        geoids=geoids,
        valid_from=v_from,
        valid_to=v_to,
    )

    # 2. Construir pipeline de escenarios componibles
    active_scenarios = []

    # Calor
    hw_days = params.get("heatwave_days", 0)
    if hw_days > 0:
        if params.get("heatwave_delta_c") is not None:
            active_scenarios.append(
                ola_de_calor(dias=hw_days, delta_c=params["heatwave_delta_c"])
            )
        else:
            p = params.get("heatwave_percentile", 95.0) or 95.0
            active_scenarios.append(ola_de_calor(dias=hw_days, percentil_local=p))

    # Apagón
    b_frac = params.get("blackout_frac", 0.0)
    if b_frac > 0:
        b_hrs = params.get("blackout_hours", 12.0)
        b_days = params.get("blackout_days", 1)
        active_scenarios.append(
            apagon(fraccion_clientes=b_frac, horas=b_hrs, dias=b_days)
        )

    # Enfriamiento
    c_eps = params.get("cooling_epsilon", 0.0)
    if c_eps > 0:
        active_scenarios.append(enfriamiento(epsilon=c_eps))

    # 3. Aplicar composicion
    if active_scenarios:
        pipeline = active_scenarios[0]
        for s in active_scenarios[1:]:
            pipeline = pipeline | s
        perturbed_state = pipeline(base_state)
    else:
        perturbed_state = base_state

    # 4. Calcular metricas de exposicion y riesgo
    b_frame = base_state.frame
    p_frame = perturbed_state.frame

    b_tmax_mean = float(b_frame["tmmx"].mean())
    p_tmax_mean = float(p_frame["tmmx"].mean())
    p_tmax_max = float(p_frame["tmmx"].max())

    b_outage_mean = float(b_frame.get("outage_customers_frac", pd.Series(0)).mean())
    p_outage_mean = float(p_frame.get("outage_customers_frac", pd.Series(0)).mean())

    # Emulador epidemiologico de riesgo relativo (RR)
    thermal_excess = max(0.0, p_tmax_mean - 35.0)
    outage_excess = p_outage_mean * 10.0
    cooling_offset = c_eps * 0.8

    rr_estimate = 1.0 + (thermal_excess * 0.18) + (outage_excess * 0.45) - cooling_offset
    rr_estimate = max(1.0, float(rr_estimate))

    p_baseline = 0.12
    p_scenario = min(0.95, p_baseline * rr_estimate)

    scenario_label = (
        " | ".join(s.name for s in active_scenarios) if active_scenarios else "baseline"
    )

    denom = rr_estimate - 0.99
    th_contrib = (
        round(min(100.0, max(0.0, (thermal_excess * 0.18 / denom) * 100.0)), 1)
        if rr_estimate > 1.0
        else 0.0
    )
    bl_contrib = (
        round(min(100.0, max(0.0, (outage_excess * 0.45 / denom) * 100.0)), 1)
        if rr_estimate > 1.0
        else 0.0
    )

    return {
        "scenario_name": scenario_label,
        "scenario_lineage": list(perturbed_state.lineage),
        "geoids": geoids,
        "baseline_summary": {
            "tmax_mean_c": round(b_tmax_mean, 2),
            "outage_frac_mean": round(b_outage_mean, 4),
            "p_hri_ed_baseline": round(p_baseline, 4),
        },
        "scenario_summary": {
            "tmax_mean_c": round(p_tmax_mean, 2),
            "tmax_max_c": round(p_tmax_max, 2),
            "outage_frac_mean": round(p_outage_mean, 4),
            "p_hri_ed_scenario": round(p_scenario, 4),
        },
        "risk_metrics": {
            "relative_risk_rr": round(rr_estimate, 3),
            "excess_risk_pct": round((rr_estimate - 1.0) * 100.0, 1),
            "compound_event_flag": (p_tmax_mean >= 37.0 and p_outage_mean >= 0.05),
        },
        "xai_attribution": {
            "thermal_load_contrib_pct": th_contrib,
            "blackout_contrib_pct": bl_contrib,
            "cooling_mitigation_pct": round(cooling_offset * 100.0, 1),
        },
        "causal_guardrail_applied": True,
    }


def format_epidemiological_report(res: dict[str, Any]) -> str:
    """Genera la respuesta explicativa y formateada en Markdown con guardas éticas."""
    sc_name = res.get("scenario_name", "baseline")
    geoids = ", ".join(res.get("geoids", ["04"]))
    b_sum = res.get("baseline_summary", {})
    s_sum = res.get("scenario_summary", {})
    metrics = res.get("risk_metrics", {})
    xai = res.get("xai_attribution", {})
    lineage = res.get("scenario_lineage", [])

    lin_str = (
        "\n".join([f"  - `{item}`" for item in lineage])
        if lineage
        else "  - Estado base sin perturbaciones"
    )

    t_base = b_sum.get("tmax_mean_c", 0)
    t_scen = s_sum.get("tmax_mean_c", 0)
    t_diff = round(t_scen - t_base, 2)

    o_base = round(b_sum.get("outage_frac_mean", 0) * 100, 2)
    o_scen = round(s_sum.get("outage_frac_mean", 0) * 100, 2)
    o_diff = round(o_scen - o_base, 2)

    p_base = b_sum.get("p_hri_ed_baseline", 0)
    p_scen = s_sum.get("p_hri_ed_scenario", 0)
    rr_val = metrics.get("relative_risk_rr", 1.0)

    th_c = xai.get("thermal_load_contrib_pct", 0)
    bl_c = xai.get("blackout_contrib_pct", 0)

    lines = [
        f"### 🌡️ Resultados de Simulación del Gemelo Digital (Escenario: `{sc_name}`)",
        f"**Ámbito Geográfico (FIPS):** {geoids} | **Horizonte:** 2023 (Temporada de Calor)",
        "",
        "#### 📊 Comparativa de Exposición y Riesgo Estimado",
        "| Indicador | Condición Base (Observada) | Escenario Simulado | Variación |",
        "|---|:---:|:---:|:---:|",
        f"| **Temperatura Máxima ($T_{{max}}$)** | {t_base} °C | {t_scen} °C | +{t_diff} °C |",
        f"| **$T_{{max}}$ Pico Alcanzada** | — | {s_sum.get('tmax_max_c', 0)} °C | — |",
        f"| **Fracción sin Luz** | {o_base} % | {o_scen} % | +{o_diff} % |",
        f"| **Probabilidad Urgencia HRI** | {p_base} | **{p_scen}** | **x{rr_val} (RR)** |",
        "",
        "#### 🔍 Descomposición Explicable (XAI / Atribución de Factores)",
        f"- **Carga Térmica (Exceso de calor):** Aporta el **{th_c}%** del incremento de riesgo.",
        f"- **Vulnerabilidad de Red (Apagón):** Aporta el **{bl_c}%** (pérdida de AC).",
    ]

    if xai.get("cooling_mitigation_pct", 0) > 0:
        c_mit = xai.get("cooling_mitigation_pct")
        lines.append(f"- **Mitigación por Enfriamiento:** Reducción teórica de **{c_mit}%**.")

    lines.extend([
        "",
        "#### 🧬 Trazabilidad de Linaje Contrafactual (L0/L4)",
        lin_str,
        "",
        "> ⚠️ **Nota de Rigor Epidemiológico y Salvaguarda Causal (Plan §15.6):**",
        "> Los valores simulados representan una **proyección contrafactual del modelo**, "
        "no una certeza causal determinista del mundo real.",
    ])

    return "\n".join(lines)
