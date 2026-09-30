"""Tests para el módulo de agentes inteligentes (LangGraph + LangFlow)."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from xdt.agent.graph import run_copilot
from xdt.agent.langflow_export import export_langflow_json, get_rag_flow_definition
from xdt.agent.tools import (
    execute_scenario_simulation,
    format_epidemiological_report,
    parse_scenario_intent,
)


def test_parse_scenario_intent_arizona_heat_blackout():
    query = (
        "Simula en Arizona una ola de calor de 7 días (+3 C) "
        "con un corte de luz del 25% y 18 horas"
    )
    params = parse_scenario_intent(query)

    assert "04" in params["geoids"]
    assert params["heatwave_days"] == 7
    assert params["heatwave_delta_c"] == 3.0
    assert params["blackout_frac"] == 0.25
    assert params["blackout_hours"] == 18.0


def test_parse_scenario_intent_louisiana_cooling():
    query = "En Louisiana queremos evaluar un centro de enfriamiento del 20% con percentil 95"
    params = parse_scenario_intent(query)

    assert "22" in params["geoids"]
    assert params["cooling_epsilon"] == 0.20
    assert params["heatwave_percentile"] == 95.0


def test_execute_scenario_simulation():
    con = duckdb.connect(":memory:")
    params = {
        "geo_level": "state",
        "geoids": ["04"],
        "valid_from": "2023-06-01",
        "valid_to": "2023-08-31",
        "heatwave_days": 5,
        "heatwave_delta_c": 2.5,
        "blackout_frac": 0.20,
        "blackout_hours": 12.0,
        "blackout_days": 1,
        "cooling_epsilon": 0.10,
    }

    res = execute_scenario_simulation(params, con)
    con.close()

    assert res["scenario_name"] == "ola_de_calor | apagon | enfriamiento"
    assert len(res["scenario_lineage"]) == 3
    assert res["risk_metrics"]["relative_risk_rr"] >= 1.0
    assert "thermal_load_contrib_pct" in res["xai_attribution"]
    assert res["causal_guardrail_applied"] is True


def test_format_epidemiological_report():
    con = duckdb.connect(":memory:")
    params = {
        "geo_level": "state",
        "geoids": ["04"],
        "heatwave_days": 3,
        "heatwave_delta_c": 1.5,
        "blackout_frac": 0.15,
        "blackout_hours": 8.0,
        "cooling_epsilon": 0.05,
    }
    res = execute_scenario_simulation(params, con)
    report = format_epidemiological_report(res)
    con.close()

    assert "Resultados de Simulación del Gemelo Digital" in report
    assert "ola_de_calor | apagon | enfriamiento" in report
    assert "Nota de Rigor Epidemiológico" in report


def test_langgraph_copilot_end_to_end():
    query = "Simula en Phoenix 5 días de calor extremo y 20% de apagón"
    result = run_copilot(query)

    assert result["geo_level"] == "state"
    assert "04" in result["geoids"]
    assert result["heatwave_days"] == 5
    assert result["blackout_frac"] == 0.20
    assert result["causal_guardrail_applied"] is True
    assert "Resultados de Simulación del Gemelo Digital" in result["response"]


def test_langflow_export_structure(tmp_path: Path):
    flow = get_rag_flow_definition()
    assert "name" in flow
    assert "nodes" in flow["data"]
    assert "edges" in flow["data"]
    assert any(n["type"] == "File" for n in flow["data"]["nodes"])
    assert any(n["type"] == "RecursiveCharacterTextSplitter" for n in flow["data"]["nodes"])
    assert any(n["type"] == "ChatOpenAI" for n in flow["data"]["nodes"])

    out_file = tmp_path / "flow.json"
    exported = export_langflow_json(out_file)
    assert exported.exists()

    with open(exported, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["name"] == flow["name"]
