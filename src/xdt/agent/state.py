"""Estado del grafo del agente copiloto (LangGraph)."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentCopilotState(TypedDict, total=False):
    """Estado tipado para el grafo de ejecucion de LangGraph."""

    # Consulta y contexto de conversacion
    query: str
    messages: list[dict[str, str]]

    # Parametros extraidos del escenario
    geo_level: str
    geoids: list[str]
    valid_from: str | None
    valid_to: str | None
    heatwave_days: int
    heatwave_delta_c: float | None
    heatwave_percentile: float | None
    blackout_frac: float
    blackout_hours: float
    blackout_days: int
    cooling_epsilon: float

    # Resultados intermedios y de simulacion
    scenario_name: str
    scenario_lineage: list[str]
    baseline_summary: dict[str, Any]
    scenario_summary: dict[str, Any]
    risk_metrics: dict[str, Any]
    xai_attribution: dict[str, Any]

    # Guardas y respuesta final
    causal_guardrail_applied: bool
    response: str
    error: str | None
