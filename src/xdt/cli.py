"""Interfaz de linea de comandos del motor.

Es la superficie que `dvc.yaml` invoca en cada etapa del pipeline, de modo que
`dvc repro` no dependa de notebooks ni de pasos manuales (§12.2).
"""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from xdt.config import get_settings
from xdt.ingest.epht import MEASURES, EphtConnector
from xdt.storage import code_version, session
from xdt.twin.state import StateStore

app = typer.Typer(help="Gemelo digital de riesgo de morbilidad por calor", no_args_is_help=True)
ingest_app = typer.Typer(help="Conectores de ingesta (L2)", no_args_is_help=True)
state_app = typer.Typer(help="Estado del gemelo (L1)", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
app.add_typer(state_app, name="state")

console = Console()


@app.command()
def info() -> None:
    """Muestra la configuracion efectiva."""
    s = get_settings()
    t = Table(title="xai-dt-heat", show_header=False)
    t.add_row("raiz de datos", str(s.data_root))
    t.add_row("base DuckDB", str(s.twin_db_path))
    t.add_row(
        "token EPHT",
        "presente" if s.epht_token else "[red]ausente[/] (429 tras ~8 peticiones)",
    )
    t.add_row("modo offline", str(s.offline))
    t.add_row("version de codigo", code_version())
    console.print(t)


@ingest_app.command("epht")
def ingest_epht(
    measure: Annotated[str, typer.Option(help=f"Una de: {', '.join(MEASURES)}")],
    temporal: Annotated[
        str, typer.Option(help="Periodo(s) separados por coma. Ej: 2020,2021,2022")
    ] = "ALL",
    refresh: Annotated[bool, typer.Option(help="Ignora la cache y vuelve a descargar")] = False,
) -> None:
    """Ingesta una medida de EPHT al estado del gemelo."""
    if measure not in MEASURES:
        raise typer.BadParameter(f"medida desconocida. Opciones: {', '.join(MEASURES)}")

    periods = [p.strip() for p in temporal.split(",")] if temporal != "ALL" else ["ALL"]
    conn = EphtConnector()
    try:
        with session() as con:
            stats = conn.run(con, measure=measure, temporal=periods, refresh=refresh)
    finally:
        conn.close()

    console.print_json(json.dumps(stats, default=str))


@ingest_app.command("discover")
def ingest_discover(measure_id: int) -> None:
    """Descubre los ids reales que getCoreHolder necesita para una medida.

    Usar SIEMPRE antes de anadir una medida al catalogo: pasar el id
    equivocado devuelve HTTP 200 con resultado vacio, un fallo silencioso.
    """
    conn = EphtConnector()
    try:
        console.print_json(json.dumps(conn.discover(measure_id), default=str))
    finally:
        conn.close()


@state_app.command("coverage")
def state_coverage() -> None:
    """Cobertura y huecos por fuente, nivel y variable (entregable de M0)."""
    with session(read_only=False) as con:
        df = StateStore(con).coverage()
    if df.empty:
        console.print("[yellow]El estado esta vacio. Ejecuta una ingesta primero.[/]")
        raise typer.Exit(0)

    t = Table(title="Cobertura del estado del gemelo")
    for c in df.columns:
        t.add_column(str(c))
    for _, r in df.iterrows():
        t.add_row(*[str(v) for v in r])
    console.print(t)


@state_app.command("show")
def state_show(
    geo_level: Annotated[str, typer.Option()] = "state",
    valid_from: Annotated[str, typer.Option()] = "2023-01-01",
    valid_to: Annotated[str | None, typer.Option()] = None,
    known_at: Annotated[
        str | None,
        typer.Option(help="Viaje en el tiempo: estado conocido en esta fecha"),
    ] = None,
    limit: Annotated[int, typer.Option()] = 20,
) -> None:
    """Imprime un snapshot del estado, opcionalmente en un corte pasado."""
    import pandas as pd

    with session() as con:
        snap = StateStore(con).snapshot(
            geo_level=geo_level,
            valid_from=valid_from,
            valid_to=valid_to,
            known_at=pd.to_datetime(known_at).to_pydatetime() if known_at else None,
        )
    console.print(repr(snap))
    if not snap.frame.empty:
        console.print(snap.frame.head(limit).to_string())


@state_app.command("history")
def state_history(
    geoid: str,
    variable: str,
    valid_date: str,
    geo_level: Annotated[str, typer.Option()] = "state",
) -> None:
    """Todas las revisiones de un hecho. Auditoria de una correccion de fuente."""
    with session() as con:
        df = StateStore(con).history(
            geo_level=geo_level, geoid=geoid, valid_date=valid_date, variable=variable
        )
    console.print(df.to_string() if not df.empty else "[yellow]sin registros[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
