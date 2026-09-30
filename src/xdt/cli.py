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
from xdt.export import write_payload
from xdt.ingest.eaglei import EagleiConnector
from xdt.ingest.epht import MEASURES, EphtConnector
from xdt.ingest.gridmet import DEFAULT_VARIABLES as GRIDMET_DEFAULT_VARIABLES
from xdt.ingest.gridmet import GridmetConnector
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


def _parse_years(spec: str) -> list[int]:
    """'2017-2026' o '2022,2023' -> lista de anos."""
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            years.extend(range(a, b + 1))
        elif part:
            years.append(int(part))
    return years


@ingest_app.command("gridmet")
def ingest_gridmet(
    years: Annotated[str, typer.Option(help="Ej: 2017-2026 o 2022,2023")],
    states: Annotated[
        str | None, typer.Option(help="FIPS de estado separados por coma. Ej: 04,22")
    ] = None,
    counties: Annotated[
        str | None, typer.Option(help="FIPS de condado (5 digitos) separados por coma")
    ] = None,
    variables: Annotated[str, typer.Option()] = ",".join(GRIDMET_DEFAULT_VARIABLES),
    refresh: Annotated[bool, typer.Option(help="Ignora la cache y vuelve a descargar")] = False,
    pop_coverage: Annotated[
        float | None,
        typer.Option(help="Solo los condados mas poblados de cada estado hasta esta "
                          "fraccion de su poblacion (p.ej. 0.8). Requiere --states"),
    ] = None,
    workers: Annotated[
        int, typer.Option(help="Descargas en paralelo antes de cargar (solo red)")
    ] = 1,
) -> None:
    """Ingesta gridMET por condado (centroide de poblacion) y por estado (ponderado)."""
    from concurrent.futures import ThreadPoolExecutor

    from rich.progress import Progress

    from xdt.harmonize.centroids import (
        NON_CONUS_STATES,
        fetch_county_centroids,
        parse_county_centroids,
        top_counties_by_population,
    )

    state_list = [s.strip().zfill(2) for s in states.split(",")] if states else None
    county_list = [c.strip() for c in counties.split(",")] if counties else None
    var_list = [v.strip() for v in variables.split(",") if v.strip()]
    year_list = _parse_years(years)

    conn = GridmetConnector()
    try:
        if pop_coverage is not None:
            if not state_list:
                raise typer.BadParameter("--pop-coverage requiere --states")
            centroids = parse_county_centroids(fetch_county_centroids(conn.census))
            conus = [s for s in state_list if s not in NON_CONUS_STATES]
            county_list = top_counties_by_population(
                centroids, states=conus, coverage=pop_coverage
            )
            state_list = None
            console.print(f"{len(county_list)} condados cubren >= {pop_coverage:.0%} "
                          f"de la poblacion de {len(conus)} estados")

        params = dict(years=year_list, variables=var_list, states=state_list,
                      counties=county_list, refresh=refresh)

        with Progress(console=console, transient=True) as progress:
            if workers > 1 and county_list:
                # Precarga a data/raw en paralelo; la carga en L1 de abajo
                # lee todo de la cache en un solo proceso (DuckDB es monoescritor).
                conn.fetch(**{**params, "counties": county_list[:1]})  # DAS, DDS, centroides
                chunks = [county_list[i::workers] for i in range(workers)]
                pre = progress.add_task("gridMET (descarga)", total=len(county_list))

                def _prefetch(chunk: list[str]) -> None:
                    c = GridmetConnector()
                    try:
                        for geoid in chunk:
                            c.fetch(**{**params, "counties": [geoid]})
                            progress.advance(pre)
                    finally:
                        c.close()

                with ThreadPoolExecutor(max_workers=workers) as ex:
                    list(ex.map(_prefetch, chunks))

            task = progress.add_task("gridMET (carga)", total=None)
            with session() as con:
                stats = conn.run(
                    con, **params,
                    on_progress=lambda d, t: progress.update(task, completed=d, total=t),
                )
    finally:
        conn.close()

    console.print_json(json.dumps(stats, default=str))


@ingest_app.command("eaglei")
def ingest_eaglei(
    years: Annotated[str, typer.Option(help="Ej: 2017-2025 o 2023")],
    states: Annotated[str, typer.Option(help="FIPS de estado separados por coma. Ej: 04,22")],
    refresh: Annotated[bool, typer.Option(help="Ignora la cache y vuelve a descargar")] = False,
) -> None:
    """Ingesta apagones de EAGLE-I (~1.2 GB por ano) como series diarias locales."""
    from rich.progress import DownloadColumn, Progress, TransferSpeedColumn

    conn = EagleiConnector()
    try:
        with Progress(
            *Progress.get_default_columns(), DownloadColumn(), TransferSpeedColumn(),
            console=console, transient=True,
        ) as progress:
            task = progress.add_task("EAGLE-I", total=None)
            with session() as con:
                stats = conn.run(
                    con,
                    years=_parse_years(years),
                    states=[s.strip() for s in states.split(",")],
                    refresh=refresh,
                    on_bytes=lambda n, t: progress.update(task, completed=n, total=t),
                )
    finally:
        conn.close()

    console.print_json(json.dumps({**stats, "report": conn.report}, default=str))


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


@app.command("compound")
def compound(
    valid_from: Annotated[str, typer.Option(help="Inicio de la ventana analizada")],
    valid_to: Annotated[str, typer.Option(help="Fin de la ventana analizada")],
    ref_start: Annotated[str, typer.Option(help="Inicio del periodo de referencia del p95")],
    ref_end: Annotated[str, typer.Option(help="Fin del periodo de referencia del p95")],
    geo_level: Annotated[str, typer.Option()] = "county",
    min_frac: Annotated[float, typer.Option(help="X de §9.2: fraccion sin luz")] = 0.01,
    min_hours: Annotated[float, typer.Option(help="Y de §9.2: horas seguidas")] = 4.0,
    top: Annotated[int, typer.Option(help="Dias compuestos a listar")] = 10,
) -> None:
    """Exploratorio de eventos compuestos calor ∧ apagon (M2, §9.2). Descriptivo."""
    from xdt.features.compound import compound_features, compound_summary

    with session() as con:
        snap = StateStore(con).snapshot(
            geo_level=geo_level, valid_from=min(ref_start, valid_from),
            valid_to=max(ref_end, valid_to),
            variables=["tmmx", "outage_customers_frac", "outage_max_hours"],
        )
    flags = compound_features(
        snap.frame, ref_start=ref_start, ref_end=ref_end, min_frac=min_frac, min_hours=min_hours
    )
    dates = flags.index.get_level_values("valid_date")
    flags = flags[(dates >= valid_from) & (dates <= valid_to)]

    summary = compound_summary(flags)
    summary = summary[summary["days"] > 0].sort_values("compound_days", ascending=False)
    console.print(summary.head(25).round(2).to_string())
    tot = summary[["days", "hot_days", "outage_days", "compound_days", "expected_compound"]].sum()
    console.print(
        f"\n[bold]total[/] dias={tot['days']:.0f} calientes={tot['hot_days']:.0f} "
        f"apagon={tot['outage_days']:.0f} compuestos={tot['compound_days']:.0f} "
        f"esperados_bajo_independencia={tot['expected_compound']:.1f}"
    )
    hits = flags[flags["compound"] == 1.0].join(snap.frame)
    if not hits.empty:
        console.rule(f"[bold]{min(top, len(hits))} dias compuestos con mayor corte[/]")
        cols = ["tmmx", "outage_customers_frac", "outage_max_hours", "hot_streak"]
        console.print(hits.sort_values("outage_customers_frac", ascending=False)[cols]
                      .head(top).round(3).to_string())
    console.print(
        "[dim]Descriptivo: coincidencia de calor y apagon, no efecto sobre la salud "
        "(eso requiere el desenlace y RERI, H2).[/]"
    )


@app.command("calibrate")
def calibrate(
    model: Annotated[str, typer.Option(help="model_name registrado en L0")],
    quantity: Annotated[str, typer.Option(help="Cantidad predicha y observada")],
    model_version: Annotated[str | None, typer.Option()] = None,
    window_start: Annotated[str | None, typer.Option()] = None,
    window_end: Annotated[str | None, typer.Option()] = None,
    obs_known_at: Annotated[
        str | None, typer.Option(help="Reproduce una calibracion pasada: corte bitemporal")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="No persiste la corrida")] = False,
) -> None:
    """Calibracion retrospectiva (L5): predicciones de L0 contra observaciones."""
    import pandas as pd

    from xdt.twin.calibrate import Calibrator, result_dict

    with session() as con:
        r = Calibrator(con).run(
            model_name=model, quantity=quantity, model_version=model_version,
            window_start=window_start, window_end=window_end,
            obs_known_at=pd.to_datetime(obs_known_at).to_pydatetime() if obs_known_at else None,
            persist=not dry_run,
        )
    console.print_json(json.dumps(result_dict(r), default=str))
    if r.drift_flag:
        console.print("[bold red]DERIVA[/] " + "; ".join(r.drift_reasons))


@app.command("scenario")
def scenario_run(
    geo_level: Annotated[str, typer.Option()] = "state",
    valid_from: Annotated[str, typer.Option()] = "2023-01-01",
    valid_to: Annotated[
        str | None, typer.Option(help="Fin de la ventana; por defecto valid_from + 20 dias")
    ] = None,
    fraccion_clientes: Annotated[float, typer.Option(help="Clientes sin luz (0-1)")] = 0.25,
    horas: Annotated[float, typer.Option(help="Duracion maxima del corte")] = 18.0,
    dias: Annotated[int, typer.Option(help="Dias afectados al final de la serie")] = 3,
    delta_c: Annotated[
        float | None,
        typer.Option(help="Ola de calor: +grados sobre tmax (requiere gridMET en el estado)"),
    ] = None,
) -> None:
    """Aplica un escenario compuesto (L4) sobre el estado real e imprime el contrafactual.

    Perturba el estado del gemelo; NO estima riesgo. El efecto sobre el
    desenlace requiere el emulador de L3 (§15.6: contrafactual del modelo,
    no del mundo).
    """
    from xdt.twin.scenarios import apagon, baseline, ola_de_calor

    with session() as con:
        # Ventana de 21 dias (§7 M2): el escenario perturba los ultimos `dias`
        # y el encoder necesita ver el tramo previo intacto.
        import pandas as pd

        vt = valid_to or str((pd.Timestamp(valid_from) + pd.Timedelta(days=20)).date())
        estado = StateStore(con).snapshot(
            geo_level=geo_level, valid_from=valid_from, valid_to=vt
        )

    esc = baseline() | apagon(fraccion_clientes=fraccion_clientes, horas=horas, dias=dias)
    if delta_c is not None:
        esc = esc | ola_de_calor(dias=dias, delta_c=delta_c)

    console.rule("[bold]estado base[/]")
    console.print(repr(estado))
    console.print(f"state_id  {estado.state_id()}   baseline={estado.is_baseline}")

    console.rule("[bold]escenario[/]")
    console.print(f"[cyan]{esc.name}[/]   id={esc.scenario_id()}")
    console.print(esc.spec_json())

    try:
        contra = esc(estado)
    except KeyError as exc:
        console.print(f"[red]el estado no tiene la variable requerida:[/] {exc}")
        console.print("ola_de_calor necesita tmax (gridMET, conector pendiente en L2).")
        raise typer.Exit(code=1) from exc

    console.rule("[bold]contrafactual[/]")
    console.print(repr(contra))
    console.print(f"state_id  {contra.state_id()}   baseline={contra.is_baseline}")
    console.print("linaje:")
    for i, step in enumerate(contra.lineage, 1):
        console.print(f"  {i}. {step}")

    # Variables que el escenario cambio (existentes modificadas o nuevas), en
    # las filas donde cambiaron: base -> contrafactual.
    base = estado.frame.reindex(columns=contra.frame.columns)
    diff = ~((base == contra.frame) | (base.isna() & contra.frame.isna()))
    cambiadas = [v for v in contra.frame.columns if diff[v].any()]
    # Las marcas scn_* cubren toda la ventana; las filas se eligen por los
    # cambios en variables del mundo (tmmx, outage_*).
    reales = [v for v in cambiadas if not v.startswith("scn_")] or cambiadas
    filas = diff[reales].any(axis=1)
    tabla = Table(title="variables perturbadas: base -> contrafactual")
    tabla.add_column("geoid")
    tabla.add_column("valid_date")
    for v in cambiadas:
        tabla.add_column(v, justify="right")
    for (geoid, vd) in contra.frame.index[filas.to_numpy()][:12]:
        celdas = []
        for v in cambiadas:
            b, c = base.at[(geoid, vd), v], contra.frame.at[(geoid, vd), v]
            celdas.append(f"{c:.3f}" if pd.isna(b) else f"{b:.3f} -> {c:.3f}")
        tabla.add_row(geoid, str(vd)[:10], *celdas)
    console.print(tabla)
    console.print(
        "[dim]Estado perturbado, no riesgo: traducir esto a morbilidad requiere L3.[/]"
    )


@app.command("export")
def export_front(
    out: Annotated[
        str, typer.Option(help="Ruta del JSON de salida")
    ] = "data/features/front_payload.json",
    html: Annotated[
        str | None, typer.Option(help="Consola HTML donde inyectar el payload ('' = no)")
    ] = "app/consola.html",
    db: Annotated[str | None, typer.Option(help="Base DuckDB alternativa")] = None,
) -> None:
    """Exporta el payload que consume la interfaz y lo inyecta en la consola.

    La frontera motor <-> visor es este archivo, no un servidor (§12.2).
    """
    from pathlib import Path

    models_dir = None if db is None else Path(db).parent / "models"
    with session(db, read_only=True) as con:
        path = write_payload(con, out, html=html or None, models_dir=models_dir)
    console.print(f"[green]escrito[/] {path}" + (f" e inyectado en {html}" if html else ""))


@app.command("crisp")
def crisp(
    out: Annotated[str, typer.Option(help="Reporte HTML de salida")] = "reports/crisp_dm.html",
    years: Annotated[str, typer.Option(help="Anos con desenlace a modelar")] = "2023",
    ref_years: Annotated[
        str, typer.Option(help="Anos de referencia del p95 local (nunca de prueba)")
    ] = "2019-2022",
    n_iter: Annotated[
        int | None, typer.Option(help="Configuraciones por busqueda (defecto: por modelo)")
    ] = None,
    n_boot: Annotated[int, typer.Option(help="Replicas del bootstrap por bloques")] = 500,
    n_perm: Annotated[int, typer.Option(help="Permutaciones circulares")] = 300,
    db: Annotated[str | None, typer.Option(help="Base DuckDB alternativa")] = None,
    deploy_model: Annotated[
        bool, typer.Option("--deploy/--no-deploy",
                           help="Guarda el modelo elegido y registra sus predicciones en L0")
    ] = True,
) -> None:
    """Motor predictivo L3 de punta a punta con reporte CRISP-DM (EDA -> evaluacion)."""
    from pathlib import Path

    from xdt.features.panel import build_panel
    from xdt.models.train import run_training
    from xdt.report.crisp import build_report, write_metrics

    with session(db, read_only=True) as con:
        panel = build_panel(con, years=_parse_years(years), ref_years=_parse_years(ref_years))
        f = panel.frame
        console.print(f"panel: {len(f):,} estado-dias, "
                      f"{f.index.get_level_values('geoid').nunique()} estados, "
                      f"prevalencia {f['y'].mean():.1%}")
        sources = con.execute(
            """
            SELECT s.source AS fuente, count(*) AS hechos,
                   count(DISTINCT s.geoid) AS geografias,
                   strftime(min(s.valid_date), '%Y-%m-%d') AS desde,
                   strftime(max(s.valid_date), '%Y-%m-%d') AS hasta,
                   any_value(a.n) AS artefactos_crudos
            FROM twin_state s
            LEFT JOIN (SELECT source, count(*) AS n FROM raw_artifact GROUP BY 1) a
              ON a.source = s.source
            GROUP BY 1 ORDER BY 1
            """
        ).df()

    console.print("entrenando la escalera de modelos (CV anidada)...")
    run = run_training(panel, n_iter=n_iter, log=console.print)
    sel = run.selection
    console.print(f"elegido: [bold]{sel['elegido']}[/] (mayor AUC: {sel['mejor_auc']}; "
                  f"no peores: {', '.join(sel['no_peores'])})")

    out_path = Path(out)
    console.print("pruebas estadisticas, explicabilidad y reporte...")
    stats_out = build_report(run, sources=sources, out=out_path, n_boot=n_boot, n_perm=n_perm)
    metrics = out_path.with_suffix(".json")
    write_metrics(run, stats_out, metrics)
    console.print(f"[green]escrito[/] {out_path} y {metrics}")

    if deploy_model:
        from xdt.models.deploy import deploy

        models_dir = get_settings().models_dir if db is None else Path(db).parent / "models"
        with session(db) as con:
            deploy(run, con, models_dir=models_dir, report_path=out_path.as_posix(),
                   log=console.print)


if __name__ == "__main__":  # pragma: no cover
    app()
