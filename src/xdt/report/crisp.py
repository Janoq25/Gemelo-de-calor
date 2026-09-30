"""Reporte CRISP-DM del motor predictivo (L3) sobre datos publicos reales.

Fases: 1 negocio · 2 datos (EDA) · 3 preparacion · 4 modelado (entrenamiento
e hiperparametros) · 5 evaluacion (CV, seleccion, pruebas robustas,
explicabilidad) · 6 despliegue en el gemelo.

Todo numero citado en un texto se calcula en esta corrida; no hay cifras
escritas a mano. Si los datos cambian, el texto cambia con ellos.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from xdt.explain import xai  # noqa: E402
from xdt.features.panel import FEATURES, OUTCOME_VAR  # noqa: E402
from xdt.models.train import TrainingRun  # noqa: E402
from xdt.report.html import Report  # noqa: E402
from xdt.validate import stats as st  # noqa: E402

PALETTE = {"umbral": "#8c8c8c", "logistica": "#2f6fb0", "bosque": "#2e7d4f",
           "lightgbm": "#b8461b"}
STATE_POSTAL = {
    "01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI",
    "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK",
    "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX",
    "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY",
}
#: Variables del estado termico del dia (se suman para la contribucion termica).
THERMAL_DAY = ("heat_index", "tmax", "tmin", "rh_min", "hi_excess_p95", "hi_mean_3d",
               "hi_lag1", "tmin_mean_3d", "hot_streak", "first_heatwave")
MONTHS_ES = {5: "may", 6: "jun", 7: "jul", 8: "ago", 9: "sep"}


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f} %".replace(".", ",")


def _n(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def _p(p: float) -> str:
    """'p' ya incluye el signo: '< 0,001' o '= 0,123'."""
    return "< 0,001" if p < 0.001 else f"= {_n(p, 3)}"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)


# =================================================================== fases
def build_report(run: TrainingRun, *, sources: pd.DataFrame, out: Path,
                 n_boot: int = 500, n_perm: int = 300) -> dict[str, Any]:
    panel = run.panel
    f = panel.frame
    y = panel.y.to_numpy()
    sel = run.selection
    mask = sel["mask"]
    chosen = sel["elegido"]
    best = sel["mejor_auc"]
    names = list(run.results)

    rep = Report(
        "Motor predictivo del gemelo digital de calor — CRISP-DM",
        f"Generado el {dt.datetime.now():%d/%m/%Y %H:%M} por <code>xdt crisp</code> · "
        f"corte bitemporal {panel.known_at:%Y-%m-%d %H:%M} UTC · "
        f"{len(f):,} estado-días · ".replace(",", ".")
        + f"{f.index.get_level_values('geoid').nunique()} estados",
    )

    # ------------------------------------------------------------ resumen
    auc_c, lo_c, hi_c = sel["auc_ci"][chosen]
    auc_u = sel["auc_ci"]["umbral"][0]
    rep.kpis([
        (chosen, "modelo elegido"),
        (_n(auc_c), f"AUC fuera de muestra (IC95 {_n(lo_c, 2)}–{_n(hi_c, 2)})"),
        (_n(auc_u), "AUC de la regla de umbral (referencia)"),
        (_pct(y.mean()), "estado-días con ≥1 visita por calor"),
    ])

    stats_out: dict[str, Any] = {}

    # ================================================= 1. negocio
    rep.section(
        "1. Comprensión del negocio",
        "El gemelo digital debe anticipar <b>en qué estados y qué días habrá visitas a "
        "urgencias por enfermedad relacionada con el calor</b> (HRI), para que una "
        "alerta llegue antes que el daño. Los sistemas vigentes alertan con un umbral "
        "meteorológico fijo; la pregunta del modelo es si aprender de los datos mejora "
        "esa regla de forma <i>estadísticamente demostrable</i>.",
    )
    rep.table(pd.DataFrame([
        ["Unidad de análisis", "estado × día, temporada cálida (mayo–septiembre)"],
        ["Desenlace (y)", f"1 si la tasa diaria de urgencias por HRI (CDC EPHT, medida 1385, "
                          f"población VA; variable {OUTCOME_VAR}) es > 0; 0 si no"],
        ["Tipo de tarea", "clasificación binaria probabilística"],
        ["Criterio de éxito 1", "AUC fuera de muestra significativamente mayor que la regla "
                                "de umbral (DeLong + Holm, α = 0,05)"],
        ["Criterio de éxito 2", "calibración aceptable: pendiente cercana a 1, "
                                "Spiegelhalter sin rechazo"],
        ["Criterio de éxito 3", "cada predicción explicable (SHAP) y trazable en L0"],
        ["Fuera de alcance", "efecto causal del calor; riesgo individual; población no VA"],
    ], columns=["Elemento", "Definición"]), "Ficha del problema",
        como_leer="Cada fila fija un elemento del problema antes de ver los resultados.",
        que_explica="Fijar el criterio de éxito antes de modelar evita elegir la métrica que "
                    "mejor sale. La referencia es la regla de umbral porque es lo que un "
                    "servicio meteorológico ya hace: el modelo solo aporta si la supera.")

    # ================================================= 2. datos / EDA
    rep.section("2. Comprensión de los datos (EDA)",
                "Análisis exploratorio del panel real, antes de cualquier modelo.")
    rep.table(sources, "Fuentes de datos y trazabilidad",
              como_leer="Una fila por fuente pública cargada en el estado bitemporal del gemelo: "
                        "hechos cargados, rango de fechas y artefactos crudos con su hash.",
              que_explica="Todo dato del reporte puede rastrearse hasta el archivo descargado "
                          "(hash SHA-256 en <code>raw_artifact</code>). Si el CDC revisa una "
                          "cifra, el gemelo guarda ambas versiones y este reporte se puede "
                          "reproducir con el corte bitemporal indicado arriba.")

    desc = f[list(FEATURES)].describe().T[["mean", "std", "min", "50%", "max"]]
    desc.columns = ["media", "desv. est.", "mín", "mediana", "máx"]
    desc.insert(0, "descripción", [FEATURES[c] for c in desc.index])
    desc["% faltante"] = 0.0
    rep.table(desc, "Estadística descriptiva de las variables predictoras",
              index=True, floatfmt="{:.2f}",
              como_leer="Una fila por variable: centro (media, mediana), dispersión "
                        "(desviación estándar) y extremos. Unidades en la descripción.",
              que_explica=f"El índice de calor va de {_n(desc.loc['heat_index', 'mín'], 1)} a "
                          f"{_n(desc.loc['heat_index', 'máx'], 1)} °C: el panel cubre desde "
                          f"días frescos hasta calor extremo, así que el modelo ve el rango "
                          f"completo sobre el que luego predice. No hay faltantes porque los "
                          f"estado-días sin clima o con cobertura de población menor al 70 % "
                          f"se excluyen antes (sección 3), no se imputan.")

    # prevalencia por mes
    month = f.index.get_level_values("valid_date").month
    by_m = f.groupby(month)["y"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([MONTHS_ES[m] for m in by_m.index], 100 * by_m["mean"], color="#b8461b")
    for i, v in enumerate(by_m["mean"]):
        ax.text(i, 100 * v, _pct(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% de estado-días con visita")
    _style(ax)
    m_hi, m_lo = by_m["mean"].idxmax(), by_m["mean"].idxmin()
    rep.figure(fig, "Frecuencia del desenlace por mes",
               como_leer="Cada barra es el porcentaje de estado-días del mes con al menos una "
                         "visita a urgencias por calor (y = 1).",
               que_explica=f"El desenlace se concentra en {MONTHS_ES[m_hi]} "
                           f"({_pct(by_m['mean'].max())}) y es mínimo en {MONTHS_ES[m_lo]} "
                           f"({_pct(by_m['mean'].min())}). La clase positiva es minoritaria "
                           f"({_pct(y.mean())} en total), por eso se reporta también la "
                           f"precisión promedio (AP) y no solo la exactitud, que sería alta "
                           f"prediciendo siempre 0.")

    # prevalencia por estado
    geo = f.index.get_level_values("geoid")
    by_s = f.groupby(geo)["y"].mean().sort_values()
    reg = f.groupby(geo)["hhs_region"].first()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    cmap = plt.get_cmap("tab10")
    ax.bar(range(len(by_s)), 100 * by_s.values,
           color=[cmap((int(reg[g]) - 1) % 10) for g in by_s.index])
    ax.set_xticks(range(len(by_s)), [STATE_POSTAL.get(g, g) for g in by_s.index],
                  rotation=90, fontsize=7)
    ax.set_ylabel("% de días con visita")
    _style(ax)
    rep.figure(fig, "Frecuencia del desenlace por estado (color = región HHS)",
               como_leer="Una barra por estado, ordenadas de menor a mayor. El color indica "
                         "la región HHS, que es la unidad que se deja fuera en la validación "
                         "cruzada espacial.",
               que_explica=f"La frecuencia va de {_pct(by_s.min())} "
                           f"({STATE_POSTAL.get(by_s.index[0])}) a {_pct(by_s.max())} "
                           f"({STATE_POSTAL.get(by_s.index[-1])}). Esta heterogeneidad entre "
                           f"estados es la razón de validar dejando fuera regiones enteras: "
                           f"un modelo que solo memorizara qué estados tienen más visitas "
                           f"fallaría en la región que nunca vio.")

    # curva empirica y vs indice de calor
    bins = pd.qcut(f["heat_index"], 12, duplicates="drop")
    emp = f.groupby(bins, observed=True)["y"].agg(["mean", "size"])
    se = np.sqrt(emp["mean"] * (1 - emp["mean"]) / emp["size"])
    mid = [iv.mid for iv in emp.index]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.errorbar(mid, 100 * emp["mean"], yerr=196 * se, fmt="o-", color="#b8461b", capsize=3)
    ax.set_xlabel("índice de calor máximo del día (°C), grupos de igual tamaño")
    ax.set_ylabel("% de días con visita")
    _style(ax)
    rho = pd.Series(f["heat_index"]).corr(f["y"], method="spearman")
    rep.figure(fig, "Relación entre índice de calor y desenlace",
               como_leer="Los días se agrupan en 12 grupos de igual tamaño según su índice de "
                         "calor. Cada punto es el % de días con visita en ese grupo; las barras "
                         "son intervalos de confianza del 95 %.",
               que_explica=f"La frecuencia sube de {_pct(emp['mean'].iloc[0])} en los días más "
                           f"frescos a {_pct(emp['mean'].iloc[-1])} en los más calurosos "
                           f"(Spearman ρ = {_n(rho, 2)}). La curva no es una recta, lo que "
                           f"justifica probar modelos no lineales (árboles) además de la "
                           f"logística. Es una asociación descriptiva, no un efecto causal.")

    # correlaciones
    corr = f[list(FEATURES)].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ")
    pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
    top = pairs.abs().sort_values(ascending=False).head(3)
    rep.figure(fig, "Correlación entre variables predictoras (Spearman)",
               como_leer="Cada celda es la correlación de rangos entre dos variables: rojo = "
                         "suben juntas, azul = una sube cuando la otra baja, blanco = sin "
                         "relación monótona.",
               que_explica="Los pares más correlacionados son "
                           + "; ".join(f"{a}–{b} (ρ = {_n(pairs[(a, b)], 2)})"
                                       for a, b in top.index)
                           + ". Esta colinealidad no daña la capacidad predictiva, pero "
                             "reparte la importancia entre variables parecidas: por eso la "
                             "explicación (sección 5) se lee por grupos de variables térmicas "
                             "y la logística lleva regularización L2.")

    # serie de un estado
    g_top = by_s.index[-1]
    s = f.xs(g_top, level="geoid")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(s.index, s["heat_index"], color="#555", lw=0.9, label="índice de calor")
    ev = s[s["y"] == 1]
    ax.scatter(ev.index, ev["heat_index"], color="#b8461b", s=14, zorder=3,
               label="día con visita")
    ax.set_ylabel("°C")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    rep.figure(fig, f"Serie diaria de {STATE_POSTAL.get(g_top)}: calor y días con visita",
               como_leer="La línea gris es el índice de calor diario; los puntos naranjas "
                         "marcan los días con al menos una visita por calor.",
               que_explica=f"En {STATE_POSTAL.get(g_top)}, el estado con más días positivos, "
                           f"las visitas se agrupan en los tramos más calurosos, pero también "
                           f"hay días calurosos sin visitas y visitas en días moderados. Esa "
                           f"superposición es la que ningún umbral fijo resuelve y es el "
                           f"margen que un modelo puede o no aprovechar.")

    pos_rate = f.loc[f["y"] == 1, "rate"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(np.log10(pos_rate), bins=30, color="#2f6fb0")
    ax.set_xlabel("log10(tasa diaria), solo días con visita")
    ax.set_ylabel("estado-días")
    _style(ax)
    rep.figure(fig, "Distribución de la tasa en los días con visita",
               como_leer="Histograma del logaritmo de la tasa diaria cuando es mayor que 0.",
               que_explica=f"La tasa positiva va de {_n(pos_rate.min(), 1)} a "
                           f"{_n(pos_rate.max(), 0)} (mediana {_n(pos_rate.median(), 0)}). La "
                           f"población VA es pequeña en cada estado, así que una sola visita "
                           f"produce tasas altas y muy variables. Por eso el desenlace se "
                           f"modela como ocurrencia (0/1) y no como tasa: predecir el valor "
                           f"exacto sería predecir ruido de conteos pequeños.")

    # ================================================= 3. preparacion
    rep.section("3. Preparación de los datos",
                "Transformaciones del estado bitemporal del gemelo a la tabla de modelado "
                "(<code>xdt.features.panel</code>).")
    rep.table(pd.DataFrame([
        ["Exposición estatal", "gridMET por condado (centroide de población) → promedio "
                               "ponderado por población. Se descargan los condados más poblados "
                               "hasta cubrir el 80 % de cada estado", "evita describir el clima "
                               "del desierto vacío en vez del de las ciudades"],
        ["Cobertura mínima", "se descartan estado-días con < 70 % de la población con dato",
         "un estado medio vacío no es el estado"],
        ["Umbral local p95", f"calculado solo con {min(panel.ref_years)}–{max(panel.ref_years)} "
                             f"(años sin desenlace en el panel)", "sin fuga: el umbral nunca "
                                                                 "ve el año que se evalúa"],
        ["Retardos y medias móviles", "solo días pasados y el presente",
         "una variable no puede usar el futuro"],
        ["Temporada cálida", "mayo–septiembre", "fuera de ella el desenlace es ~0 y el modelo "
                                                "aprendería estación, no calor"],
        ["Desenlace binario", "y = 1 si tasa > 0", "tasas de conteos pequeños son ruidosas"],
    ], columns=["Paso", "Regla", "Por qué"]), "Reglas de preparación y control de fuga",
        como_leer="Cada fila es un paso aplicado en orden, con la razón que lo justifica.",
        que_explica="La fuga de información (usar datos del futuro o del conjunto de prueba) "
                    "es la forma más común de obtener un AUC que no se reproduce. Cada regla "
                    "cierra una vía concreta de fuga.")
    feat_tbl = pd.DataFrame({"variable": list(FEATURES), "descripción": list(FEATURES.values())})
    rep.table(feat_tbl, "Catálogo de variables del modelo",
              como_leer="Nombre de la columna en el código y su significado físico.",
              que_explica="Todas son observables el mismo día o antes: el modelo puede usarse "
                          "en tiempo real con el pronóstico de gridMET. "
                          "<code>hi_clim_ref</code> captura la aclimatación: el mismo índice "
                          "de calor pesa distinto en Maine que en Arizona.")

    # ================================================= 4. modelado
    rep.section("4. Modelado: entrenamiento e hiperparámetros",
                "Escalera de modelos de complejidad creciente (§8 del plan). Cada peldaño "
                "solo se justifica si supera al anterior con significancia.")
    ladder = pd.DataFrame([{
        "modelo": r.spec.name, "nivel": r.spec.nivel, "descripción": r.spec.descripcion,
        "hiperparámetros buscados": ", ".join(k.split("__")[-1] for k in r.spec.param_space)
        or "umbral elegido por Youden",
    } for r in run.results.values()])
    rep.table(ladder, "Escalera de modelos",
              como_leer="Nivel 0 es la regla de umbral (referencia); los niveles superiores "
                        "son más flexibles pero menos transparentes.",
              que_explica="Construirlos en orden permite responder no solo "
                          "«¿qué modelo gana?» sino «¿cuánta complejidad hace falta?». Si la "
                          "logística empata con LightGBM, se prefiere la logística porque "
                          "sus coeficientes se leen directamente.")

    hp_rows = []
    for k, prm in run.final_params.items():
        for pk, pv in prm.items():
            hp_rows.append({"modelo": k, "hiperparámetro": pk.split("__")[-1],
                            "valor elegido": f"{pv:.4g}" if isinstance(pv, float) else str(pv)})
    if hp_rows:
        rep.table(pd.DataFrame(hp_rows), "Hiperparámetros finales (búsqueda sobre todo el panel)",
                  como_leer="Valor elegido por búsqueda aleatoria con validación cruzada "
                            "agrupada por estado (GroupKFold), maximizando el AUC.",
                  que_explica="Agrupar por estado dentro de la búsqueda impide que el mismo "
                              "estado aparezca a ambos lados, lo que favorecería "
                              "hiperparámetros que memorizan estados. Estos son los valores "
                              "del modelo final que se explica en la sección 5 y se "
                              "despliega en el gemelo; los de cada pliegue están en el JSON "
                              "de métricas.")

    for k in ("lightgbm", "logistica"):
        if k not in run.results or not run.results[k].search_tables:
            continue
        t = pd.concat(run.results[k].search_tables, ignore_index=True)
        if k == "lightgbm":
            fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
            for ax, prm, lab in ((axs[0], "param_learning_rate", "learning_rate (escala log)"),
                                 (axs[1], "param_num_leaves", "num_leaves")):
                ax.scatter(t[prm].astype(float), t["mean_test_score"], s=10, alpha=0.5,
                           color=PALETTE[k])
                ax.set_xlabel(lab)
                _style(ax)
            axs[0].set_xscale("log")
            axs[0].set_ylabel("AUC de validación interna")
            best_row = t.loc[t["mean_test_score"].idxmax()]
            rng = t["mean_test_score"].max() - t["mean_test_score"].quantile(0.1)
            rep.figure(fig, "Búsqueda de hiperparámetros de LightGBM",
                       como_leer="Cada punto es una configuración probada en un pliegue "
                                 "externo; el eje vertical es su AUC en la validación interna.",
                       que_explica=f"La mejor configuración usó learning_rate ≈ "
                                   f"{_n(float(best_row['param_learning_rate']), 3)} y "
                                   f"num_leaves = {int(best_row['param_num_leaves'])}. Entre "
                                   f"el decil inferior y la mejor configuración hay "
                                   f"{_n(rng, 3)} de AUC: si esa diferencia es pequeña, el "
                                   f"modelo es poco sensible a los hiperparámetros y el "
                                   f"resultado no depende de haber afinado de más.")
        else:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.scatter(t["param_logisticregression__C"].astype(float), t["mean_test_score"],
                       s=12, alpha=0.6, color=PALETTE[k])
            ax.set_xscale("log")
            ax.set_xlabel("C (inversa de la regularización, escala log)")
            ax.set_ylabel("AUC de validación interna")
            _style(ax)
            rep.figure(fig, "Búsqueda de la regularización de la logística",
                       como_leer="Cada punto es un valor de C probado. C pequeño = coeficientes "
                                 "más encogidos hacia 0.",
                       que_explica="Una curva plana indica que la regularización apenas "
                                   "cambia el resultado: las variables tienen señal estable y "
                                   "el modelo no depende de un ajuste fino.")

    # ================================================= 5. evaluacion
    rep.section("5. Evaluación",
                f"Validación cruzada anidada: externo = <b>{run.cv_scheme}</b> "
                f"({len(run.folds)} pliegues), interno = GroupKFold por estado. Todas las "
                f"métricas de esta sección son <b>fuera de muestra</b>.")

    blocks = (geo.astype(str) + "-" + f["year"].astype(str) + "-" +
              f["week"].astype(str)).to_numpy()
    rows = []
    for k, r in run.results.items():
        p = r.oof[mask]
        auc, alo, ahi = sel["auc_ci"][k]
        _, blo, bhi = st.block_bootstrap_ci(y[mask], p, blocks[mask], roc_auc_score,
                                            n_boot=n_boot)
        rows.append({
            "modelo": k, "AUC (OOF)": auc, "IC95 DeLong": f"{_n(alo, 3)}–{_n(ahi, 3)}",
            "IC95 bootstrap bloques": f"{_n(blo, 3)}–{_n(bhi, 3)}",
            "AUC medio pliegues": r.fold_scores["auc"].mean(),
            "desv. pliegues": r.fold_scores["auc"].std(),
            "AP": average_precision_score(y[mask], p),
            "Brier": brier_score_loss(y[mask], p),
            "AUC temporal": r.temporal_auc,
        })
    perf = pd.DataFrame(rows)
    stats_out["rendimiento"] = perf.to_dict(orient="records")
    rep.table(perf, "Rendimiento fuera de muestra por modelo",
              como_leer="AUC: probabilidad de que un día con visita reciba más riesgo que uno "
                        "sin visita (0,5 = azar, 1 = perfecto). AP: precisión promedio, "
                        f"sensible a la clase rara (azar = {_n(y[mask].mean(), 3)}). Brier: "
                        "error cuadrático de la probabilidad (menor es mejor). «AUC temporal»: "
                        f"entrenar hasta el {run.temporal_split['corte']} y evaluar desde el "
                        f"{run.temporal_split['inicio_prueba']}.",
              que_explica=f"{best} obtiene el mayor AUC global ({_n(sel['auc_ci'][best][0])}). "
                          f"El AUC global junta las predicciones de todos los pliegues y premia "
                          f"también distinguir estados entre sí; el «AUC medio pliegues» solo "
                          f"mide ordenar días dentro de una región no vista. Para {chosen}, "
                          f"global {_n(sel['auc_ci'][chosen][0])} frente a "
                          f"{_n(run.results[chosen].fold_scores['auc'].mean())} dentro de región: "
                          f"parte del poder del modelo es saber qué estados tienen más riesgo "
                          f"en general (aclimatación, clima), no solo qué días. Dos "
                          f"intervalos se reportan a propósito: el de DeLong supone días "
                          f"independientes y el bootstrap por bloques (estado × semana) no. "
                          f"Cuando el segundo es más ancho, esa diferencia es la dependencia "
                          f"temporal que una prueba ingenua habría ignorado. El AUC temporal "
                          f"comprueba que el modelo también funciona hacia adelante en el "
                          f"tiempo, no solo en regiones nuevas.")

    # ROC
    fig, ax = plt.subplots(figsize=(5.8, 5))
    for k, r in run.results.items():
        fpr, tpr, _ = roc_curve(y[mask], r.oof[mask])
        ax.plot(fpr, tpr, color=PALETTE.get(k), lw=2 if k == chosen else 1.2,
                label=f"{k} (AUC {_n(sel['auc_ci'][k][0], 3)})")
    ax.plot([0, 1], [0, 1], ls="--", color="#bbb", lw=1)
    ax.set_xlabel("tasa de falsas alarmas (1 − especificidad)")
    ax.set_ylabel("tasa de detección (sensibilidad)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    rep.figure(fig, "Curvas ROC fuera de muestra",
               como_leer="Cada curva muestra, para todos los umbrales posibles, cuántos días con "
                         "visita se detectan (vertical) a cambio de cuántas falsas alarmas "
                         "(horizontal). La diagonal es el azar.",
               que_explica="Una curva más cercana a la esquina superior izquierda detecta más "
                           "días de riesgo con menos falsas alarmas. Si dos curvas se cruzan, "
                           "ningún modelo domina en todos los umbrales y la elección depende "
                           "de cuántas falsas alarmas tolere el servicio de salud.")

    # AUC por pliegue
    fs = pd.DataFrame({k: r.fold_scores.set_index("pliegue")["auc"] for k, r in
                       run.results.items()})
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for k in fs.columns:
        ax.plot(range(len(fs)), fs[k], "o-", color=PALETTE.get(k), label=k, ms=4)
    ax.set_xticks(range(len(fs)), fs.index, rotation=30, fontsize=8)
    ax.set_ylabel("AUC en el pliegue")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    _style(ax)
    worst = fs[chosen].idxmin()
    rep.figure(fig, "AUC por pliegue externo",
               como_leer="Cada punto es el AUC de un modelo en un pliegue (una región o un año "
                         "que el modelo no vio al entrenar).",
               que_explica=f"La dispersión entre pliegues muestra cuánto depende el resultado "
                           f"de la región evaluada. El peor pliegue de {chosen} es «{worst}» "
                           f"(AUC {_n(fs.loc[worst, chosen])}): es donde el gemelo sería "
                           f"menos confiable y donde conviene revisar datos o calibrar "
                           f"localmente. Estas son las unidades que usa la prueba de Friedman.")

    # ---- pruebas estadisticas
    rep.h3("5.1 Pruebas estadísticas robustas")
    rep.p("El panel tiene dependencia temporal (días consecutivos) y espacial (estados "
          "vecinos). Las pruebas clásicas suponen independencia y dan p-valores demasiado "
          "optimistas; cada prueba elegida corrige una de esas violaciones (Tabla siguiente).")
    rep.table(pd.DataFrame([
        ["DeLong", "¿el AUC de A difiere del de B?", "compara curvas ROC pareadas (mismos "
         "días), sin suponer normalidad de las puntuaciones"],
        ["Bootstrap por bloques", "IC de AUC, AP, Brier", "remuestrea bloques estado × semana "
         "completos: conserva la autocorrelación"],
        ["Friedman + Nemenyi", "¿hay diferencias entre los k modelos a lo largo de los pliegues?",
         "no paramétrica, basada en rangos; recomendada por Demšar (2006) para comparar "
         "clasificadores"],
        ["t corregida de Nadeau–Bengio", "A vs B con las métricas de cada pliegue",
         "los pliegues comparten datos de entrenamiento; la corrección infla la varianza"],
        ["Permutación por desplazamiento circular", "¿el modelo aprende algo o es azar?",
         "desplaza la serie de cada estado: rompe la alineación sin destruir la "
         "autocorrelación"],
        ["Calibración + Spiegelhalter", "¿las probabilidades son honestas?",
         "no depende de agrupar en deciles, a diferencia de Hosmer–Lemeshow"],
        ["Diebold–Mariano (HAC)", "¿la pérdida diaria de A es menor que la de B?",
         "varianza de Newey–West: tolera errores correlacionados en el tiempo"],
        ["Holm–Bonferroni", "control de comparaciones múltiples",
         "más potente que Bonferroni y controla el error de familia"],
    ], columns=["Prueba", "Pregunta", "Por qué es robusta aquí"]),
        "Pruebas estadísticas aplicadas y su justificación",
        como_leer="Una fila por prueba: qué pregunta responde y qué supuesto clásico evita.",
        que_explica="Con datos de panel, usar una prueba t sobre filas o un bootstrap "
                    "ordinario llevaría a declarar diferencias que no existen. Estas pruebas "
                    "son las que corresponden a un gemelo digital con datos espacio-temporales.")

    # DeLong + Holm
    dl = pd.DataFrame([{
        "comparación": f"{best} vs {k}", "AUC A": t.auc_a, "AUC B": t.auc_b,
        "diferencia": t.diff, "IC95 dif.": f"{_n(t.ci_low, 3)}–{_n(t.ci_high, 3)}",
        "z": t.z, "p": t.p_value,
    } for k, t in sel["delong"].items()])
    if not dl.empty:
        dl = dl.merge(sel["holm"][["comparacion", "p_holm", "rechaza_H0"]],
                      left_on="comparación", right_on="comparacion").drop(columns="comparacion")
        stats_out["delong_holm"] = dl.to_dict(orient="records")
        rep.table(dl, "DeLong: mejor modelo contra cada alternativa (corrección de Holm)",
                  como_leer="Cada fila compara el modelo con mayor AUC (A) contra otro (B). "
                            "p_holm es el p-valor corregido por hacer varias comparaciones; "
                            "«rechaza_H0 = sí» significa diferencia significativa (α = 0,05).",
                  que_explica=f"<b>Filtro A de la selección.</b> Modelos que no son "
                              f"significativamente peores que {best} en AUC global: "
                              f"{', '.join(sel['no_peores'])}. El resto queda descartado.")

    # Friedman + Nemenyi
    try:
        fr = sel["friedman"]
        if fr is None:
            raise ValueError("sin Friedman")
        stats_out["friedman"] = st.as_dict(fr)
        ranks = pd.Series(fr.mean_ranks).sort_values()
        fig, ax = plt.subplots(figsize=(7, 2.2))
        ax.hlines(0, 1, len(ranks), color="#999")
        for i, (k, v) in enumerate(ranks.items()):
            ax.plot(v, 0, "o", color=PALETTE.get(k), ms=9)
            ax.annotate(f"{k}\n{_n(v, 2)}", (v, 0), (0, 12 if i % 2 else -30),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.plot([ranks.iloc[0], ranks.iloc[0] + fr.critical_difference], [0.75, 0.75],
                color="#b3261e", lw=3)
        ax.text(ranks.iloc[0], 0.85, f"CD = {_n(fr.critical_difference, 2)}", fontsize=8,
                color="#b3261e")
        ax.set_xlim(0.5, len(ranks) + 0.5)
        ax.set_ylim(-0.9, 1.05)
        ax.set_yticks([])
        ax.set_xlabel("rango medio (1 = mejor)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        rep.figure(fig, "Friedman + Nemenyi: rangos medios entre pliegues",
                   como_leer="En cada pliegue se ordenan los modelos por AUC (1 = mejor) y se "
                             "promedia su posición. La barra roja es la diferencia crítica "
                             "(CD): dos modelos cuyos rangos se separan menos que la CD no "
                             "difieren significativamente.",
                   que_explica=f"Friedman χ² = {_n(fr.statistic, 2)}, p {_p(fr.p_value)} "
                               f"sobre {fr.n_blocks} pliegues. "
                               + (f"Hay diferencias entre modelos dentro de las regiones. "
                                  f"<b>Filtro B de la selección:</b> quedan a menos de una CD "
                                  f"del mejor rango {', '.join(sel['nemenyi_ok'])}."
                                  if fr.p_value < 0.05 else
                                  "No se detectan diferencias globales entre modelos a lo "
                                  "largo de los pliegues: con pocos pliegues esta prueba tiene "
                                  "poca potencia, por eso la decisión se apoya también en "
                                  "DeLong sobre todas las predicciones."))
    except ValueError:
        pass

    # Nadeau-Bengio, DM, permutacion, calibracion
    extra = []
    ntr = int(np.mean([len(fo.train) for fo in run.folds]))
    nte = int(np.mean([len(fo.test) for fo in run.folds]))
    dates = f.index.get_level_values("valid_date")
    for k in names:
        if k == chosen:
            continue
        diffs = (fs[chosen] - fs[k]).dropna()
        t_nb, p_nb = st.corrected_resampled_t(diffs.to_numpy(), ntr, nte)
        la = pd.Series((run.results[chosen].oof[mask] - y[mask]) ** 2).groupby(
            dates[mask]).mean()
        lb = pd.Series((run.results[k].oof[mask] - y[mask]) ** 2).groupby(dates[mask]).mean()
        dm, p_dm = st.diebold_mariano(la, lb)
        extra.append({"comparación": f"{chosen} vs {k}", "Δ AUC medio pliegues": diffs.mean(),
                      "t Nadeau–Bengio": t_nb, "p NB": p_nb, "DM (Brier diario)": dm,
                      "p DM": p_dm})
    if extra:
        ex = pd.DataFrame(extra)
        stats_out["nadeau_bengio_dm"] = ex.to_dict(orient="records")
        rep.table(ex, f"Nadeau–Bengio y Diebold–Mariano: {chosen} contra las alternativas",
                  como_leer="Δ AUC > 0 favorece al modelo elegido. En Diebold–Mariano, un "
                            "estadístico negativo significa que el modelo elegido tiene "
                            "menor error de Brier diario.",
                  que_explica="Son dos vistas complementarias a DeLong: Nadeau–Bengio usa las "
                              "métricas por pliegue corrigiendo que los pliegues se solapan; "
                              "Diebold–Mariano compara el error día a día como una serie de "
                              "tiempo, que es como opera el gemelo. Si las tres pruebas "
                              "coinciden, la conclusión no depende de la prueba elegida.")

    pr = []
    geos = geo.to_numpy()
    for k, r in run.results.items():
        obs, p_perm = st.circular_shift_permutation_test(
            y[mask], r.oof[mask], geos[mask], roc_auc_score, n_perm=n_perm)
        cal = st.calibration(y[mask], r.oof[mask])
        pr.append({"modelo": k, "AUC": obs, "p permutación": p_perm,
                   "pendiente calibración": cal.slope, "intercepto": cal.intercept,
                   "z Spiegelhalter": cal.spiegelhalter_z, "p Spiegelhalter": cal.spiegelhalter_p})
    prd = pd.DataFrame(pr)
    stats_out["permutacion_calibracion"] = prd.to_dict(orient="records")
    cal_c = prd.set_index("modelo").loc[chosen]
    rep.table(prd, "Permutación (¿aprende algo?) y calibración (¿probabilidades honestas?)",
              como_leer="p permutación < 0,05: el AUC no se explica por azar. Pendiente de "
                        "calibración 1 e intercepto 0 = probabilidades perfectas; "
                        "p Spiegelhalter < 0,05 = mal calibrado.",
              que_explica=f"Para {chosen}: pendiente {_n(cal_c['pendiente calibración'], 2)} "
                          + ("(< 1: probabilidades algo extremas, típico al transferir a "
                             "regiones no vistas)" if cal_c["pendiente calibración"] < 0.9 else
                             "(> 1: probabilidades algo tímidas)"
                             if cal_c["pendiente calibración"] > 1.1 else "(cercana a 1)")
                          + f", Spiegelhalter p {_p(cal_c['p Spiegelhalter'])}. "
                          f"<b>Filtro C de la selección:</b> modelos calibrados (p ≥ 0,05): "
                          f"{', '.join(sel['calibrados']) or 'ninguno'}. "
                          "Un modelo puede ordenar bien (AUC alto) y aun así dar "
                          "probabilidades mal escaladas; para emitir alertas con un % de "
                          "riesgo, lo segundo importa. La capa L5 del gemelo recalibra con "
                          "las observaciones nuevas.")

    fig, ax = plt.subplots(figsize=(5.8, 5))
    for k, r in run.results.items():
        pt, pp = calibration_curve(y[mask], r.oof[mask], n_bins=10, strategy="quantile")
        ax.plot(pp, pt, "o-", color=PALETTE.get(k), label=k, ms=4)
    lim = max(0.05, float(np.nanmax([r.oof[mask].max() for r in run.results.values()])))
    ax.plot([0, lim], [0, lim], ls="--", color="#bbb")
    ax.set_xlabel("probabilidad predicha")
    ax.set_ylabel("frecuencia observada")
    ax.legend(frameon=False, fontsize=8)
    rep.figure(fig, "Diagrama de fiabilidad (calibración)",
               como_leer="Los días se agrupan por probabilidad predicha (deciles). Si el modelo "
                         "está calibrado, los puntos caen sobre la diagonal: de los días con "
                         "20 % de riesgo, el 20 % tuvo visita.",
               que_explica="Puntos bajo la diagonal = el modelo sobreestima el riesgo; sobre la "
                           "diagonal = lo subestima. Complementa la pendiente y la prueba de "
                           "Spiegelhalter de la tabla anterior.")

    # ---- seleccion
    rep.h3("5.2 Selección del mejor modelo")
    holm_tbl = sel["holm"].set_index("comparacion") if len(sel["holm"]) else None
    key_u = f"{best} vs umbral"
    if chosen == "umbral":
        vs_umbral = ("Ningún modelo aprendido supera a la regla de umbral bajo los tres "
                     "filtros: con estos datos, aprender del histórico no mejora la alerta "
                     "meteorológica.")
    elif "umbral" in sel["no_peores"]:
        vs_umbral = ("La regla de umbral no es significativamente peor en AUC global, pero no "
                     "supera los filtros de ranking dentro de región o de calibración.")
    else:
        p_u = (f", p Holm {_p(float(holm_tbl.loc[key_u, 'p_holm']))}"
               if holm_tbl is not None and key_u in holm_tbl.index else "")
        vs_umbral = (f"La regla de umbral queda significativamente por debajo en AUC global "
                     f"({_n(auc_u)} frente a {_n(sel['auc_ci'][best][0])} de {best}{p_u}).")
    recal = ("<br><b>Advertencia:</b> ningún modelo que supera A y B está calibrado; el elegido "
             "se despliega marcado para recalibración en L5 antes de emitir probabilidades."
             if sel["requiere_recalibracion"] else "")
    rep.raw(
        f'<div class="verdict"><b>Modelo elegido: {chosen}</b><br>'
        f"Regla: el modelo <b>más simple</b> que cumple los tres criterios de éxito:<br>"
        f"A · AUC global no significativamente peor que el mejor (DeLong + Holm): "
        f"{', '.join(sel['no_peores'])}<br>"
        f"B · ranking dentro de región no significativamente peor (Friedman + Nemenyi): "
        f"{', '.join(sel['nemenyi_ok'])}<br>"
        f"C · probabilidades calibradas (Spiegelhalter p ≥ 0,05): "
        f"{', '.join(sel['calibrados']) or 'ninguno'}<br>"
        f"Cumplen A y B: {', '.join(sel['filtro_ab']) or 'ninguno'} · cumplen A, B y C: "
        f"{', '.join(sel['filtro_abc']) or 'ninguno'}.<br>{vs_umbral}{recal}</div>"
    )
    rep.p("<i>Nota de trazabilidad:</i> la primera corrida de este motor seleccionaba solo "
          "con el filtro A. Los filtros B y C se añadieron al comprobar que el modelo así "
          "elegido incumplía el criterio de calibración de la ficha del problema (sección 1); "
          "ambos eran criterios de éxito declarados, no métricas nuevas.")

    # ---- explicabilidad
    rep.h3("5.3 Explicabilidad del modelo")
    expl_model = chosen if run.results[chosen].spec.explicacion != "regla" else "lightgbm"
    fm = run.final_models[expl_model]
    X = panel.X
    if run.results[expl_model].spec.explicacion in ("shap_tree", "coef"):
        ex_, Xs = xai.shap_values(fm, X)
        imp = xai.global_importance(ex_)
        stats_out["shap_global"] = imp.round(5).to_dict()
        signs = {v: np.sign(np.corrcoef(Xs[v], ex_.values[:, list(Xs.columns).index(v)])[0, 1])
                 for v in imp.index}
        fig, ax = plt.subplots(figsize=(7, 4.4))
        imp[::-1].plot.barh(ax=ax, color=PALETTE.get(expl_model))
        ax.set_xlabel("|SHAP| medio (log-odds)")
        _style(ax)
        rep.figure(fig, f"Importancia global SHAP ({expl_model})",
                   como_leer="Longitud de la barra = cuánto mueve en promedio cada variable la "
                             "predicción (en log-odds), sin importar el signo.",
                   que_explica="Las tres variables que más pesan son "
                               + "; ".join(
                                   f"{v} (valores altos {'suben' if sgn > 0 else 'bajan'} "
                                   f"el riesgo)" for v, sgn in list(signs.items())[:3])
                               + ". Explica en qué se apoya el modelo, no qué causa las "
                                 "visitas: si dos variables están correlacionadas (Figura de "
                                 "correlación), el mérito se reparte entre ellas. Que "
                                 "hi_clim_ref suba el riesgo significa que los estados de "
                                 "clima cálido tienen más días con visitas en general (más "
                                 "exposición acumulada), no que la aclimatación proteja menos.")

        import shap

        plt.figure(figsize=(8, 5))
        shap.plots.beeswarm(ex_, max_display=14, show=False)
        fig = plt.gcf()
        rep.figure(fig, "Resumen SHAP: dirección de cada efecto",
                   como_leer="Cada punto es un estado-día. Horizontal: cuánto sube (derecha) o "
                             "baja (izquierda) el riesgo esa variable en ese día. Color: valor "
                             "de la variable (rojo alto, azul bajo).",
                   que_explica="Si los puntos rojos de una variable térmica quedan a la derecha, "
                               "más calor sube el riesgo, que es lo físicamente esperable; un "
                               "patrón al revés sería una señal de alerta sobre los datos. "
                               "Permite auditar que el modelo aprendió relaciones con sentido.")

        # Interaccion calor x aclimatacion: requiere un modelo de arboles; en la
        # logistica cada efecto es aditivo por construccion y no puede mostrarla.
        is_tree = run.results[expl_model].spec.explicacion == "shap_tree"
        inter_model = expl_model if is_tree else "lightgbm"
        if inter_model == expl_model:
            ex_t, Xt = ex_, Xs
        else:
            ex_t, Xt = xai.shap_values(run.final_models[inter_model], X)
        thermal = [c for c in THERMAL_DAY if c in Xt.columns]
        idx = [list(Xt.columns).index(c) for c in thermal]
        load = ex_t.values[:, idx].sum(axis=1)
        clim = Xt["hi_clim_ref"].to_numpy()
        hi = Xt["heat_index"].to_numpy()
        band = (hi >= 32) & (hi <= 36)
        cool = band & (clim < np.median(clim))
        warm = band & (clim >= np.median(clim))
        both = cool.any() and warm.any()
        d_cw = float(load[cool].mean() - load[warm].mean()) if both else float("nan")
        stats_out["interaccion_aclimatacion"] = {
            "modelo": inter_model, "banda_hi": [32, 36],
            "shap_termico_estados_frescos": float(load[cool].mean()) if cool.any() else None,
            "shap_termico_estados_calidos": float(load[warm].mean()) if warm.any() else None,
            "diferencia": d_cw,
        }
        fig, ax = plt.subplots(figsize=(7, 4))
        sc = ax.scatter(hi, load, c=clim, s=8, cmap="coolwarm", alpha=0.7)
        fig.colorbar(sc, ax=ax, label="clima de referencia del estado (°C)")
        ax.axhline(0, color="#999", lw=0.8)
        ax.axvspan(32, 36, color="#f2c14e", alpha=0.15)
        ax.set_xlabel("índice de calor del día (°C)")
        ax.set_ylabel("contribución SHAP térmica total (log-odds)")
        _style(ax)
        lin_note = ("" if inter_model == expl_model else
                    f" Se usa {inter_model} porque en la {expl_model} cada efecto es aditivo "
                    f"por construcción y no puede mostrar esta interacción.")
        rep.figure(fig, f"Calor × aclimatación: contribución térmica ({inter_model})",
                   como_leer="Cada punto es un estado-día: horizontal su índice de calor, "
                             "vertical la suma de las contribuciones SHAP de todas las "
                             "variables térmicas del día (" + ", ".join(thermal) + "). Color: "
                             "clima habitual del estado (azul = fresco, rojo = cálido). La "
                             "franja amarilla es la banda 32–36 °C que se compara.",
                   que_explica=f"En la banda de 32–36 °C, la contribución térmica media es "
                               f"{_n(float(load[cool].mean()) if cool.any() else float('nan'), 2)} "
                               f"en estados frescos y "
                               f"{_n(float(load[warm].mean()) if warm.any() else float('nan'), 2)} "
                               f"en estados cálidos (diferencia {_n(d_cw, 2)} log-odds). "
                               + ("A igual calor, el modelo asigna más riesgo donde la población "
                                  "está menos acostumbrada: es consistente con la hipótesis de "
                                  "aclimatación del plan (asociación del modelo, no prueba "
                                  "causal)." if d_cw > 0.05 else
                                  "El modelo no asigna más riesgo al mismo calor en estados "
                                  "frescos: con estos datos no aparece la señal de aclimatación "
                                  "como interacción." if d_cw < -0.05 else
                                  "La diferencia es mínima: el modelo no distingue el efecto del "
                                  "mismo calor según el clima habitual del estado.")
                               + lin_note)

        k_top = int(np.argmax(fm.predict_proba(Xs)[:, 1]))
        row = Xs.index[k_top]
        plt.figure(figsize=(8, 4.5))
        shap.plots.waterfall(ex_[k_top], max_display=10, show=False)
        fig = plt.gcf()
        rep.figure(fig, f"Explicación local: {STATE_POSTAL.get(row[0], row[0])} el "
                        f"{row[1]:%d/%m/%Y}",
                   como_leer="Parte del riesgo promedio (E[f(x)], abajo) y cada barra suma "
                             "(roja) o resta (azul) la contribución de una variable ese día, "
                             "hasta el riesgo final f(x) arriba, en log-odds.",
                   que_explica="Es la explicación que el gemelo entregaría junto a una alerta: "
                               "no solo «riesgo alto», sino qué condiciones del día lo "
                               "provocaron. Es el estado-día con mayor riesgo predicho en la "
                               "muestra explicada.")

    if "logistica" in run.final_models:
        orr = xai.odds_ratios(run.final_models["logistica"], list(X.columns))
        stats_out["odds_ratios"] = orr.round(4).to_dict(orient="records")
        rep.table(orr, "Logística: odds ratio por +1 desviación estándar",
                  como_leer="Odds ratio > 1: al subir la variable una desviación estándar "
                            "(columna desv_std, en sus unidades), las chances de visita se "
                            "multiplican por ese factor, manteniendo el resto fijo.",
                  que_explica=f"El efecto más fuerte es {orr.iloc[0]['variable']} "
                              f"(OR = {_n(orr.iloc[0]['odds_ratio_1sd'], 2)}). Con variables "
                              "correlacionadas, un OR aislado puede cambiar de signo respecto "
                              "a la relación bivariada: se interpreta en conjunto con el "
                              "SHAP, no por separado.")

    pim = xai.permutation_importance_table(fm, X.iloc[run.temporal_split["idx_test"]],
                                           y[run.temporal_split["idx_test"]])
    rep.table(pim, f"Importancia por permutación ({expl_model}, periodo de prueba temporal)",
              como_leer="Cuánto cae el AUC al desordenar al azar cada variable en datos que el "
                        "modelo no usó para elegir hiperparámetros.",
              que_explica="Es independiente del tipo de modelo y verifica el ranking SHAP con "
                          "otra técnica. Una variable con caída ≈ 0 podría eliminarse sin "
                          "pérdida; una caída negativa indica que solo aporta ruido.")

    # ================================================= 6. despliegue
    rep.section("6. Despliegue en el gemelo digital",
                "Cómo se integra el modelo en las capas del gemelo (§6 del plan).")
    rep.table(pd.DataFrame([
        ["L1 estado", "las variables se calculan desde el estado bitemporal: una predicción "
                      "puede reproducirse con el corte bitemporal con que se emitió"],
        ["L0 trazabilidad", "cada predicción se registra con versión de modelo, hash de "
                            "entrada y semilla"],
        ["L4 escenarios", "un escenario (ola de calor +3 °C, apagón) perturba el estado y el "
                          "modelo traduce la perturbación en riesgo"],
        ["L5 calibración", "las observaciones posteriores recalibran el modelo si la pendiente "
                           "de calibración se aleja de 1"],
        ["Limitaciones", "desenlace de población VA (mayor, masculina), un solo año sin token "
                         "EPHT, exposición estatal (no intraurbana), sin apagones en el modelo "
                         "porque EAGLE-I aún cubre solo AZ y LA"],
    ], columns=["Capa", "Uso del modelo"]), "Integración y limitaciones",
        como_leer="Qué hace cada capa del gemelo con el modelo elegido.",
        que_explica="El modelo no es un entregable aislado: es el emulador de la capa L3. "
                    "Las limitaciones se declaran porque acotan a quién y a qué escala se "
                    "pueden aplicar las conclusiones.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rep.render(), encoding="utf-8")
    return stats_out


def write_metrics(run: TrainingRun, stats_out: dict[str, Any], path: Path) -> None:
    sel = run.selection
    payload = {
        "esquema_cv": run.cv_scheme,
        "pliegues": [fo.name for fo in run.folds],
        "split_temporal": {k: v for k, v in run.temporal_split.items()
                           if not k.startswith("idx")},
        "elegido": sel["elegido"], "mejor_auc": sel["mejor_auc"],
        "auc_ci": {k: list(v) for k, v in sel["auc_ci"].items()},
        "hiperparametros_finales": run.final_params,
        "hiperparametros_por_pliegue": {k: r.best_params for k, r in run.results.items()},
        "por_pliegue": {k: r.fold_scores.to_dict(orient="records")
                        for k, r in run.results.items()},
        **stats_out,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
