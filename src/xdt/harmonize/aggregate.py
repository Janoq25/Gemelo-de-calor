"""Agregacion espacial ponderada (M0).

La agregacion condado -> estado es donde un promedio simple miente: Arizona
tiene 15 condados y Maricopa concentra ~60 % de la poblacion. Un promedio sin
ponderar da el mismo peso al desierto de La Paz (17.000 habitantes) que a
Phoenix, y describe una exposicion que casi nadie vive.

Regla de cobertura: un agregado se emite solo si las unidades con dato
suman al menos `min_weight_coverage` del peso total esperado del grupo. Un
estado con la mitad de sus condados sin dato NO es "el estado"; emitirlo
contaminaria en silencio el panel estado-dia de la Etapa A (§5.1).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def population_weighted(
    facts: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    parent_col: str,
    parent_level: str,
    min_weight_coverage: float = 0.99,
    by: Sequence[str] = ("valid_date", "variable"),
) -> pd.DataFrame:
    """Promedio ponderado de hechos hijos agregado al nivel padre.

    Parametros
    ----------
    facts
        Hechos en esquema canonico (geoid, valid_date, variable, value) del
        nivel hijo.
    weights
        Una fila por unidad hija esperada: `geoid`, `population` y
        `parent_col`. Define el universo: el peso total de cada padre sale de
        aqui, no de los hechos, para que un hijo ausente cuente como hueco.
    """
    if not 0.0 < min_weight_coverage <= 1.0:
        raise ValueError("min_weight_coverage debe estar en (0, 1]")

    w = weights[["geoid", "population", parent_col]]
    total = w.groupby(parent_col)["population"].sum().rename("w_total")

    df = facts.dropna(subset=["value"]).merge(w, on="geoid", how="inner")
    if df.empty:
        return pd.DataFrame(
            columns=["geo_level", "geoid", *by, "value", "weight_coverage"]
        )

    df["wv"] = df["value"] * df["population"]
    keys = [parent_col, *by]
    agg = df.groupby(keys, as_index=False).agg(wv=("wv", "sum"), w=("population", "sum"))
    agg = agg.merge(total, left_on=parent_col, right_index=True)
    agg["weight_coverage"] = agg["w"] / agg["w_total"]
    agg = agg[agg["weight_coverage"] >= min_weight_coverage - 1e-12]

    agg["value"] = agg["wv"] / agg["w"]
    agg["geo_level"] = parent_level
    agg = agg.rename(columns={parent_col: "geoid"})
    return agg[["geo_level", "geoid", *by, "value", "weight_coverage"]].reset_index(drop=True)
