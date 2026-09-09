"""Conector del CDC Environmental Public Health Tracking Network (EPHT).

Fuente del desenlace (§4.1 del plan).

Estructura de la API verificada en vivo el 9/9/2026
----------------------------------------------------
Base: https://ephtracking.cdc.gov/apigateway/api/v1

  /contentAreas/json                                   areas tematicas
  /indicators/{contentAreaId}                          indicadores del area
  /measures/{indicatorId}                              medidas del indicador
  /geographicTypes/{measureId}                         niveles geograficos
  /stratificationlevel/{measureId}/{geoTypeId}/{isSmoothed}
  /stratificationTypes/{measureId}/{geoTypeId}/{isSmoothed}
  /getCoreHolder/{measureId}/{stratLevelId}/{geoTypeFilter}/{geoItemsFilter}
                /{temporalTypeIdFilter}/{temporalItemsFilter}/{isSmoothed}/{full}

Dos trampas que costaron tiempo y quedan documentadas para no repetirlas:

1. `/geographicLevels/` devuelve **410 Gone**. Fue sustituido por
   `/geographicTypes/`. Es exactamente el riesgo R6 del plan, ya materializado.

2. El id que devuelve `/geographicTypes/{measureId}` (p.ej. 962 para la medida
   440) **no** es el `stratificationLevelId` que espera `getCoreHolder`. El
   correcto sale de `/stratificationlevel/` y para el nivel estatal es 1.
   Pasar 962 devuelve HTTP 200 con `tableResult` vacio: un fallo silencioso,
   que es la peor clase. `discover()` existe para no volver a adivinar.

Limite de tasa: sin token se observa HTTP 429 tras ~8 peticiones. El token
gratuito se solicita a trackingsupport@cdc.gov y es un bloqueante real de la
ingesta sistematica (checklist §19).
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from xdt.ingest.base import Connector
from xdt.ingest.http import InvalidBody, RawArtifact, RetryableBody

# geographicTypeId de EPHT -> geo_level canonico del gemelo
GEO_TYPE_TO_LEVEL = {1: "state", 2: "county", 13: "hhs_region"}

# Buckets de resultado del sobre de getCoreHolder que contienen filas de datos.
# El sobre trae ~12 buckets y solo uno suele estar poblado: cual, depende de la
# medida. Las series diarias no usan `tableResult`.
RESULT_BUCKETS = (
    "tableResult",
    "dailyEstimatesTableResult",
    "dailyTemperatureTableResult",
    "healthImpactTableResult",
)


@dataclass(frozen=True)
class Measure:
    """Una medida de EPHT del inventario verificado del plan (§4.1)."""

    measure_id: int
    variable: str  # nombre canonico en el estado del gemelo
    geo_type_id: int
    strat_level_id: int
    cadence: str  # 'annual' | 'daily' | 'weekly' | 'quinquennial'
    note: str = ""

    @property
    def geo_level(self) -> str:
        return GEO_TYPE_TO_LEVEL.get(self.geo_type_id, f"geotype_{self.geo_type_id}")


# Catalogo verificado. Los `strat_level_id` de nivel estatal/condado se
# confirmaron contra /stratificationlevel/; los marcados abajo siguen
# pendientes de confirmar con token (el 429 impidio agotar la verificacion).
MEASURES: dict[str, Measure] = {
    "hri_ed_rate_age_adj": Measure(
        440, "hri_ed_rate_age_adj", geo_type_id=1, strat_level_id=1, cadence="annual",
        note="Tasa ajustada por edad de urgencias por HRI. Verificado: 16 estados en 2023.",
    ),
    "hri_ed_count": Measure(
        438, "hri_ed_count", geo_type_id=1, strat_level_id=1, cadence="annual",
        note="Numero de urgencias por HRI.",
    ),
    "hri_ed_rate_crude": Measure(
        439, "hri_ed_rate_crude", geo_type_id=1, strat_level_id=1, cadence="annual",
        note="Tasa cruda de urgencias por HRI.",
    ),
    "hri_ed_rate_daily_va": Measure(
        1385, "hri_ed_rate_daily_va", geo_type_id=1, strat_level_id=1, cadence="daily",
        note=(
            "Serie diaria estatal, poblacion Veterans Affairs. Etapa A del plan (§5.1). "
            "Poblacion centinela, NO representativa: mayor edad, predominio masculino "
            "(limitacion §15.3). strat_level_id pendiente de confirmar con token."
        ),
    ),
    "hri_ed_rate_daily_nonva": Measure(
        1238, "hri_ed_rate_daily_nonva", geo_type_id=13, strat_level_id=1, cadence="daily",
        note=(
            "Serie diaria por region HHS (10 unidades), poblacion no-VA. Serie de "
            "contraste de §5.1 para verificar que los patrones no son artefacto de "
            "la poblacion VA. strat_level_id pendiente de confirmar con token."
        ),
    ),
    "heat_deaths_5yr": Measure(
        1034, "heat_deaths_5yr", geo_type_id=2, strat_level_id=2, cadence="quinquennial",
        note=(
            "Muertes por calor, quinquenio, nivel condado (~3.000 unidades). "
            "Etapa B del plan (§5.2). strat_level_id pendiente de confirmar."
        ),
    ),
}


class EphtConnector(Connector):
    name = "epht"
    cadence = "annual/daily segun medida"
    base_url = "https://ephtracking.cdc.gov/apigateway/api/v1"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.settings.epht_token:
            self.client.headers["Authorization"] = f"Bearer {self.settings.epht_token}"
        self.client.body_check = _check_epht_body

    # ------------------------------------------------------------ descubrir
    def discover(self, measure_id: int, is_smoothed: int = 0) -> dict[str, Any]:
        """Devuelve los ids reales que `getCoreHolder` necesita para una medida.

        Usar esto antes de anadir una medida a MEASURES. Evita el fallo
        silencioso descrito en el docstring del modulo.
        """
        geo_types = self.client.get(f"/geographicTypes/{measure_id}").json()
        out: dict[str, Any] = {"measure_id": measure_id, "geographic_types": geo_types,
                               "stratification_levels": {}}
        for gt in geo_types if isinstance(geo_types, list) else []:
            gt_id = gt.get("geographicTypeId")
            levels = self.client.get(
                f"/stratificationlevel/{measure_id}/{gt_id}/{is_smoothed}"
            ).json()
            out["stratification_levels"][gt_id] = levels
        return out

    # --------------------------------------------------------------- fetch
    def fetch(
        self,
        *,
        measure: str | Measure,
        temporal: Sequence[str | int] | str | int = "ALL",
        geo_items: str = "ALL",
        is_smoothed: int = 0,
        refresh: bool = False,
        **_: Any,
    ) -> list[RawArtifact]:
        """Descarga una medida. Una peticion por periodo temporal.

        Se pide periodo a periodo (no "ALL" en un solo golpe) para que la cache
        sea granular: si un ano falla o se revisa, se re-descarga solo ese ano.
        """
        m = MEASURES[measure] if isinstance(measure, str) else measure
        periods = [temporal] if isinstance(temporal, (str, int)) else list(temporal)

        artifacts: list[RawArtifact] = []
        for period in periods:
            path = (
                f"/getCoreHolder/{m.measure_id}/{m.strat_level_id}/{m.geo_type_id}"
                f"/{geo_items}/0/{period}/{is_smoothed}/0"
            )
            artifacts.append(
                self.client.get(
                    path,
                    resource=f"epht:getCoreHolder:{m.measure_id}:{period}",
                    refresh=refresh,
                )
            )
        return artifacts

    # ----------------------------------------------------------- normalize
    def normalize(self, artifacts: Iterable[RawArtifact]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for art in artifacts:
            payload = art.json()
            for bucket in RESULT_BUCKETS:
                for rec in payload.get(bucket) or []:
                    parsed = self._parse_row(rec, art.raw_hash)
                    if parsed is not None:
                        rows.append(parsed)

        if not rows:
            return pd.DataFrame(columns=[
                "geo_level", "geoid", "valid_date", "variable", "value",
                "suppressed", "ci_low", "ci_high", "raw_hash",
            ])
        return pd.DataFrame(rows)

    @staticmethod
    def _parse_row(rec: dict[str, Any], raw_hash: str) -> dict[str, Any] | None:
        geo_type_id = rec.get("geographicTypeId")
        geoid = rec.get("geoId")
        if geoid is None:
            return None

        valid_date = _parse_temporal(rec.get("temporal"), rec.get("temporalId"))
        if valid_date is None:
            return None

        # `dataValue` llega como cadena; "" y None significan sin dato.
        raw_value = rec.get("dataValue")
        value = pd.to_numeric(raw_value, errors="coerce") if raw_value not in (None, "") else None

        # suppressionFlag "1" = celda suprimida por conteo bajo. NO es un dato
        # faltante: es censura por intervalo (§14 R2 del plan). Se marca para
        # que las capas superiores la traten como tal y no la borren.
        suppressed = str(rec.get("suppressionFlag") or "0") == "1"

        calc = (rec.get("calculationType") or "").strip()
        return {
            "geo_level": GEO_TYPE_TO_LEVEL.get(geo_type_id, f"geotype_{geo_type_id}"),
            "geoid": str(geoid),
            "valid_date": valid_date,
            "variable": _variable_name(rec, calc),
            "value": value,
            "suppressed": suppressed,
            "ci_low": pd.to_numeric(rec.get("confidenceIntervalLow"), errors="coerce"),
            "ci_high": pd.to_numeric(rec.get("confidenceIntervalHigh"), errors="coerce"),
            "raw_hash": raw_hash,
        }

    def source_version(self, artifacts: Iterable[RawArtifact]) -> str | None:
        stamps = [a.fetched_at for a in artifacts]
        return f"epht-{max(stamps):%Y%m%d}" if stamps else None


def _variable_name(rec: dict[str, Any], calc: str) -> str:
    """Nombre canonico de variable a partir de la medida y su tipo de calculo."""
    mid = rec.get("measureId") or rec.get("groupById")
    slug = {
        "Age Adjusted Rate": "age_adj_rate",
        "Crude Rate": "crude_rate",
        "Number": "count",
        "Percent": "pct",
    }.get(calc, calc.lower().replace(" ", "_") or "value")
    return f"epht_m{mid}_{slug}" if mid else f"epht_{slug}"


def _parse_temporal(temporal: Any, temporal_id: Any) -> dt.date | None:
    """Convierte el campo temporal de EPHT en una fecha canonica.

    EPHT mezcla granularidades en el mismo campo:
      "2023"        -> anual        -> 2023-01-01
      "2023-07-15"  -> diario       -> esa fecha
      "2018-2022"   -> quinquenio   -> inicio del periodo
    La granularidad real queda registrada en la cadencia de la medida; aqui
    solo se necesita un ancla comparable.
    """
    s = str(temporal).strip() if temporal is not None else ""
    if not s or s.lower() == "none":
        if temporal_id is None:
            return None
        s = str(temporal_id)

    if len(s) == 4 and s.isdigit():
        return dt.date(int(s), 1, 1)
    if "-" in s:
        head = s.split("-")[0].strip()
        if len(s) >= 8:
            try:
                return pd.to_datetime(s).date()
            except (ValueError, TypeError):
                pass
        if head.isdigit() and len(head) == 4:
            return dt.date(int(head), 1, 1)
    try:
        return pd.to_datetime(s).date()
    except (ValueError, TypeError):
        return None


def _check_epht_body(content: bytes) -> None:
    """Detecta el sobre de error que EPHT devuelve DENTRO de un HTTP 200.

    La API no usa el codigo HTTP para senalar el limite de tasa: responde 200
    con un cuerpo `{"code":429,"status":"Too Many Requests",...}`. Sin esta
    comprobacion el cliente cachearia ese documento en data/raw como si fuera
    dato, y `normalize()` produciria cero filas sin que nada fallara. Es
    exactamente la clase de fallo silencioso contra la que el plan pide tests
    de contrato (§12.2).
    """
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return  # no es JSON; no es asunto de este validador

    if not isinstance(payload, dict):
        return
    code = payload.get("code")
    if not isinstance(code, int) or code < 400:
        return

    msg = payload.get("message", "")
    if code == 429:
        raise RetryableBody(
            f"EPHT limita la tasa (429 en cuerpo 200): {msg}. "
            "Solicita el token gratuito a trackingsupport@cdc.gov."
        )
    raise InvalidBody(f"EPHT devolvio error {code} en cuerpo 200: {msg}")
