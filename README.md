# xai-dt-heat

Motor del **gemelo digital explicable de riesgo de morbilidad por calor con enfoque de
equidad**. Implementa las capas L0–L5 descritas en `plan_gemelo_digital_calor.md` §6.

> Estado: **fundamento construido y probado** (L0, L1, L2, L4 parcial).
> Faltan L3 (modelos), L5 (calibración) y el resto de conectores. Ver *Siguientes pasos*.

## Puesta en marcha

```bash
uv sync --python 3.12          # crea .venv e instala el nucleo
cp .env.example .env           # y rellena XDT_EPHT_TOKEN cuando llegue
uv run pytest                  # 53 tests, sin red
uv run xdt info
```

El stack pesado está en extras, para no arrastrarlo antes de necesitarlo:

```bash
uv sync --extra geo    # geopandas, xarray, rioxarray  (M0, Landsat, gridMET)
uv sync --extra ml     # lightgbm, shap, statsmodels    (M3)
uv sync --extra fair   # fairlearn                      (M5)
```

Python se fija en **3.12**: PyTorch y LightGBM aún no publican wheels para 3.14.

## Uso

```bash
# Descubrir los ids reales de una medida ANTES de anadirla al catalogo
uv run xdt ingest discover 1385

# Ingesta
uv run xdt ingest epht --measure hri_ed_rate_age_adj --temporal 2022,2023

# Estado del gemelo
uv run xdt state coverage
uv run xdt state show --geo-level state --valid-from 2023-01-01

# Viaje en el tiempo: el estado tal como se conocia en una fecha pasada
uv run xdt state show --valid-from 2023-01-01 --known-at 2024-06-01

# Auditar las revisiones de un hecho concreto
uv run xdt state history 04 epht_m1_age_adj_rate 2023-01-01
```

## Arquitectura

| Capa | Modulo | Estado |
|---|---|---|
| **L0** trazabilidad | `twin/registry.py` | ✅ registro de predicciones + observaciones censuradas |
| **L1** estado | `twin/state.py` | ✅ bitemporal, con viaje en el tiempo |
| **L2** sincronizacion | `ingest/` | 🟡 base + conector EPHT; faltan 7 conectores |
| **L3** emulador + XAI | `models/`, `explain/` | ⬜ pendiente |
| **L4** escenarios | `twin/scenarios.py` | 🟡 calor, apagon, enfriamiento, composicion |
| **L5** calibracion | `twin/calibrate.py` | ⬜ pendiente (el esquema ya lo soporta) |

### La decisión de diseño que sostiene el resto: estado bitemporal

`twin_state` guarda **dos** ejes de tiempo:

- `valid_date` — la fecha del mundo real que el hecho describe
- `known_at` — el instante en que el gemelo se enteró del hecho

Las fuentes del plan se revisan: EPHT republica años anteriores, gridMET llega con
rezago y luego se corrige, SVI y PLACES cambian de añada. Si el estado guardara solo
"el último valor conocido", evaluar en 2026 una predicción emitida en 2023 usaría datos
que en 2023 no existían — fuga de información que invalidaría tanto la validación
temporal (§10.2) como la calibración retrospectiva de L5.

Con `known_at`, la pregunta *"¿qué creía el gemelo el 1 de julio de 2023?"* es
contestable. Un valor revisado **no sobrescribe**: crea una revisión nueva y la
anterior sigue siendo recuperable.

```python
# lo que sabemos hoy
store.snapshot(geo_level="state", valid_from="2023-01-01")

# lo que el gemelo creia entonces
store.snapshot(geo_level="state", valid_from="2023-01-01",
               known_at=datetime(2023, 7, 1))
```

### Escenarios componibles (L4)

Firma uniforme de §9, `escenario(estado) -> estado`, con estados inmutables:

```python
from xdt.twin.scenarios import ola_de_calor, apagon

compuesto = ola_de_calor(dias=7, percentil_local=95.0) | apagon(
    fraccion_clientes=0.25, horas=18.0, dias=3
)
contrafactual = compuesto(estado)
contrafactual.lineage   # ('ola_de_calor(...)', 'apagon(...)')
```

Cada estado derivado arrastra su linaje, así que un contrafactual siempre puede
explicar cómo se construyó.

## Hallazgos de la verificación de la API (9/9/2026)

Verificados en vivo; reponen el contenido del anexo `verificacion_datos_epht.md`,
que no está en el repositorio.

| Hallazgo | Consecuencia |
|---|---|
| `/geographicLevels/` devuelve **410 Gone**; el reemplazo es `/geographicTypes/{measureId}` | R6 del plan, ya materializado |
| **429 tras ~8 peticiones** sin token | R6b confirmado. El token es bloqueante real |
| El id de `/geographicTypes/` **no** es el `stratificationLevelId` de `getCoreHolder` | Pasar el equivocado devuelve HTTP 200 con resultado **vacío**: fallo silencioso. Usar `xdt ingest discover` |
| `stratificationLevelId` = **1** para nivel estatal (medida 440) | Verificado |
| Cobertura medida 440: **2023 → 16 estados** (4.2–57.7), **2022 → 26 estados** (4.6–47.0) | Confirma §15.2 y resuelve un pendiente del checklist §19: **los años previos sí tienen mayor cobertura** |
| El sobre de `getCoreHolder` trae ~12 buckets de resultado | Las series diarias no usan `tableResult`. El parser recorre varios buckets |

## Siguientes pasos

En orden, según el cronograma (§13) y el principio de cerrar el circuito pronto (§12.3):

1. **Solicitar el token de EPHT** — bloquea toda ingesta sistemática.
2. Conector `gridmet.py` (4 km diario, netCDF vía OPeNDAP) — fuente climática primaria.
3. Conector `eaglei.py` (apagones por condado, 15 min) — núcleo del evento compuesto.
4. `harmonize/` — crosswalks ZCTA ↔ tract ↔ condado. Es donde §13 avisa que se
   consume más tiempo del previsto.
5. **Hito del mes 4**: baseline de umbral + LightGBM + AUC. Circuito cerrado.
6. `twin/calibrate.py` (L5) — el esquema (`observation`, `calibration_run`) ya está listo.

## Convenciones

- `data/` lo gestiona DVC; nunca entra en git. `data/raw/` es **inmutable**.
- Un conector = una fuente, idempotente, con test de contrato.
- Los tests de contrato golpean la API viva y no corren por defecto:
  `uv run pytest -m contract`. Conviene un cron semanal, no cada commit.
- Los escenarios son contrafactuales **del modelo**, no del mundo (§15.6). El código
  no afirma causalidad y el lenguaje de los reportes tampoco debe hacerlo.
