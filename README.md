# xai-dt-heat

Motor del **gemelo digital explicable de riesgo de morbilidad por calor con enfoque de
equidad**. Implementa las capas L0–L5 descritas en `plan_gemelo_digital_calor.md` §6.

> Estado: **L0, L1, L5 construidos y probados; L2 con 3 de 8 conectores (EPHT, gridMET,
> EAGLE-I); L4 parcial.** Falta L3 (modelos), bloqueado por el token de EPHT para el
> desenlace diario. Ver *Siguientes pasos*.

## Puesta en marcha

```bash
uv sync --python 3.12          # crea .venv e instala el nucleo
cp .env.example .env           # y rellena XDT_EPHT_TOKEN cuando llegue
uv run pytest                  # 112 tests, sin red
uv run xdt info
```

El stack pesado está en extras, para no arrastrarlo antes de necesitarlo:

```bash
uv sync --extra geo    # geopandas, xarray, rioxarray  (M0 areal, Landsat)
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

# Clima (gridMET): condado en su centroide de poblacion + estado ponderado
uv run xdt ingest gridmet --states 04,22 --years 2017-2026

# Apagones (EAGLE-I, ~1.2 GB/ano): series diarias en hora local
uv run xdt ingest eaglei --states 04,22 --years 2023

# Estado del gemelo
uv run xdt state coverage
uv run xdt state show --geo-level state --valid-from 2023-01-01

# Viaje en el tiempo: el estado tal como se conocia en una fecha pasada
uv run xdt state show --valid-from 2023-01-01 --known-at 2024-06-01

# Auditar las revisiones de un hecho concreto
uv run xdt state history 04 epht_m1_age_adj_rate 2023-01-01

# Escenario compuesto sobre el estado real (L4)
uv run xdt scenario --geo-level county --valid-from 2023-07-10 --fraccion-clientes 0.25

# Calibracion retrospectiva (L5); --obs-known-at reproduce una corrida pasada
uv run xdt calibrate --model lgbm --quantity hri_ed_rate_daily_va

# Motor predictivo L3 con reporte CRISP-DM (EDA, CV anidada, hiperparametros,
# seleccion, pruebas robustas, SHAP). Requiere `uv sync --extra ml`.
uv run xdt ingest gridmet --years 2019-2023 --states <FIPS CONUS> --pop-coverage 0.8 --workers 8
uv run xdt crisp --years 2023 --ref-years 2019-2022 --out reports/crisp_dm.html

# Consola: lee el estado, L0 y el modelo desplegado, y se inyecta en app/consola.html
uv run xdt export
```

### Motor predictivo y reporte CRISP-DM (L3)

`xdt crisp` construye el panel estado-dia (`features/panel.py`), entrena la escalera
umbral -> logistica -> Random Forest -> LightGBM (`models/`) con validacion cruzada
anidada, aplica las pruebas de `validate/stats.py` y escribe un reporte HTML en el que
cada figura y tabla lleva un bloque *Como leerlo* y otro *Que explica*, con cifras de
la propia corrida.

| Pregunta | Prueba robusta | Por que no la clasica |
|---|---|---|
| AUC de A vs B | DeLong | ROC pareadas sobre los mismos dias |
| IC de metricas | bootstrap por bloques estado x semana | dias consecutivos no son independientes |
| k modelos en pliegues | Friedman + Nemenyi | no parametrica, por rangos |
| A vs B por pliegue | t de Nadeau-Bengio | los pliegues comparten entrenamiento |
| aprende algo? | permutacion por desplazamiento circular | conserva la autocorrelacion |
| probabilidades honestas | pendiente de calibracion + Spiegelhalter | Hosmer-Lemeshow depende de los deciles |
| perdida diaria A vs B | Diebold-Mariano con HAC | errores autocorrelacionados |
| multiples comparaciones | Holm | controla el error de familia |

Despliegue (`models/deploy.py`, por defecto en `xdt crisp`): guarda el modelo elegido en
`data/models/` con version determinista, registra en L0 sus predicciones fuera de muestra
(con las 5 contribuciones SHAP de cada una) y las de los escenarios de calentamiento
+1/+2/+3 °C, y carga el desenlace observado en el almacen de L5 sellado con el `known_at`
real de EPHT. Por eso L5 las cuenta como post-dicciones y no como pronosticos genuinos.
`xdt export` lleva todo a la seccion *Riesgo predicho por el gemelo* de la consola.

Seleccion: gana el modelo **mas simple** cuyo AUC fuera de muestra no sea
significativamente peor que el del mejor (DeLong + Holm, alfa 0,05).

## Arquitectura

| Capa | Modulo | Estado |
|---|---|---|
| **L0** trazabilidad | `twin/registry.py` | ✅ registro de predicciones + observaciones censuradas |
| **L1** estado | `twin/state.py` | ✅ bitemporal, con viaje en el tiempo |
| **L2** sincronizacion | `ingest/` | 🟡 EPHT, gridMET, EAGLE-I; faltan SVI, PLACES, LACE, Landsat, NWS |
| **L3** emulador + XAI | `models/`, `explain/`, `report/` | 🟡 escalera de 4 modelos, CV anidada, SHAP; desenlace solo 2023 hasta tener token |
| **L4** escenarios | `twin/scenarios.py` | 🟡 calor, apagon, enfriamiento, composicion |
| **L5** calibracion | `twin/calibrate.py` | ✅ parejas genuinas, censura como intervalo, deriva a priori |
| M2 features | `features/` | 🟡 índice de calor NWS, p95 local sin fuga, rachas, primera ola, evento compuesto |
| M0 armonizacion | `harmonize/` | 🟡 centroides de poblacion + agregacion ponderada; falta promedio areal (extra `geo`) |

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
| **EPHT señala el 429 dentro de un cuerpo HTTP 200** | El más peligroso. Un cliente que solo mire `status_code` cachea el documento de error como si fuera dato y `normalize()` devuelve cero filas **sin que nada falle**. `ingest/http.py` valida el cuerpo antes de escribir en caché, y revalida al leerla |

## Hallazgos de gridMET y EAGLE-I (23/9/2026)

Verificados en vivo; cada uno está fijado en un test (de contrato o unitario).

| Fuente | Hallazgo | Consecuencia |
|---|---|---|
| gridMET | NCSS tarda ~14 s por punto-año; OPeNDAP ASCII ~0.8 s por celda-año | El conector usa OPeNDAP ASCII: sin netCDF4, corre con el núcleo |
| gridMET | Los valores llegan **empaquetados**: Phoenix 1/7/2023 = `971.0` "K" | Se desempaqueta con `scale_factor`/`add_offset` → 43.95 °C |
| gridMET | **tmmx usa add_offset 220, tmmn usa 210** | El empaquetado se lee del DAS de cada variable; una constante desplazaría las mínimas 10 °C |
| gridMET | El océano devuelve **HTTP 200 con 32767**; fuera de CONUS, HTTP 400 | Relleno → ausente. Monroe (FL, los Cayos) queda fuera de la malla y cuenta como hueco de cobertura |
| EAGLE-I | Formato **disperso**: la ausencia es "cero o hueco de recolección, no se distingue" (artículo) | Serie densa con ceros + cobertura estado-año (`eaglei_cov_*_pct`) para filtrar años pobres |
| EAGLE-I | Marcas de tiempo en **UTC** | Se agrega por día civil local (huso predominante del estado); el último día local de un archivo, incompleto, se descarta |
| EAGLE-I | El artículo documenta fecha `MM/DD/YY`; los archivos usan ISO | Se aceptan ambos; cualquier otra forma detiene la ingesta |
| EAGLE-I | `MCC.csv` pierde el cero inicial del FIPS (`1001`) | Normalizado a 5 dígitos; si no, AL-CA quedarían sin denominador |
| EAGLE-I | ~1.2 GB por año | Descarga en streaming con MD5 de figshare; DuckDB agrega leyendo del disco |

## Primer exploratorio de eventos compuestos (AZ + LA, verano 2023)

```bash
uv run xdt compound --valid-from 2023-05-01 --valid-to 2023-09-30     --ref-start 2017-01-01 --ref-end 2022-12-31
```

| | condado-días |
|---|---|
| Analizados (79 condados × 153 días) | 12.087 |
| Calientes (> p95 local de 2017–2022) | 3.136 (26 %) |
| Con apagón (≥ 1 % sin luz ≥ 4 h seguidas) | 561 |
| **Compuestos observados** | **112** |
| Esperados si calor y apagón fueran independientes | 150,4 |

Descriptivo, sin desenlace de salud. Dos lecturas:

- Que el 26 % de los días supere un p95 muestra lo atípico de 2023 frente a 2017–2022, y
  por qué el umbral debe salir de un periodo de referencia explícito.
- En conjunto, **calor y apagón coinciden menos de lo que el azar predice (O/E ≈ 0,74)**.
  Hipótesis por contrastar, no hallazgo: los apagones de verano en Luisiana son
  convectivos y la tormenta que corta la luz también enfría ese día. El evento que
  importa podría ser el corte que **persiste** hacia días calientes posteriores. El
  mayor corte estatal del año (LA, 16–20 de junio: pico de 7,7 % de clientes, rachas
  de 22–24 h/día) tiene esa forma. Requiere una definición con retardos antes de H2.

## Siguientes pasos

En orden, según el cronograma (§13) y el principio de cerrar el circuito pronto (§12.3):

1. **Solicitar el token de EPHT** — sigue bloqueando la serie diaria 1385/1238, que es
   el desenlace de la Etapa A. Sin ella no hay hito del mes 4.
2. ~~Conector `gridmet.py`~~ ✅ · ~~Conector `eaglei.py`~~ ✅ · ~~`twin/calibrate.py` (L5)~~ ✅
3. Ampliar gridMET y EAGLE-I a los estados reportantes de EPHT (2022: 26 estados).
   gridMET cuesta ~0.8 s por celda-año: todo CONUS 2017–2026 son ~27 h de descarga
   secuencial, conviene lanzarlo por lotes de estados.
4. `features/` (M2): evento compuesto **con retardos** (corte en t, calor en t+1..t+3) y
   barrido de sensibilidad sobre X (fracción) e Y (horas); materializar la tabla canónica.
   Ampliar EAGLE-I a 2017–2025 (~11 GB) para tener referencia plurianual de apagones.
5. **Hito del mes 4**: baseline de umbral + LightGBM + AUC, en cuanto llegue el token.
6. `harmonize/` completo — crosswalks ZCTA ↔ tract ↔ condado y promedio areal (extra `geo`).
7. Conectores SVI, PLACES, LACE (estáticos, sin límite de tasa conocido).

## Convenciones

- `data/` lo gestiona DVC; nunca entra en git. `data/raw/` es **inmutable**.
- Un conector = una fuente, idempotente, con test de contrato.
- Los tests de contrato golpean la API viva y no corren por defecto:
  `uv run pytest -m contract`. Conviene un cron semanal, no cada commit.
- Los escenarios son contrafactuales **del modelo**, no del mundo (§15.6). El código
  no afirma causalidad y el lenguaje de los reportes tampoco debe hacerlo.
- `known_at` se guarda **siempre** en UTC naive: usar `xdt.twin.state.now_utc()`, nunca
  `datetime.now()`. El estado rechaza revisiones antedatadas y hechos fechados en el
  futuro, porque ambas cosas corrompen el eje bitemporal en silencio.
