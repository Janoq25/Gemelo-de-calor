# Plan de Trabajo de Investigación
## Gemelo Digital Explicable para Riesgo de Morbilidad por Calor con Enfoque de Equidad

**Versión:** 2.0 · **Fecha:** 2 de septiembre de 2026
**Documento:** reestructuración del tema original para viabilidad con datasets públicos y para constituir un gemelo digital genuino.
**Anexo obligatorio:** [verificacion_datos_epht.md](verificacion_datos_epht.md) — auditoría de la API del CDC ejecutada el 2/9/2026, cuyos resultados determinan el diseño de este plan.

> **Cambio de v1.0 a v2.0.** La v1.0 asumía que el desenlace de morbilidad por calor estaba disponible a nivel condado. **La consulta directa a la API demostró que no lo está.** Todo el §4 y §5 fueron reescritos sobre datos verificados, no sobre supuestos.

---

## 0. Resumen ejecutivo de los cambios

El tema original es sólido en su motivación, pero tenía defectos que lo hacían inviable o lo reducían a "un modelo predictivo con nombre bonito". Esta reestructuración los corrige.

| # | Problema del planteamiento original | Corrección adoptada |
|---|---|---|
| 1 | **CDC WONDER no contiene hospitalizaciones.** Solo mortalidad, a nivel condado y con supresión de celdas <10. La variable dependiente, tal como estaba escrita, no existía. | Se adopta el **CDC Environmental Public Health Tracking Network (EPHT)** como fuente de outcome, con las restricciones de resolución documentadas en la fila 4. |
| 2 | **Resolución de código postal inalcanzable para el outcome**, y criterio de inclusión (≥100 hospitalizaciones/año por ZIP) que habría eliminado prácticamente toda la muestra: la morbilidad por calor es un evento raro. | Se abandona la pretensión de observar el desenlace a nivel ZIP con datos públicos nacionales. Ver fila 4 y §5. |
| 3 | **No era un gemelo digital**, sino un modelo entrenado offline. Faltaban estado persistente, sincronización con el mundo real y bucle de retroalimentación. | Arquitectura formal de **6 capas** (L0–L5) con sincronización diaria automatizada, motor de escenarios contrafactuales y capa de calibración retrospectiva. |
| 4 | **Verificado el 2/9/2026 contra la API real:** no existe ningún desenlace de morbilidad por calor con resolución sub-estatal y temporalidad fina. El dato diario solo existe por región HHS (10 unidades) o por estado en la subpoblación VA. El desenlace sub-estatal más fino son muertes agregadas a 5 años por condado. Además, **solo 16 estados reportaron en 2023**, no ~25. | Diseño de **fusión de dos escalas** (§5): motor temporal entrenado con el panel estado-día, modificadores espaciales estimados con variación transversal, y desagregación a ZCTA validada contra el Heat & Health Index del propio CDC. Con vía paralela hacia datos restringidos de California. |

### Las dos contribuciones diferenciadoras

1. **Evento compuesto calor–apagón.** El dataset **EAGLE-I** (apagones por condado, cada 15 minutos, 2014–2025, público, con artículo en *Scientific Data*) convierte este escenario de simulación hipotética en fenómeno empíricamente observable. Ni el CDC Heat & Health Index ni CalHeatScore incorporan infraestructura eléctrica.
2. **Riesgo dinámico frente a vulnerabilidad estática.** El CDC ya publica un ranking de carga histórica calor-salud por ZCTA, y California opera CalHeatScore con puntajes diarios por código postal. Ambos son referencias obligadas — y ambos son, en lo esencial, **estáticos o sin explicabilidad ni auditoría de equidad**. El posicionamiento del proyecto es preciso: *ellos dicen qué zonas son crónicamente vulnerables; el gemelo dice qué zonas están en riesgo hoy, bajo estas condiciones, y por qué*.

---

## 1. Título revisado

> **A Fairness-Aware Explainable Digital Twin for Compound Heat–Blackout Health Risk in U.S. Urban Communities**

Justificación del cambio: se elimina "predicting heat-related morbidity" (genérico, muy poblado en la literatura) y se incorpora el evento compuesto calor–apagón, que es donde está la novedad y donde los datos públicos son sorprendentemente buenos.

---

## 2. Problema, brecha y contribución

### Problema
Los sistemas de alerta temprana por calor operan con umbrales meteorológicos uniformes que ignoran tres factores que determinan quién termina en urgencias: la vulnerabilidad social, la microclimatología urbana (islas de calor) y la carga de enfermedad crónica preexistente. Además, ignoran los **eventos compuestos**: una ola de calor con apagón eléctrico simultáneo elimina el acceso al aire acondicionado, que es el principal factor protector conocido.

### Estado del arte operativo — el punto de partida obligado

Dos sistemas gubernamentales ya desplegados definen el listón. **Ignorarlos en la revisión de literatura sería un error fatal ante un revisor.**

| Sistema | Qué hace | Resolución | Qué **no** hace |
|---|---|---|---|
| **CDC Heat & Health Index** (EPHT ind. 225) | Ranking percentil de carga histórica calor-salud, con 4 subíndices (carga histórica, sensibilidad, sociodemografía, entorno natural y construido) | **ZCTA**, estático | No es dinámico: no responde al clima de hoy. No modela apagones. No explica predicciones individuales |
| **CalHeatScore** (CalEPA/OEHHA) | Puntaje diario 0–4 de riesgo por calor con pronóstico a 7 días, API pública sin token | **Código postal de California**, diario | Según su documentación de métodos, combina temperatura pronosticada con tasas históricas; no incorpora infraestructura eléctrica, no publica explicabilidad por predicción, ni auditoría de equidad |

### Brecha específica
1. Los gemelos digitales climáticos urbanos existentes se orientan a energía, movilidad o confort térmico — no a desenlaces de salud.
2. **Ningún sistema operativo de alerta por calor incorpora la interrupción del suministro eléctrico como covariable dinámica**, pese a que el apagón elimina el principal factor protector conocido (el aire acondicionado).
3. La auditoría de equidad algorítmica es prácticamente inexistente en sistemas de alerta por calor, pese a que las poblaciones afectadas son sistemáticamente las más desfavorecidas.
4. Los índices existentes son **estáticos** (HHI) o **sin explicabilidad** (CalHeatScore): ninguno responde a la pregunta *"¿por qué esta zona, hoy?"*.

### Contribución declarada
1. Un gemelo digital operativo, reproducible y de código abierto, sincronizado diariamente con fuentes públicas.
2. La primera caracterización (hasta donde alcanza la revisión) del riesgo compuesto calor–apagón sobre morbilidad, usando datos observados de interrupción eléctrica.
3. Una auditoría de equidad completa (equalized odds, paridad demográfica) con intervalos de confianza bootstrap, sobre quintiles de vulnerabilidad social.
4. Un análisis de umbral de intervención que invierte la pregunta de política: no "cuántas hospitalizaciones evita un centro de enfriamiento", sino "qué efectividad mínima debería tener para justificar su despliegue en cada zona".

---

## 3. Preguntas de investigación e hipótesis

### Preguntas
- **PI1.** ¿Un modelo que integra clima, vulnerabilidad social, morbilidad crónica basal y microclima urbano supera a un modelo de umbral meteorológico en la predicción de tasas de visitas a urgencias por calor?
- **PI2.** ¿Los apagones eléctricos concurrentes modifican (interacción, no solo suma) la relación entre temperatura y morbilidad por calor?
- **PI3.** ¿El rendimiento predictivo del modelo es equitativo entre estratos de vulnerabilidad social y composición racial?
- **PI4.** ¿Qué efectividad mínima debe tener una intervención de enfriamiento comunitario para producir una reducción de riesgo detectable, y dónde es más plausible alcanzarla?

### Hipótesis (reformuladas)

Se elimina el umbral arbitrario de AUC-ROC ≥ 0.85 del planteamiento original. Ese valor podía no alcanzarse por razones ajenas a la calidad del modelo (eventos raros, agregación del outcome, ruido de codificación), lo que habría convertido un buen trabajo en un "resultado negativo" artificial.

| ID | Hipótesis nula | Hipótesis alterna | Contraste |
|---|---|---|---|
| **H1** | El AUC-ROC del gemelo no difiere del baseline de umbral meteorológico | El gemelo es superior | **Test de DeLong** para curvas ROC correlacionadas |
| **H2** | No hay interacción entre apagón y temperatura sobre el riesgo | Existe interacción supra-aditiva | Término de interacción en GLMM + **RERI** (Relative Excess Risk due to Interaction) con IC bootstrap |
| **H3** | La diferencia de equalized odds entre quintiles extremos de SVI es 0 | Existe disparidad | IC 95% bootstrap sobre `equalized_odds_difference`; se declara disparidad si el IC excluye 0 |
| **H4** | La relación temperatura–morbilidad es lineal y sin retardo | Es no lineal y con retardo distribuido | **DLNM** (modelo no lineal de retardos distribuidos), test de razón de verosimilitud contra el modelo lineal |
| **H5** | La superficie de riesgo desagregada a ZCTA no correlaciona con el CDC Heat & Health Index (medida 1504) más allá del azar | Correlaciona sustancialmente en su componente estático | ρ de Spearman con IC bootstrap. **Validación de la desagregación** (§5.3) |
| **H6** | El riesgo estimado por el gemelo no diverge de CalHeatScore en días con apagón | Diverge significativamente, y la divergencia crece con la extensión del corte | Diferencia de puntajes estandarizados, estratificada por presencia de apagón. **Es la prueba de que el componente eléctrico aporta información que el estado del arte no captura** |

> **Nota sobre H3:** se reporta la magnitud de la disparidad con su IC, no solo la significación. Un umbral fijo de 0.1 se conserva únicamente como *criterio de aceptabilidad operativa declarado a priori*, no como hipótesis estadística.
>
> **Nota sobre H5 y H6:** son las hipótesis que sustituyen a la validación directa que los datos no permiten. H5 verifica la desagregación espacial; H6 verifica que la contribución dinámica es real. Juntas convierten la principal debilidad del diseño en un resultado contrastable.

---

## 4. Datos: inventario verificado

Todas las fuentes fueron verificadas en septiembre de 2026 en cuanto a existencia, acceso público y resolución.

### 4.1 Variable de desenlace (outcome) — **verificado contra la API**

Resultados de la auditoría del 2/9/2026. Detalle completo en [verificacion_datos_epht.md](verificacion_datos_epht.md).

| ID | Medida | Geografía | Temporalidad | Cobertura |
|---|---|---|---|---|
| 440 | Tasa ajustada por edad de urgencias por HRI | **Estado** | Anual | 2000–2023 (24 años) |
| 438 / 439 | Número y tasa cruda de urgencias por HRI | **Estado** | Anual | 2000–2023 |
| **1385** | Tasa diaria de urgencias por HRI (**población VA**) | **Estado** | **Diaria** | 2017–2026 (3,530 registros) |
| **1238** | Tasa diaria de urgencias por HRI (no-VA) | **Región HHS** (10) | **Diaria** | 2017–2026 (3,166 registros) |
| 1237 | Tasa semanal de urgencias por HRI (no-VA) | Región HHS | Semanal | 2017–2026 |
| 431–433 | Hospitalizaciones por HRI | **Estado** | Anual | — |
| 370 | Muertes por calor | **Estado** | Anual | — |
| **1034** | Muertes por calor en período de 5 años | **Estado, Condado** | Quinquenal | — |

**La disyuntiva es estricta y no tiene escapatoria con datos públicos nacionales:**

```
   resolución temporal fina  ──►  geografía muy gruesa (10 regiones, o estado)
   geografía sub-estatal     ──►  solo mortalidad, agregada a 5 años
```

**Cobertura real:** en 2023 solo **16 estados** reportaron la medida 440 — Louisiana (57.7) y Arizona (48.1) en el extremo alto, Rhode Island (4.2) y Nueva York (6.3) en el bajo. Un factor de 14× que mezcla señal climática real con **heterogeneidad de codificación entre sistemas estatales de vigilancia**. Todo modelo que agrupe estados debe incluir efectos fijos estatales, o el ranking reflejará prácticas administrativas tanto como riesgo.

**Advertencia sobre la medida 1385:** es la única serie diaria a nivel estatal, pero cubre la población de Veterans Affairs — mayoritariamente masculina, de mayor edad y con perfil de comorbilidad distinto al general. Es una **población centinela legítima**, no una muestra representativa. Debe declararse en métodos y discutirse su efecto sobre la validez externa.

### 4.1b Recursos de resolución fina que sí existen (sin ser desenlace)

| ID | Medida | Geografía | Rol en el proyecto |
|---|---|---|---|
| 1504 | Historical Heat & Health Burden Percentile Rank | **ZCTA** | **Anclaje de validación externa** de la capa desagregada |
| 1505–1508 | Sensibilidad, sociodemografía, entorno, ranking global (HHI) | ZCTA (por confirmar con token) | Comparación y posicionamiento |
| 1018–1023 | Proyecciones mensuales de olas de calor: conteo, duración, intensidad, temperatura máxima y nocturna; categoría de pronóstico CPC | **Condado**, mensual | Covariables de exposición y motor de escenarios prospectivos |
| 364, 367, 368 | Mayores de 65 viviendo solos; cobertura forestal; suelo urbanizado | **Census Tract**, condado | Moderadores del DAG a resolución fina |

### 4.1c Vía California (acceso restringido, alto valor)

| Fuente | Contenido | Acceso |
|---|---|---|
| **CalHeatScore API** | Puntaje diario 0–4 por código postal + pronóstico 7 días | **Público, sin token** (ArcGIS Feature Service + CA GeoPortal) |
| **HCAI · encuentros de urgencias** | Tasas diarias de enfermedad por calor por **código postal de residencia**, base de 1,678 ZIPs | **Restringido** — *Limited Data Request* |

Esta es la única vía conocida hacia el dato que el planteamiento original quería: morbilidad por calor, **diaria y por código postal**. Ver la estrategia de doble vía en §5.3.

### 4.2 Clima y exposición

| Fuente | Contenido | Resolución | Rol |
|---|---|---|---|
| **gridMET** (Climatology Lab) | Meteorología diaria de superficie: `tmmx`, `tmmn`, `rmax`, `rmin`, `srad`, `vs` | **4 km, diario, CONUS, 1979–presente** | **Fuente climática primaria.** Cobertura completa y continua |
| NOAA GHCN-Daily | Observaciones de estación | Puntual, diario | **Validación** de gridMET (test KS) — ya no es la fuente primaria |
| **NWS API** (`api.weather.gov`) | Pronóstico en malla de 2.5 km | 2.5 km, horizonte 7 días | **Capa de sincronización prospectiva** del gemelo. Gratuita, sin API key (requiere header `User-Agent` identificativo) |
| NWS HeatRisk (NDFD XML) | Índice operativo de riesgo por calor | Malla NDFD, diario | Baseline comparativo externo de referencia operacional |
| **Landsat C2 L2 Surface Temperature** | Temperatura superficial (isla de calor) | 30 m, revisita ~16 días, 1982–presente | Caracterización de microclima intra-condado |

> **Impacto del cambio a gridMET:** elimina por completo el criterio de inclusión "estación NOAA ≤ 50 km" del planteamiento original. Toda geografía del CONUS recibe una serie climática completa y sin huecos. Se recuperan cientos de condados que el diseño original habría descartado.

### 4.3 Vulnerabilidad, salud basal e infraestructura

| Fuente | Contenido | Geografía | Notas |
|---|---|---|---|
| **CDC/ATSDR SVI 2022** | Índice de vulnerabilidad social, 16 variables, 4 temas | Census tract **y ZCTA** | Existe versión ZCTA oficial documentada — habilita el nivel de desagregación |
| **CDC PLACES 2025** | 40 medidas de salud: 12 desenlaces crónicos, 7 discapacidades, 7 necesidades sociales | Condado, tract, **ZCTA**, place | Resuelve la variable "comorbilidades" del DAG original, que no tenía fuente asignada |
| **Census LACE 2023** | Estimaciones locales de acceso a aire acondicionado | Nación, estado, condado, **census tract** | Producto experimental reciente. Resuelve el moderador "acceso a AC" |
| Census CRE-Heat | Community Resilience Estimates para calor | Tract | Alternativa/complemento a LACE |
| Romitti et al. (Harvard Dataverse) | Probabilidad empírica de AC en 45,995 tracts de 115 áreas metropolitanas | Tract | Fuente de contraste para LACE |
| **EAGLE-I** (ORNL) | Clientes sin suministro eléctrico | **Condado, cada 15 minutos, 2014–2025** | Núcleo del componente de evento compuesto. Publicado en *Scientific Data* |
| NLCD Tree Canopy | Cobertura arbórea | 30 m | Moderador "vegetación urbana" |
| US Census ACS | Demografía, ingreso, composición racial | Tract, ZCTA | Estratificación para auditoría de equidad |

### 4.4 Lo que sigue sin cubrirse (declarar como limitación)

- **Ubicación de centros de enfriamiento:** no existe un registro nacional. Se trata paramétricamente (§9.3).
- **Comorbilidades individuales:** PLACES da prevalencia agregada, no carga individual. Todo el estudio es **ecológico**; la falacia ecológica debe declararse explícitamente en la discusión.
- **Aclimatación y migración diaria:** la exposición se asigna por residencia, no por movilidad real.

---

## 5. Unidad de análisis: fusión de dos escalas

La auditoría dejó una restricción dura: **la resolución temporal y la espacial no coexisten en ningún desenlace público**. En lugar de fingir lo contrario, el diseño convierte esa restricción en el problema metodológico del trabajo: **fusión de datos multiescala**.

```
ESCALA TEMPORAL                        ESCALA ESPACIAL
"¿cómo responde el riesgo              "¿quién es más susceptible
 al clima, día a día?"                  ante la misma exposición?"
─────────────────────────              ────────────────────────────
Panel estado-día (VA) 2017–2026        Panel estado-año 2000–2023
+ región HHS-día (no-VA)               + condado-quinquenio (mortalidad)

~165,000 obs · alta potencia           16–25 estados · 3,000 condados
Estima: función exposición–respuesta   Estima: modificadores de
        retardos, no linealidad,               vulnerabilidad
        interacción con apagón
                    │                            │
                    └──────────┬─────────────────┘
                               ▼
                    CAPA DE FUSIÓN Y DESAGREGACIÓN
                    Superficie de riesgo por ZCTA
                               │
                               ▼
                    VALIDACIÓN EXTERNA
                    CDC Heat & Health Index (ZCTA, medida 1504)
                    CalHeatScore (ZIP de California, diario)
```

### 5.1 Etapa A — Motor temporal (estado-día)

**Outcome:** medida 1385 (tasa diaria de urgencias por HRI, población VA, por estado, 2017–2026) como serie principal; medida 1238 (región HHS, no-VA) como serie de contraste para verificar que los patrones no son artefactos de la población VA.

**Qué estima:** la función exposición–respuesta completa — forma no lineal, estructura de retardos, efecto de la duración de la ola, efecto de la primera ola de temporada, e **interacción con apagones** (EAGLE-I agregado a estado).

**Por qué funciona:** ~50 estados × ~3,300 días es una muestra amplia y bien poblada de eventos de calor, con efectos fijos estatales que absorben la heterogeneidad de codificación.

### 5.2 Etapa B — Modificadores espaciales (transversal)

**Outcome:** panel estado-año 2000–2023 (medida 440) y **mortalidad por calor a nivel condado en período de 5 años** (medida 1034). Este último aporta ~3,000 unidades espaciales, que es donde vive la variación de vulnerabilidad.

**Qué estima:** cuánto modifica la composición de vulnerabilidad (SVI, prevalencia crónica de PLACES, acceso a AC de LACE, isla de calor de Landsat, dosel arbóreo) la pendiente de la relación exposición–respuesta estimada en la Etapa A.

### 5.3 Etapa C — Fusión, desagregación a ZCTA y validación

La superficie de riesgo por ZCTA combina la función temporal (A) con los modificadores espaciales (B):

```
riesgo(z, t) = f_exposición( clima(z,t), apagón(z,t) ; parámetros de A )
               × g_vulnerabilidad( SVI(z), PLACES(z), AC(z), LST(z) ; parámetros de B )
```

**Aquí está el aporte metodológico más defendible del trabajo:** el resultado no se presenta como un hecho, sino que se **valida contra dos referencias externas independientes**:

1. **CDC Heat & Health Index, medida 1504** (carga histórica calor-salud por ZCTA). El componente estático del riesgo debe correlacionar fuertemente con este ranking. Si no correlaciona, el modelo está mal.
2. **CalHeatScore** (puntaje diario por ZIP de California, API pública). El componente dinámico debe correlacionar con este puntaje **en días sin apagón** — y debe **divergir en días con apagón**. Esa divergencia, si aparece, es precisamente el resultado que justifica el trabajo.

> Esta doble validación convierte un supuesto de desagregación no verificable en una hipótesis contrastable. Es la diferencia entre "creo que mi desagregación es razonable" y "aquí está la evidencia de que lo es".

### 5.4 Estrategia de doble vía para California

**No espere por una solicitud de datos.** El plan corre en dos carriles paralelos e independientes:

| Carril | Qué requiere | Cuándo arranca | Si falla |
|---|---|---|---|
| **Principal — nacional** | Nada. Todo público | Mes 1 | No aplica: no depende de nadie |
| **Profundización — California** | *Limited Data Request* a HCAI (posible IRB, posible costo, meses de trámite) | Solicitud en el mes 1, uso cuando llegue | El carril principal ya está completo; California pasa de resultado a "trabajo futuro" |

**Si la solicitud HCAI prospera**, California aporta lo que ninguna otra fuente da: **desenlace diario, por código postal, real**. Con eso se puede (a) validar directamente la desagregación de la Etapa C en vez de inferirla, y (b) ejecutar la auditoría de equidad sobre desenlaces observados, no modelados. Eso eleva el trabajo de "publicable" a "publicable en Q1".

**Si no prospera**, no se pierde nada: el diseño nacional está completo y CalHeatScore sigue disponible como baseline público.

### 5.5 Población de estudio

- **Etapa A:** todos los estados con serie diaria disponible, 2017–2026.
- **Etapa B:** los 16–25 estados que reportan a EPHT (verificar la lista año por año; en 2023 fueron 16) + condados con mortalidad no suprimida en la medida 1034.
- **Etapa C:** ZCTAs urbanas de esos estados.
- **Partición temporal:** entrenamiento hasta 2022, validación 2023–2026.

---

## 6. Arquitectura del gemelo digital

Un modelo entrenado offline no es un gemelo digital. Lo que lo convierte en uno es la conjunción de **estado persistente**, **sincronización con el sistema real** y **bucle de retroalimentación**. Estas seis capas son el argumento de defensa ante la pregunta "¿por qué llama a esto un gemelo?".

```
┌──────────────────────────────────────────────────────────────────────┐
│  L5 · CALIBRACIÓN Y APRENDIZAJE                                      │
│  Compara predicciones históricas contra observaciones que llegan     │
│  después. Detecta drift. Re-calibra. ── ESTA CAPA DEFINE EL TWIN     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ retroalimenta
┌───────────────────────────────┴──────────────────────────────────────┐
│  L4 · MOTOR DE ESCENARIOS                                            │
│  Perturba el estado y re-simula:                                     │
│    · olas de calor de 3 / 5 / 7 días                                 │
│    · apagón concurrente (parametrizado desde EAGLE-I real)           │
│    · intervención de enfriamiento comunitario                        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ consulta
┌───────────────────────────────┴──────────────────────────────────────┐
│  L3 · EMULADOR PREDICTIVO + EXPLICABILIDAD                           │
│  Encoder temporal → embedding ⊕ features estáticas → LightGBM        │
│  SHAP sobre el modelo completo · DLNM como contraste epidemiológico  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ lee
┌───────────────────────────────┴──────────────────────────────────────┐
│  L2 · SINCRONIZACIÓN                                                 │
│  gridMET (diario, con rezago) · NWS forecast (7d) · EAGLE-I (15min)  │
│  EPHT (anual) · SVI/PLACES/LACE (bienal) · Landsat (~16d)            │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ actualiza
┌───────────────────────────────┴──────────────────────────────────────┐
│  L1 · ESTADO DEL GEMELO                                              │
│  Snapshot versionado por (geografía, fecha):                         │
│  exposición · vulnerabilidad · salud basal · infraestructura         │
│  Consultable en cualquier instante pasado ("time travel")            │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ registra
┌───────────────────────────────┴──────────────────────────────────────┐
│  L0 · TRAZABILIDAD                                                   │
│  Cada predicción se persiste con timestamp, versión de modelo,       │
│  hash de datos de entrada y semilla. Auditoría retrospectiva total.  │
└──────────────────────────────────────────────────────────────────────┘
```

### Por qué cada capa importa para la defensa

| Capa | Sin ella, el revisor dice... |
|---|---|
| L0 | "No puedo verificar qué predijo el sistema ni cuándo" |
| L1 | "Esto es un dataset, no un gemelo" |
| L2 | "Es un modelo estático; el gemelo debe reflejar el sistema vivo" |
| L3 | — (es lo único que el planteamiento original tenía) |
| L4 | "¿Dónde está la simulación? Un gemelo responde preguntas contrafactuales" |
| L5 | "Sin retroalimentación no hay gemelo, hay un simulador de un solo sentido" |

**L5 es la capa que casi ningún trabajo estudiantil implementa y la que más peso tiene.** Concretamente: cada vez que EPHT publica un año nuevo de datos, el sistema recupera automáticamente las predicciones que había emitido para ese período (desde L0), calcula el error, actualiza las métricas de calibración y registra si hubo deriva. Implementarlo cuesta poco y es el argumento decisivo.

---

## 7. Módulos de trabajo

Reestructurados respecto al planteamiento original: se añaden M0 (armonización espacial, que estaba implícita y es donde se consume más tiempo real) y M6 (calibración).

### M0 · Armonización espacio-temporal
Construcción del esqueleto geográfico y los crosswalks. Sin esto nada más funciona.
- Crosswalk **ZCTA ↔ census tract ↔ condado** con ponderación por población (crosswalk HUD-USPS + relaciones del Census).
- Agregación de `gridMET` (4 km) a polígonos por promedio ponderado por área.
- Agregación de Landsat LST (30 m) a estadísticos distribucionales por polígono: media, p90, desviación, e **índice de desigualdad térmica intra-condado** (Gini de LST) — esta última es una feature original y defendible.
- Alineación temporal: EPHT anual, gridMET diario, EAGLE-I 15-min → tabla canónica `(geoid, fecha)`.
- **Entregable:** tabla canónica en Parquet + reporte de cobertura y huecos.

### M1 · Ingesta y estado del gemelo (L1 + L2)
Un conector por fuente, idempotente, con caché local y reintentos.
- **Entregable:** `dvc repro ingest` reconstruye todo el estado desde cero.

### M2 · Ingeniería de características
- **Térmicas:** tmax, tmin, índice de calor (a partir de `tmmx` + `rmax`), noches tropicales, grados-día acumulados, rachas de días consecutivos sobre percentil 95 local, anomalía respecto a la normal climática local (crítico: el umbral de daño es relativo a la aclimatación regional, no absoluto).
- **Retardos:** ventana de 21 días para el encoder temporal.
- **Estacionales:** día del año cíclico, indicador de primera ola de la temporada (efecto de no aclimatación).
- **Apagón:** fracción de clientes sin suministro, duración máxima del corte, y **feature de compuesto** = coincidencia de corte con día de calor extremo.
- **Estáticas:** SVI (4 temas + total), PLACES (crónicas relevantes: EPOC, enfermedad cardíaca, diabetes, salud mental), LACE (% sin AC), NLCD (% dosel arbóreo), Gini de LST.

### M3 · Modelado y explicabilidad (L3)
Ver §8.

### M4 · Motor de escenarios (L4)
Ver §9.

### M5 · Validación y equidad
Ver §10 y §11.

### M6 · Calibración retrospectiva (L5)
Ver §6.

---

## 8. Estrategia de modelado

### Escalera de modelos (obligatorio construirla en este orden)

| Nivel | Modelo | Propósito |
|---|---|---|
| 0 | Umbral meteorológico (percentil 95 de tmax local) | Baseline, replica el estado del arte operativo. **Es H0** |
| 1 | Regresión de Poisson/binomial negativa con offset poblacional | Baseline estadístico interpretable |
| 2 | **LightGBM** con features de retardo | Baseline fuerte de ML |
| 3 | **Híbrido:** encoder temporal → embedding ⊕ estáticas → LightGBM | Modelo propuesto |
| 4 | **DLNM** (retardos distribuidos no lineales) | Contraste epidemiológico, no compite: complementa |

> **Advertencia sobre el nivel 3.** En datos tabulares con series cortas y outcome agregado, LightGBM con retardos bien construidos frecuentemente iguala o supera a un LSTM. **Si el híbrido no supera al nivel 2, eso es un hallazgo publicable, no un fracaso** — pero solo si construyó el nivel 2 correctamente. Nunca reporte un modelo complejo sin su baseline simple bien afinado.

### Decisión arquitectónica clave: cómo hacer el híbrido

**No haga un ensemble de dos modelos separados.** Use el componente temporal como **encoder** que produce un embedding de la ventana de 21 días, concaténelo con las features estáticas, y alimente el vector resultante a LightGBM.

```
serie climática 21d ──► encoder (LSTM o TCN) ──► embedding (dim ~16)
                                                        │
features estáticas (SVI, PLACES, LACE, LST, canopy) ────┤
                                                        ▼
                                                    LightGBM ──► riesgo
```

**Razón:** con esta arquitectura, `shap.TreeExplainer` funciona sobre el modelo final completo — es exacto y rápido (segundos). Con un ensemble quedaría atrapado en `KernelSHAP`, que es aproximado y computacionalmente prohibitivo a esta escala; el módulo de explicabilidad se volvería inviable. Esta decisión, tomada al inicio, salva semanas.

Considere **TCN (Temporal Convolutional Network)** en lugar de LSTM: entrena más rápido, es más estable con series cortas, y el campo receptivo es explícito.

### Explicabilidad
- **Global:** importancia SHAP agregada; dependencia parcial de temperatura estratificada por quintil de SVI (este gráfico es probablemente la figura principal del artículo).
- **Local:** waterfall SHAP por condado-día, para justificar alertas específicas.
- **Interacción:** valores de interacción SHAP para el par (temperatura × apagón) y (temperatura × SVI). Sustituye al ANOVA del planteamiento original, que era inapropiado (ver §10).

---

## 9. Motor de escenarios

Un escenario es una **función que transforma el estado del gemelo**, no un conjunto de datos aparte. Firma uniforme:

```python
def escenario(estado: EstadoGemelo, **params) -> EstadoGemelo: ...
```

Esto permite componer escenarios (`calor_7d ∘ apagon`) y es lo que hace que L4 sea un motor y no una colección de scripts.

### 9.1 Olas de calor sintéticas
Inyección de secuencias de 3, 5 y 7 días sobre el percentil 95 local, con dos variantes: temprano en temporada (población no aclimatada) y tardío. La comparación entre ambas es un resultado interesante en sí mismo.

### 9.2 Evento compuesto calor + apagón — **componente central**

A diferencia del resto, este escenario **no es puramente sintético**: EAGLE-I proporciona el historial real de cortes por condado cada 15 minutos desde 2014.

**Doble uso del dataset:**
1. **Estimación:** identificar los eventos compuestos históricos (día sobre p95 de temperatura ∧ >X% de clientes sin suministro ≥ Y horas) y estimar el efecto de interacción observado sobre la morbilidad. Contraste mediante **RERI** con IC bootstrap.
2. **Simulación:** parametrizar cortes hipotéticos usando la distribución empírica de duración y extensión observada en cada condado — los escenarios son realistas porque provienen de la historia del propio condado.

Este componente responde PI2 y es el que sostiene la novedad del trabajo.

### 9.3 Intervención de enfriamiento comunitario

**Problema honesto:** no existe registro nacional de centros de enfriamiento, y el efecto causal de abrir uno no es estimable con estos datos.

**Solución — invertir la pregunta.** En lugar de afirmar "abrir N centros evita M hospitalizaciones" (afirmación causal no sustentada), se ejecuta un **análisis de umbral de efectividad**:

> ¿Qué reducción mínima de exposición efectiva (ε) tendría que lograr una intervención de enfriamiento en cada ZCTA para producir una reducción de riesgo predicho estadísticamente detectable?

El resultado es un mapa de ZCTAs ordenadas por *facilidad de intervención*: aquellas donde un ε pequeño basta son las prioritarias. Esto es defendible, políticamente útil, y evita una afirmación causal que un revisor rechazaría.

Se acompaña de análisis de sensibilidad sobre ε ∈ [0.05, 0.40] y se declara explícitamente: **efecto asumido, no estimado**.

---

## 10. Validación y pruebas estadísticas

El planteamiento original tenía tres pruebas mal especificadas. Se corrigen aquí.

### 10.1 Correcciones a las pruebas propuestas

| Prueba original | Problema | Reemplazo |
|---|---|---|
| **KS** para temperaturas simuladas vs. observadas | Uso poco claro | **Se conserva con nuevo propósito:** validar gridMET contra observaciones de estación GHCN. Se convierte en control de calidad de datos, que es un uso correcto y necesario |
| **Mann-Whitney U** predicho vs. observado | Inapropiado. Los datos son pareados por geografía, y "no rechazar H0" **no es evidencia de buen ajuste** — es ausencia de evidencia. Además la potencia crece con n, penalizando muestras grandes | **Análisis de calibración:** curva de calibración, pendiente e intercepto, **Brier score**, **ICI** (Integrated Calibration Index) y **gráfico de Bland-Altman** |
| **ANOVA** para SVI × temperatura | Viola independencia: hay autocorrelación espacial y temporal severa. Los errores estándar estarían sesgados a la baja | **Término de interacción en GLMM** con efecto aleatorio por condado y estructura AR(1) temporal + **valores de interacción SHAP** como evidencia complementaria del modelo ML |

### 10.2 Esquema de validación

**Validación temporal (principal):** entrenar 2015–2021, validar 2022–2024. Refleja el uso real: predecir el futuro.

**Validación espacial:** `GroupKFold` agrupando por **región climática NOAA** o por CBSA — **no por condado individual**. Agrupar por condado permite fuga espacial entre condados vecinos del mismo área metropolitana; agrupar por ciudad individual produce pliegues muy desbalanceados. La propuesta original ("entrenar en ciudades del norte, validar en el sur") es un caso extremo válido, pero debe reportarse como **análisis de transferibilidad** aparte, no como la validación principal — el cambio de régimen climático y de aclimatación hace que el rendimiento caiga por razones sustantivas, no por sobreajuste.

**Validación externa (deseable):** un estado completo excluido del entrenamiento.

### 10.3 Métricas

- **Discriminación:** AUC-ROC, AUC-PR (más informativa con eventos raros — repórtela como métrica principal junto a ROC), F1.
- **Calibración:** Brier, ICI, curva de calibración.
- **Conteo:** RMSE, MAE, y **desviación de Poisson** (más apropiada que RMSE para conteos).
- **Comparación entre modelos:** test de DeLong para AUC-ROC.
- **Incertidumbre:** bootstrap por bloques (**block bootstrap**, respetando la autocorrelación temporal — el bootstrap i.i.d. subestimaría los IC), 2,000 réplicas, IC 95%.

---

## 11. Auditoría de equidad

**Herramienta:** `fairlearn`. No implementar las métricas a mano.

**Atributos de estratificación:**
- Quintiles de SVI total y de cada uno de los 4 temas.
- Composición racial/étnica por ACS (% población no blanca, tratada como variable continua **y** en terciles).
- Ingreso mediano del hogar en quintiles.
- Estatus urbano/suburbano dentro del CBSA.

**Métricas:** `equalized_odds_difference`, `demographic_parity_ratio`, y adicionalmente **tasa de falsos negativos por estrato** — en un sistema de alerta, un falso negativo en una zona vulnerable es el error con mayor costo humano. Esta métrica debe ser la principal del análisis de equidad y merece un párrafo propio en la discusión.

**Incertidumbre:** IC bootstrap para cada métrica de equidad. Reportar magnitud, no solo significación.

**Advertencia metodológica obligatoria:** todos los datos son agregados. La equidad se evalúa entre **áreas geográficas** caracterizadas por su composición, no entre **individuos**. Es una auditoría de equidad ecológica. Debe declararse en resultados y discusión; omitirlo sería una sobreinterpretación grave.

**Si se detecta disparidad:** no basta con reportarla. Ejecutar al menos una mitigación (reponderación de muestra, o post-procesamiento con `ThresholdOptimizer`) y documentar el compromiso resultante entre rendimiento global y equidad. Esa curva de trade-off es un resultado valioso.

---

## 12. Arquitectura de software

### 12.1 Estructura del repositorio

```
xai-dt-heat/
├── data/                        # gestionado por DVC, nunca en git
│   ├── raw/                     # descargas crudas, inmutables
│   ├── interim/                 # parquet normalizado por fuente
│   └── features/                # tabla canónica (geoid, fecha)
├── src/
│   ├── ingest/                  # un conector por fuente, idempotente
│   │   ├── gridmet.py
│   │   ├── epht.py
│   │   ├── eaglei.py
│   │   ├── svi.py
│   │   ├── places.py
│   │   ├── lace.py
│   │   ├── landsat_lst.py
│   │   └── nws_forecast.py
│   ├── harmonize/               # crosswalks, agregación espacial
│   ├── features/
│   ├── models/
│   │   ├── baseline_threshold.py    # H0
│   │   ├── baseline_poisson.py
│   │   ├── gbm.py
│   │   ├── encoder.py               # LSTM o TCN
│   │   ├── hybrid.py
│   │   └── dlnm.py
│   ├── explain/                 # SHAP global, local, interacciones
│   ├── fairness/                # fairlearn + bootstrap
│   ├── validate/                # CV espacial/temporal, DeLong, calibración
│   └── twin/                    # ── EL GEMELO ──
│       ├── state.py             # L1 · estado versionado
│       ├── sync.py              # L2 · actualización programada
│       ├── scenarios.py         # L4 · perturbaciones componibles
│       ├── registry.py          # L0 · registro de predicciones
│       └── calibrate.py         # L5 · retroalimentación
├── app/                         # Streamlit + pydeck
├── notebooks/                   # solo exploración, nunca resultados finales
├── tests/
├── dvc.yaml                     # pipeline reproducible end-to-end
└── README.md
```

### 12.2 Stack y justificación de cada elección

| Componente | Elección | Por qué |
|---|---|---|
| Almacenamiento | **DuckDB + Parquet** | A esta escala (~1,200 condados × 3,650 días ≈ 4.4M filas) es más rápido que PostGIS, no requiere servidor, tiene extensión espacial, y es un solo archivo. **No monte PostgreSQL.** |
| Reproducibilidad | **DVC** | No opcional: el pre-registro en OSF exige que `dvc repro` regenere todo desde cero. Los revisores de *Journal of Biomedical Informatics* lo valoran explícitamente |
| Datos geoespaciales | GeoPandas + Shapely 2.x | Estándar |
| Rasters | rioxarray + xarray | Necesario para gridMET (netCDF) y Landsat |
| Acceso satelital | `pystac-client` + Planetary Computer | Evita descargar escenas completas; procesamiento por ventana |
| Modelos | LightGBM, PyTorch (encoder) | LightGBM primero, siempre |
| Explicabilidad | SHAP (`TreeExplainer`) | Exacto sobre el modelo final gracias a la arquitectura de §8 |
| Equidad | `fairlearn` | Implementa exactamente las métricas requeridas |
| Interfaz | **Streamlit + pydeck** | Un mapa coroplético con panel de escenarios son ~300 líneas. El equivalente en React/FastAPI son tres semanas que el cronograma no tiene. **Resista la tentación.** |
| Orquestación | `dvc.yaml` + tarea programada del SO | Prefect/Airflow es sobreingeniería aquí |
| Testing | pytest, con tests de contrato por conector | Los conectores fallan silenciosamente cuando cambia un endpoint; los tests son la única defensa |

### 12.3 Principio rector de construcción

> **Cierre el circuito completo lo antes posible, aunque el resultado sea malo.**

Un pipeline end-to-end mediocre en el mes 4 vale más que tres módulos excelentes y desconectados en el mes 10. El pipeline completo revela los problemas de integración —que son los que consumen tiempo— cuando todavía hay margen para resolverlos.

---

## 13. Cronograma (15 meses)

| Meses | Fase | Entregables verificables |
|---|---|---|
| **1** | Pre-registro y solicitud de datos | ~~Confirmación por API de resolución EPHT~~ ✅ **completada el 2/9/2026**, ver [anexo](verificacion_datos_epht.md). Token EPHT solicitado. **Limited Data Request enviada a HCAI** (carril California, §5.4). Protocolo pre-registrado en **OSF** |
| **1–2** | Revisión sistemática | Matriz de literatura. Posicionamiento de la brecha. Justificación del evento compuesto |
| **2–3** | **M0 + M1** · Armonización e ingesta | Crosswalks validados. 8 conectores funcionando. Tabla canónica en Parquet. `dvc repro` verde |
| **4** | **Circuito cerrado mínimo** ⚑ | Baseline de umbral + LightGBM + AUC reportado. **Hito crítico: a partir de aquí siempre hay un resultado que presentar** |
| **5–6** | **M2** · Ingeniería de características | Catálogo documentado de features. Análisis exploratorio de eventos compuestos |
| **7–8** | **M3** · Modelado | Escalera de 5 modelos completa. Comparación con DeLong |
| **8–9** | **M3** · Explicabilidad | SHAP global/local/interacciones. Figura principal (dependencia térmica × quintil SVI) |
| **9–10** | **M4** · Motor de escenarios (L4) | Escenarios componibles. Análisis de umbral de intervención. App Streamlit funcional |
| **10–11** | **L0/L2/L5** · Cierre del gemelo | Sincronización programada operativa. Registro de predicciones. Calibración retrospectiva. **El sistema ya es un gemelo digital** |
| **11–12** | **M5** · Validación | CV espacial y temporal. Calibración. Transferibilidad norte→sur. DLNM. **Contraste H5 (vs. HHI) y H6 (vs. CalHeatScore)** |
| **12–13** | **M5** · Equidad | Auditoría completa con IC bootstrap. Mitigación y curva de trade-off |
| **13–14** | Redacción | Manuscrito + documento de tesis |
| **15** | Cierre | Repositorio público con DOI (Zenodo). Envío a revista |

**Holgura deliberada:** el cronograma asume ~15% de deslizamiento. Los meses 2–3 son los de mayor riesgo: la armonización espacial consume sistemáticamente más tiempo del previsto en todo proyecto geoespacial.

---

## 14. Riesgos y planes de contingencia

| ID | Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|---|
| ~~**R1**~~ | ~~EPHT solo ofrece el outcome de calor en agregación anual~~ | — | — | ✅ **Materializado y resuelto por diseño.** La auditoría del 2/9/2026 confirmó que no hay desenlace sub-estatal con temporalidad fina. El diseño de fusión de dos escalas (§5) absorbe la restricción. Ya no es un riesgo abierto |
| **R1b** | La solicitud HCAI (California) es denegada, demorada más allá del mes 9, o exige costo/IRB inasumibles | **Media-alta** | **Bajo por diseño** | El carril principal es 100% público y no depende de ella (§5.4). California se degrada a "trabajo futuro" sin afectar ningún entregable mínimo. **Nunca poner un entregable obligatorio en ese carril** |
| **R1c** | La población VA (medida 1385) resulta demasiado atípica y sus patrones no se replican en la serie no-VA por región HHS | Media | Medio | La comparación cruzada 1385 vs. 1238 está incorporada al diseño (§5.1) como verificación, no como opcional. Si divergen, se reporta como hallazgo sobre validez de datos centinela y se pondera hacia la serie regional |
| **R2** | Supresión de celdas y cobertura estatal escasa (solo 16 estados en 2023) | **Confirmada** | Medio | Verificar la lista de estados reportantes año por año — la cobertura puede ser mayor en años previos. Efectos fijos estatales obligatorios. Modelar conteos con binomial negativa y offset; tratar la supresión como censura por intervalo, no como dato faltante |
| **R2b** | La heterogeneidad de codificación entre estados (rango observado de 4.2 a 57.7 por 100,000) domina la señal de vulnerabilidad | **Alta** | **Alto** | Efectos fijos estatales en toda la Etapa B; identificación de modificadores basada en variación **intra-estatal** (entre condados del mismo estado), no entre estados. Declarar como limitación central |
| **R3** | El modelo híbrido no supera a LightGBM | Media | **Bajo** | Es un resultado reportable si el baseline está bien afinado. Reencuadrar la contribución hacia el evento compuesto y la arquitectura del gemelo, que no dependen de ello |
| **R4** | Procesamiento de Landsat excede la capacidad de cómputo | Media | Medio | Usar composiciones estacionales pre-agregadas en lugar de escenas individuales; limitar a los meses de verano; procesar por ventana vía STAC sin descargar escenas completas |
| **R5** | Falta de datos de centros de enfriamiento | **Cierta** | Bajo | Ya mitigado por diseño: análisis de umbral de efectividad (§9.3) |
| **R6** | Cambio no anunciado en un endpoint de API rompe el pipeline | **Confirmada** | Medio | Ya observado: `/geographicLevels/` devuelve **410 Gone** y fue sustituido por `/geographicTypes/`. Tests de contrato por conector; caché local de todo lo descargado; `data/raw/` inmutable y versionado |
| **R6b** | Límite de tasa de la API bloquea la ingesta sistemática | **Confirmada** | Bajo | Observado repetidamente (HTTP 429 sin token). Token gratuito de EPHT + retroceso exponencial desde 5 s + caché agresiva. Ya diseñado en §12 |
| **R8** | Un revisor señala que CalHeatScore ya resuelve el problema | Media | **Alto si no está previsto** | Posicionamiento explícito desde la revisión de literatura (§2) y contraste empírico formalizado en H6. La diferenciación es concreta: apagones, explicabilidad y equidad. **No es un riesgo si se enfrenta de frente; es fatal si se ignora** |
| **R7** | Alcance excesivo para 15 meses | Alta | Alto | El hito del mes 4 garantiza un resultado presentable. Prioridad de recorte declarada: (1) Landsat, (2) DLNM, (3) validación externa estatal. **Nunca recortar: equidad ni capa L5** |

---

## 15. Limitaciones a declarar explícitamente

Declararlas es señal de rigor; omitirlas es lo que un revisor detecta primero.

1. **Diseño ecológico.** Todas las asociaciones son entre áreas, no entre individuos. La falacia ecológica impide inferencia a nivel personal.
2. **Cobertura parcial y verificada.** Solo **16 estados** reportaron la medida de urgencias en 2023. Los resultados no son nacionalmente representativos, y los estados reportantes probablemente difieren sistemáticamente de los no reportantes (capacidad de vigilancia epidemiológica correlaciona con recursos estatales).
3. **Población centinela en la serie diaria.** La única serie diaria a nivel estatal cubre a beneficiarios de Veterans Affairs: mayor edad, predominio masculino, perfil de comorbilidad distinto. La función exposición–respuesta estimada puede sobreestimar la susceptibilidad de la población general.
4. **Heterogeneidad de codificación entre estados.** El rango observado de 4.2 a 57.7 casos por 100,000 mezcla riesgo real con diferencias en práctica de codificación y en sistemas de vigilancia. La identificación de modificadores se restringe a variación intra-estatal por esta razón.
5. **Desagregación a ZCTA no observada.** No existe desenlace público a esa resolución. La capa ZCTA es una estimación de área pequeña, validada **indirectamente** contra el CDC Heat & Health Index y CalHeatScore (H5, H6), no contra casos observados — salvo que prospere la solicitud HCAI.
6. **Naturaleza predictiva, no causal.** El DAG orienta la selección de variables, pero el modelo no identifica efectos causales. Los escenarios son contrafactuales *del modelo*, no del mundo. **El lenguaje debe reflejarlo en todo el manuscrito.**
7. **Exposición por residencia.** No se captura movilidad diaria ni exposición ocupacional.
8. **Sub-registro diferencial.** Más allá de la heterogeneidad entre estados (punto 4), la enfermedad por calor está sistemáticamente sub-codificada, y el sub-registro puede correlacionar con acceso a atención, que a su vez correlaciona con vulnerabilidad. Es un sesgo diferencial que puede **atenuar o amplificar** las disparidades observadas. Discutir su dirección probable.
9. **Efecto de intervención asumido, no estimado.**
10. **Validación circular parcial.** H5 valida contra el CDC Heat & Health Index, que comparte fuentes de entrada con este proyecto (SVI, sociodemografía, entorno construido). La correlación esperada no es evidencia independiente plena. **H6 es la validación menos circular** porque contrasta contra un desenlace operativo con fuente de datos distinta (HCAI). Declararlo.

---

## 16. Criterios de éxito

**Mínimos (tesis aprobable):**
- Pipeline reproducible end-to-end con `dvc repro`.
- Baseline y modelo propuesto comparados con test estadístico formal.
- Auditoría de equidad con intervalos de confianza.
- Las seis capas del gemelo implementadas y demostrables.

**Objetivo (tesis destacada):**
- Efecto de interacción calor × apagón caracterizado con datos observados.
- Aplicación interactiva funcional con escenarios componibles.
- Repositorio público con DOI y protocolo pre-registrado.

**Ambicioso (publicable en Q1):**
- Validación externa en un estado excluido, incluyendo el nivel ZCTA.
- Análisis de mitigación de sesgo con curva de trade-off rendimiento–equidad.
- Sincronización operativa en funcionamiento continuo con calibración retrospectiva documentada durante ≥ 6 meses.

---

## 17. Ética y datos sensibles

- Todos los datos son **agregados y públicos**. No se manejan datos identificables. Probable exención de IRB, pero **debe consultarse formalmente** al comité de la universidad y documentarse la respuesta.
- **Zonas con alta población indígena:** aplicar principios **CARE** para gobernanza de datos indígenas, además de FAIR. Considerar suprimir la desagregación en geografías donde el tamaño poblacional permita re-identificación indirecta, y no presentar rankings de vulnerabilidad que puedan estigmatizar comunidades.
- **Riesgo de uso indebido del modelo:** un mapa de priorización puede usarse para asignar recursos, pero también para justificar decisiones de aseguramiento o inversión que perjudiquen a las zonas señaladas. Incluir una declaración de uso previsto en el repositorio y en el artículo.
- **Pre-registro en OSF** antes de ver los datos de validación temporal (2022–2024). Esto es lo que separa una predicción de una post-dicción.

---

## 18. Revistas objetivo

Se mantienen las del planteamiento original, con el orden ajustado a la contribución reformulada.

| Revista | Afinidad con el trabajo reestructurado |
|---|---|
| *Journal of Biomedical Informatics* | Integración de datos heterogéneos + XAI + arquitectura de sistema. Primera opción |
| *Environment International* / *Environmental Research: Health* | Añadida: el componente de evento compuesto calor–apagón encaja mejor aquí que en las originales |
| *Frontiers in Public Health* | Salud ambiental + vulnerabilidad. Buena opción de respaldo |
| *Scientific Reports* | Datasets abiertos + modelado predictivo. Respaldo amplio |

---

## 19. Checklist de arranque (primeras dos semanas)

- [x] ~~Consultar la API de EPHT y confirmar la resolución real del outcome de calor~~ ✅ **Completado el 2/9/2026** → [verificacion_datos_epht.md](verificacion_datos_epht.md)
- [ ] **Solicitar token de API de EPHT** a `trackingsupport@cdc.gov` — bloquea cualquier ingesta sistemática por límite de tasa
- [ ] **Iniciar el Limited Data Request ante HCAI** (carril California). Es un trámite largo: cuanto antes se envíe, mayor probabilidad de que llegue a tiempo de ser útil
- [ ] Con el token, completar la verificación pendiente: indicadores 173 (Historical Temperature & Heat Index) y 228 (HeatRisk), y medidas 1505–1508 del Heat & Health Index
- [ ] Descargar la serie completa de la medida 1385 (diaria, estado, VA) y 1238 (diaria, región HHS) — es el insumo de la Etapa A
- [ ] Verificar la lista de estados reportantes **año por año** para la medida 440: la cobertura de 2023 fue de 16 estados, pero puede ser mayor en años anteriores
- [ ] Descargar y explorar una muestra de EAGLE-I (un año, un estado con clima cálido)
- [ ] Consultar la API pública de CalHeatScore y guardar una serie de referencia — es el baseline de H6
- [ ] Descargar SVI 2022 (tract y ZCTA), PLACES 2025 (ZCTA) y el HHI por ZCTA (medida 1504)
- [ ] Verificar acceso a gridMET vía OPeNDAP para un condado y un año
- [ ] Verificar disponibilidad y estructura de Census LACE 2023
- [ ] Inicializar repositorio git + DVC
- [ ] Escribir los dos primeros conectores (`epht.py`, `gridmet.py`) con sus tests de contrato
- [ ] **Presentar al profesor asesor** la auditoría y esta v2.0, con tres puntos: (a) CDC WONDER no tiene hospitalizaciones; (b) el desenlace no existe a nivel sub-estatal con temporalidad fina, lo que obliga al diseño de fusión de dos escalas; (c) CalHeatScore es arte previo operativo y el proyecto debe posicionarse frente a él

---

## Fuentes verificadas

Verificadas en septiembre de 2026.

Inventario detallado de medidas, endpoints y cobertura en [verificacion_datos_epht.md](verificacion_datos_epht.md).

- CDC Environmental Public Health Tracking Network — https://ephtracking.cdc.gov/
- CDC EPHT · ayuda de la API — https://ephtracking.cdc.gov/apihelp
- CalHeatScore (CalEPA/OEHHA) — https://calheatscore.calepa.ca.gov/
- CalHeatScore · API pública — https://calheatscore.calepa.ca.gov/api/
- CalHeatScore · documentación de métodos v2.0 — https://calheatscore.calepa.ca.gov/sites/default/files/2026-02/methods_v2.pdf
- HCAI · Limited Data Request — https://hcai.ca.gov/data-and-reports/request-data/limited-data-request-information
- CDC EPHT · Heat & Heat-related Illness — https://www.cdc.gov/environmental-health-tracking/php/data-research/heat-heat-related-illness.html
- EPHTrackR, interfaz R oficial a la API — https://github.com/CDCgov/EPHTrackR
- CDC PLACES 2025, datos por census tract — https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-Census-Tract-D/cwsq-ngmh
- CDC/ATSDR SVI · descargas — https://svi.cdc.gov/dataDownloads/data-download.html
- CDC/ATSDR SVI 2022 · documentación ZCTA — https://svi.cdc.gov/map25/data/docs/SVI2022Documentation_ZCTA.pdf
- gridMET · Climatology Lab — https://www.climatologylab.org/gridmet.html
- EAGLE-I Power Outage Data 2014–2022 (ORNL) — https://doi.ccs.ornl.gov/dataset/ccec86f0-e144-5de8-aee0-fb26028b26e1
- EAGLE-I · artículo de datos en *Scientific Data* — https://www.nature.com/articles/s41597-024-03095-5
- NWS API · documentación de gridpoints — https://weather-gov.github.io/api/gridpoints
- NDFD XML Web Service, elemento HeatRisk — https://graphical.weather.gov/xml/rest.php
- Landsat Collection 2 Level-2 Science Products (USGS) — https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products
- Census Bureau · Local Air Conditioning Estimates (LACE) 2023 — https://www.census.gov/newsroom/press-releases/2026/2023-lace.html
- Romitti et al. · prevalencia de AC por tract (Harvard Dataverse) — https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HWFVP6
- Tracking California · Heat-Related Illness — https://trackingcalifornia.org/topics/heat-related-illness
