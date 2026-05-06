### Diagnóstico
El dataset actual tiene duplicados evidentes: mismo inmueble publicado en múltiples plataformas o varias veces en la misma plataforma con pequeñas variaciones. Sin deduplicación, el modelo aprende con sesgo (observaciones correlacionadas en train y test) y las métricas de negocio (cantidad de inmuebles reales en el mercado) son incorrectas.

### Propuesta

**Fase 1 — Bloqueo (blocking):** antes de comparar todos contra todos, agrupar candidatos por criterios que reduzcan el espacio de búsqueda:
- Celda geográfica H3 (resolución 9 ≈ 150m de lado) como primera clave de bloqueo.
- Segundo bloqueo por `tipo` + `ambientes` dentro de cada celda.
- Esto lleva el problema de O(n²) a O(k²) por bloque, con k << n.

**Fase 2 — Scoring de similitud:** para cada par de candidatos dentro de un bloque, calcular un score compuesto:

| Feature | Método | Peso sugerido |
|---------|--------|---------------|
| Distancia geográfica | Haversine < 50m | Alto |
| Precio | Diferencia porcentual < 15% | Alto |
| Superficie | Diferencia absoluta < 5m² | Alto |
| Ambientes / dormitorios | Igualdad exacta | Medio |
| Descripción libre | TF-IDF cosine similarity | Medio |
| Fecha publicación | Diferencia en días | Bajo (penaliza) |

**Fase 3 — Decisión:** umbral de score ≥ 0.80 → mismo inmueble. Casos borderline (0.65–0.80) → revisión manual o regla de desempate por plataforma de mayor confianza.

**Fase 4 — Evaluación:** construir un golden set de ~200 pares anotados manualmente. Métricas: precisión, recall y F1 sobre pares positivos (mismo inmueble). Target mínimo: F1 ≥ 0.85 antes de usar en producción.

### Esfuerzo estimado
2–3 semanas: 1 para implementar bloqueo + scoring, 1 para golden set + tuning de umbrales, ½–1 para integración al pipeline.

### Riesgos
- Inmuebles con coordenadas imprecisas (scrapers distintos pueden geocodificar diferente): mitigar con radio de bloqueo más generoso (100m) y segundo criterio textual.
- Falsos positivos en edificios grandes (mismo edificio, distintos pisos): precio/superficie/piso como discriminadores adicionales.

---

## 4.2 Arquitectura de tablas

### Diagnóstico
Hoy el output del scraping llega directamente al dataset de entrenamiento, mezclando concerns: datos crudos con datos limpios, alquileres con ventas, sin versionado ni linaje. Esto hace imposible reproducir un entrenamiento anterior o depurar un error de scraping.

### Propuesta: arquitectura en capas (medallion)

```
┌─────────────────────────────────────────────────────────────────┐
│  BRONZE — datos crudos del scraper                              │
│  raw_listings (id_scraping, plataforma, json_raw, scraped_at)   │
│  → append-only, nunca se modifica, particionado por fecha       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ limpieza + homogeneización
┌──────────────────────────▼──────────────────────────────────────┐
│  SILVER — datos limpios y homogéneos                            │
│  clean_listings (id_aviso, plataforma, tipo_operacion,          │
│                  precio_ars, superficie_m2, zona_barrio,        │
│                  lat, lon, ..., cleaned_at, pipeline_version)   │
│  → una fila por publicación, con tipo_operacion explícito       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ deduplicación (matching)
┌──────────────────────────▼──────────────────────────────────────┐
│  GOLD — inmuebles únicos para modelado y UI                     │
│  unique_properties (id_property, listing_ids[], precio_median,  │
│                     ultima_actualizacion, activo)               │
│  → una fila por inmueble físico real                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     training_dataset             inference_api
     (snapshot con versión)       (Gold + modelo registrado)
```

**Criterios de versionado:**
- Cada tabla Silver y Gold lleva columna `pipeline_version` (semver del código de limpieza).
- El dataset de entrenamiento es un snapshot inmutable con fecha y hash del commit que lo generó.
- Los modelos en MLflow referencian el `dataset_version` con el que fueron entrenados.

### Esfuerzo estimado
1–2 semanas para definir schemas y migrar el pipeline existente. La capa Bronze probablemente ya existe parcialmente (output del scraper actual).

### Riesgos
- Cambios de schema en plataformas externas rompen Bronze→Silver: mitigar con validación de schema (Great Expectations o Pydantic) en la ingesta.
- Costos de storage si Bronze crece sin TTL: definir política de retención (ej. Bronze: 90 días, Silver/Gold: indefinido).

---

## 4.3 Modelado avanzado para R² = 0.80

### Diagnóstico
El modelo actual (XGBoost) ya alcanza R² ≈ 0.89 en alquileres una vez resuelto el problema de mezcla alquiler/venta. Los frentes de mejora ahora son más finos: reducir el error en segmentos específicos (casas grandes, barrios con alta dispersión) y mejorar la generalización ante inflación y estacionalidad.

### Propuesta priorizada por impacto/costo

**1. Target encoding suavizado por barrio (alto impacto, bajo costo)**  
Reemplazar el one-hot de `zona_barrio` por el precio mediano histórico por barrio + tipo, suavizado con la media global (m-estimate encoding). Captura el efecto barrio de forma continua sin explosión de dimensionalidad.

**2. Features de contexto externo (alto impacto, costo medio)**  
- Distancia a estaciones de subte/colectivos (datos GTFS de la Ciudad de Buenos Aires, disponibles en data.buenosaires.gob.ar).
- Precio m² histórico del barrio como feature lagged (mediana de los últimos 90 días).
- Índice de valorización por zona (proxy de gentrificación).

**3. Segmentación de modelos (impacto medio, costo medio)**  
Entrenar modelos separados para `Departamento`, `PH` y `Casa`, dado que tienen distribuciones de precio y features distintas. Alternativa más simple: agregar interacciones `tipo × superficie` y `tipo × zona` como features explícitas.

**4. LightGBM con early stopping (impacto incremental, bajo costo)**  
LightGBM suele superar XGBoost en datasets tabulares medianos (~4k filas) por su mejor manejo de variables categóricas y velocidad de entrenamiento. Vale la pena compararlo directamente.

**5. Stacking / blending (impacto incremental, costo alto)**  
Meta-modelo que combina Ridge (captura relaciones lineales estables) + XGBoost (captura no linealidades). Agrega complejidad operativa; reservar para cuando los otros frentes estén agotados.

### Esfuerzo estimado
Ítems 1–2: 1 semana. Ítem 3–4: 3–4 días adicionales. Ítem 5: 1 semana extra.

### Riesgos
- Inflación argentina distorsiona precios históricos: normalizar por índice inflacionario (IPC o tipo de cambio) antes de usar features lagged.
- Sobreajuste en barrios con pocas observaciones (ej. San Telmo, 490 filas): regularización más agresiva o agrupación de barrios similares.

---

## 4.4 MLOps / CI-CD

### Diagnóstico
Hoy no hay tracking sistemático de experimentos, el modelo se retrain manualmente y no hay monitoreo de drift. Esto hace imposible saber si un modelo en producción está degradado y dificulta comparar enfoques entre iteraciones.

### Stack mínimo viable (semanas 1–3)

**Tracking de experimentos:** MLflow local (ya implementado en `02_modelado.py`). Cada run registra parámetros, métricas CV por fold, feature importance y el modelo serializado. El Model Registry de MLflow gestiona versiones con estados: `Staging` → `Production` → `Archived`.

**Promoción de modelos:** regla simple — un modelo pasa a `Production` solo si supera al actual en R² CV y RMSE, validado contra el mismo snapshot de dataset. La promoción es manual con un script `scripts/promote_model.py` que compara runs en MLflow.

**Retraining trigger:** script de Airflow (ya existe en el stack) que corre semanalmente: descarga nuevos listings, corre `01_limpieza.py` + `02_modelado.py`, loggea a MLflow, y envía alerta si el R² nuevo < umbral configurable.

**CI básico:** GitHub Actions con un workflow que en cada PR corre: lint (ruff), tests unitarios de limpieza (pytest), y smoke test del pipeline de inferencia con 10 filas.

### Stack ideal (mes 2 en adelante)

**Monitoreo de drift:** comparar distribución de features de producción (precio, superficie, zona) contra el dataset de entrenamiento usando PSI (Population Stability Index). Alertar si PSI > 0.2 en alguna feature clave. Herramienta: Evidently AI (open source, integra con MLflow).

**Serving:** FastAPI con endpoint `/predict` que carga el modelo `Production` desde MLflow y devuelve precio estimado + intervalo de confianza (usando quantile regression o conformal prediction). Containerizado en Docker, deployado en el infra existente.

**CD automatizado:** si el retraining semanal produce un modelo con mejora > 2% en R² y pasa smoke tests, se promueve automáticamente a `Staging`. La promoción a `Production` sigue siendo manual (aprobación en GitHub PR o Slack).

**Data versioning:** DVC para versionar snapshots de datasets junto al código, de modo que cada modelo registrado en MLflow apunte a un hash de DVC reproducible.

### Esfuerzo estimado
Stack mínimo viable: 2–3 semanas. Stack ideal completo: 6–8 semanas adicionales.

### Riesgos
- Drift acelerado por inflación argentina: el modelo necesita retraining más frecuente que en mercados estables (considerar quincenal en lugar de semanal).
- Complejidad operativa del stack ideal: priorizar observabilidad (saber cuándo el modelo falla) antes que automatización completa del deploy.

---

## Priorización: si tuviese 1 mes

| Semana | Frente | Justificación |
|--------|--------|---------------|
| **1** | **MLOps mínimo viable** | Sin tracking no se puede iterar con criterio. Costo bajo, beneficio inmediato para todo lo demás. |
| **2** | **Arquitectura de tablas (Bronze→Silver→Gold)** | Desbloquea el matching y el retraining sistemático. Sin capas limpias, cada mejora de modelo requiere trabajo manual de datos. |
| **3** | **Modelado avanzado (target encoding + features externas)** | Con el pipeline limpio y tracking funcionando, iterar sobre features tiene ROI alto y bajo riesgo. |
| **4** | **Matching de inmuebles** | El más costoso de implementar bien. Requiere golden set manual. Impacto en calidad de datos a largo plazo, pero no desbloquea otras cosas en el corto. |

**Razonamiento:** MLOps y arquitectura son habilitadores — sin ellos, el trabajo de modelado y matching se hace a ciegas y no es reproducible. El matching es el más valioso para el producto final pero el más caro de validar correctamente; vale más hacerlo bien en el mes 2 que apurado en el mes 1.
