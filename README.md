# Proyecto Final — Analítica Avanzada: NBA Intelligence Platform

> **Universidad Anáhuac** · Ingeniería en Tecnologías de la Información · 4° Semestre  
> Curso: *Analítica Avanzada*

---

## Autores

| Nombre | Contribución Principal |
|---|---|
| **Luis Alberto Arias Llaguno** | Modelos LSTM, XGBoost, Predictor de partidos (XGBoost calibrado), API Flask |
| **David Ceballos Mata** | Clustering de jugadores (K-Means + PCA), integración con MLflow, generación de joblibs |
| **Rubén Octavio Flores Ramos** | Predictor de playoffs (Regresión Logística + Naive Bayes), recolección de standings históricos |

---

## Descripción General

Este repositorio contiene el **proyecto final** del curso de Analítica Avanzada, donde se aplican técnicas de **machine learning** y **deep learning** para analizar y resolver problemáticas del mundo de la NBA. El proyecto integra datos reales de la NBA API oficial, múltiples modelos predictivos optimizados con búsqueda de hiperparámetros, seguimiento de experimentos con MLflow, y una **API REST en Flask** que expone los modelos entrenados para consumo externo.

Los tres problemas principales que se abordan son:

1. **Predicción de puntos partido a partido** de los Lakers usando redes neuronales LSTM.
2. **Agrupación de jugadores** por estilo de juego usando K-Means con visualización PCA.
3. **Predicción de clasificación a playoffs** de equipos usando modelos de clasificación clásicos.
4. **Predicción del ganador de un partido** entre dos equipos usando XGBoost calibrado (el modelo final deployado).

---

## Estructura del Repositorio

```
proyecto-analitica-nba/
│
├── app.py                          # API REST Flask con los 3 endpoints principales
├── project.py                      # Archivo consolidado del proyecto final
├── requirements.txt                # Dependencias del proyecto
│
├── tests_luis.ipynb                # Notebook de Luis: LSTM, XGBoost, predictor de partidos
├── tests_david.ipynb               # Notebook de David: Clustering K-Means + PCA
├── tests_ruben.ipynb               # Notebook de Rubén: Predicción de playoffs
│
├── xgb_isotonic_nba_homewin.joblib # Modelo XGBoost calibrado (predicción de partidos)
├── modelo_nba.joblib               # Modelo champion de playoffs (LogReg o Naive Bayes)
├── player_cluster_model_k6.joblib  # Modelo K-Means con k=6 para clustering de jugadores
│
├── mlruns/                         # Experimentos y métricas registrados en MLflow
├── kt_lakers_runs/                 # Resultados de Keras Tuner (búsqueda LSTM)
└── models/                         # Directorio de modelos adicionales
```

---

## Tecnologías y Dependencias

```txt
flask         # API REST
nba_api       # Fuente de datos oficial de la NBA
pandas        # Manipulación de datos
numpy         # Operaciones numéricas
scikit-learn  # K-Means, PCA, Regresión Logística, Naive Bayes, pipelines
joblib        # Serialización y carga de modelos
mlflow        # Seguimiento de experimentos y registro de modelos
xgboost       # Modelo XGBoost para predicción de partidos
pydantic      # Validación de datos
```

### Dependencias adicionales (notebooks)

```txt
tensorflow / keras   # Redes neuronales LSTM y densas
keras-tuner          # Búsqueda automática de hiperparámetros
seaborn              # Visualizaciones estadísticas
matplotlib           # Gráficas
```

Para instalar todas las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

---

## Notebooks de Experimentación

### `tests_luis.ipynb` — Luis Alberto Arias Llaguno

Este notebook es el más extenso del proyecto (128 celdas) y contiene todos los experimentos de Luis, que culminaron en el modelo final deployado en la API.

#### Secciones:

**0. Importación de librerías**  
Carga de todas las bibliotecas necesarias: `tensorflow`, `keras`, `sklearn`, `xgboost`, `mlflow`, `nba_api`.

**1. Exploración de Datos**  
Análisis exploratorio usando la NBA API para obtener el historial completo de juegos de los Lakers.

---

**2.1 — Predicción LSTM Parte 1 (solo stats Lakers)**

- **Problema:** Predecir los puntos que anotará el equipo de los Lakers en su próximo partido, usando exclusivamente las estadísticas de los últimos N juegos del propio equipo.
- **Enfoque:** Red neuronal LSTM (Long Short-Term Memory) con secuencias de longitud 3, 5 y 10 partidos.
- **Features utilizadas:** `ES_CASA`, `WIN`, `MIN`, `FGM`, `FGA`, `FG_PCT`, `FG3M`, `FG3A`, `FG3_PCT`, `FTM`, `FTA`, `FT_PCT`, `OREB`, `DREB`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `PF`, `DAYS_REST`.
- **Target:** `PTS` (puntos anotados en el siguiente partido).
- **Optimización:** Keras Tuner con búsqueda aleatoria sobre unidades LSTM, tasa de aprendizaje y dropout. Experimentos registrados en MLflow.
- **Mejor configuración:** `SEQ_LEN=10` — MAE ≈ 11.33 puntos (el modelo que usa los 10 juegos anteriores como contexto obtuvo las mejores métricas).

---

**2.2 — Predicción LSTM Parte 2 (stats Lakers + contrincante)**

- **Problema:** Igual que la sección 2.1, pero ahora se enriquecen las features con las estadísticas del equipo contrincante en cada partido.
- **Justificación:** Conocer el rendimiento histórico del rival da más contexto al modelo para predecir el marcador.
- **Features adicionales (prefijo `OPP_`):** `OPP_PTS`, `OPP_FGM`, `OPP_FGA`, `OPP_FG_PCT`, `OPP_FG3A`, `OPP_FG3_PCT`, `OPP_FTA`, `OPP_FT_PCT`, `OPP_REB`, `OPP_AST`, `OPP_STL`, `OPP_BLK`, `OPP_TOV`, entre otras.
- **Proceso:** Se realizó un merge entre el DataFrame de los Lakers y el de los rivales usando el `GAME_ID` como llave.
- **Resultado:** Se evalúa si las features del contrincante mejoran la predicción respecto a la Parte 1.

---

**2.3 — Red Neuronal Densa (baseline ChatGPT)**  
Experimento de red neuronal densa como punto de comparación (baseline) para las LSTM.

---

**2.4 — Red Neuronal Densa con Feature Engineering Avanzado**  
Versión mejorada de la red densa con un nuevo proceso de ingeniería de características, explorando si un modelo más simple puede competir con LSTM cuando las features son mejores.

---

**2.5 — XGBoost (Lakers)**  
Primera exploración con XGBoost para predicción de puntos de los Lakers, incluyendo:
- Obtención, transformación y limpieza de datos con rolling features (ventanas móviles).
- Búsqueda de hiperparámetros óptimos.
- Entrenamiento y evaluación de métricas.

---

**2.6 — XGBoost Calibrado: Predictor General de Partidos NBA (Modelo Final)**

Esta es la propuesta más ambiciosa y el modelo que finalmente se deployó en la API.

- **Problema:** Dado un partido entre cualquier equipo local (home) y visitante (away) de la NBA, predecir la probabilidad de que gane el equipo local.
- **Datos:** Temporadas regulares de la NBA del 2015 al 2025, obtenidas usando `LeagueGameFinder`.
- **Pipeline de datos:**
  1. `fetch_official_nba_regular_season()` — Descarga datos de cada temporada.
  2. `build_team_game_table_from_raw()` — Construye tabla de partidos por equipo con columnas limpias, agrega `WIN`, `IS_HOME` y `DAYS_REST`.
  3. `add_pregame_features_teamgame()` — Agrega **rolling features** de los últimos 5 partidos por equipo (media móvil de PTS, FG%, REB, AST, TOV, OREB, FGA, FTA, PLUS_MINUS; desviación estándar de PTS, FG%, FG3%; win rate; y racha de victorias).
  4. `build_single_game_features()` — Construye el vector de features para un partido específico entre dos equipos, calculando las diferencias (`DIFF_`) entre el equipo local y visitante.
- **Features del modelo (por equipo, home `H_` y away `A_`):**
  - Rolling medias (5 juegos): puntos, porcentaje de tiro, rebotes, asistencias, pérdidas, rebotes ofensivos, intentos, etc.
  - Desviaciones estándar de PTS, FG%, FG3%.
  - Win rate en los últimos 5 partidos.
  - Racha de victorias (`WIN_STREAK`).
  - Días de descanso.
  - Diferenciales de todas las features anteriores (`DIFF_`).
  - `ES_CASA = 1` (siempre el equipo local).
- **Calibración:** El modelo XGBoost base se calibra con `CalibratedClassifierCV` usando método isotónico, para obtener probabilidades bien calibradas en lugar de solo clasificaciones binarias.
- **Serialización:** El modelo calibrado + columnas de features se guardan en `xgb_isotonic_nba_homewin.joblib`.
- **MLflow:** Todos los experimentos (búsqueda de hiperparámetros, métricas finales) quedan registrados.
- **Visualizaciones generadas:**
  - Curvas de calibración.
  - Feature Importance.
  - Curva ROC, matriz de confusión, distribución de probabilidades.

---

### `tests_david.ipynb` — David Ceballos Mata

Este notebook (12 celdas) contiene el desarrollo completo del modelo de **agrupación de jugadores NBA por estilo de juego**.

#### Secciones:

**1.1 — Obtención de Datos**  
Uso de `LeagueDashPlayerStats` de la NBA API para obtener estadísticas de todos los jugadores de la temporada regular 2023-24.

**1.2 — Preprocesamiento**
- Filtro de jugadores con **≥ 500 minutos** jugados para eliminar ruido.
- Estadísticas normalizadas **por 36 minutos** para comparación justa entre jugadores.
- Features usadas: `PTS_PER36`, `REB_PER36`, `AST_PER36`, `STL_PER36`, `BLK_PER36`, `FGA_PER36`, `FG_PCT`, `FG3A_PER36`, `FG3_PCT`, `FTA_PER36`, `FT_PCT`, `TOV_PER36`.

**1.3 — Método del Codo para K óptimo**  
Cálculo de inercia para k de 2 a 14 y visualización para identificar el punto de inflexión óptimo.

**1.4 — K-Means con k=6**
- Se seleccionó `k=6` como el número óptimo de clusters.
- Los clusters agrupan a los jugadores en arquetipos de juego (ej. bases anotadores, bases de paso, aleros versátiles, pívots defensivos, etc.).
- Análisis de características promedio por cluster.

**1.5 — Visualización PCA**  
Reducción de dimensionalidad con PCA (2 componentes) para visualizar los clusters en 2D con un scatter plot coloreado por cluster.

**Generación del Joblib**  
Función `generate_model(k)` que encapsula todo el flujo: obtener datos → preprocesar → escalar → entrenar K-Means → entrenar PCA → guardar en `player_cluster_model_k{k}.joblib` con el formato:

```python
{
    'scaler': StandardScaler(),
    'kmeans': KMeans(n_clusters=k),
    'pca': PCA(n_components=2),
    'features': list_of_feature_names
}
```

**MLflow**  
Dos implementaciones de logging:
1. Logging básico del modelo KMeans a MLflow con el nombre registrado `nba_player_cluster_k6`.
2. Flujo completo de MLflow con obtención de datos, preprocesamiento, entrenamiento y log de métricas (inercia).

---

### `tests_ruben.ipynb` — Rubén Octavio Flores Ramos

Este notebook (12 celdas) contiene el modelo de **predicción de playoffs** usando datos de standings históricos de la NBA.

#### Secciones:

**Importación de Librerías**  
`pandas`, `scikit-learn` (LogisticRegression, GaussianNB, Pipeline, GridSearchCV), `mlflow`, `joblib`, `nba_api`.

**Recolección de Datos**  
Función `obtener_datos_entrenamiento(anios)` que itera sobre múltiples temporadas usando `LeagueStandings` de la NBA API:
- Variable target: `Playoffs = 1` si el `PlayoffRank ≤ 8`, `0` en caso contrario.
- Features usadas: `WinPCT`, `PointsPG`, `OppPointsPG`, `DiffPointsPG`.
- Se agrega un `time.sleep(0.6)` entre requests para respetar el rate limit de la API.

**MLflow Setup**  
Experimento `NBA_Playoff_Predicciones_LogReg` configurado en MLflow.

**Modelo 1: Regresión Logística**
- Pipeline: `StandardScaler → LogisticRegression`.
- Optimización con `GridSearchCV` (5-fold CV):
  - `clf__C`: [0.1, 1, 10]
  - `clf__solver`: ['liblinear', 'lbfgs']
- Métrica principal: `accuracy`.
- Resultados y modelo registrados en MLflow.

**Modelo 2: Naive Bayes**
- Pipeline: `StandardScaler → GaussianNB`.
- Sin hiperparámetros a tunear (Naive Bayes es simple y robusto).
- Resultados registrados en MLflow.

**Selección del Modelo Campeón**
- Se comparan las accuracies de ambos modelos.
- El modelo con mayor accuracy se guarda como `modelo_nba.joblib`.
- Este es el modelo que usa la API en el endpoint `/predict_playoffs`.

---

## API REST (Flask)

El archivo `app.py` implementa una API REST en Flask que expone tres endpoints de predicción. Al iniciar la aplicación, se cargan automáticamente los modelos desde sus archivos `.joblib` y se descargan y preparan los datos base de la NBA (temporadas 2015-2025).

### Iniciar el servidor

```bash
python app.py
```

El servidor inicia en `http://localhost:8000`.

> **Nota:** Al iniciar, la aplicación descarga datos de la NBA API para las temporadas 2015-2025. Esto puede tardar algunos minutos debido al rate limiting de la API.

---

### Endpoints

#### `GET /cluster_players/`

Agrupa a todos los jugadores de la NBA (temporada 2023-24) en `k` clusters según su estilo de juego, usando el modelo K-Means pre-entrenado.

**Parámetro de query:**
| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `k` | `int` | `6` | Número de clusters. Debe ser > 1. |

**Respuesta exitosa (200):**
```json
{
  "k": 6,
  "player_clusters": [
    {
      "PLAYER_NAME": "LeBron James",
      "TEAM_ABBREVIATION": "LAL",
      "cluster": 2,
      "pc1": 1.452,
      "pc2": -0.873
    },
    ...
  ]
}
```

**Errores posibles:**
- `400` — `k` ≤ 1.
- `404` — No existe un modelo guardado para el `k` solicitado (solo existe `k=6` por defecto).
- `500` — Error al cargar el modelo o al obtener datos frescos de la API.

**Ejemplo de uso:**
```bash
curl "http://localhost:8000/cluster_players/?k=6"
```

---

#### `POST /predict_matchup`

Predice la probabilidad de victoria del equipo local en un partido entre dos equipos NBA, usando el modelo XGBoost calibrado.

**Body (JSON):**
```json
{
  "home_abbr": "LAL",
  "away_abbr": "GSW",
  "as_of_date": "2024-01-15"
}
```

| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `home_abbr` | `string` | ✅ | Abreviatura del equipo local (ej. `"LAL"`, `"GSW"`, `"BOS"`). |
| `away_abbr` | `string` | ✅ | Abreviatura del equipo visitante. |
| `as_of_date` | `string` (YYYY-MM-DD) | ❌ | Fecha del partido. Si se omite, usa la última fecha disponible en los datos. |

**Respuesta exitosa (200):**
```json
{
  "home_team": "LAL",
  "away_team": "GSW",
  "prediction_date": "2024-01-15",
  "home_win_probability": 0.5832,
  "away_win_probability": 0.4168
}
```

**Errores posibles:**
- `400` — Faltan `home_abbr` o `away_abbr`, fecha inválida, o no hay suficiente historial para alguno de los equipos.
- `500` — Features incompletas (falta historial de rolling).
- `503` — El modelo no está inicializado (falló la carga al inicio).

**Ejemplo de uso:**
```bash
curl -X POST http://localhost:8000/predict_matchup \
  -H "Content-Type: application/json" \
  -d '{"home_abbr": "LAL", "away_abbr": "GSW"}'
```

---

#### `POST /predict_playoffs`

Predice si un equipo clasificará a playoffs basándose en sus estadísticas de la temporada.

**Body (JSON):**
```json
{
  "WinPCT": 0.61,
  "PointsPG": 115.3,
  "OppPointsPG": 110.1,
  "DiffPointsPG": 5.2
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `WinPCT` | `float` | Porcentaje de victorias (0.0 – 1.0). |
| `PointsPG` | `float` | Puntos anotados por partido (promedio). |
| `OppPointsPG` | `float` | Puntos recibidos por partido (promedio). |
| `DiffPointsPG` | `float` | Diferencial de puntos por partido (`PointsPG - OppPointsPG`). |

**Respuesta exitosa (200):**
```json
{
  "input_stats": {
    "WinPCT": 0.61,
    "PointsPG": 115.3,
    "OppPointsPG": 110.1,
    "DiffPointsPG": 5.2
  },
  "prediction_label": "Hace Playoffs",
  "prediction_value": 1,
  "probability_no_playoffs": 0.0823,
  "probability_playoffs": 0.9177
}
```

**Errores posibles:**
- `400` — Faltan alguna de las 4 features requeridas, o los valores no son numéricos.
- `503` — El modelo no está inicializado.

**Ejemplo de uso:**
```bash
curl -X POST http://localhost:8000/predict_playoffs \
  -H "Content-Type: application/json" \
  -d '{"WinPCT": 0.61, "PointsPG": 115.3, "OppPointsPG": 110.1, "DiffPointsPG": 5.2}'
```

---

## Modelos Entrenados

### `xgb_isotonic_nba_homewin.joblib`
- **Tipo:** `CalibratedClassifierCV` (XGBoost + calibración isotónica)
- **Objetivo:** Predecir la probabilidad de victoria del equipo local en un partido de la NBA.
- **Datos de entrenamiento:** Temporadas regulares 2015–2025 (todos los equipos NBA).
- **Features:** Rolling stats (ventana de 5 partidos) de ambos equipos + diferenciales.
- **Contenido del bundle:**
  ```python
  {
      "model_calibrated": CalibratedClassifierCV,
      "feature_cols": List[str]
  }
  ```

### `modelo_nba.joblib`
- **Tipo:** Pipeline de scikit-learn (`StandardScaler` + clasificador campeón: Regresión Logística o Naive Bayes)
- **Objetivo:** Predecir si un equipo hará playoffs dado su rendimiento en la temporada.
- **Features:** `WinPCT`, `PointsPG`, `OppPointsPG`, `DiffPointsPG`.
- **Selección:** El modelo con mayor accuracy entre Regresión Logística y Naive Bayes.

### `player_cluster_model_k6.joblib`
- **Tipo:** Bundle con `StandardScaler`, `KMeans(k=6)` y `PCA(n_components=2)`.
- **Objetivo:** Agrupar jugadores NBA por estilo de juego usando estadísticas por 36 minutos.
- **Datos de entrenamiento:** Jugadores con ≥500 minutos en la temporada 2023-24.
- **Contenido del bundle:**
  ```python
  {
      "scaler": StandardScaler,
      "kmeans": KMeans,
      "pca": PCA,
      "features": List[str]
  }
  ```

---

## Seguimiento de Experimentos con MLflow

El proyecto usa **MLflow** para registrar todos los experimentos de entrenamiento. Los artefactos se guardan en el directorio `mlruns/`.

Para visualizar los experimentos en la interfaz web de MLflow:

```bash
mlflow ui
```

Luego abre `http://localhost:5000` en tu navegador.

**Experimentos registrados:**
- `NBA_Playoff_Predicciones_LogReg` — Regresión Logística y Naive Bayes para playoffs (Rubén).
- Experimentos de clustering K-Means (David).
- Experimentos de LSTM con Keras Tuner (Luis).
- Experimentos de XGBoost con búsqueda de hiperparámetros y calibración (Luis).

---

## Consideraciones Técnicas

### Rate Limiting de la NBA API

La NBA API tiene límites de peticiones. El proyecto incluye `time.sleep(0.6)` entre requests de temporada para evitar ser bloqueado. Al iniciar el servidor Flask, la descarga de datos (temporadas 2015-2025) puede tardar **entre 5 y 15 minutos** dependiendo de la velocidad de conexión.

### Abreviaturas de Equipos Históricas

Algunos equipos han cambiado de ciudad o nombre a lo largo del tiempo. El notebook de Luis incluye un mapa de corrección de abreviaturas históricas:

```python
abbr_fix_map = {
    "UTH": "UTA",   # Utah Jazz
    "SDC": "LAC",   # San Diego Clippers → LA Clippers
    "SEA": "OKC",   # Seattle SuperSonics → Oklahoma City Thunder
    "GOS": "GSW",   # Golden State Warriors
    "SAN": "SAS",   # San Antonio Spurs
    "KCK": "SAC",   # Sacramento Kings
    ...
}
```

### Rolling Features y Partidos Tempranos

Las rolling features requieren al menos N partidos anteriores (donde N es el tamaño de ventana, por defecto 5). Los partidos al inicio de cada temporada pueden no tener features completas. La columna `HAS_ROLL_FEATURES` indica si un registro tiene todas las features de rolling disponibles.

### Modelos Pre-entrenados

Si los archivos `.joblib` no están presentes, el servidor iniciará pero los endpoints correspondientes responderán con `503 Service Unavailable`. Para regenerar los modelos, ejecuta las secciones correspondientes de cada notebook.

---

## Guía de Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd proyecto-analitica-nba
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Para los notebooks (LSTM y visualizaciones):

```bash
pip install tensorflow keras-tuner seaborn matplotlib jupyter
```

### 4. Ejecutar la API

```bash
python app.py
```

### 5. Ejecutar los notebooks

```bash
jupyter notebook
```

Abre cualquiera de los archivos `.ipynb` para explorar los experimentos.

---

## Flujo de Datos

```
NBA API (nba_api)
       │
       ▼
LeagueGameFinder          LeagueDashPlayerStats      LeagueStandings
(juegos 2015-2025)        (jugadores 2023-24)        (standings históricos)
       │                          │                          │
       ▼                          ▼                          ▼
build_team_game_table     Normalización por 36min    Etiqueta Playoffs
       │                  StandardScaler             (PlayoffRank ≤ 8)
       ▼                          │                          │
add_pregame_features       K-Means (k=6)              Pipeline
(rolling window=5)         + PCA (2D)                 LogReg / Naive Bayes
       │                          │                          │
       ▼                          ▼                          ▼
XGBoost + Calibración    player_cluster_model_k6.joblib   modelo_nba.joblib
xgb_isotonic_nba_homewin.joblib
       │                          │                          │
       └──────────────────────────┴──────────────────────────┘
                                  │
                                  ▼
                           Flask API (app.py)
                    ┌──────────────────────────┐
                    │  /cluster_players/       │
                    │  /predict_matchup        │
                    │  /predict_playoffs       │
                    └──────────────────────────┘
```

---

## Notas Académicas

- Este proyecto fue desarrollado para el curso de **Analítica Avanzada** en la Universidad Anáhuac.
- Todos los datos provienen de la **NBA API oficial** (`nba_api`), que es una librería de Python que consume los endpoints públicos de `stats.nba.com`.
- El proyecto sigue buenas prácticas de MLOps: separación entre exploración (notebooks) y producción (API), serialización de modelos con `joblib`, y seguimiento de experimentos con `MLflow`.
- Los modelos no están diseñados para uso comercial ni para apuestas deportivas.

---