# Proyecto-20261

Repositorio de análisis MIP/IIT con dos implementaciones principales:

1. **QNodes** — estrategia `KQNodes` basada en búsqueda heurística (simulated annealing) sobre particiones.
2. **KGeoMIP** — estrategia `KGeoMIP` y `GeometricSIAGeneralizada` basadas en programación dinámica sobre una tabla geométrica de costos, con soporte para k ≥ 2 particiones.

## Requisitos

- Python 3.11+
- `uv` instalado

```bash
pip install uv
```

## Estructura del repositorio

```
QNodes/
  exec.py                   # Punto de entrada
  src/
    main.py                 # Configuración del caso a analizar
    controllers/strategies/KQNodes/   # Estrategia KQNodes (annealing)
    models/                 # NCube, System, Solution, SIA
    funcs/                  # IIT, fuerza bruta, visualización

KGeoMIP/
  exec.py                   # Punto de entrada
  data/creation.py          # Generación de datasets TPM
  src/
    main.py                 # Flujo de ejecución desde Excel
    controllers/strategies/KGeoMIP/          # Estrategia k-partición DP
    controllers/strategies/geometric_generalized/  # GeometricSIAGeneralizada
    graphics/               # Generación de figuras comparativas (matplotlib)
    models/                 # NCube, System, Solution, SIA
  results/                  # Excels de entrada y salida
  tests/                    # Tests de regresión
```

## 1) QNodes

### Instalación

```bash
cd QNodes
uv sync
```

### Ejecución

```bash
uv run exec.py
```

### Configuración

Edita `QNodes/src/main.py` para ajustar:

- `estado_inicial` — estado de inicio de la red.
- `condiciones`, `alcance`, `mecanismo` — parámetros del subsistema a analizar.

Las redes se cargan desde `QNodes/src/.samples/`. Si la ejecución termina rápido no es necesariamente un error: puede ser un caso pequeño o `phi = 0`.

### Tests

```bash
cd QNodes
uv run pytest tests/
```

## 2) KGeoMIP

### Instalación

```bash
cd KGeoMIP
uv sync
```

### Ejecución

```bash
uv run exec.py
```

El punto de entrada `exec.py` llama a `iniciar()` en `src/main.py`, que lee un Excel de entrada y escribe los resultados en otro Excel.

### Variables de entorno

| Variable              | Descripción                        | Valor por defecto                           |
|-----------------------|------------------------------------|---------------------------------------------|
| `GEOMIP_INPUT_XLSX`   | Excel con los subsistemas a evaluar | `KGeoMIP/results/Pruebas_video.xlsx`        |
| `GEOMIP_OUTPUT_XLSX`  | Excel de resultados                | `KGeoMIP/results/k3/resultados_Geometric_12A3.xlsx` |

### Estrategias disponibles

- **`KGeoMIP`** (`src/controllers/strategies/KGeoMIP/`) — k-partición de mínima información mediante DP con tabla geométrica truncada adaptativamente, greedy jerárquico y hill climbing. Parámetro `k` (≥ 2).
- **`GeometricSIAGeneralizada`** (`src/controllers/strategies/geometric_generalized/`) — variante generalizada que opera sobre el sistema completo (sin necesidad de columnas alcance/mecanismo en el Excel).

### Datos TPM

Los archivos `NxA.csv` se ubican en `KGeoMIP/data/samples/`. En la primera ejecución se convierten automáticamente a `.npy` (float32) para reducir el uso de RAM mediante memmap.

### Tests de regresión

```bash
cd KGeoMIP
uv run pytest tests/
```
