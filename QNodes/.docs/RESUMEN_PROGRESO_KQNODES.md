# Resumen de progreso — Corrección de QNodes y desarrollo de KQNodes

> Documento de seguimiento del trabajo realizado en la sesión: diagnóstico y
> corrección de la estrategia QNodes, y diseño + implementación de **KQNodes**
> (extensión a k-particiones). Incluye estado actual, lo pendiente, mejoras
> posibles y las técnicas/estrategias empleadas.
>
> Ámbito acordado: **solo el módulo `QNodes/`** (no se toca GeoMIP).

---

## 1. Contexto y objetivo

El proyecto calcula **φ** de la Teoría de la Información Integrada (IIT)
buscando la **Partición de Mínima Información (MIP)** de un subsistema. El
objetivo del semestre (doc `docs/Proyecto_KGeoMIP.docx`) es **extender el
algoritmo de biparticiones al caso general de k-particiones (k-MIP)** con
k ∈ {2,3,4,5}. Se decidió hacerlo sobre QNodes, construyendo **KQNodes**.

---

## 2. Cronología de lo realizado

### 2.1 Exploración y planteamiento
- Exploración del repositorio y de los documentos de `docs/`; se identificó que
  el entregable es la extensión a k-particiones.
- Análisis de cómo funcionan las **biparticiones** en QNodes: conjunto base de
  vértices bitemporales `(tiempo, índice)`, corte vía `System.bipartir`, pérdida
  con la EMD-efecto, y minimización con el **algoritmo de Queyranne** (submodular).

### 2.2 Baseline y hallazgo crítico
- No existían tests reales. Se creó un **baseline de regresión** (`tests/baseline.py`
  + `baseline_qnodes.csv`, 1235 casos N3–N5) que compara QNodes contra
  **BruteForce** (referencia exacta).
- **Hallazgo:** QNodes coincidía con la MIP real en **0 %** de los casos (siempre
  daba pérdida mayor) y devolvía `φ = inf` en subsistemas de 1 futuro + 1 presente.

### 2.3 Corrección de QNodes
- Se separaron dos causas:
  - **Bug de implementación (dominante):** en cada fase de Queyranne se guardaba
    la pérdida del candidato equivocado — `f({s})` (penúltimo) en vez de
    `f({pendiente t})` —, y el caso `m+n=2` quedaba en infinito.
  - **No-submodularidad (límite teórico, menor):** se verificó empíricamente que
    la EMD-efecto es **simétrica pero no submodular** (ni en redes estocásticas ni
    deterministas), por lo que Queyranne no puede ser exacto al 100 %.
- Prueba decisiva: un Queyranne **correcto** con la misma función da 97–100 %,
  frente al 0–28 % del original → el grueso del fallo era el bug.
- **Fix** en `q_nodes.py`: evaluar explícitamente el vértice pendiente y guardar
  `f({pendiente})`. Resultado: **0 % → 96–98 %** de coincidencia exacta, sin casos
  `inf`. El golden baseline se actualizó al comportamiento corregido.

### 2.4 Limpieza segura
- `q_nodes.py`: eliminado código muerto (`self.tiempos`, `self.m`, `self.n`,
  `indices_alcance/mecanismo`, import `INFTY_POS`).
- `system.py`: eliminado el no-op `else: self.memo[clave] = self.memo[clave]`.
- `slogger.py`: corregido el `RecursionError` (colorama se reinicializaba por
  instancia) y la fuga de descriptores (FileHandler reabiertos): `init()` una
  vez a nivel de módulo y reutilización del logger por nombre.
- Verificado con el baseline: comportamiento idéntico (1235 casos).

### 2.5 KQNodes — Fase 0 (formulación)
- Definición de la pérdida **δ_k** generalizando el corte: cada nodo futuro
  conserva solo los presentes de su bloque.
- Referencia exacta `kbruteforce` (enumera particiones de los 2N vértices).
- **Hallazgo clave — separabilidad:** `δ_k = Σ_i c_i(P_i)`, es decir, depende solo
  de, para cada nodo futuro, el conjunto de presentes de su bloque.
- **Algoritmo exacto por separabilidad:** particiona solo los presentes y asigna
  cada futuro de forma voraz. Coincide con la fuerza bruta hasta `float32`
  (diff 1.5×10⁻⁸) y reduce candidatos hasta ~**800×** (`k^N` en vez de `k^{2N}`).
- Documento matemático: `.docs/.strategies/kqnodes/formulacion.md`.

### 2.6 KQNodes — Fase 1 (clase exacta)
- Clase **`KQNodes(SIA)`** en `src/strategies/KQNodes.py`, con
  `aplicar_estrategia(estado, condicion, alcance, mecanismo, k)` → `Solution`.
- Formateador `fmt_kparticion` y constantes `KQNODES_*`.
- **Validación:** 15/15 OK (KQNodes == kbruteforce, k=2..5; k=2 == BruteForce).

### 2.7 Benchmark de límites
- El costo del exacto depende del nº de **nodos presente** `n`, no del tamaño N
  de la red. Límite práctico: **n ≈ 10–11 presentes** (≤ ~30 s). A partir de
  n ≈ 13–14, minutos→horas.

### 2.8 KQNodes — Fase 2 (heurísticas)
- **Voraz** (búsqueda local), **recocido simulado** (con *warm start* desde el
  voraz, vecindario reetiquetar/intercambiar, opción multi-arranque) y selección
  **`auto`** por umbral de presentes.
- **Validación:** recocido **19/20** óptimos exactos (gap máx 0.007) vs 10/20 del
  voraz; resuelve n=14 en ~1.5 s.

### 2.9 KQNodes — Fase 3 (optimización de rendimiento para pruebas grandes)

> Objetivo: que la estrategia (`src/strategies/KQNodes.py`) escale a sistemas
> grandes. Todas las optimizaciones se validaron contra el comportamiento previo:
> **mismos δ** en todos los benchmarks y **34/34 tests pytest** verdes.

- **Marginalización "condicionar-primero" (mejora decisiva).** `_marginal_futuro`
  promediaba el arreglo completo `2^n` por cada coste `c_i(Q)`. Reformulado para
  **fijar primero al estado inicial los ejes conservados `Q`** (un *slice* que
  reduce el arreglo a `2^(n-|Q|)`) y promediar después → coste por evaluación de
  `Θ(2^n)` a `Θ(2^{n-|Q|})`. Se detectó que `NCube.condicionar` asume dims
  contiguas (falla en subsistemas asimétricos), así que el *slice* se hace con el
  convenio de ejes locales de `marginalizar` (`(ndims-1)-dim_idx`). **Verificado
  idéntico al cálculo original (≤6e-8) en N3–N6, simétricos y asimétricos.**
  - **Recocido (mismos δ):** N16 32.6 s → 0.9 s (**~37×**), N18 155 s → 4 s
    (**~38×**), N20 pasó de inviable a ~21 s.

- **Camino escalar `_delta_grupos`.** Separa el cálculo de `δ_k` (escalar) de la
  reconstrucción de bloques de vértices. El lazo de búsqueda (exacto y heurísticas)
  ya no reconstruye la k-partición por candidato; la reconstrucción se difiere a una
  única llamada sobre el ganador.

- **Microoptimizaciones del lazo caliente.** Grupos como `frozenset` (clave directa
  del caché) y `_costo` con un solo `dict.get` por acierto.

- **Recocido con evaluación INCREMENTAL** (`_RecocidoIncremental`, Python puro).
  Mantiene una matriz `coste[lab][i]` y, tras mover un presente, recalcula solo las
  **≤2 columnas afectadas** en vez de barrer `m·k` costes y reconstruir grupos cada
  paso. **Hallazgo:** una primera versión con numpy era ~7 % *más lenta* a tamaños
  chicos por overhead fijo de numpy (cProfile lo ocultaba); la versión
  lista-de-listas pura gana en **ambos** regímenes: **1.5–1.8×** en corridas largas
  (caché caliente) y ~1.15× en N grande. La δ devuelta se recalcula con
  `_evaluar_grupos` para consistencia exacta con el método exacto. **δ bit-exacta
  vs `_delta_grupos`: 9035 comprobaciones, diferencia 0.0.**

- **`UMBRAL_EXACTO` remedido y dependiente de k.** Como el exacto cuesta
  `Σ_{r=1}^k S(n,r)` candidatos, el techo práctico depende de k. La marginalización
  optimizada subió esos techos (p.ej. **n=13, k=3: 31 s → 4.6 s = 6.8×**). Medido en
  `fase1_bench.py` (presupuesto ≈6 s): **k=2 → n≤14, k=3 → n≤13, k=4 → n≤11,
  k=5 → n≤11**. Implementado como `UMBRAL_EXACTO_POR_K = {2:14, 3:13, 4:11, 5:11}`
  + `_umbral_exacto(k)` (k≥6 → 10); el despacho `auto` lo usa en vez del escalar 11.
  Antes `auto` mandaba k=2 y k=3 a recocido ya en n=12; ahora usan el óptimo exacto
  hasta n=14 y n=13.

- **Cumplimiento de nomenclatura del spec.** Añadidos los nombres canónicos
  `find_k_mip(k, metodo)` (alias despachador) y `marginal_distributions(particion)`
  (envoltorio público de `_distribucion_kparticion`).

### 2.10 Integración de usuario, persistencia `.npy` y visualización

> Cierre de los puntos de entrada y de la infraestructura de E/S para poder
> ejecutar KQNodes desde la terminal y escalar a N grandes. Verificado con el
> baseline (comportamiento idéntico de QNodes) y con corridas de KQNodes N25.

- **Punto de entrada CLI (`exec.py` + `src/main.py`).** `exec.py` ahora usa
  `argparse`: `--estrategia {qnodes,kqnodes}`, `--k`, `--metodo {auto,exacto,
  voraz,recocido}` y `--viz ARCHIVO.png`. Nueva función `iniciar_kqnodes(k,
  metodo, guardar_viz)` en `src/main.py` que carga la red, ejecuta
  `KQNodes.aplicar_estrategia` y, opcionalmente, guarda la visualización. El demo
  de QNodes pasó a N22 y el de KQNodes a N25.
- **Migración de la TPM de CSV de texto a binario `.npy` (decisiva para RAM).**
  `Manager` persiste ahora en `.npy` (float32 crudo, `np.save`): el archivo ocupa
  exactamente `2^n · n · bytes_por_celda` y se recarga con `np.load` **sin parseo
  ni pico de memoria**, lo que destraba N grandes (N22+). `cargar_red` prioriza el
  `.npy` y **recae en el CSV heredado** (`np.loadtxt`) si es lo único disponible,
  de modo que las muestras antiguas siguen siendo legibles. Nuevas propiedades
  `npy_filename`/`csv_filename` y constante `NPY_EXTENSION`. El estimador de tamaño
  distingue `int8` (determinista) vs `float32` (estocástico).
- **Visualización del hipercubo (`src/funcs/visualize.py`).** Dibuja el hipercubo
  binario de `n` dimensiones coloreado por bloque de la k-partición (cuadrado 2D
  para n=2, cubo 3D para n=3, **proyección PCA 2D para n≥4**); presentes con
  relleno sólido y futuros con borde grueso. Expuesta vía `--viz`.

---

## 3. Estado actual (qué funciona)

| Componente | Estado |
|---|---|
| QNodes (biparticiones) | Corregido: 96–98 % exacto, sin `inf` |
| Baseline de regresión | Operativo (1235 casos) |
| Limpieza de código/logger | Aplicada y verificada |
| KQNodes exacto (separabilidad) | Validado (exacto hasta float32) |
| KQNodes heurístico (voraz/recocido) | Validado (recocido 19/20) |
| Selección automática exacto/heurístico | Operativa (umbral dependiente de k) |
| Optimización de rendimiento (Fase 3) | Aplicada y validada (recocido ~37–38× en N16–18; N20 viable) |
| Punto de entrada de usuario (CLI) | Operativo (`exec.py --estrategia kqnodes --k --metodo --viz`) |
| Persistencia de TPM en `.npy` (+ fallback CSV) | Aplicada (carga sin parseo, viable N22+) |
| Visualización del hipercubo | Implementada (2D / 3D / PCA n≥4) |
| Documento de formulación matemática | Escrito |

**KQNodes es funcionalmente completo**: exacto para n pequeño/moderado +
heurístico para n grande, con despacho automático.

---

## 4. Lo que falta por implementar

1. **Tests unitarios formales** (pytest): consolidar las validaciones ad-hoc de
   `tests/fase*.py` en una suite reproducible (incluida la consistencia k=2).
2. **Experimentos del proyecto**: tablas y gráficas de exactitud y tiempos vs `n`
   y `k`, speedups frente a fuerza bruta, patrones en las k-particiones óptimas.
3. **Manuales** (entregables): Manual Técnico (incorporar la formulación y el
   análisis de complejidad) y Manual de Usuario + video tutorial.
4. **Comparación con referencias**: PyPhi no importa en Python 3.12; decidir si se
   arregla el entorno o se documenta el uso de BruteForce como referencia exacta.
5. **(Opcional) Harness batch**: exponer el parámetro `k` también en
   `src/networks/run_tests.py` (hoy KQNodes se ejecuta vía `exec.py`/`main.py`).
6. **(Opcional) k-MIP global**: actualmente se calcula la k-MIP para un k dado;
   podría añadirse el barrido k ∈ {2..5} y un criterio de selección.

> **Hecho desde la versión previa de este documento** (antes en esta lista):
> el *punto de entrada de usuario* ya está integrado (CLI en `exec.py` +
> `iniciar_kqnodes`), junto con la persistencia de TPM en `.npy` y la
> visualización del hipercubo. Ver §2.10.

---

## 5. Mejoras posibles

### Algorítmicas
- **Cerrar el residual ~3 % de QNodes (k=2)**: multi-arranque de Queyranne,
  híbrido con búsqueda exhaustiva en n pequeño, o branch-and-bound.
- **Bajar el costo del exacto k**: explotar más estructura de `c_i(Q)` (¿monotonía,
  programación dinámica sobre presentes, poda con cotas?) para superar la barrera
  exponencial en presentes. *(Parcial — Fase 3: la marginalización "condicionar-
  primero" abarató cada `c_i(Q)`; la barrera de enumeración Stirling sigue.)*
- **Recocido más fino**: esquema de enfriamiento adaptativo, vecindarios
  adicionales (mover bloques completos), *warm start* incremental entre k y k+1.
  El recálculo **incremental** de δ **ya está hecho** (Fase 3, `_RecocidoIncremental`).
- **Multi-arranque eficiente**: hoy es caro por fallos de caché; precalcular
  `c_i(Q)` para los subconjuntos relevantes reduciría ese costo.

### Ingeniería
- **Vectorizar/cachear marginales** (`NCube.marginalizar`) y usar `float64` donde
  importe la precisión. *(Parcial — Fase 3: `_marginal_futuro` hace slice antes de
  promediar, reduciendo el arreglo a `2^{n-|Q|}`.)*
- **`delta()` adaptativo en el recocido**: usar la versión vectorizada con numpy
  solo cuando `m·k` supere un umbral (para N muy grande), Python puro por debajo.
- **Paralelizar** el método exacto (las particiones de presentes son independientes)
  y el multi-arranque del recocido.
- **Arreglar de raíz el `SafeLogger`** (handlers/colorama) en vez de neutralizarlo
  en los tests.
- **Reproducibilidad**: fijar semillas y versionar los datasets generados.

---

## 6. Técnicas y estrategias utilizadas

- **Algoritmo de Queyranne** — minimización de funciones submodulares simétricas
  (la base de QNodes para biparticiones).
- **Verificación de submodularidad y simetría** — pruebas empíricas exhaustivas
  sobre subconjuntos para explicar el comportamiento.
- **EMD-efecto factorizada** — solución analítica de la Earth Mover's Distance
  para repertorios condicionalmente independientes (`Σ|u−v|`).
- **Pruebas de regresión tipo *golden master*** — congelar el comportamiento
  actual para detectar cambios no intencionados.
- **Referencia exacta por fuerza bruta** — enumeración completa como verdad de
  terreno para validar.
- **Descomposición / separabilidad del problema** — reducir el espacio de
  búsqueda de `S(2N,k)` a las particiones de los presentes.
- **Enumeración combinatoria** — particiones de conjunto (números de Stirling de
  2.ª especie).
- **Asignación voraz** — cada futuro elige su mejor bloque (óptimo dado el
  particionado de presentes).
- **Búsqueda local (ascenso/hill climbing)** — heurística voraz.
- **Recocido simulado** — metaheurística con aceptación probabilística y
  enfriamiento.
- **Warm start y multi-arranque** — inicializar la metaheurística desde una buena
  solución / reiniciar para robustez.
- **Memoización/caché** — de costos `c_i(Q)` y de marginalizaciones.
- **Benchmarking y profiling** — caracterización empírica de los límites prácticos.

---

## 7. Archivos creados / modificados

### Creados
- `tests/baseline.py`, `tests/baseline_qnodes.csv` — baseline de regresión de QNodes.
- `tests/fase0_kloss.py` — definición de δ_k, `kbruteforce`, validación de separabilidad.
- `tests/fase0_separabilidad.py` — algoritmo exacto por separabilidad y su validación.
- `tests/fase1_kqnodes.py` — validación de la clase KQNodes.
- `tests/fase1_bench.py` — benchmark de los límites del método exacto.
- `tests/fase2_heuristica.py` — validación de voraz y recocido.
- `src/strategies/KQNodes.py` — **clase `KQNodes`** (exacto + heurísticas).
- `src/funcs/visualize.py` — visualización del hipercubo por k-partición (2D/3D/PCA).
- `.docs/.strategies/kqnodes/formulacion.md` — formulación matemática.
- `.docs/RESUMEN_PROGRESO_KQNODES.md` — este documento.

### Modificados
- `src/strategies/KQNodes.py` — **optimización de rendimiento (Fase 3)**:
  `_marginal_futuro` condicionar-primero, `_delta_grupos` escalar, grupos
  `frozenset`, recocido incremental (`_RecocidoIncremental`), umbral dependiente de
  k (`UMBRAL_EXACTO_POR_K`/`_umbral_exacto`), alias del spec `find_k_mip` y
  `marginal_distributions`.
- `src/strategies/q_nodes.py` — corrección del bug de Queyranne + limpieza.
- `src/models/core/system.py` — eliminación de no-op en `bipartir`.
- `src/middlewares/slogger.py` — corrección de `RecursionError` y fuga de FDs.
- `src/constants/models.py` — etiquetas `KQNODES_*`.
- `src/constants/base.py` — constante `NPY_EXTENSION`.
- `src/funcs/format.py` — `fmt_kparticion`.
- `src/controllers/manager.py` — **persistencia de TPM en `.npy`** (`np.save`/
  `np.load`) con fallback al CSV heredado; propiedades `npy_filename`/`csv_filename`.
- `src/main.py` — función `iniciar_kqnodes`; demos a N22 (QNodes) y N25 (KQNodes).
- `exec.py` — **CLI** con `argparse` (`--estrategia`, `--k`, `--metodo`, `--viz`).

---

## 8. Cómo ejecutar (desde `QNodes/`)

```bash
# CLI de usuario (exec.py)
.venv/bin/python exec.py                                   # QNodes bipartición (default)
.venv/bin/python exec.py --estrategia kqnodes --k 3        # k-MIP, despacho auto
.venv/bin/python exec.py --estrategia kqnodes --k 4 --metodo recocido
.venv/bin/python exec.py --estrategia kqnodes --k 3 --viz particion.png

# Baseline de QNodes (regresión)
PYTHONPATH=. .venv/bin/python tests/baseline.py            # verifica
PYTHONPATH=. .venv/bin/python tests/baseline.py --update   # regenera

# Validaciones de KQNodes
PYTHONPATH=. .venv/bin/python tests/fase0_kloss.py
PYTHONPATH=. .venv/bin/python tests/fase0_separabilidad.py
PYTHONPATH=. .venv/bin/python tests/fase1_kqnodes.py
PYTHONPATH=. .venv/bin/python tests/fase1_bench.py
PYTHONPATH=. .venv/bin/python tests/fase2_heuristica.py
```

Uso programático:

```python
from src.controllers.manager import Manager
from src.strategies.KQNodes import KQNodes

estado, cond, alcance, mecanismo, k = "1000", "1111", "1111", "1111", 3
tpm = Manager(estado).cargar_red()
sol = KQNodes(tpm).aplicar_estrategia(estado, cond, alcance, mecanismo, k, metodo="auto")
print(sol)   # k-partición, δ_k y distribución
```
