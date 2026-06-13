# Extensión de GeoMIP a k > 2 particiones

## Contexto

La implementación original de `GeometricSIA` calculaba únicamente **biparticiones** (k = 2): dividía el subsistema en exactamente dos grupos de variables y medía la pérdida de información (EMD-Effect) al cortar las conexiones entre ellos. Este documento describe los cambios realizados para soportar **k-particiones arbitrarias** (k ≥ 2) dentro del mismo framework EMD-based, junto con las optimizaciones de rendimiento introducidas simultáneamente.

---

## 1. Archivos modificados

| Archivo | Tipo de cambio |
|---------|---------------|
| `src/controllers/strategies/geometric.py` | Refactorización + extensión k > 2 |
| `src/models/core/system.py` | Nuevo método `bipartir_k` |
| `src/main.py` | Parámetro `k` en el flujo Excel |
| `exec.py` | Protección `if __name__ == '__main__'` |

---

## 2. Cambios en `geometric.py`

### 2.1 Parámetro `k`

```python
# Antes
class GeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        ...

# Después
class GeometricSIA(SIA):
    def __init__(self, gestor: Manager, k: int = 2):
        self.k = k
        ...
```

`aplicar_estrategia` también acepta `k` como parámetro opcional, con prioridad sobre el valor del constructor. El default `k=2` garantiza compatibilidad total con el código existente.

---

### 2.2 Optimizaciones de la tabla de transiciones

La implementación original usaba estructuras lentas para almacenar los costos de transición `t_x(i, j)`:

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Clave del diccionario** | `(tuple(estado_i), tuple(estado_j))` — hash de dos tuplas de longitud n | `j_int: int` — un solo entero |
| **Valor almacenado** | `list[float]` — lista Python | `np.ndarray` — array numpy |
| **Conversión estado → índice** | `int("".join(map(str, estado[::-1])), 2)` — concatenación de strings | `int(np.dot(estado, self._powers))` — producto punto vectorizado |
| **Lookup de vecinos** | Reconstruir lista + conversión a string | `k_int = j_int ^ (1 << bit)` — operación XOR |
| **Lookup de probabilidades** | `[flat[idx] for flat in self._flat_data]` — loop Python | `self._flat_matrix[:, j_int]` — indexación matricial |
| **Conjunto de visitados** | `set[tuple]` — hash de tupla de n enteros | `set[int]` — hash de un entero |

#### Precomputación de potencias de 2

```python
self._powers = 2 ** np.arange(n)   # [1, 2, 4, 8, ..., 2^(n-1)]
self._i0_int  = int(self.estado_inicial @ self._powers)
self._i_final_int = int(self.estado_final  @ self._powers)
```

`estado[i]` contribuye `2^i` al índice entero. Esto hace que `estado[0]` sea el bit menos significativo, preservando la convención little-endian original.

#### Matriz de probabilidades aplanada

```python
self._flat_matrix = np.stack(
    [ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos]
)  # shape: (n_vars, 2^n_present)
```

Permite calcular `|X[i0] - X[j]|` para todas las variables en una sola operación:

```python
raw = np.abs(self._flat_matrix[:, self._i0_int] - self._flat_matrix[:, j_int])
```

#### XOR para lookup de vecinos

Para sumar los costos de los vecinos de `j` en el camino hacia `i0`, la implementación original reconstruía la lista del estado y la convertía a string. La versión optimizada usa XOR a nivel de bits:

```python
# El vecino de j al que se le flipea el bit `bit` está en:
k_int = j_int ^ (1 << bit)
raw = raw + self.tabla_np[k_int]
```

`1 << bit` crea una máscara con un único bit activo en posición `bit`. XOR con `j_int` invierte exactamente ese bit, obteniendo el estado vecino en O(1).

---

### 2.3 Separación del flujo en dos caminos

```python
def find_mip(self, k: int = 2):
    # ... construcción de la tabla ...
    if k == 2:
        return self._find_mip_k2()   # bipartición clásica
    else:
        return self._find_mip_kn(k)  # k-partición generalizada
```

El camino k = 2 es funcionalmente idéntico al original pero usa las nuevas estructuras optimizadas. El camino k > 2 es completamente nuevo.

---

### 2.4 Generación de candidatos para k > 2 — `_candidatos_k`

Para k > 2, generar todos los posibles agrupamientos de n variables en k grupos es computacionalmente inviable (número de Stirling). En su lugar se generan hasta **150 candidatos** mediante tres estrategias complementarias:

#### Estrategia 1 — Cortes en saltos naturales de costo

```
costos = tabla_np[i_final_int]   # costo t_x(i0, i_final) por variable
orden  = argsort(costos)         # variables ordenadas de menor a mayor costo
gaps   = diff(costos_ordenados)  # saltos entre costos consecutivos
```

Se identifican los k-1 mayores saltos en el vector de costos ordenado y se colocan los puntos de corte justo ahí. Variables con costo similar quedan en el mismo grupo; variables separadas por un salto grande quedan en grupos distintos.

```
costos ordenados: [0.1, 0.15, 0.18, | 0.50, 0.52, | 0.90]
                                     ^             ^
                               corte 1 (gap 0.32)  corte 2 (gap 0.38)
→ grupos: {var_0, var_2, var_4} | {var_1, var_3} | {var_5}
```

#### Estrategia 2 — Variaciones ±1, ±2

Por cada punto de corte natural se generan variaciones desplazando el corte ±1 y ±2 posiciones. Esto explora el vecindario local de la solución natural sin costo exponencial.

#### Estrategia 3 — División equitativa

Un candidato adicional que divide las variables en k grupos de tamaño aproximadamente igual, como baseline.

#### Estrategia 4 — Candidatos desde niveles intermedios del camino

Los estados en `caminos[nivel]` representan estados a distancia Hamming `nivel` desde el estado inicial. Las variables que "cambiaron" hacia el estado final en ese nivel tienden a moverse juntas, lo que sugiere afinidad causal. Se usan para construir candidatos adicionales.

---

### 2.5 Asignación de mecanismos — `_asignar_mecanismos`

Este método resuelve el problema de cómo asignar variables **presentes** (mecanismo, t) a cada grupo de variables **futuras** (alcance, t+1) cuando alcance ≠ mecanismo.

**Regla principal:** la variable presente `p` pertenece al mecanismo del grupo i si y solo si su índice global coincide con alguna variable futura del grupo i. Esto refleja la correspondencia natural en IIT: la variable A en t influye primariamente sobre la variable A en t+1.

```python
mec_match = pres_set & set(alc)   # intersección de índices globales
```

**Variables sin par:** las variables presentes cuyo índice global no existe en ningún grupo de futuras (porque estaban en mecanismo pero no en alcance) se distribuyen en round-robin empezando por el grupo de mayor tamaño:

```python
sin_asignar = sorted(pres_set - asignadas)
orden = sorted(range(len(grupos)), key=lambda i: -len(grupos[i][0]))
for j, p in enumerate(sin_asignar):
    grupos[orden[j % len(orden)]][1].append(p)
```

Esto garantiza que **todas las variables presentes del subsistema queden asignadas** a exactamente un grupo, haciendo que el mecanismo total sea una partición completa del conjunto de variables presentes.

---

### 2.6 Evaluación de candidatos — `_evaluar_particion_k`

```python
def _evaluar_particion_k(self, grupos_locales):
    particion_k = self._asignar_mecanismos(grupos_locales)
    dist = self.sia_subsistema.bipartir_k(particion_k).distribucion_marginal()
    emd_val = emd_efecto(dist, self.sia_dists_marginales)
    return emd_val, dist
```

Para cada candidato se llama a `bipartir_k` (ver sección 3), se obtiene la distribución marginal del sistema particionado y se calcula la EMD-Effect contra la distribución del subsistema original. Se selecciona el candidato con **menor pérdida** (MIP generalizado).

---

### 2.7 Formato de salida — `_fmt_particion_k`

El resultado se formatea igual que la bipartición k = 2, extendido a k columnas:

```
| A,E,F || B,C,D || G,H,I,J |
| a,e,f || b,c,d || g,h,i,j |
```

- Fila superior: variables futuras (t+1) en mayúscula.
- Fila inferior: variables presentes (t) en minúscula, siguiendo la misma asignación de mecanismos.
- `∅` cuando un grupo no tiene variables presentes asignadas (variable futura sin par en el mecanismo).

---

## 3. Cambios en `system.py` — `bipartir_k`

### Motivación

El método `bipartir` existente solo soporta dos partes: variables en alcance (conservan mecanismo) y variables fuera de alcance (pierden mecanismo). Para k > 2 partes se necesita que cada grupo conserve solo las conexiones de su propio mecanismo.

### Implementación

```python
def bipartir_k(self, particion_k: list[tuple]) -> "System":
    # Mapa: índice global de variable futura → su mecanismo
    fut_to_mec: dict[int, NDArray] = {}
    for alcance_i, mecanismo_i in particion_k:
        for f in alcance_i:
            fut_to_mec[int(f)] = mecanismo_i

    new_sys = System.__new__(System)
    new_sys.estado_inicial = self.estado_inicial
    new_sys.ncubos = tuple(
        cube.marginalizar(np.setdiff1d(cube.dims, fut_to_mec[int(cube.indice)]))
        if int(cube.indice) in fut_to_mec
        else cube.marginalizar(cube.dims)
        for cube in self.ncubos
    )
    return new_sys
```

**Lógica por ncubo:**

- Si el ncubo pertenece al alcance de algún grupo i → marginaliza sobre `cube.dims - mecanismo_i`. Conserva solo las dimensiones del mecanismo de su grupo; las conexiones con otros grupos quedan cortadas.
- Si el ncubo no pertenece a ningún grupo (no debería ocurrir si la partición es completa) → marginaliza sobre todas sus dimensiones, volviéndolo constante.

### Comparación con `bipartir` (k = 2)

```python
# bipartir (k=2): un mecanismo global para todos los del alcance
cube.marginalizar(np.setdiff1d(cube.dims, mecanismo))   # si en alcance
cube.marginalizar(mecanismo)                             # si fuera de alcance

# bipartir_k (k≥2): mecanismo específico por grupo
cube.marginalizar(np.setdiff1d(cube.dims, fut_to_mec[cube.indice]))  # si en alcance_i
cube.marginalizar(cube.dims)                             # si sin asignar
```

---

## 4. Cambios en `main.py`

### Parámetro `k` en el flujo Excel

```python
# Antes
def ejecutar_con_tiempo(config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm):
    analizador_fi = GeometricSIA(config_sistema)
    ...

def ejecutar_desde_excel(ruta_excel, ruta_salida, inicio=0, cantidad=50, ...):
    ...
    proceso = multiprocessing.Process(target=ejecutar_con_tiempo, args=(..., tpm))
```

```python
# Después
def ejecutar_con_tiempo(..., tpm, k=2):
    analizador_fi = GeometricSIA(config_sistema, k=k)
    ...

def ejecutar_desde_excel(..., k=2):
    ...
    proceso = multiprocessing.Process(target=ejecutar_con_tiempo, args=(..., tpm, k))
```

El Excel de salida incluye una columna `k` para identificar con qué número de particiones se ejecutó cada fila.

### Lectura del estado inicial desde Excel (celda E2)

```python
df_header = pd.read_excel(ruta_excel, sheet_name=1, header=None, usecols="E", nrows=2)
raw = df_header.iloc[1, 0]
estado_inicio_excel = str(int(raw)).strip() if not pd.isna(raw) else None
```

La conversión `int(raw)` elimina el `.0` que pandas agrega al leer números enteros almacenados como float en Excel (e.g., `1000000000.0` → `"1000000000"`).

---

## 5. Cambios en `exec.py`

En Windows, `multiprocessing` usa el método `spawn` para crear procesos hijos, lo que implica reimportar el módulo principal. Si `iniciar()` está al nivel de módulo (fuera de `if __name__ == '__main__'`), el proceso hijo lo ejecuta nuevamente al importar, causando un `RuntimeError`.

```python
# Antes
iniciar()

if __name__ == "__main__":
    main()

# Después
if __name__ == "__main__":
    iniciar()
    main()
```

---

## 6. Resumen de estructuras de datos

| Estructura | Descripción | Complejidad |
|-----------|-------------|-------------|
| `tabla_np: dict[int, np.ndarray]` | Tabla de costos `t_x(i0, j)`. Clave = representación entera del estado j. Valor = array de costos por variable. | Escritura O(1) amortizado; lookup O(1) |
| `_flat_matrix: np.ndarray` shape `(n_vars, 2^n_pres)` | Datos aplanados de todos los ncubos. Permite lookup vectorizado de probabilidades por estado. | Construcción O(n · 2^n); lookup O(1) |
| `_powers: np.ndarray` | Potencias de 2 `[1, 2, 4, ..., 2^(n-1)]`. Convierte estado binario a entero mediante producto punto. | Construcción O(n) |
| `caminos: dict[int, list[list[int]]]` | Estados por nivel de distancia Hamming desde i0. Reutilizados para generación de candidatos k > 2. | O(C(n, nivel)) por nivel |
| `fut_to_mec: dict[int, NDArray]` | Mapa variable futura → mecanismo de su grupo. Construido en `bipartir_k`. | O(n_fut) |

---

## 7. Algoritmo completo k > 2 (flujo)

```
aplicar_estrategia(condicion, alcance, mecanismo, tpm, k)
│
├── sia_preparar_subsistema(...)       ← conditioning + substraction
├── _flat_matrix = stack(ncubos)       ← matriz de probabilidades
│
└── find_mip(k)
    ├── _powers, _i0_int, _i_final_int ← precomputación
    ├── tabla_np[i0_int] = zeros       ← caso base
    │
    ├── para nivel = 1..n:
    │   └── _calcular_costos_nivel()
    │       ├── generar estados a distancia `nivel` de i0
    │       └── _calcular_costo(j, j_int, d)
    │           ├── raw = |flat_matrix[:,i0] - flat_matrix[:,j]|  ← vectorizado
    │           ├── para cada bit donde j != i0:
    │           │   └── raw += tabla_np[j_int ^ (1<<bit)]         ← XOR lookup
    │           └── tabla_np[j_int] = (0.5^d) * raw
    │
    └── _find_mip_kn(k)
        ├── _candidatos_k(k)
        │   ├── cortes en gaps naturales del vector de costos
        │   ├── variaciones ±1, ±2 de los cortes
        │   ├── división equitativa
        │   └── candidatos desde caminos intermedios
        │
        ├── para cada candidato:
        │   └── _evaluar_particion_k(grupos)
        │       ├── _asignar_mecanismos(grupos)
        │       │   ├── mec_i = pres_set ∩ alc_i  (intersección de índices)
        │       │   └── sobrantes → round-robin al grupo mayor
        │       ├── bipartir_k(particion_k)
        │       └── emd_efecto(dist_particion, dist_subsistema)
        │
        └── retornar grupos con menor EMD  ← MIP generalizado
```

---

## 8. Cómo usar

### Desde código

```python
from pathlib import Path
from src.main import ejecutar_desde_excel

ejecutar_desde_excel(
    ruta_excel=Path("results/Pruebas_iniciales_Metodo2.xlsx"),
    ruta_salida=Path("results/resultados_k3.xlsx"),
    k=3,          # número de particiones (2 = comportamiento original)
    inicio=0,
    cantidad=50,
)
```

### Desde `exec.py`

Editar la llamada en `iniciar()` dentro de `src/main.py`:

```python
def iniciar():
    ruta_entrada = Path(...)
    ruta_salida  = Path(...)
    ejecutar_desde_excel(ruta_entrada, ruta_salida, k=3)
```

### Formato del Excel de entrada

- **Hoja 2** (índice 1)
- **Celda E2**: estado inicial en binario (e.g., `1000000000`)
- **Columna B** desde fila 3: especificaciones de subsistema con formato `"ABC t+1|BCD t"`

### Formato del Excel de salida

| Columna | Descripción |
|---------|-------------|
| Iteración | Número de fila procesada |
| k | Número de particiones usado |
| Alcance | String binario de variables futuras activas |
| Mecanismo | String binario de variables presentes activas |
| Partición | Resultado en formato `\| G1 \|\| G2 \|\| ... \|` |
| Pérdida | EMD-Effect de la partición mínima (φ) |
| Tiempo de ejecución (s) | Segundos de cómputo |
