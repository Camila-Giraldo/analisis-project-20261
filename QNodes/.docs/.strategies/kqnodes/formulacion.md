# KQNodes — Formulación matemática de la k-Partición de Mínima Información

> Extensión de QNodes (biparticiones) al caso general de **k-particiones**
> (k ≥ 2). Este documento define la pérdida δₖ, demuestra la propiedad de
> **separabilidad** de la EMD-efecto y deriva a partir de ella un **algoritmo
> exacto** cuyo costo depende de los nodos *presente* y no del total de vértices.
>
> Resultados validados empíricamente en `tests/fase0_kloss.py` y
> `tests/fase0_separabilidad.py`.

---

## 1. Notación y preparación del subsistema

Tras la preparación del subsistema (condiciones de fondo + substracción), el
subsistema queda descrito por dos conjuntos de variables binarias en dos tiempos:

- **Nodos futuro** (alcance / *purview*, en t+1): `F = {1, …, m}`. Cada nodo
  futuro `i ∈ F` está representado por un n-cubo (su repertorio-efecto
  condicional).
- **Nodos presente** (mecanismo, en t): `P = {1, …, n}`.

El **conjunto base** sobre el que se particiona es el de los vértices bitemporales

$$ V = \{(1, i) : i \in F\} \;\cup\; \{(0, j) : j \in P\}, \qquad |V| = m + n .$$

donde el primer componente es el tiempo (`1` futuro/efecto, `0` presente). Para el
sistema completo `m = n = N`, de modo que `|V| = 2N`.

### 1.1 Marginal-efecto y función de coste por nodo

Para un nodo futuro `i` y un conjunto de presentes `Q ⊆ P`, sea

$$ m_i(Q) = \text{marginal del cubo } i \text{ conservando } Q \text{ y marginalizando } P\setminus Q, $$

evaluada en el estado inicial (esto es lo que produce `NCube.marginalizar`
seguido de la selección de estado en `System.distribucion_marginal`). El
**marginal del subsistema** es `μ_i = m_i(P)` (se conserva todo el presente).

Definimos la **función de coste del nodo futuro** como

$$ c_i(Q) \;=\; \bigl|\, m_i(Q) - \mu_i \,\bigr| \;\ge\; 0 . $$

Es la pérdida que aporta el nodo futuro `i` cuando los presentes de su bloque son
exactamente `Q`. Nótese que `c_i(P) = 0` (sin corte, sin pérdida).

### 1.2 La EMD-efecto es factorizada

QNodes usa la solución analítica de la *Earth Mover's Distance* en el efecto, que
para repertorios condicionalmente independientes equivale a la EMD entre los
productos de marginales:

$$ \mathrm{EMD}(a, b) \;=\; \sum_{i \in F} \lvert a_i - b_i \rvert . $$

Esta forma factorizada (suma de diferencias absolutas por nodo) es la clave de
toda la formulación que sigue.

---

## 2. Bipartición (k = 2) — repaso

Una bipartición está dada por un bloque `S ⊆ V` y su complemento `V \ S`. El nodo
futuro `i ∈ S` ve los presentes `Q_i = S ∩ P`; el nodo futuro `i ∈ V\S` ve
`Q_i = P \ (S ∩ P)`. La pérdida es

$$ \varphi(S) \;=\; \sum_{i \in F} c_i(Q_i). $$

Esto es exactamente lo que computa `System.bipartir(alcance, mecanismo)` seguido
de `distribucion_marginal` y `emd_efecto`. **Verificado:** la formulación
generalizada con k=2 coincide con `bipartir` con diferencia 0.0 sobre todas las
biparticiones de N3 y N4.

---

## 3. k-Partición — definición

Una **k-partición** de `V` es una colección `{B₁, …, B_k}` de bloques disjuntos,
no vacíos, cuya unión es `V`. Cada bloque `B_t` induce:

- un **conjunto de presentes** `A_t = B_t ∩ P`; los `{A₁, …, A_k}` forman una
  partición de `P` en `k` partes (algunas posiblemente vacías);
- un conjunto de futuros, donde cada futuro `i` pertenece a un único bloque
  `t(i)`.

La regla de corte generaliza la bipartición: **cada nodo futuro conserva
únicamente los presentes de su propio bloque** y margina el resto. La pérdida es

$$ \boxed{\; \delta_k(\{B_t\}) \;=\; \sum_{i \in F} c_i\!\bigl(A_{t(i)}\bigr). \;} $$

Para `k = 2` se reduce a `φ(S)`. **Verificado:** `δₖ` crece monótonamente con `k`
(más bloques ⇒ más pérdida).

---

## 4. Propiedad de separabilidad

**Proposición (separabilidad).** `δ_k` depende únicamente de la aplicación que a
cada nodo futuro le asigna el conjunto de presentes de su bloque,
`i ↦ A_{t(i)}`. En consecuencia:

1. **Independencia de los futuros.** Para una partición de los presentes
   `{A₁,…,A_k}` fija, `δ_k` se minimiza asignando cada futuro de forma
   independiente: `t(i) = argmin_t c_i(A_t)` (sujeto a la restricción de bloques
   no vacíos del §5).
2. **Irrelevancias.** `δ_k` es invariante a:
   - reagrupar nodos futuros entre bloques que comparten el mismo conjunto de
     presentes (en particular, bloques sin presentes, `A_t = ∅`);
   - reagrupar nodos presente que no acompañan a ningún futuro.

**Demostración.** Inmediata de la forma factorizada: el término `i`-ésimo de
`δ_k` es `c_i(A_{t(i)})`, que depende solo de `A_{t(i)}`. La regla de corte hace
que el marginal particionado del futuro `i` sea `m_i(A_{t(i)})`,
independientemente de qué otros futuros estén en su bloque y de los presentes
fuera de él. La suma de términos no acopla a los futuros entre sí. ∎

**Verificación empírica.** Reagrupar futuros sin presentes (juntos vs separados)
o presentes sin futuro (juntos vs separados) deja `δ_k` idéntico (ver
`check_separabilidad` en `tests/fase0_kloss.py`).

---

## 5. Algoritmo exacto por separabilidad

La separabilidad reduce el problema de minimizar sobre las `S(|V|, k)`
particiones de los vértices al de **particionar solo los presentes** y resolver
los futuros de forma voraz.

### 5.1 Reformulación

Buscar el mínimo de `δ_k` equivale a elegir:

1. una partición de `P` en `r` grupos no vacíos `G₁, …, G_r`, con
   `1 ≤ r ≤ min(n, k)`;
2. los `k − r` bloques restantes son **vacíos de presente** (`A_t = ∅`) y cada
   uno debe recibir al menos un futuro;
3. una asignación de cada futuro a uno de los `k` bloques.

### 5.2 Asignación óptima de los futuros (partición de presentes fija)

Para cada futuro `i`:

- `mejor_grupo_i = min_g c_i(G_g)` (mejor entre los grupos con presentes),
- `c∅_i = c_i(∅)` (coste si se va a un bloque vacío).

Sin restricciones, `i` elige `min(mejor_grupo_i, c∅_i)`. La única restricción es
que cada uno de los `k − r` bloques vacíos reciba ≥ 1 futuro. Como todos los
bloques vacíos son idénticos (`A_t = ∅`), basta colocar `≥ (k − r)` futuros en
`∅`. Si menos de `k − r` los eligieron de forma natural, se **fuerzan** los más
baratos, ordenando por el sobrecosto `Δ_i = c∅_i − mejor_grupo_i`. Los grupos con
presentes nunca necesitan futuros (ya son no vacíos). Factible si `m ≥ k − r`.

### 5.3 Pseudocódigo

```
ENTRADA: subsistema (cubos futuro, presentes P), k
SALIDA: mínimo δ_k

mejor ← +∞
para r = 1 … k:
    si r > |P|: terminar
    vacios ← k − r
    si vacios > m: continuar              # futuros insuficientes
    para cada partición {G_1..G_r} de P en r grupos:
        total ← 0 ; deltas ← [] ; en_vacio ← 0
        para cada futuro i:
            mejor_grupo ← min_g c_i(G_g)
            si vacios > 0:
                c0 ← c_i(∅)
                si c0 ≤ mejor_grupo:  total += c0 ; en_vacio += 1
                si no:                total += mejor_grupo ; deltas.add(c0 − mejor_grupo)
            si no:
                total += mejor_grupo
        si vacios > 0 y en_vacio < vacios:
            faltan ← vacios − en_vacio
            si |deltas| < faltan: continuar
            ordenar deltas asc ; total += suma(deltas[0:faltan])
        mejor ← min(mejor, total)
devolver mejor
```

`c_i(Q)` se memoiza por `(i, Q)`.

### 5.4 Correctitud

Toda k-partición de `V` se proyecta en (partición de presentes en
`r = #bloques-con-presente` grupos) + (asignación de futuros), y recíprocamente
toda configuración del §5.1 es una k-partición válida. El algoritmo minimiza
exactamente sobre ese conjunto.

**Verificado:** coincide con la fuerza bruta `kbruteforce` (que enumera
`S(|V|, k)`) hasta la precisión de `float32` (diferencia máxima **1.5 × 10⁻⁸**)
en N3, N4 y N5 para `k = 2 … 6`.

---

## 6. Complejidad y resultados

- **Candidatos evaluados:** `Σ_{r=1}^{k} S(n, r)`, función *únicamente* del número
  de nodos **presente** `n` (Stirling de 2.ª especie), frente a los `S(m+n, k)`
  de la fuerza bruta. Los futuros se resuelven de forma voraz en `O(m · r)` por
  candidato (con `c_i` memoizado).
- **Reducción medida** (mismo mínimo exacto):

  | Caso        | Separabilidad | Fuerza bruta | Reducción |
  |-------------|--------------:|-------------:|----------:|
  | N5, k = 3   |            41 |        9 330 |    ~230×  |
  | N5, k = 4   |            51 |       34 105 |    ~670×  |
  | N5, k = 5   |            52 |       42 525 |    ~800×  |

- **Naturaleza asintótica:** exponencial solo en el número de presentes `n`
  (aprox. `k^n / k!` para `k` fija), polinomial en los futuros `m`. Pasa de
  `k^{2N}` a `k^{N}` en el caso completo: la *raíz cuadrada* del espacio de
  búsqueda, manteniendo exactitud.

---

## 7. Alcance y límites

- Para `N` pequeño/moderado, el algoritmo por separabilidad calcula la **k-MIP
  exacta global** mucho más rápido que la enumeración completa, y sirve como
  **referencia exacta** para validar variantes heurísticas.
- Para `N` grande (20+), sigue siendo exponencial en los presentes; ahí se
  recurre a heurísticas (**Queyranne voraz** por bisección recursiva,
  **recocido simulado**, *warm start*), que pueden operar sobre el espacio
  reducido de particiones de presentes.
- La pérdida usada es la EMD-efecto factorizada (consistente con QNodes). La
  exactitud de la **estrategia** QNodes/Queyranne tiene un residual ~3% por la
  no-submodularidad de esta misma `f`; por ello la validación de KQNodes se hace
  contra `kbruteforce` (exacto), no contra QNodes.

---

## 8. Heurísticas para n grande (Fase 2)

Cuando el número de presentes supera el umbral práctico del método exacto
(≈ 10-11), KQNodes recurre a heurísticas que operan sobre el **espacio reducido
de etiquetaciones de los presentes** (cada presente → uno de k bloques). La
rutina común `_evaluar_grupos` resuelve los futuros de forma óptima para cada
etiquetación, de modo que toda la búsqueda heurística es sobre los presentes.

- **Voraz** (`find_kmip_voraz`): arranque factible + ascenso por mínimos locales
  (mueve cada presente al bloque que más reduce δ_k hasta no mejorar).
  Determinista, muy rápida, pero se atasca en óptimos locales (~10/20 óptimos en
  las pruebas).
- **Recocido simulado** (`find_kmip_recocido`): metaheurística con **warm start**
  desde la solución voraz; vecindario de reetiquetar/intercambiar presentes;
  acepta empeoramientos con probabilidad `exp(-Δ/T)` y enfría `T`. Opción de
  **multi-arranque** (`reinicios`) para más robustez a costa de tiempo.
- **Selección automática** (`metodo="auto"`): exacto si `n_presentes ≤
  UMBRAL_EXACTO` (=11), recocido en caso contrario.

**Resultados (Fase 2):** el recocido alcanzó **19/20** óptimos exactos (gap máx
0.007) frente a 10/20 del voraz, y resuelve subsistemas de 14 presentes en ~1.5 s
(donde el exacto sería de minutos a horas). Detalles del límite en
`tests/fase1_bench.py`; validación en `tests/fase2_heuristica.py`.

## 9. Referencias en el código

- `src/models/core/system.py` — `bipartir`, `distribucion_marginal` (caso k=2).
- `src/models/core/ncube.py` — `marginalizar` (cálculo de `m_i(Q)`).
- `src/funcs/iit.py` — `emd_efecto` (EMD-efecto factorizada).
- `tests/fase0_kloss.py` — definición de `δ_k`, `kbruteforce`, verificación de
  reducción k=2 y de separabilidad.
- `tests/fase0_separabilidad.py` — algoritmo exacto por separabilidad y su
  validación.
