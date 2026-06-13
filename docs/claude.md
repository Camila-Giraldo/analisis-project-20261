# CLAUDE.md — Reglas del Proyecto K-QGMIP

## Contexto del Proyecto

Este proyecto extiende las estrategias de bi-partición **GeoMIP** y **QNodes** al caso general de **k-particiones (k ∈ {2, 3, 4, 5})** para encontrar la Partición de Mínima Información (k-MIP) en el marco de la Teoría de la Información Integrada (IIT). Las nuevas implementaciones se denominan **KGeoMIP** y **KQNodes**.

El objetivo es identificar la k-partición de un sistema V con n variables binarias que minimice la Earth Mover's Distance (EMD) entre la distribución original del sistema y el producto tensorial de las distribuciones marginales de cada parte.

---

## Convenciones de Nomenclatura

| Estrategia               | Nombre de clase / repositorio |
|--------------------------|-------------------------------|
| GeoMIP → k-particiones   | `KGeoMIP`                     |
| QNodes → k-particiones   | `KQNodes`                     |

- La `K` inicial identifica explícitamente que son extensiones a k-particiones.
- Aplicar este nombre de forma consistente en: clase principal, repositorio Git, carpeta raíz del proyecto, y referencias en documentación.

---

## Arquitectura y Estructura del Código

### Ubicación de archivos

```
src/
  controllers/
    strategies/
      KGeoMIP.py       ← implementación de KGeoMIP (hereda de SIA)
      KQNodes.py       ← implementación de KQNodes (hereda de SIA)
  models/              ← modelos de datos (N-Cubos, particiones, etc.)
  utils/               ← funciones auxiliares compartidas
tests/                 ← tests unitarios y de integración
```

### Herencia obligatoria

Ambas clases **deben heredar de la clase base `SIA`**, siguiendo el patrón de diseño establecido en semestres anteriores:

```python
class KGeoMIP(SIA):
    ...

class KQNodes(SIA):
    ...
```

### Reutilización de componentes existentes

- Reutilizar **siempre** la infraestructura de N-Cubos y la tabla de costos de transiciones de GeoMIP.
- La **tabla de costos de transiciones se calcula una única vez** y se reutiliza para evaluar todas las k-particiones candidatas, independientemente del valor de k.
- Solo modificar estructuras de datos o procedimientos cuando no alteren la fundamentación teórica conceptual de GeoMIP o QNodes.

---

## Implementación Requerida

### Métodos mínimos obligatorios

Cada clase debe implementar como mínimo:

1. **`find_k_mip(k)`** — Encuentra la k-MIP para un valor de k dado (2 ≤ k ≤ 5).
2. **Métodos de evaluación** — Evalúan k-particiones candidatas usando la tabla de costos existente.
3. **`marginal_distributions(partition)`** — Calcula distribuciones marginales de cada parte de la partición.
4. **`tensor_product(distributions)`** — Calcula el producto tensorial de k distribuciones marginales.
5. **`generate_k_partitions(k)`** — Genera o enumera k-particiones candidatas para evaluación.

### Compatibilidad con k=2

La extensión **debe reproducir exactamente los resultados** de GeoMIP original y QNodes original cuando k=2. Esto es un criterio de validación obligatorio.

### Rango de k soportado

- Para k pequeño y sistemas de pocos nodos (3–6 nodos): búsqueda exhaustiva y resultado óptimo garantizado.
- Para sistemas medianos (10–15 nodos) o k mayor: estrategia heurística o aproximada que encuentre soluciones de alta calidad en tiempo razonable.
- Para sistemas grandes (20+ nodos): solo aproximaciones son aceptables.

---

## Documentación del Código

- Todos los métodos públicos deben tener **docstrings** que describan: función, parámetros de entrada/salida y precondiciones importantes.
- Incluir **comentarios en línea** en secciones de código complejo o no obvio.
- Usar **fuente monoespaciada** (Courier New / Consolas) en pseudocódigo del manual técnico con sangrado consistente.
- Documentar el **uso de herramientas de IA generativa** (Claude, Copilot, ChatGPT, etc.) de forma transparente: etapas donde se usó, prompts representativos, partes del código influenciadas, y reflexión crítica. Esto no penaliza la evaluación.

---

## Tests

- Incluir **tests unitarios** para los componentes principales.
- Validar obligatoriamente:
  - Evaluación correcta de k-particiones.
  - Consistencia de resultados con GeoMIP/QNodes originales para k=2.
  - Robustez ante diferentes inputs y casos edge.

---

## Fundamentos Teóricos a Respetar

### Definición de k-partición válida

Una k-partición de V = {v₁, ..., vₙ} en S₁, S₂, ..., Sₖ debe cumplir:
- **Cobertura:** S₁ ∪ S₂ ∪ ... ∪ Sₖ = V
- **Disjunción:** Sᵢ ∩ Sⱼ = ∅ para todo i ≠ j
- **No vacío:** Sᵢ ≠ ∅ para todo i

### Función de pérdida

Minimizar la EMD (Earth Mover's Distance con métrica de Hamming) entre:
- La distribución de probabilidad del sistema original.
- El producto tensorial de las distribuciones marginales de cada parte: P(S₁) ⊗ P(S₂) ⊗ ... ⊗ P(Sₖ).

### Complejidad del espacio de búsqueda

El número de k-particiones está dado por los **números de Stirling del segundo tipo S(n, k)**. Tener presente que este espacio crece exponencialmente; el diseño algorítmico debe justificar explícitamente cómo se aborda esta explosión combinatoria.

### Relación geométrica

- Una k-partición corresponde a dividir el hipercubo n-dimensional con **k-1 hiperplanos**.
- La tabla de costos de GeoMIP (basada en distancia de Hamming y topología del hipercubo) es aplicable directamente al caso k-partito.

---

## Análisis de Complejidad Requerido

El código y la documentación deben incluir análisis explícito de:

- **Complejidad temporal:** expresada en notación asintótica Θ(·) en función de `n` (número de variables) y `k` (número de particiones).
- **Complejidad espacial:** memoria de estructuras permanentes y temporales.
- **Mejor caso / peor caso:** identificar qué configuraciones conducen a cada uno.
- **Comparación:** contrastar con búsqueda exhaustiva y con las estrategias originales de bi-partición.

---

## Datos de Prueba

Los datasets de prueba están en los archivos del proyecto:
- `DatosPruebas2026_1.xlsx` — sistemas de prueba 2026-1.
- `Ejemplos.xlsx` — ejemplos de referencia.

Validar el algoritmo en:
- **Sistemas pequeños (3–6 nodos):** comparar con búsqueda exhaustiva y medir tasa de acierto exacto.
- **Sistemas medianos (10–15 nodos):** medir speedup respecto a fuerza bruta.
- **Sistemas grandes (20+ nodos):** reportar soluciones cuasi-óptimas y tiempos.

---

## Resultados Experimentales Requeridos

- **Tablas** con métricas: tiempo de ejecución, tasa de acierto, error relativo, speedup.
- **Gráficas** de escalabilidad: tiempo vs n, tiempo vs k.
- **Visualización** de k-particiones sobre el hipercubo (especialmente para la presentación final).
- **Comparación** entre KGeoMIP y KQNodes cuando ambas estén implementadas.
- Análisis crítico de casos donde el método encuentra soluciones subóptimas.

---

## Presentación Final

- Debe incluir demostración en vivo del software ejecutando búsqueda de k-MIP en al menos un sistema de prueba.
- Si es posible, visualización de las k-particiones sobre la representación del hipercubo.

---

## Criterios de Evaluación (resumen)

| Dimensión              | Aspectos clave                                                                                  |
|------------------------|------------------------------------------------------------------------------------------------|
| Correctitud            | Resultados correctos en casos de validación; consistencia con GeoMIP/QNodes para k=2; robustez |
| Eficiencia             | Speedup respecto a búsqueda exhaustiva; escalabilidad con n y k                                |
| Calidad del código     | Claridad, organización, docstrings, convenciones de estilo, tests                              |
| Documentación técnica  | Claridad matemática, análisis de complejidad, calidad experimental, reflexión crítica          |
| Presentación           | Comunicación clara, visualizaciones efectivas, demo en vivo, respuesta a preguntas             |

---

## Observaciones Generales

- Mantener **compatibilidad con la arquitectura existente** en todo momento.
- No recalcular la tabla de costos de transiciones más de una vez por ejecución.
- Justificar explícitamente todas las decisiones de diseño algorítmico en la documentación técnica.
- El balance entre calidad de solución y tiempo de cómputo es el desafío central; la creatividad algorítmica para explotar la estructura del espacio de soluciones es altamente valorada.