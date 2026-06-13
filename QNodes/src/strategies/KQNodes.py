import random
import time
from math import exp
from typing import Iterable, Optional

import numpy as np

from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import gestor_perfilado, profile
from src.funcs.iit import ABECEDARY, emd_efecto
from src.funcs.format import fmt_kparticion
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.models.base.application import aplicacion
from src.constants.models import (
    KQNODES_ANALYSIS_TAG,
    KQNODES_LABEL,
    KQNODES_STRAREGY_TAG,
)
from src.constants.base import (
    ACTUAL,
    COLS_IDX,
    EFFECT,
    INFTY_POS,
    NET_LABEL,
    TYPE_TAG,
)


#: Conjunto de presentes vacío reutilizable (clave de coste de un bloque sin
#: presentes). Se define una sola vez para no reconstruir ``frozenset()`` en cada
#: evaluación del lazo caliente.
_VACIO: frozenset = frozenset()


def particiones_en_k(elementos: list, k: int):
    """
    Genera todas las particiones de `elementos` en EXACTAMENTE k bloques no vacíos
    (particiones de conjunto sin etiqueta, número de Stirling de 2.ª especie).

    Args:
        elementos: lista de elementos a particionar.
        k: número exacto de bloques no vacíos.

    Yields:
        list[list]: una partición como lista de k bloques (listas no vacías).
    """
    n = len(elementos)
    if k < 1 or k > n:
        return
    if k == 1:
        yield [list(elementos)]
        return
    if k == n:
        yield [[e] for e in elementos]
        return
    primero, resto = elementos[0], elementos[1:]
    # `primero` en su propio bloque -> resto en k-1 bloques
    for parte in particiones_en_k(resto, k - 1):
        yield [[primero]] + parte
    # `primero` se une a cada bloque de una k-partición del resto
    for parte in particiones_en_k(resto, k):
        for i in range(len(parte)):
            yield parte[:i] + [[primero] + parte[i]] + parte[i + 1 :]


class _RecocidoIncremental:
    """
    Evaluador INCREMENTAL de ``δ_k`` para el recocido simulado de KQNodes.

    Mantiene una matriz de costes ``coste[lab][i]`` —coste del futuro ``i``
    respecto al bloque ``lab`` (``None`` para bloques sin presentes)— y el vector
    constante ``c0`` (coste de cada futuro al bloque-vacío, que NO cambia durante
    la búsqueda). Tras mover UN presente de bloque solo se recalculan las **≤ 2
    columnas afectadas** (``2·m`` costes, casi todos aciertos de caché) en lugar
    de reconstruir los grupos y barrer los ``m·k`` costes con ``_costo`` (cada uno
    con hash de ``frozenset`` + ``dict.get``) desde cero.

    Implementación en **Python puro** a propósito: a los tamaños típicos (m·k ≤
    unas decenas) el barrido del óptimo se hace leyendo ``coste[lab][i]`` —dos
    indexaciones de lista, ~5× más baratas que un ``_costo``— y se evita el
    overhead fijo de numpy, que a esa escala superaba al barrido directo. El
    recálculo de columnas sí llama a ``_costo`` (reusa la caché de marginales).

    El ``δ`` producido coincide EXACTAMENTE con :meth:`KQNodes._delta_grupos`
    (misma asignación voraz de futuros y misma regla de cubrimiento de bloques
    vacíos; verificado bit a bit). La δ que ``find_kmip_recocido`` DEVUELVE se
    recalcula al final con ``_evaluar_grupos`` para garantizar consistencia con el
    método exacto.
    """

    __slots__ = (
        "_costo", "k", "m", "presentes", "cubos", "base_f", "c0",
        "etq", "miembros", "coste",
    )

    def __init__(self, costo_fn, etq, k, presentes, cubos_futuro, base_f, c0):
        self._costo = costo_fn
        self.k = k
        self.m = len(cubos_futuro)
        self.presentes = presentes
        self.cubos = cubos_futuro
        self.base_f = base_f
        self.c0 = c0  # list[float] (m,), constante
        self.etq = list(etq)
        # miembros[lab] = ids globales de presentes con esa etiqueta.
        self.miembros: list[set] = [set() for _ in range(k)]
        for pos, lab in enumerate(etq):
            self.miembros[lab].add(presentes[pos])
        # coste[lab] = lista de m costes (o None si el bloque no tiene presentes).
        self.coste: list = [None] * k
        for lab in range(k):
            self._recompute(lab)

    def _recompute(self, lab: int) -> None:
        """Recalcula la columna del bloque `lab` tras cambiar sus miembros.

        Reemplaza la lista completa (no la muta in situ), de modo que un snapshot
        puede conservar la referencia anterior sin copiarla."""
        miembros = self.miembros[lab]
        if miembros:
            q = frozenset(miembros)
            costo, cubos, base_f = self._costo, self.cubos, self.base_f
            self.coste[lab] = [costo(cubos[i], base_f[i], q) for i in range(self.m)]
        else:
            self.coste[lab] = None

    def delta(self) -> float:
        """δ_k actual: asignación voraz de cada futuro a su mejor bloque + cubrimiento
        de los ``k - r`` bloques vacíos (idéntico a ``KQNodes._delta_grupos``)."""
        coste = self.coste
        nonempty = [c for c in coste if c is not None]
        r = len(nonempty)
        vacios = self.k - r
        if vacios < 0 or vacios > self.m:
            return INFTY_POS

        total = 0.0
        if vacios == 0:
            for i in range(self.m):
                mejor = INFTY_POS
                for col in nonempty:
                    v = col[i]
                    if v < mejor:
                        mejor = v
                total += mejor
            return total

        c0 = self.c0
        n_en_vacio = 0
        sobrecostos: list[float] = []
        for i in range(self.m):
            mejor = INFTY_POS
            for col in nonempty:
                v = col[i]
                if v < mejor:
                    mejor = v
            c0i = c0[i]
            if c0i <= mejor:
                total += c0i
                n_en_vacio += 1
            else:
                total += mejor
                sobrecostos.append(c0i - mejor)
        faltan = vacios - n_en_vacio
        if faltan > 0:
            if len(sobrecostos) < faltan:
                return INFTY_POS
            sobrecostos.sort()
            total += sum(sobrecostos[:faltan])
        return total

    def _snapshot(self, labs, etq_entries):
        """Captura columnas/miembros de `labs` y entradas de etq para revertir.

        ``coste[lab]`` se reemplaza (no se muta) en ``_recompute``, así que basta
        guardar la referencia; ``miembros[lab]`` sí se muta, por lo que se copia."""
        return (
            {lab: (self.coste[lab], set(self.miembros[lab])) for lab in set(labs)},
            dict(etq_entries),
        )

    def restaurar(self, undo) -> None:
        """Deshace un movimiento (restaura columnas, miembros y etiquetas)."""
        cols, etq_entries = undo
        for lab, (col, miembros) in cols.items():
            self.coste[lab] = col
            self.miembros[lab] = miembros
        for pos, lab in etq_entries.items():
            self.etq[pos] = lab

    def mover(self, pos: int, nuevo_lab: int):
        """Reetiqueta el presente en `pos` a `nuevo_lab` (≠ actual). Devuelve token undo."""
        viejo = self.etq[pos]
        undo = self._snapshot((viejo, nuevo_lab), {pos: viejo})
        pid = self.presentes[pos]
        self.miembros[viejo].discard(pid)
        self.miembros[nuevo_lab].add(pid)
        self.etq[pos] = nuevo_lab
        self._recompute(viejo)
        self._recompute(nuevo_lab)
        return undo

    def swap(self, a: int, b: int):
        """Intercambia las etiquetas de los presentes en `a` y `b` (labs ≠). Token undo."""
        la, lb = self.etq[a], self.etq[b]
        undo = self._snapshot((la, lb), {a: la, b: lb})
        pa, pb = self.presentes[a], self.presentes[b]
        self.miembros[la].discard(pa)
        self.miembros[la].add(pb)
        self.miembros[lb].discard(pb)
        self.miembros[lb].add(pa)
        self.etq[a], self.etq[b] = lb, la
        self._recompute(la)
        self._recompute(lb)
        return undo


class KQNodes(SIA):
    """
    Estrategia KQNodes: extensión de QNodes a la **k-Partición de Mínima
    Información** (k-MIP) para k ≥ 2.

    A diferencia de QNodes (biparticiones vía Queyranne), KQNodes resuelve el
    caso general explotando la **separabilidad** de la EMD-efecto: la pérdida
    ``δ_k = Σ_i c_i(P_i)`` depende solo, para cada nodo futuro ``i``, del conjunto
    de nodos presente ``P_i`` de su bloque. Esto permite un algoritmo EXACTO que
    enumera únicamente las particiones de los nodos presente (en r grupos,
    r = 1..k) y asigna cada futuro de forma voraz a su mejor bloque, cubriendo
    los ``k - r`` bloques sin presentes.

    La formulación y demostración están en
    ``.docs/.strategies/kqnodes/formulacion.md``.

    Args:
    ----
        tpm (np.ndarray): Matriz de Probabilidad de Transición del sistema.

    API pública requerida (docs/claude.md) — mapeo a implementación real:
    -----------------------------------------------------------------------
    El spec exige los nombres canónicos indicados a la izquierda; la
    implementación usa los equivalentes de la derecha por razones de diseño:

    - ``find_k_mip(k)``             → ``find_k_mip(k, metodo)`` [implementado: alias
                                       despachador] sobre ``find_kmip`` (exacto),
                                       ``find_kmip_voraz`` (heurístico) y
                                       ``find_kmip_recocido`` (metaheurístico).
    - ``generate_k_partitions(k)``  → ``generate_k_partitions(k)`` [implementado].
    - ``marginal_distributions(p)`` → ``marginal_distributions(p)`` [implementado:
                                       envoltorio público de ``_distribucion_kparticion``].
    - ``tensor_product(dists)``     → **no aplica**: la EMD-efecto factorizada
                                       ``Σ_i |a_i − b_i|`` reemplaza el producto
                                       tensorial explícito (ver formulacion.md §1.2).
                                       El cálculo equivalente está en ``emd_efecto``.

    Notas:
    -----
    - Para k=2 produce la MIP exacta (coincide con BruteForce; QNodes/Queyranne
      tiene un ~3 % residual por la no-submodularidad de la misma función).
    - El costo exacto es exponencial en el número de nodos PRESENTE ``n`` y
      polinomial en los futuros ``m``. Para ``n > _umbral_exacto(k)`` (techo
      medido que depende de k) se activa la heurística de recocido simulado
      automáticamente (``metodo="auto"``).
    """

    #: nº de nodos presente hasta el cual el método exacto es práctico, POR k.
    #: El costo del exacto es Σ_{r=1}^{k} S(n, r) candidatos (Stirling), que crece
    #: con k, de modo que el techo práctico depende de k. Medido en la PC de
    #: referencia (`tests/fase1_bench.py`) con la marginalización optimizada y un
    #: presupuesto ≈ 6 s por corrida:
    #:     k=2 → n≤14 (3.2 s; n=15 = 7.4 s)
    #:     k=3 → n≤13 (4.6 s; n=14 = 12.8 s)
    #:     k=4 → n≤11 (2.1 s; n=12 = 9.7 s)
    #:     k=5 → n≤11 (5.9 s; n=12 = 28.8 s)
    #: La marginalización "condicionar-primero" subió estos techos (p.ej. n=13,k=3
    #: pasó de 31 s a 4.6 s; n=11,k=5 de 8.6 s a 5.9 s).
    UMBRAL_EXACTO_POR_K: dict[int, int] = {2: 14, 3: 13, 4: 11, 5: 11}

    #: Respaldo/escalar (peor caso del rango del spec, k=5). Para k≥6 el espacio
    #: crece aún más, así que se usa un techo más conservador (ver `_umbral_exacto`).
    UMBRAL_EXACTO: int = 11

    def _umbral_exacto(self, k: int) -> int:
        """Umbral de presentes para usar el método exacto en "auto", según k.

        Devuelve el techo medido para `k` (``UMBRAL_EXACTO_POR_K``); para k≥6,
        donde los candidatos crecen hacia el número de Bell, usa un valor más
        conservador (10) por seguridad.
        """
        if k in self.UMBRAL_EXACTO_POR_K:
            return self.UMBRAL_EXACTO_POR_K[k]
        return 10 if k > 5 else self.UMBRAL_EXACTO

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.memoria_costos: dict[tuple[int, frozenset], float] = {}
        self.logger = SafeLogger(KQNODES_STRAREGY_TAG)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int,
        metodo: str = "auto",
        **kwargs,
    ) -> Solution:
        """
        Ejecuta la búsqueda de la k-MIP sobre el subsistema indicado.

        Args:
            estado_inicial, condicion, alcance, mecanismo: igual que en QNodes.
            k (int): número de bloques de la partición (2 ≤ k ≤ |V|).
            metodo (str): "exacto", "voraz", "recocido" o "auto" (exacto si el nº
                de presentes ≤ `_umbral_exacto(k)`, de lo contrario "recocido").
            **kwargs: parámetros del recocido (iteraciones, t0, alpha, semilla).

        Returns:
            Solution: con la k-partición encontrada, su pérdida δ_k y la distribución.
        """
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        num_vertices = (
            self.sia_subsistema.indices_ncubos.size
            + self.sia_subsistema.dims_ncubos.size
        )
        if k < 2 or k > num_vertices:
            raise ValueError(
                f"k debe estar en [2, {num_vertices}] para este subsistema; recibido k={k}."
            )

        n_presentes = int(self.sia_subsistema.dims_ncubos.size)
        if metodo == "auto":
            metodo = "exacto" if n_presentes <= self._umbral_exacto(k) else "recocido"

        if metodo == "exacto":
            _, bloques = self.find_kmip(k)
        elif metodo == "voraz":
            _, bloques = self.find_kmip_voraz(k)
        elif metodo == "recocido":
            _, bloques = self.find_kmip_recocido(k, **kwargs)
        else:
            raise ValueError(f"método desconocido: {metodo!r}")

        # Guardar para uso externo (p.ej. visualización del hipercubo).
        self.ultima_kparticion: list[list[tuple[int, int]]] = bloques

        distribucion_particion = self._distribucion_kparticion(bloques)
        perdida = float(emd_efecto(distribucion_particion, self.sia_dists_marginales))

        return Solution(
            estrategia=f"{KQNODES_LABEL} (k={k}, {metodo})",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=distribucion_particion,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_kparticion(bloques),
        )

    def generate_k_partitions(self, k: int) -> Iterable[list[list]]:
        """
        Genera todas las k-particiones de los nodos presente del subsistema actual.

        Requiere que ``sia_preparar_subsistema`` haya sido llamado primero.
        Delega en la función de módulo ``particiones_en_k`` sobre la lista de
        nodos presente ``self.sia_subsistema.dims_ncubos``.

        Args:
            k (int): número exacto de bloques (1 ≤ k ≤ n_presentes).

        Yields:
            list[list]: una partición como lista de k bloques de índices de nodos.

        Complejidad:
            Genera S(n, k) particiones (número de Stirling de 2.ª especie),
            donde n = len(dims_ncubos). Θ(S(n, k)) en tiempo y espacio de salida.
        """
        presentes = [int(i) for i in self.sia_subsistema.dims_ncubos]
        return particiones_en_k(presentes, k)

    # --- API canónica del spec (docs/claude.md) ---------------------------

    def find_k_mip(
        self, k: int, metodo: str = "auto"
    ) -> tuple[float, list[list[tuple[int, int]]]]:
        """
        Nombre canónico requerido por ``docs/claude.md`` para la búsqueda de la
        k-MIP. Despacha al método concreto según ``metodo`` (igual criterio que
        ``aplicar_estrategia``): exacto/voraz/recocido, o ``"auto"`` (exacto si el
        nº de presentes ≤ ``_umbral_exacto(k)``, recocido en caso contrario).

        Requiere ``sia_preparar_subsistema`` previo. Devuelve ``(δ_k, k-partición)``.
        """
        if metodo == "auto":
            n_presentes = int(self.sia_subsistema.dims_ncubos.size)
            metodo = "exacto" if n_presentes <= self._umbral_exacto(k) else "recocido"
        if metodo == "exacto":
            return self.find_kmip(k)
        if metodo == "voraz":
            return self.find_kmip_voraz(k)
        if metodo == "recocido":
            return self.find_kmip_recocido(k)
        raise ValueError(f"método desconocido: {metodo!r}")

    def marginal_distributions(
        self, particion: list[list[tuple[int, int]]]
    ) -> np.ndarray:
        """
        Nombre canónico del spec. Distribución marginal del sistema k-particionado:
        para cada nodo futuro, su marginal-efecto conservando únicamente los
        presentes de su bloque (la forma factorizada de la EMD-efecto reemplaza el
        producto tensorial explícito; ver ``formulacion.md`` §1.2).

        Args:
            particion: k-partición como lista de bloques de vértices ``(tiempo, i)``.

        Returns:
            np.ndarray: vector de marginales en el orden de los n-cubos futuro.
        """
        return self._distribucion_kparticion(particion)

    # --- Evaluación común (separabilidad) ---------------------------------

    def _delta_grupos(
        self,
        grupos: list[frozenset],
        k: int,
        cubos_futuro: list,
        base: np.ndarray,
    ) -> float:
        """
        Versión SOLO-ESCALAR de :meth:`_evaluar_grupos`: devuelve únicamente la
        pérdida ``δ_k`` de una partición de los presentes en ``grupos`` (con
        ``k - len(grupos)`` bloques sin presentes), sin reconstruir la lista de
        bloques de vértices.

        Es la rutina del LAZO CALIENTE: el método exacto la invoca una vez por
        candidato (S(n, r) veces) y las heurísticas una vez por probe (miles de
        veces). Evitar la reconstrucción de bloques —innecesaria mientras solo se
        compara δ_k— es la optimización dominante para correr pruebas grandes.

        Produce EXACTAMENTE el mismo ``δ_k`` que ``_evaluar_grupos(...)[0]`` (misma
        asignación voraz y misma regla de cubrimiento de bloques vacíos); la
        reconstrucción de bloques se difiere a una única llamada final sobre el
        ganador.
        """
        r = len(grupos)
        vacios = k - r
        if vacios < 0 or vacios > len(cubos_futuro):
            return INFTY_POS
        permite_vacio = vacios > 0

        total = 0.0
        n_en_vacio = 0
        sobrecostos: list[float] = []

        for idx, cubo in enumerate(cubos_futuro):
            base_i = float(base[idx])
            costo_mejor = INFTY_POS
            for g in grupos:
                c = self._costo(cubo, base_i, g)
                if c < costo_mejor:
                    costo_mejor = c

            if permite_vacio:
                costo_vacio = self._costo(cubo, base_i, _VACIO)
                if costo_vacio <= costo_mejor:
                    total += costo_vacio
                    n_en_vacio += 1
                else:
                    total += costo_mejor
                    sobrecostos.append(costo_vacio - costo_mejor)
            else:
                total += costo_mejor

        if permite_vacio and n_en_vacio < vacios:
            faltan = vacios - n_en_vacio
            if len(sobrecostos) < faltan:
                return INFTY_POS
            sobrecostos.sort()
            total += sum(sobrecostos[:faltan])

        return total

    def _evaluar_grupos(
        self,
        grupos: list[set],
        k: int,
        cubos_futuro: list,
        base: np.ndarray,
    ) -> tuple[float, Optional[list[list[tuple[int, int]]]]]:
        """
        Evalúa una partición de los presentes en `grupos` (r = len(grupos) grupos
        no vacíos) con `k - r` bloques sin presentes. Asigna cada futuro a su
        bloque de mínimo coste y cubre los bloques vacíos forzando los futuros más
        baratos. Devuelve (δ_k, bloques) o (INFTY_POS, None) si es inviable.

        Esta rutina es la base COMÚN del método exacto y de las heurísticas:
        garantiza que ambos optimizan exactamente sobre los futuros.
        """
        r = len(grupos)
        vacios = k - r
        if vacios < 0 or vacios > len(cubos_futuro):
            return INFTY_POS, None
        permite_vacio = vacios > 0

        total = 0.0
        asignacion: list[int] = []
        deltas: list[tuple[float, int]] = []
        en_vacio: list[int] = []

        for idx, cubo in enumerate(cubos_futuro):
            base_i = float(base[idx])
            if grupos:
                costos = [self._costo(cubo, base_i, g) for g in grupos]
                g_mejor = min(range(len(grupos)), key=costos.__getitem__)
                costo_mejor = costos[g_mejor]
            else:
                g_mejor, costo_mejor = -1, INFTY_POS

            if permite_vacio:
                costo_vacio = self._costo(cubo, base_i, _VACIO)
                if costo_vacio <= costo_mejor:
                    total += costo_vacio
                    asignacion.append(-1)
                    en_vacio.append(idx)
                else:
                    total += costo_mejor
                    asignacion.append(g_mejor)
                    deltas.append((costo_vacio - costo_mejor, idx))
            else:
                total += costo_mejor
                asignacion.append(g_mejor)

        if permite_vacio and len(en_vacio) < vacios:
            faltan = vacios - len(en_vacio)
            if len(deltas) < faltan:
                return INFTY_POS, None
            deltas.sort(key=lambda d: d[0])
            for sobrecosto, idx in deltas[:faltan]:
                total += sobrecosto
                asignacion[idx] = -1
                en_vacio.append(idx)

        bloques = self._reconstruir_bloques(grupos, asignacion, en_vacio, vacios, cubos_futuro)
        return total, bloques

    @staticmethod
    def _grupos_de_etiquetas(
        etiquetas: list[int], presentes: list[int]
    ) -> list[frozenset]:
        """Convierte una etiquetación presente→bloque en la lista de grupos no vacíos.

        Devuelve ``frozenset`` (no ``set``) para que sirvan directamente como clave
        del caché de costes en ``_costo`` sin reconstruirse en cada evaluación.
        """
        por_bloque: dict[int, set] = {}
        for pos, lab in enumerate(etiquetas):
            por_bloque.setdefault(lab, set()).add(presentes[pos])
        return [frozenset(s) for s in por_bloque.values()]

    # --- Método exacto ----------------------------------------------------

    @profile(context={TYPE_TAG: KQNODES_ANALYSIS_TAG})
    def find_kmip(self, k: int) -> tuple[float, list[list[tuple[int, int]]]]:
        """
        Algoritmo EXACTO de la k-MIP por separabilidad: enumera las particiones de
        los presentes en r grupos (r = 1..k) y evalúa cada una con asignación
        voraz óptima de los futuros (ver formulacion.md §5).

        Args:
            k (int): número de bloques (2 ≤ k ≤ n + m).

        Returns:
            tuple[float, list]: (pérdida mínima δ_k, k-partición óptima).

        Complejidad temporal:
            Θ(Σ_{r=1}^{k} S(n, r) · m · r), donde n = nodos presente,
            m = nodos futuro, S(n, r) = Stirling de 2.ª especie.
            Con caché de c_i(Q): cada costo se computa una vez → Θ(n · 2^n)
            entradas distintas en ``memoria_costos``.

        Complejidad espacial:
            Θ(n · 2^n) para la caché de costos (permanente en la instancia).
            Θ(n) para el estado de la partición candidata activa.

        Mejor caso:
            k ≥ n → el bucle r=1..k termina en r=n (S(n,r)=0 para r>n).
        Peor caso:
            k ≈ n/2, donde los números de Stirling S(n, k) son máximos.
        """
        cubos_futuro = list(self.sia_subsistema.ncubos)
        presentes = [int(i) for i in self.sia_subsistema.dims_ncubos]
        base = self.sia_dists_marginales
        n = len(presentes)

        mejor_delta = INFTY_POS
        mejor_grupos: Optional[list[frozenset]] = None

        for r in range(1, k + 1):
            if r > n:
                break
            for particion_presente in particiones_en_k(presentes, r):
                grupos = [frozenset(g) for g in particion_presente]
                # Lazo caliente: solo el escalar δ_k; NO se reconstruyen bloques
                # por candidato (se difiere al ganador).
                total = self._delta_grupos(grupos, k, cubos_futuro, base)
                if total < mejor_delta:
                    mejor_delta, mejor_grupos = total, grupos

        if mejor_grupos is None:
            return INFTY_POS, None
        # Reconstrucción de la k-partición una sola vez, sobre el óptimo.
        _, mejor_bloques = self._evaluar_grupos(mejor_grupos, k, cubos_futuro, base)
        return mejor_delta, mejor_bloques

    # --- Heurísticas ------------------------------------------------------

    @profile(context={TYPE_TAG: KQNODES_ANALYSIS_TAG})
    def find_kmip_voraz(self, k: int) -> tuple[float, list[list[tuple[int, int]]]]:
        """
        Heurística VORAZ: arranque factible + búsqueda local (mueve cada presente
        al bloque que más reduce δ_k, hasta no mejorar). Determinista.

        Args:
            k (int): número de bloques (2 ≤ k ≤ n + m).

        Returns:
            tuple[float, list]: (pérdida δ_k encontrada, k-partición).

        Complejidad temporal:
            Θ(I · n · k · m) en el peor caso, donde I = número de iteraciones
            hasta convergencia, n = nodos presente, m = nodos futuro.
            En la práctica I es pequeño (< n² en todos los benchmarks).

        Complejidad espacial:
            Θ(n) para el vector de etiquetas; Θ(n · 2^n) para la caché de costos
            compartida con ``find_kmip``.

        Mejor caso:
            El arranque factible ya es óptimo → una sola pasada sin mejoras (Θ(n·k·m)).
        Peor caso:
            El paisaje tiene muchos óptimos locales → convergencia lenta,
            solución subóptima. Usar ``find_kmip_recocido`` para más robustez.
        """
        cubos_futuro = list(self.sia_subsistema.ncubos)
        presentes = [int(i) for i in self.sia_subsistema.dims_ncubos]
        base = self.sia_dists_marginales
        etiquetas, delta = self._construir_voraz(k, presentes, cubos_futuro, base)
        _, bloques = self._evaluar_grupos(
            self._grupos_de_etiquetas(etiquetas, presentes), k, cubos_futuro, base
        )
        return delta, bloques

    def _construir_voraz(
        self, k: int, presentes: list[int], cubos_futuro: list, base: np.ndarray
    ) -> tuple[list[int], float]:
        """Etiquetación inicial factible + ascenso por mínimos locales."""
        n = len(presentes)
        # Arranque factible: reparte los presentes en min(n, k) bloques.
        etiquetas = [pos % min(n, k) for pos in range(n)] if n else []

        def evalua(etq: list[int]) -> float:
            return self._delta_grupos(
                self._grupos_de_etiquetas(etq, presentes), k, cubos_futuro, base
            )

        mejor = evalua(etiquetas)
        mejorando = True
        while mejorando:
            mejorando = False
            for pos in range(n):
                actual = etiquetas[pos]
                mejor_lab, mejor_local = actual, mejor
                for lab in range(k):
                    if lab == actual:
                        continue
                    etiquetas[pos] = lab
                    d = evalua(etiquetas)
                    if d < mejor_local:
                        mejor_local, mejor_lab = d, lab
                etiquetas[pos] = mejor_lab
                if mejor_lab != actual:
                    mejor, mejorando = mejor_local, True
        return etiquetas, mejor

    @profile(context={TYPE_TAG: KQNODES_ANALYSIS_TAG})
    def find_kmip_recocido(
        self,
        k: int,
        iteraciones: int = 3000,
        t0: Optional[float] = None,
        alpha: float = 0.997,
        semilla: int = 0,
        reinicios: int = 1,
    ) -> tuple[float, list[list[tuple[int, int]]]]:
        """
        Heurística de RECOCIDO SIMULADO sobre la etiquetación de los presentes.

        - WARM START: la primera corrida parte de la solución voraz; las demás de
          etiquetaciones aleatorias factibles (multi-arranque, ``reinicios`` > 1).
          Más reinicios mejoran la robustez pero cuestan tiempo.
        - Vecindario: reetiquetar un presente a otro bloque o intercambiar las
          etiquetas de dos presentes (ambas operaciones con probabilidad 0.5 si n≥2).
        - Acepta empeoramientos con probabilidad ``exp(-Δ/T)`` y enfría ``T *= alpha``.

        Args:
            k (int): número de bloques.
            iteraciones (int): pasos de la cadena de Markov por corrida.
            t0 (float | None): temperatura inicial; si None se estima como
                ``max(δ_voraz * 0.5, 1e-3)``.
            alpha (float): factor de enfriamiento (0 < alpha < 1).
            semilla (int): semilla para reproducibilidad.
            reinicios (int): número de corridas con warm-start/aleatorio.

        Returns:
            tuple[float, list]: (mejor pérdida δ_k, k-partición correspondiente).

        Complejidad temporal:
            Θ(reinicios · iteraciones · m · r) en el peor caso, pero con evaluación
            INCREMENTAL (``_RecocidoIncremental``) cada paso recalcula solo las ≤2
            columnas afectadas → Θ(m) costes (aciertos de caché) + un ``min``/
            cubrimiento vectorizados Θ(m·r), sin reconstruir grupos ni bloques.
            En caché caliente acelera ≈ 2-3× frente al barrido completo.

        Complejidad espacial:
            Θ(n) para la etiquetación; Θ(n · 2^n) para la caché de costos
            compartida con ``find_kmip``.

        Mejor caso:
            warm start ya es óptimo y la primera iteración no acepta cambios → Θ(m·n).
        Peor caso:
            reinicios altos, caché fría en cada reinicio → Θ(reinicios · iteraciones · m · n).

        Devuelve la mejor solución sobre todas las corridas.
        """
        cubos_futuro = list(self.sia_subsistema.ncubos)
        presentes = [int(i) for i in self.sia_subsistema.dims_ncubos]
        base = self.sia_dists_marginales
        rng = random.Random(semilla)

        warm, _ = self._construir_voraz(k, presentes, cubos_futuro, base)
        mejor_global, mejor_delta = warm[:], INFTY_POS

        for corrida in range(max(1, reinicios)):
            inicio = warm[:] if corrida == 0 else self._etiquetas_factibles(k, presentes, rng)
            etq, delta = self._recocido_una_corrida(
                inicio, k, presentes, cubos_futuro, base, iteraciones, t0, alpha, rng
            )
            if delta < mejor_delta:
                mejor_global, mejor_delta = etq, delta

        # δ autoritativa: se recalcula con _evaluar_grupos sobre la mejor
        # etiquetación para garantizar consistencia EXACTA con el método exacto
        # (la δ incremental se usa solo para decidir aceptar/rechazar en la cadena).
        delta_auth, bloques = self._evaluar_grupos(
            self._grupos_de_etiquetas(mejor_global, presentes), k, cubos_futuro, base
        )
        return delta_auth, bloques

    def _recocido_una_corrida(
        self, inicio, k, presentes, cubos_futuro, base, iteraciones, t0, alpha, rng
    ) -> tuple[list[int], float]:
        """Una corrida de recocido simulado desde la etiquetación `inicio`, con
        evaluación INCREMENTAL de δ_k (cada paso recalcula solo los bloques
        afectados; ver :class:`_RecocidoIncremental`).

        Movimientos: intercambiar las etiquetas de dos presentes (prob. 0.5 si
        n≥2) o reetiquetar un presente. Los movimientos nulos (misma etiqueta) se
        descartan consumiendo la misma aleatoriedad que antes, manteniendo el
        enfriamiento; los empeoramientos se aceptan con prob. ``exp(-Δ/T)`` y los
        rechazos se revierten en O(m) sin recomputar costes.
        """
        n = len(presentes)
        if n == 0:
            return inicio[:], 0.0

        m = len(cubos_futuro)
        base_f = [float(base[idx]) for idx in range(m)]
        c0 = [self._costo(cubos_futuro[i], base_f[i], _VACIO) for i in range(m)]
        estado = _RecocidoIncremental(
            self._costo, inicio, k, presentes, cubos_futuro, base_f, c0
        )
        actual_delta = estado.delta()
        mejor, mejor_delta = estado.etq[:], actual_delta

        temperatura = t0 if t0 is not None else max(actual_delta * 0.5, 1e-3)
        for _ in range(iteraciones):
            if n >= 2 and rng.random() < 0.5:  # intercambio de dos presentes
                a, b = rng.sample(range(n), 2)
                if estado.etq[a] == estado.etq[b]:  # movimiento nulo
                    temperatura *= alpha
                    continue
                undo = estado.swap(a, b)
            else:  # reetiquetar un presente
                p = rng.randrange(n)
                nuevo = rng.randrange(k)
                if estado.etq[p] == nuevo:  # movimiento nulo
                    temperatura *= alpha
                    continue
                undo = estado.mover(p, nuevo)

            d = estado.delta()
            if d <= actual_delta or (
                d < INFTY_POS
                and rng.random() < exp(-(d - actual_delta) / max(temperatura, 1e-12))
            ):
                actual_delta = d
                if d < mejor_delta:
                    mejor, mejor_delta = estado.etq[:], d
            else:
                estado.restaurar(undo)
            temperatura *= alpha
        return mejor, mejor_delta

    def _etiquetas_factibles(self, k: int, presentes: list[int], rng: random.Random) -> list[int]:
        """Etiquetación aleatoria factible: garantiza ≥ min(n,k) bloques con presentes."""
        n = len(presentes)
        base_labels = list(range(min(n, k)))
        etq = base_labels + [rng.randrange(k) for _ in range(n - len(base_labels))]
        rng.shuffle(etq)
        return etq

    def _reconstruir_bloques(
        self,
        grupos: list[set],
        asignacion: list[int],
        en_vacio: list[int],
        vacios: int,
        cubos_futuro: list,
    ) -> list[list[tuple[int, int]]]:
        """Construye la k-partición (lista de bloques de vértices) a partir de la
        asignación óptima."""
        bloques: list[list[tuple[int, int]]] = []

        # Bloques con presentes (grupo g + futuros asignados a g).
        for g_idx, grupo in enumerate(grupos):
            bloque: list[tuple[int, int]] = [(ACTUAL, p) for p in sorted(grupo)]
            for idx, g in enumerate(asignacion):
                if g == g_idx:
                    bloque.append((EFFECT, int(cubos_futuro[idx].indice)))
            bloques.append(bloque)

        # Bloques sin presentes: un futuro por bloque (cubrimiento) + el resto.
        if vacios > 0:
            bloques_vacios: list[list[tuple[int, int]]] = [[] for _ in range(vacios)]
            for b in range(vacios):
                idx = en_vacio[b]
                bloques_vacios[b].append((EFFECT, int(cubos_futuro[idx].indice)))
            for idx in en_vacio[vacios:]:
                bloques_vacios[0].append((EFFECT, int(cubos_futuro[idx].indice)))
            bloques.extend(bloques_vacios)

        return bloques

    def _marginal_futuro(self, cubo, presentes: Iterable[int]) -> float:
        """Marginal-efecto del cubo futuro conservando solo los presentes dados.

        Estrategia CONDICIONAR-PRIMERO (clave para sistemas grandes): se fija al
        estado inicial los ejes conservados ``Q`` —un slice del n-cubo que reduce
        el arreglo a ``2^(n-|Q|)`` elementos— y se promedia (marginaliza) el resto
        a un escalar. Es matemáticamente equivalente a promediar ``P\\Q`` primero y
        seleccionar el estado después (operaciones sobre ejes disjuntos conmutan;
        verificado en N3-N5 con diferencia ≤ 6e-8), pero evita el ``np.mean`` sobre
        el arreglo completo de ``2^n`` del orden inverso. Reduce el coste por
        evaluación de ``Θ(2^n)`` a ``Θ(2^{n-|Q|})``, lo que multiplica el techo
        práctico de las heurísticas en sistemas grandes.

        El slice respeta el mismo convenio de ejes locales que ``NCube.marginalizar``
        —el eje local del dim en la posición ``dim_idx`` es ``(ndims-1) - dim_idx``,
        es decir, el primer dim es el más significativo— por lo que es correcto
        también para subsistemas asimétricos con dimensiones NO contiguas (donde
        ``NCube.condicionar`` no aplica, pues asume índices globales == ejes locales).
        Verificado idéntico al cálculo original (≤ 6e-8) en N3-N6, simétricos y
        asimétricos.
        """
        conservar = set(presentes)
        dims = cubo.dims
        ndims = dims.size
        if ndims == 0:  # cubo ya reducido a escalar (sin presentes que conservar)
            return float(cubo.data)

        estado = self.sia_subsistema.estado_inicial
        # Fija al estado inicial los ejes conservados Q (slice que reduce el arreglo
        # a 2^(n-|Q|)); los ejes restantes P\Q quedan libres para promediarse.
        seleccion: list = [slice(None)] * ndims
        for dim_idx, g in enumerate(dims):
            g = int(g)
            if g in conservar:
                seleccion[(ndims - 1) - dim_idx] = int(estado[g])
        return float(np.mean(cubo.data[tuple(seleccion)]))

    def _costo(self, cubo, base_i: float, presentes: frozenset) -> float:
        """c_i(Q) = |marginal del futuro i con presentes Q − marginal subsistema|.

        Memoizado por ``(índice del futuro, Q)``. ``presentes`` debe llegar ya como
        ``frozenset`` (todos los llamadores internos lo garantizan), de modo que la
        clave se usa directamente sin reconstruirla. Una sola consulta al diccionario
        por acierto (``get``) en el lazo caliente.
        """
        clave = (cubo.indice, presentes)
        val = self.memoria_costos.get(clave)
        if val is None:
            val = abs(self._marginal_futuro(cubo, presentes) - base_i)
            self.memoria_costos[clave] = val
        return val

    def _distribucion_kparticion(
        self, bloques: list[list[tuple[int, int]]]
    ) -> np.ndarray:
        """Vector de marginales del sistema k-particionado (orden de los n-cubos)."""
        futuro_a_presentes: dict[int, frozenset] = {}
        for bloque in bloques:
            presentes_bloque = frozenset(i for (t, i) in bloque if t == ACTUAL)
            for (t, i) in bloque:
                if t == EFFECT:
                    futuro_a_presentes[i] = presentes_bloque

        distribucion = np.empty(
            self.sia_subsistema.indices_ncubos.size, dtype=np.float32
        )
        for idx, cubo in enumerate(self.sia_subsistema.ncubos):
            presentes = futuro_a_presentes.get(int(cubo.indice), frozenset())
            distribucion[idx] = self._marginal_futuro(cubo, presentes)
        return distribucion
