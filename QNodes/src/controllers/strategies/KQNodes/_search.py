"""
Mixin de algoritmos de búsqueda para KQNodes.

Implementa los tres métodos de búsqueda de la k-MIP:
  - ``find_kmip``         — exacto por separabilidad (Stirling).
  - ``find_kmip_voraz``   — heurística de ascenso local.
  - ``find_kmip_recocido``— metaheurística de recocido simulado con evaluación
                            incremental (``RecocidoIncremental``).
"""

import random
from math import exp
from typing import Optional

from src.middlewares.profile import profile
from src.constants.models import KQNODES_ANALYSIS_TAG
from src.constants.base import INFTY_POS, TYPE_TAG
from ._partitions import particiones_en_k, VACIO
from ._annealing import RecocidoIncremental


class _SearchMixin:
    """Algoritmos de búsqueda de la k-MIP. Requiere ``_delta_grupos``,
    ``_evaluar_grupos``, ``_grupos_de_etiquetas`` y ``_costo``."""

    @profile(context={TYPE_TAG: KQNODES_ANALYSIS_TAG})
    def find_kmip(self, k: int) -> tuple[float, list[list[tuple[int, int]]]]:
        """
        Algoritmo EXACTO de la k-MIP por separabilidad.

        Enumera particiones de los presentes en r grupos (r = 1..k) y evalúa
        cada una con asignación voraz óptima de los futuros.

        Complejidad: Θ(Σ_{r=1}^{k} S(n, r) · m · r), con caché de c_i(Q).
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
                total = self._delta_grupos(grupos, k, cubos_futuro, base)
                if total < mejor_delta:
                    mejor_delta, mejor_grupos = total, grupos

        if mejor_grupos is None:
            return INFTY_POS, None

        _, mejor_bloques = self._evaluar_grupos(mejor_grupos, k, cubos_futuro, base)
        return mejor_delta, mejor_bloques

    # Heurística voraz

    @profile(context={TYPE_TAG: KQNODES_ANALYSIS_TAG})
    def find_kmip_voraz(self, k: int) -> tuple[float, list[list[tuple[int, int]]]]:
        """
        Heurística VORAZ: arranque factible + ascenso por mínimos locales.

        Complejidad: Θ(I · n · k · m), donde I = iteraciones hasta estabilizar.
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
        self, k: int, presentes: list[int], cubos_futuro: list, base
    ) -> tuple[list[int], float]:
        """Etiquetación inicial factible + ascenso por mínimos locales."""
        n = len(presentes)
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

    # Heurística de recocido simulado
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

        - WARM START en la primera corrida (parte de la solución voraz).
        - Vecindario: reetiquetar un presente o intercambiar etiquetas de dos.
        - Evaluación INCREMENTAL (``RecocidoIncremental``): solo se recalculan
          las ≤ 2 columnas afectadas por movimiento → Θ(m) por paso.

        Args:
            k: número de bloques.
            iteraciones: pasos de la cadena de Markov por corrida.
            t0: temperatura inicial; None → ``max(δ_voraz * 0.5, 1e-3)``.
            alpha: factor de enfriamiento (0 < alpha < 1).
            semilla: semilla para reproducibilidad.
            reinicios: número de corridas (warm-start + aleatorio).
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

        # δ autoritativa: recalculada con _evaluar_grupos para consistencia exacta.
        delta_auth, bloques = self._evaluar_grupos(
            self._grupos_de_etiquetas(mejor_global, presentes), k, cubos_futuro, base
        )
        return delta_auth, bloques

    def _recocido_una_corrida(
        self, inicio, k, presentes, cubos_futuro, base, iteraciones, t0, alpha, rng
    ) -> tuple[list[int], float]:
        """Una corrida de recocido simulado con evaluación INCREMENTAL de δ_k."""
        n = len(presentes)
        if n == 0:
            return inicio[:], 0.0

        m = len(cubos_futuro)
        base_f = [float(base[idx]) for idx in range(m)]
        c0 = [self._costo(cubos_futuro[i], base_f[i], VACIO) for i in range(m)]
        estado = RecocidoIncremental(
            self._costo, inicio, k, presentes, cubos_futuro, base_f, c0
        )
        actual_delta = estado.delta()
        mejor, mejor_delta = estado.etq[:], actual_delta

        temperatura = t0 if t0 is not None else max(actual_delta * 0.5, 1e-3)
        for _ in range(iteraciones):
            if n >= 2 and rng.random() < 0.5:
                a, b = rng.sample(range(n), 2)
                if estado.etq[a] == estado.etq[b]:
                    temperatura *= alpha
                    continue
                undo = estado.swap(a, b)
            else:
                p = rng.randrange(n)
                nuevo = rng.randrange(k)
                if estado.etq[p] == nuevo:
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

    def _etiquetas_factibles(
        self, k: int, presentes: list[int], rng: random.Random
    ) -> list[int]:
        """Etiquetación aleatoria factible: garantiza ≥ min(n,k) bloques con presentes."""
        n = len(presentes)
        base_labels = list(range(min(n, k)))
        etq = base_labels + [rng.randrange(k) for _ in range(n - len(base_labels))]
        rng.shuffle(etq)
        return etq
