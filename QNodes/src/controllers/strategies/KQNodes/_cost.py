"""
Mixin de costo y marginales para KQNodes.

Encapsula el cálculo del marginal-efecto (estrategia condicionar-primero),
la función de costo memoizada c_i(Q) y la distribución de la k-partición.
"""

from typing import Iterable

import numpy as np

from src.constants.base import ACTUAL, EFFECT
from ._partitions import VACIO


class _CostMixin:
    """Cálculo de costes y marginales. Requiere acceso a ``sia_subsistema``."""

    def _marginal_futuro(self, cubo, presentes: Iterable[int]) -> float:
        """Marginal-efecto del cubo futuro conservando solo los presentes dados.

        Estrategia CONDICIONAR-PRIMERO: fija al estado inicial los ejes
        conservados Q (slice que reduce el arreglo a 2^(n-|Q|) elementos) y
        promedia el resto. Reduce el coste de Θ(2^n) a Θ(2^{n-|Q|}),
        multiplicando el techo práctico para sistemas grandes.
        """
        conservar = set(presentes)
        dims = cubo.dims
        ndims = dims.size
        if ndims == 0:
            return float(cubo.data)

        estado = self.sia_subsistema.estado_inicial
        seleccion: list = [slice(None)] * ndims
        for dim_idx, g in enumerate(dims):
            g = int(g)
            if g in conservar:
                seleccion[(ndims - 1) - dim_idx] = int(estado[g])
        return float(np.mean(cubo.data[tuple(seleccion)]))

    def _costo(self, cubo, base_i: float, presentes: frozenset) -> float:
        """c_i(Q) = |marginal del futuro i con presentes Q − marginal subsistema|.

        Memoizado por ``(índice del futuro, Q)``.
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
            for t, i in bloque:
                if t == EFFECT:
                    futuro_a_presentes[i] = presentes_bloque

        distribucion = np.empty(
            self.sia_subsistema.indices_ncubos.size, dtype=np.float32
        )
        for idx, cubo in enumerate(self.sia_subsistema.ncubos):
            presentes = futuro_a_presentes.get(int(cubo.indice), VACIO)
            distribucion[idx] = self._marginal_futuro(cubo, presentes)
        return distribucion
