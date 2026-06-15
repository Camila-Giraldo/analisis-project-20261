"""
Clase principal KQNodes y punto de entrada de la estrategia.
"""

import time
from typing import Iterable

import numpy as np

from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import gestor_perfilado
from src.funcs.iit import ABECEDARY, emd_efecto
from src.funcs.format import fmt_kparticion
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.models.base.application import aplicacion
from src.constants.models import KQNODES_LABEL, KQNODES_STRAREGY_TAG
from src.constants.base import COLS_IDX, NET_LABEL
from ._partitions import particiones_en_k
from ._cost import _CostMixin
from ._evaluation import _EvaluationMixin
from ._search import _SearchMixin


class KQNodes(_SearchMixin, _EvaluationMixin, _CostMixin, SIA):
    """
    Estrategia KQNodes: extensión de QNodes a la **k-Partición de Mínima
    Información** (k-MIP) para k ≥ 2.

    Explota la **separabilidad** de la EMD-efecto (``δ_k = Σ_i c_i(P_i)``)
    para enumerar solo las particiones de los nodos presente y asignar cada
    futuro de forma voraz, cubriendo los ``k - r`` bloques sin presentes.

    Args:
        tpm (np.ndarray): Matriz de Probabilidad de Transición del sistema.

    API pública canónica:
        - ``find_k_mip(k, metodo)``     — despachador (exacto / voraz / recocido / auto).
        - ``generate_k_partitions(k)``  — genera todas las k-particiones de los presentes.
        - ``marginal_distributions(p)`` — distribución marginal de una k-partición.
    """

    #: nº de nodos presente hasta el cual el método exacto es práctico, POR k.
    UMBRAL_EXACTO_POR_K: dict[int, int] = {2: 14, 3: 13, 4: 11, 5: 11}

    #: Respaldo escalar para k ≥ 6.
    UMBRAL_EXACTO: int = 11

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.memoria_costos: dict[tuple[int, frozenset], float] = {}
        self.logger = SafeLogger(KQNODES_STRAREGY_TAG)


    def _umbral_exacto(self, k: int) -> int:
        """Umbral de presentes para el método exacto en modo "auto".

        Para k ≥ 6 usa 10 (conservador) ya que los candidatos crecen hacia el
        número de Bell.
        """
        if k in self.UMBRAL_EXACTO_POR_K:
            return self.UMBRAL_EXACTO_POR_K[k]
        return 10 if k > 5 else self.UMBRAL_EXACTO

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
            k (int): número de bloques (2 ≤ k ≤ |V|).
            metodo (str): "exacto", "voraz", "recocido" o "auto".
            **kwargs: parámetros del recocido (iteraciones, t0, alpha, semilla).
        """
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        num_vertices = (
            self.sia_subsistema.indices_ncubos.size
            + self.sia_subsistema.dims_ncubos.size
        )
        if k < 2 or k > num_vertices:
            raise ValueError(
                f"k debe estar en [2, {num_vertices}] para este subsistema; "
                f"recibido k={k}."
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

    def find_k_mip(
        self, k: int, metodo: str = "auto"
    ) -> tuple[float, list[list[tuple[int, int]]]]:
        """Despachador canónico → exacto / voraz / recocido / auto."""
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

    def generate_k_partitions(self, k: int) -> Iterable[list[list]]:
        """
        Genera todas las k-particiones de los nodos presente del subsistema.

        Requiere ``sia_preparar_subsistema`` previo.

        Args:
            k (int): número exacto de bloques (1 ≤ k ≤ n_presentes).

        Yields:
            list[list]: una partición como lista de k bloques de índices.
        """
        presentes = [int(i) for i in self.sia_subsistema.dims_ncubos]
        return particiones_en_k(presentes, k)

    def marginal_distributions(
        self, particion: list[list[tuple[int, int]]]
    ) -> np.ndarray:
        """
        Distribución marginal del sistema k-particionado.

        La forma factorizada de la EMD-efecto reemplaza el producto tensorial
        explícito (ver ``formulacion.md`` §1.2).

        Args:
            particion: k-partición como lista de bloques de vértices (tiempo, i).

        Returns:
            np.ndarray: vector de marginales en el orden de los n-cubos futuro.
        """
        return self._distribucion_kparticion(particion)
