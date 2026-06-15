"""
Clase principal KGeoMIP y punto de entrada de la estrategia.
"""

import time

import numpy as np

from src.constants.base import NET_LABEL, ACTUAL, EFECTO, TYPE_TAG
from src.constants.models import GEOMETRIC_ANALYSIS_TAG, GEOMETRIC_LABEL
from src.controllers.manager import Manager
from src.funcs.base import ABECEDARY
from src.funcs.format import fmt_biparte_q
from src.middlewares.profile import profiler_manager, profile
from src.models.base.sia import SIA
from src.models.core.solution import Solution

from ._table import _TableMixin
from ._search_k2 import _SearchK2Mixin
from ._search_kn import _SearchKNMixin
from ._partition import _PartitionMixin
from ._sampling import _SamplingMixin


class KGeoMIP(_SearchKNMixin, _SearchK2Mixin, _PartitionMixin, _TableMixin, _SamplingMixin, SIA):
    """
    Estrategia KGeoMIP: k-Partición de Mínima Información mediante
    Programación Dinámica sobre la tabla geométrica de costos.

    Extiende GeoMIP (k=2) al caso general k ≥ 2 mediante:
      - Tabla DP truncada adaptativamente (Paso 1).
      - Greedy jerárquico sobre la bipartición base (Paso 2).
      - Hill climbing first-improvement (Paso 3).

    Args:
        gestor (Manager): Gestor del sistema con estado inicial y TPM.
        k (int): Número de particiones deseado (≥ 2).
    """

    def __init__(self, gestor: Manager, k: int = 2):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.k = k
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set[tuple]
        self.memoria_particiones: dict[tuple, tuple[float, np.ndarray]] = {}
        # float32 por variable; llenado en aplicar_estrategia
        self._ncubo_data: list[np.ndarray] = []
        self._col_cache: dict[int, np.ndarray] = {}
        self._powers: np.ndarray
        self._i0_int: int
        self._i_final_int: int

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: int = None,
    ) -> Solution:
        k = k if k is not None else self.k
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        futuro = tuple(
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )

        # Memoria 1 — float32: columnas de la TPM bajo demanda
        self._ncubo_data = [
            ncubo.data.ravel().astype(np.float32)
            for ncubo in self.sia_subsistema.ncubos
        ]
        self._col_cache = {}

        self.vertices = set(presente + futuro)
        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial

        mip_result = self.find_mip(k)

        if k == 2:
            fmt_mip = fmt_biparte_q(list(mip_result), self.nodes_complement(mip_result))
            return Solution(
                estrategia=GEOMETRIC_LABEL,
                perdida=self.memoria_particiones[mip_result][0],
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.memoria_particiones[mip_result][1],
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                particion=fmt_mip,
            )
        else:
            grupos_locales, emd_val, dist = mip_result
            return Solution(
                estrategia=GEOMETRIC_LABEL + f" k={k}",
                perdida=emd_val,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist,
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                particion=self._fmt_particion_k(grupos_locales),
            )

    def nodes_complement(self, nodes) -> list:
        return list(set(self.vertices) - set(nodes))
