import time
from typing import List, Set, Optional
import numpy as np

from src.models.base.sia import SIA
from src.controllers.manager import Manager

from .tabla_t import TablaT
from .warm_start import warm_start
from .beam_search import beam_search_local, calibrar_b
from .refinamiento import refinar_fronteras
from .score import score_global
from .utils import delta_interna, construir_tpm_estado_nodo, mejor_biparticion_delta


class GeometricSIAGeneralizada(SIA):
    """
    Estrategia geométrica generalizada para 2 ≤ k ≤ 5 y n ≤ 25.

    Extiende GeoMIP original (biparticiones) a k-particiones de variables
    mediante tres capas: tabla T adaptativa, warm start espectral y
    beam search con refinamiento de fronteras.

    Adaptaciones respecto al framework base
    ----------------------------------------
    - `aplicar_estrategia` no recibe parámetros de subsistema; opera sobre
      el sistema completo cargado desde el gestor.
    - `n` y la TPM se derivan de `gestor.estado_inicial` y `gestor.tpm_filename`.
    - Retorna dict en lugar de Solution para no acoplar al flujo EMD existente.

    Uso básico
    ----------
        gestor = Manager(estado_inicial="10000")
        res = GeometricSIAGeneralizada(gestor, k=3).aplicar_estrategia()
        print(res['particion'], res['phi'])
    """

    def __init__(self, gestor: Manager, k: int = 2, b: Optional[int] = None):
        super().__init__(gestor)
        if not (2 <= k <= 5):
            raise ValueError(f"k debe estar entre 2 y 5, recibido k={k}")
        self.k = k
        self.b = b
        # Derivar n del estado inicial (longitud de la cadena de bits)
        self.n = len(gestor.estado_inicial)

    def _cargar_tpm_nodo(self) -> np.ndarray:
        """
        Carga la TPM desde el archivo CSV del gestor y la convierte a
        formato estado-nodo (2^n × n) si es necesario.

        El proyecto almacena las TPMs ya en formato estado-nodo, así que
        `construir_tpm_estado_nodo` las devuelve sin transformación.
        """
        tpm_raw = np.genfromtxt(self.sia_gestor.tpm_filename, delimiter=",")
        return construir_tpm_estado_nodo(tpm_raw, self.n)

    def aplicar_estrategia(self) -> dict:
        """
        Ejecuta la estrategia completa:

        1. Tabla T adaptativa          (Capa 1)
        2. Warm start espectral        (Capa 2)
        3. Beam search local           (Capa 3a)
        4. Refinamiento de fronteras   (Capa 3b)
        5. Limpieza con regla de parada
        6. Retornar resultado

        Returns
        -------
        dict con claves:
            particion    : List[Set[int]] — k conjuntos de variables
            k_encontrado : int
            k_natural    : int
            phi          : float — score final
            score        : float — igual que phi
            delta_global : float
            tiempo       : float — segundos de ejecución
        """
        t0 = time.time()

        # 1. Tabla T
        tpm_nodo = self._cargar_tpm_nodo()
        T = TablaT(tpm_nodo, self.n)
        delta_global = delta_interna(set(range(self.n)), T, self.n)

        # 2. Warm start
        particion_inicial, k_natural = warm_start(self.n, self.k, T)

        # 3. Beam search
        b = self.b or calibrar_b(self.n, self.k)
        particion_candidata = beam_search_local(
            particion_inicial, T, self.k, self.n, b, delta_global
        )

        # 4. Refinamiento
        particion_refinada = refinar_fronteras(
            particion_candidata, T, self.k, self.n,
            max_iter=self.n * self.k,
        )

        # 5. Limpieza
        particion_final = self._limpiar(particion_refinada, T, delta_global)

        score_final = score_global(particion_final, T, self.k, self.n)
        return {
            'particion':    particion_final,
            'k_encontrado': len(particion_final),
            'k_natural':    k_natural,
            'phi':          score_final,
            'score':        score_final,
            'delta_global': delta_global,
            'tiempo':       time.time() - t0,
        }

    def _limpiar(
        self, particion: List[Set[int]], T: TablaT, delta_global: float
    ) -> List[Set[int]]:
        """Fusiona particiones triviales que no deberían haberse separado."""
        cambio = True
        while cambio and len(particion) > 2:
            cambio = False
            for i, Si in enumerate(particion):
                if not Si:
                    particion.pop(i)
                    cambio = True
                    break
                d_star = mejor_biparticion_delta(Si, T, self.n)
                if d_star < 0.01 * delta_global and len(particion) > self.k:
                    j = min(
                        (jj for jj in range(len(particion)) if jj != i),
                        key=lambda jj: delta_interna(
                            Si | particion[jj], T, self.n
                        ),
                    )
                    particion[j] |= particion[i]
                    particion.pop(i)
                    cambio = True
                    break
        return [p for p in particion if p]


# Alias oficial: convenio de nomenclatura K-QGMIP
KQNodes = GeometricSIAGeneralizada
