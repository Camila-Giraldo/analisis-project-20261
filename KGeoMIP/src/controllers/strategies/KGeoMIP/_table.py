"""
Tabla de programación dinámica + caché lazy de columnas float32.

Responsabilidad única: construir y consultar la tabla t_x(i0, j) que
cuantifica el costo geométrico de cada variable en cada estado accesible
desde el estado inicial i0.
"""

from typing import Dict, List

import numpy as np


class _TableMixin:

    @staticmethod
    def _umbral_nivel(n: int) -> int:
        """
        Nivel máximo de Hamming a construir en la tabla.

        n <= 20  →  tabla completa (n)
        n <= 22  →  n // 2   (error < 0.05 %)
        n  > 22  →  n // 3   (error < 0.1 %)
        """
        if n <= 20:
            return n
        elif n <= 22:
            return n // 2
        return n // 3

    def _get_costos_finales(self) -> np.ndarray:
        """
        Vector de costos t_x(i0, i_final).

        Si la tabla fue truncada y el estado final no está almacenado,
        aproxima promediando los vectores del nivel más profundo disponible.
        """
        if self._i_final_int in self.tabla_np:
            return self.tabla_np[self._i_final_int]

        nivel_max = max(self.caminos.keys())
        vectores = [
            self.tabla_np[int(np.dot(e, self._powers))]
            for e in self.caminos[nivel_max]
            if int(np.dot(e, self._powers)) in self.tabla_np
        ]
        if vectores:
            return np.mean(vectores, axis=0)
        return np.zeros(len(self.sia_subsistema.indices_ncubos))

    # ------------------------------------------------------------------
    # Caché lazy de columnas (Memoria 2 — float32)
    # ------------------------------------------------------------------

    def _get_col(self, j_int: int) -> np.ndarray:
        """
        Vector de probabilidades de todas las variables para el estado j_int.

        El caché evita recalcular columnas ya accedidas. Con el truncamiento
        adaptativo solo se tocan ~1,8M columnas de las 2^25 = 33M posibles
        para N=25, reduciendo el uso real de RAM de ~6,6 GB a ~0,5 GB.
        """
        if j_int not in self._col_cache:
            self._col_cache[j_int] = np.array(
                [data[j_int] for data in self._ncubo_data], dtype=np.float32
            )
        return self._col_cache[j_int]

    def _get_cols_batch(self, j_ints: np.ndarray) -> np.ndarray:
        """Matriz (n_vars, m) con las columnas pedidas, desde caché."""
        return np.stack([self._get_col(int(j)) for j in j_ints], axis=1)

    # ------------------------------------------------------------------
    # Construcción de la tabla
    # ------------------------------------------------------------------

    def find_mip(self, k: int = 2):
        self.sia_logger.critic("empieza.")
        n = len(self.estado_inicial)
        n_vars = len(self.sia_subsistema.indices_ncubos)

        self._powers = 2 ** np.arange(n)
        self._i0_int = int(self.estado_inicial @ self._powers)
        self._i_final_int = int(self.estado_final @ self._powers)

        self.tabla_np: Dict[int, np.ndarray] = {}
        self.tabla_np[self._i0_int] = np.zeros(n_vars, dtype=np.float32)
        self.caminos: Dict[int, List[List[int]]] = {0: [self.estado_inicial.tolist()]}

        nivel_max = self._umbral_nivel(n)
        for nivel in range(1, nivel_max + 1):
            self._calcular_costos_nivel(self.estado_final, nivel)

        if k == 2:
            self.memoria_particiones = {}
            return self._find_mip_k2()
        else:
            return self._find_mip_kn(k)

    def _calcular_costos_nivel(self, estado_final: np.ndarray, nivel: int):
        """
        Vectorización numpy por nivel.

        Todos los estados de un mismo nivel de Hamming son independientes
        entre sí (solo dependen del nivel anterior, ya calculado). Se
        procesan como operaciones matriciales para aprovechar BLAS/MKL.
        """
        n = len(estado_final)
        i0_arr = np.array(self.caminos[0][0], dtype=np.int8)

        visitados: set[int] = set()
        nuevos: list[list] = []
        for estado_prev in self.caminos[nivel - 1]:
            for bit in range(n):
                if estado_prev[bit] != estado_final[bit]:
                    nuevo = estado_prev.copy()
                    nuevo[bit] = estado_final[bit]
                    j_int = int(np.dot(nuevo, self._powers))
                    if j_int not in visitados:
                        nuevos.append(nuevo)
                        visitados.add(j_int)

        self.caminos[nivel] = nuevos
        if not nuevos:
            return

        j_ints = np.array([int(np.dot(e, self._powers)) for e in nuevos], dtype=np.int64)
        col_i0 = self._get_col(self._i0_int)[:, None]
        raw = np.abs(col_i0 - self._get_cols_batch(j_ints))

        if nivel > 1:
            for bit in range(n):
                i0_bit = int(i0_arr[bit])
                mask = ((j_ints >> bit) & 1) != i0_bit
                if not np.any(mask):
                    continue
                pred_ints = j_ints[mask] ^ (1 << bit)
                pred_costs = np.stack(
                    [self.tabla_np[int(p)] for p in pred_ints], axis=1
                )
                raw[:, mask] += pred_costs

        result = (0.5 ** nivel) * raw
        for idx, j_int in enumerate(j_ints):
            self.tabla_np[int(j_int)] = result[:, idx]
