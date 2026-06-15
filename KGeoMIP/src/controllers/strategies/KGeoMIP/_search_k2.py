"""
Búsqueda de la bipartición de mínima pérdida (k = 2).

Responsabilidad única: generar candidatos de bipartición usando la tabla DP
y evaluarlos con EMD real mediante ThreadPoolExecutor.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.funcs.base import emd_efecto


class _SearchK2Mixin:

    def _find_mip_k2(self):
        """Genera y evalúa candidatos de bipartición (k=2)."""
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_pres = len(self.sia_subsistema.dims_ncubos)

        candidatos: list[tuple[list, list]] = []
        for idx in range(n_vars):
            presentes = list(range(n_pres))
            futuros = [i for i in range(n_vars) if i != idx]
            candidatos.append((presentes, futuros))

        es_par = len(self.caminos) % 2 == 0
        mitad = len(self.caminos) // 2 if es_par else (len(self.caminos) // 2) + 1

        for nivel in range(1, mitad):
            costo_mejor = 1e5
            p_mejor: list = []
            f_mejor: list = []
            for estado in self.caminos[nivel]:
                i0 = self.caminos[0][0]
                estado_int = int(np.dot(estado, self._powers))
                estado_comp = (1 - np.array(estado)).tolist()
                comp_int = int(np.dot(estado_comp, self._powers))

                actual = self.tabla_np.get(estado_int)
                complementario = self.tabla_np.get(comp_int)
                if actual is None or complementario is None:
                    continue

                presentes = [i for i, (s, i0b) in enumerate(zip(estado, i0)) if s == i0b]
                futuros: list[int] = []
                costo = 0.0
                for idx in range(n_vars):
                    if actual[idx] <= complementario[idx]:
                        futuros.append(idx)
                        costo += actual[idx]
                    else:
                        costo += complementario[idx]

                if costo < costo_mejor:
                    costo_mejor = costo
                    p_mejor = presentes
                    f_mejor = futuros

            if f_mejor:
                candidatos.append((p_mejor, f_mejor))

        top_candidatos = [(p, f) for p, f in candidatos if f]
        _n_workers = min(len(top_candidatos), os.cpu_count() or 4)

        def _eval(presentes_local, futuros_local):
            p_g = self.sia_subsistema.dims_ncubos[
                np.array(presentes_local, dtype=np.int8)
            ]
            f_g = self.sia_subsistema.indices_ncubos[
                np.array(futuros_local, dtype=np.int8)
            ]
            dist = self.sia_subsistema.bipartir(f_g, p_g).distribucion_marginal()
            emd_val = emd_efecto(dist, self.sia_dists_marginales)
            key = tuple([(0, v) for v in p_g] + [(1, v) for v in f_g])
            return key, emd_val, dist

        with ThreadPoolExecutor(max_workers=_n_workers) as pool:
            futuros_exec = [
                pool.submit(_eval, p, f) for p, f in top_candidatos
            ]
            for fut in as_completed(futuros_exec):
                res = fut.result()
                if res is not None:
                    key, emd_val, dist = res
                    self.memoria_particiones[key] = (emd_val, dist)

        return min(self.memoria_particiones, key=lambda kk: self.memoria_particiones[kk][0])
