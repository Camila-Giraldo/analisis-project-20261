"""
Asignación de mecanismos, evaluación EMD y formato de k-particiones.

Responsabilidad única: transformar listas de índices locales en la
representación canónica (alcance, mecanismo) de IIT, evaluar su pérdida
con EMD real y formatear la partición como cadena imprimible.
"""

import numpy as np

from src.funcs.base import emd_efecto, ABECEDARY


class _PartitionMixin:

    def _asignar_mecanismos(self, grupos_locales: list) -> list[tuple]:
        """
        Construye la k-partición como lista de (alc_global, mec_global).

        Mecanismo de cada grupo = variables presentes cuyo índice global
        coincide con alguna variable futura del grupo (correspondencia natural
        en IIT). Las variables presentes sin par se distribuyen en round-robin
        empezando por el grupo de mayor tamaño.
        """
        fut_globals = self.sia_subsistema.indices_ncubos
        pres_set = {int(x) for x in self.sia_subsistema.dims_ncubos}

        grupos: list[tuple[list[int], list[int]]] = []
        asignadas: set[int] = set()
        for grupo in grupos_locales:
            if not grupo:
                continue
            g_arr = np.array(sorted(grupo), dtype=np.int8)
            alc = [int(x) for x in fut_globals[g_arr]]
            mec = sorted(pres_set & set(alc))
            asignadas |= set(mec)
            grupos.append((alc, mec))

        sin_asignar = sorted(pres_set - asignadas)
        orden = sorted(range(len(grupos)), key=lambda i: -len(grupos[i][0]))
        for j, p in enumerate(sin_asignar):
            grupos[orden[j % len(orden)]][1].append(p)

        return [
            (np.array(sorted(alc), dtype=np.int8),
             np.array(sorted(mec), dtype=np.int8))
            for alc, mec in grupos
        ]

    def _evaluar_particion_k(self, grupos_locales: list) -> tuple[float, np.ndarray]:
        """Evalúa EMD de una k-partición usando asignación correcta de mecanismos."""
        particion_k = self._asignar_mecanismos(grupos_locales)
        dist = self.sia_subsistema.bipartir_k(particion_k).distribucion_marginal()
        emd_val = emd_efecto(dist, self.sia_dists_marginales)
        return emd_val, dist

    def _score_proxy(self, grupos_locales: list) -> float:
        """
        Proxy O(n) para el EMD de una k-partición.

        score = Σ_i costos[i]  -  max_g(Σ_{i∈g} costos[i])

        Costo O(n_vars): no llama a bipartir_k ni a distribucion_marginal.
        Usado en _split_mejor_grupo y _hill_climbing para n_vars >= 25.
        """
        costos = self._get_costos_finales()
        group_sums = [
            float(costos[np.array(g, dtype=np.int64)].sum()) if g else 0.0
            for g in grupos_locales
        ]
        if not group_sums:
            return float("inf")
        return float(sum(group_sums)) - max(group_sums)

    def _fmt_particion_k(self, grupos_locales: list) -> str:
        """Formatea la k-partición igual que fmt_biparte_q pero para k grupos."""
        abecedary_lower = [a.lower() for a in ABECEDARY]
        particion_k = self._asignar_mecanismos(grupos_locales)

        fut_parts = []
        pres_parts = []
        for alc, mec in particion_k:
            fut_labels = [ABECEDARY[i] for i in alc if i < len(ABECEDARY)]
            pres_labels = [abecedary_lower[i] for i in mec if i < len(abecedary_lower)]
            fut_parts.append(",".join(fut_labels) if fut_labels else "∅")
            pres_parts.append(",".join(pres_labels) if pres_labels else "∅")

        top = "| " + " || ".join(fut_parts) + " |"
        bot = "| " + " || ".join(pres_parts) + " |"
        return top + "\n" + bot
