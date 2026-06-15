"""
Búsqueda de la k-partición de mínima pérdida para k > 2.

Responsabilidad única: implementar el greedy jerárquico (Paso 2) y el
hill climbing (Paso 3) que refinan la bipartición base obtenida por
_SearchK2Mixin hasta obtener k grupos.
"""

from typing import Optional

import numpy as np


class _SearchKNMixin:

    def _find_mip_kn(self, k: int) -> tuple:
        """
        Encuentra la k-partición de menor pérdida mediante:

        Paso 2 — Greedy jerárquico:
            Parte de la mejor bipartición (k=2) y divide sucesivamente
            el grupo mayor usando la tabla de costos como guía.

        Paso 3 — Hill climbing:
            Reasigna variables de un grupo a otro mientras la pérdida
            (EMD) disminuya. Converge en O(n × k) iteraciones.
        """
        self.memoria_particiones = {}
        # No se pueden formar más grupos no vacíos que variables disponibles.
        # Sin este tope, pedir k > n_vars deja el `while` girando para siempre
        # porque _split_mejor_grupo no puede dividir grupos de tamaño 1.
        k = min(k, len(self.sia_subsistema.indices_ncubos))

        mip_k2 = self._find_mip_k2()
        grupos = self._grupos_desde_mip_k2(mip_k2)

        while len(grupos) < k:
            nuevos = self._split_mejor_grupo(grupos)
            if len(nuevos) <= len(grupos):
                # El split no progresó (grupo mayor ya es singleton): cortar
                # para no quedar en bucle infinito.
                break
            grupos = nuevos

        grupos = self._hill_climbing(grupos)

        emd_val, dist = self._evaluar_particion_k(grupos)
        return grupos, emd_val, dist

    def _grupos_desde_mip_k2(self, mip_key: tuple) -> list[list[int]]:
        """
        Convierte la clave del MIP k=2 (índices globales) a dos listas
        de índices locales (posición en indices_ncubos).
        """
        fut_globals = list(self.sia_subsistema.indices_ncubos.astype(int))
        futuros_g1 = {x for tipo, x in mip_key if tipo == 1}
        futuros_g2 = {x for x in fut_globals if x not in futuros_g1}

        def to_local(global_set: set) -> list[int]:
            return [i for i, g in enumerate(fut_globals) if g in global_set]

        g1 = to_local(futuros_g1)
        g2 = to_local(futuros_g2)
        return [g for g in [g1, g2] if g]

    def _split_mejor_grupo(self, grupos: list[list[int]]) -> list[list[int]]:
        """
        Divide el grupo con más variables evaluando todos los n-1 cortes.

        Para n_vars < 25 usa EMD exacto (bipartir_k).
        Para n_vars >= 25 usa el proxy geométrico O(n).
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        use_exact = n_vars < 25

        idx_mayor = max(range(len(grupos)), key=lambda i: len(grupos[i]))
        grupo = grupos[idx_mayor]

        if len(grupo) < 2:
            return grupos

        costos_full = self._get_costos_finales()
        costos_sub = costos_full[np.array(grupo)]
        orden_local = np.argsort(costos_sub)
        resto = [g for i, g in enumerate(grupos) if i != idx_mayor]

        mejor_score = float("inf")
        mejor_grupos: Optional[list] = None

        for corte in range(1, len(grupo)):
            parte1 = [grupo[orden_local[i]] for i in range(corte)]
            parte2 = [grupo[orden_local[i]] for i in range(corte, len(grupo))]
            candidato = resto + [parte1, parte2]
            score = (
                self._evaluar_particion_k(candidato)[0]
                if use_exact
                else self._score_proxy(candidato)
            )
            if score < mejor_score:
                mejor_score = score
                mejor_grupos = candidato

        return mejor_grupos if mejor_grupos is not None else grupos

    def _hill_climbing(self, grupos: list[list[int]]) -> list[list[int]]:
        """
        Refinamiento local de la k-partición por first-improvement.

        Para n_vars < 25 evalúa cada movimiento con EMD exacto (bipartir_k).
        Para n_vars >= 25 usa el proxy geométrico O(n).
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        use_exact = n_vars < 25

        mejor_score = (
            self._evaluar_particion_k(grupos)[0]
            if use_exact
            else self._score_proxy(grupos)
        )

        mejoro = True
        while mejoro:
            mejoro = False
            for i in range(len(grupos)):
                if len(grupos[i]) <= 1:
                    continue
                for var in list(grupos[i]):
                    for j in range(len(grupos)):
                        if i == j:
                            continue
                        nueva = [list(g) for g in grupos]
                        nueva[i].remove(var)
                        nueva[j].append(var)
                        score = (
                            self._evaluar_particion_k(nueva)[0]
                            if use_exact
                            else self._score_proxy(nueva)
                        )
                        if score < mejor_score - 1e-9:
                            mejor_score = score
                            grupos = nueva
                            mejoro = True
                            break
                    if mejoro:
                        break
                if mejoro:
                    break

        return grupos
