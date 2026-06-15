"""
Mixin de evaluación de grupos para KQNodes.

Contiene las rutinas del lazo caliente: la versión escalar ``_delta_grupos``
(invocada S(n,r) veces en el exacto y miles de veces en las heurísticas) y
``_evaluar_grupos`` (llamada una única vez sobre el candidato ganador para
reconstruir la k-partición completa).
"""

from typing import Optional

from src.constants.base import ACTUAL, EFFECT, INFTY_POS
from ._partitions import VACIO


class _EvaluationMixin:
    """Evaluación de particiones por separabilidad. Requiere ``_costo``."""

    def _delta_grupos(
        self,
        grupos: list[frozenset],
        k: int,
        cubos_futuro: list,
        base,
    ) -> float:
        """
        Versión SOLO-ESCALAR: devuelve únicamente δ_k sin reconstruir bloques.

        Rutina del lazo caliente. Produce exactamente el mismo δ_k que
        ``_evaluar_grupos(...)[0]`` (misma asignación voraz y cubrimiento de
        bloques vacíos).
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
                costo_vacio = self._costo(cubo, base_i, VACIO)
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
        base,
    ) -> tuple[float, Optional[list[list[tuple[int, int]]]]]:
        """
        Evalúa una partición y reconstruye la k-partición completa.

        Asigna cada futuro a su bloque de mínimo coste y cubre los bloques
        vacíos forzando los futuros más baratos.

        Returns:
            (δ_k, bloques) o (INFTY_POS, None) si la partición es inviable.
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
                costo_vacio = self._costo(cubo, base_i, VACIO)
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
        """Convierte una etiquetación presente→bloque en lista de grupos no vacíos."""
        por_bloque: dict[int, set] = {}
        for pos, lab in enumerate(etiquetas):
            por_bloque.setdefault(lab, set()).add(presentes[pos])
        return [frozenset(s) for s in por_bloque.values()]

    def _reconstruir_bloques(
        self,
        grupos: list[set],
        asignacion: list[int],
        en_vacio: list[int],
        vacios: int,
        cubos_futuro: list,
    ) -> list[list[tuple[int, int]]]:
        """Construye la k-partición (bloques de vértices) desde la asignación óptima."""
        bloques: list[list[tuple[int, int]]] = []

        for g_idx, grupo in enumerate(grupos):
            bloque: list[tuple[int, int]] = [(ACTUAL, p) for p in sorted(grupo)]
            for idx, g in enumerate(asignacion):
                if g == g_idx:
                    bloque.append((EFFECT, int(cubos_futuro[idx].indice)))
            bloques.append(bloque)

        if vacios > 0:
            bloques_vacios: list[list[tuple[int, int]]] = [[] for _ in range(vacios)]
            for b in range(vacios):
                bloques_vacios[b].append((EFFECT, int(cubos_futuro[en_vacio[b]].indice)))
            for idx in en_vacio[vacios:]:
                bloques_vacios[0].append((EFFECT, int(cubos_futuro[idx].indice)))
            bloques.extend(bloques_vacios)

        return bloques
