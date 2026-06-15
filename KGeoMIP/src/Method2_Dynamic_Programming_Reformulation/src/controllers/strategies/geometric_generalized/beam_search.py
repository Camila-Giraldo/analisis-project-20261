from math import ceil, log2
from typing import List, Set, Optional
from .tabla_t import TablaT
from .regla_parada import debe_parar
from .score import score_global
from .utils import delta_interna, generar_biparticiones_candidatas


def calibrar_b(n: int, k: int) -> int:
    """
    Calibra b automáticamente para mantener costo computacional constante.
    PRESUPUESTO = número máximo de evaluaciones por nivel del árbol.
    """
    PRESUPUESTO = 2 ** 15
    tam_sub = max(1, n // max(k, 1))
    costo = 2 ** min(tam_sub, 15)
    b_max = max(1, PRESUPUESTO // costo)

    if n <= 15:
        b_base = min(b_max, 5)
    elif n <= 20:
        b_base = min(b_max, 3)
    elif n <= 22:
        b_base = min(b_max, 2)
    else:
        b_base = 1

    return max(1, b_base // max(1, k // 3))


def beam_search_local(
    particion_inicial: List[Set[int]],
    T: TablaT,
    k: int,
    n: int,
    b: Optional[int] = None,
    delta_global: float = 1.0,
) -> List[Set[int]]:
    """
    Beam search sobre árbol de subdivisiones.

    Selecciona la parte con mayor discrepancia interna, genera sus
    biparticiones candidatas y conserva las b mejores particiones del haz.
    Para n=25 con b=1: greedy, compensado por refinamiento posterior.
    """
    if b is None:
        b = calibrar_b(n, k)

    max_sub = min(15, max(3, n // max(k - 1, 1)))
    haz = [particion_inicial]
    prof_max = ceil(log2(max(k, 2))) + 2

    for _ in range(prof_max):
        if haz and len(haz[0]) >= k:
            break

        nuevos: List[List[Set[int]]] = []

        for particion in haz:
            Si = max(particion, key=lambda s: delta_interna(s, T, n))
            k_rest = k - len(particion)

            if debe_parar(Si, k_rest, T, n, delta_global):
                nuevos.append(particion)
                continue

            candidatos = (
                _subproblemas(Si, T, n, max_sub)
                if len(Si) > max_sub
                else generar_biparticiones_candidatas(Si, T, n)
            )

            umbral = delta_interna(Si, T, n) * 0.5
            candidatos = [
                (s1, s2) for s1, s2 in candidatos
                if s1 and s2
                and delta_interna(s1, T, n) + delta_interna(s2, T, n) >= umbral
            ]

            for s1, s2 in candidatos:
                nueva = [p for p in particion if p != Si]
                if s1:
                    nueva.append(s1)
                if s2:
                    nueva.append(s2)
                nuevos.append(nueva)

        if not nuevos:
            break

        haz = sorted(nuevos, key=lambda p: score_global(p, T, k, n))[:b]

    return haz[0] if haz else particion_inicial


def _subproblemas(Si: Set[int], T, n: int, max_sub: int) -> list:
    """Para Si grande, genera candidatos por bloques manejables."""
    vars_ = list(Si)
    candidatos = []
    for inicio in range(0, len(vars_), max_sub):
        bloque = set(vars_[inicio:inicio + max_sub])
        resto = Si - bloque
        if len(bloque) >= 2:
            for s1, s2 in generar_biparticiones_candidatas(bloque, T, n):
                candidatos.append((s1, s2 | resto))
    return candidatos or [(Si, set())]
