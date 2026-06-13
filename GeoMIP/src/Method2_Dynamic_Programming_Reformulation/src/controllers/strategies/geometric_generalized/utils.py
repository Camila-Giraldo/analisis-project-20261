import numpy as np
from itertools import combinations
from typing import Set, List, Tuple
from numpy.typing import NDArray


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def construir_tpm_estado_nodo(
    tpm_estado_estado: NDArray[np.float64],
    n: int,
) -> NDArray[np.float64]:
    """
    Convierte TPM estado-estado (2^n × 2^n) a estado-nodo (2^n × n).
    Si la entrada ya tiene forma (2^n, n) la devuelve sin cambios.
    """
    num_estados = 2 ** n
    if tpm_estado_estado.shape == (num_estados, n):
        return tpm_estado_estado.astype(np.float64)

    tpm_nodo = np.zeros((num_estados, n))
    for estado in range(num_estados):
        fila = tpm_estado_estado[estado]
        for var in range(n):
            prob_cero = sum(
                fila[j]
                for j in range(num_estados)
                if not (j >> (n - 1 - var)) & 1
            )
            tpm_nodo[estado, var] = prob_cero
    return tpm_nodo


def delta_interna(Si: Set[int], T, n: int) -> float:
    """
    Discrepancia interna de Si (ec. 1.1 del doc. GeoMIP aplicada localmente).
    Retorna 0 para |Si| <= 1.
    """
    if len(Si) <= 1:
        return 0.0

    variables = sorted(Si)
    total = 0.0
    conteo = 0
    for i, vi in enumerate(variables):
        for vj in variables[i + 1:]:
            total += T.costos_cruzados(vi, vj)
            conteo += 1
    return total / conteo if conteo > 0 else 0.0


def mejor_biparticion_delta(Si: Set[int], T, n: int) -> float:
    """Mínima discrepancia alcanzable al bipartir Si (delta* del documento)."""
    if len(Si) <= 1:
        return 0.0

    variables = list(Si)
    mejor = float('inf')

    if len(variables) <= 10:
        for r in range(1, len(variables) // 2 + 1):
            for combo in combinations(variables, r):
                s1 = set(combo)
                s2 = Si - s1
                d = delta_interna(s1, T, n) + delta_interna(s2, T, n)
                if d < mejor:
                    mejor = d
    else:
        for s1, s2 in generar_biparticiones_candidatas(Si, T, n):
            d = delta_interna(s1, T, n) + delta_interna(s2, T, n)
            if d < mejor:
                mejor = d
    return mejor


def generar_biparticiones_candidatas(
    Si: Set[int], T, n: int, max_candidatos: int = 20
) -> List[Tuple[Set[int], Set[int]]]:
    variables = list(Si)
    m = len(variables)
    if m < 2:
        return []

    candidatos = []
    if m <= 8:
        for r in range(1, m // 2 + 1):
            for combo in combinations(variables, r):
                s1 = set(combo)
                s2 = Si - s1
                if s1 and s2:
                    score = _score_biparticion(s1, s2, T)
                    candidatos.append((score, s1, s2))
    else:
        costos_var = {
            v: sum(T.costos_cruzados(v, u) for u in variables if u != v)
            for v in variables
        }
        variables_ord = sorted(variables, key=lambda v: costos_var[v])
        for corte in range(1, min(m, max_candidatos)):
            s1 = set(variables_ord[:corte])
            s2 = Si - s1
            score = _score_biparticion(s1, s2, T)
            candidatos.append((score, s1, s2))

    candidatos.sort(key=lambda x: x[0], reverse=True)
    return [(s1, s2) for _, s1, s2 in candidatos[:max_candidatos]]


def _score_biparticion(s1: Set[int], s2: Set[int], T) -> float:
    costo_cruzado = sum(
        T.costos_cruzados(v1, v2) for v1 in s1 for v2 in s2
    ) / max(len(s1) * len(s2), 1)

    costo_int_s1 = sum(
        T.costos_cruzados(v1, v2) for v1, v2 in combinations(s1, 2)
    ) / max(len(s1), 1)

    costo_int_s2 = sum(
        T.costos_cruzados(v1, v2) for v1, v2 in combinations(s2, 2)
    ) / max(len(s2), 1)

    return (costo_int_s1 + costo_int_s2) - 2 * costo_cruzado


def variables_en_frontera(particion: List[Set[int]], k: int) -> List[int]:
    todas = []
    for Si in particion:
        todas.extend(list(Si))
    return todas
