from typing import List, Set
from .tabla_t import TablaT
from .utils import delta_interna


def score_global(
    particion: List[Set[int]], T: TablaT, k: int, n: int
) -> float:
    """
    Evalúa la calidad de una k-partición. Menor score = mejor partición.

    k=2 → minimiza discrepancia total sin penalización.
    k>2 → penaliza desbalance entre partes con factor (k-2)/k.
    """
    if not particion:
        return float('inf')

    deltas = [delta_interna(Si, T, n) for Si in particion]
    delta_total = sum(deltas)

    if k <= 2 or delta_total < 1e-10:
        return delta_total

    delta_max = max(deltas)
    delta_min = min(deltas)
    desbalance = (delta_max - delta_min) / (delta_total + 1e-10)
    penalizacion = desbalance * (k - 2) / k

    return delta_total * (1.0 + penalizacion)
