from typing import Set
from .tabla_t import TablaT
from .utils import delta_interna, mejor_biparticion_delta

ALPHA = 0.05    # fracción de δ_global como umbral absoluto
BETA = 0.05     # ganancia relativa mínima para justificar subdivisión
MIN_SIZE = 2    # tamaño mínimo para subdividir


def debe_parar(
    Si: Set[int],
    k_restante: int,
    T: TablaT,
    n: int,
    delta_global: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    min_size: int = MIN_SIZE,
) -> bool:
    """
    Retorna True si NO se debe subdividir Si.

    Criterios (por prioridad):
    1. DURO: k_restante == 0
    2. DURO: |Si| < min_size
    3. ADAPT: ganancia < β · δ_interna  → subdivisión no aporta
    4. ADAPT: δ*(Si) < α · δ_global    → Si es causalmente cohesiva
    """
    if k_restante <= 0:
        return True
    if len(Si) < min_size:
        return True

    delta_si = delta_interna(Si, T, n)
    delta_star = mejor_biparticion_delta(Si, T, n)

    ganancia = delta_si - delta_star
    if delta_si > 1e-10 and ganancia / delta_si < beta:
        return True

    if delta_star < alpha * delta_global:
        return True

    return False
