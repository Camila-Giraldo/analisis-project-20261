"""
Helpers autocontenidos de la suite KQNodes (reemplazan a los antiguos ``fase*.py``).

Responsabilidad única: utilidades de referencia/medición compartidas por los
módulos ``test_kqnodes_*.py``. No contiene tests; el prefijo ``_`` evita que
pytest lo recolecte.

La referencia de fuerza bruta (``kbruteforce``) evalúa cada k-partición con la
distribución completa (``_distribucion_kparticion`` + ``emd_efecto``), un camino
de código DISTINTO al de la búsqueda por separabilidad (coste factorizado). Que
ambos coincidan cross-valida tanto el teorema de separabilidad como la búsqueda.
"""

import math

from src.controllers.manager import Manager
from src.strategies.KQNodes import KQNodes
from src.strategies.force import BruteForce
from src.funcs.iit import emd_efecto
from src.constants.base import ACTUAL, EFFECT

TOL = 1e-5  # tolerancia float32


def stirling2(n: int, k: int) -> int:
    """Número de Stirling de 2.ª especie S(n, k) por recurrencia (DP)."""
    if k < 0 or k > n:
        return 0
    if n == k == 0:
        return 1
    if k == 0 or n == 0:
        return 0
    fila = [0] * (k + 1)
    fila[0] = 1
    for _ in range(1, n + 1):
        for j in range(min(_, k), 0, -1):
            fila[j] = j * fila[j] + fila[j - 1]
        fila[0] = 0
    return fila[k]


def set_partitions_exact(elementos: list, k: int):
    """Genera todas las particiones de ``elementos`` en EXACTAMENTE k bloques no
    vacíos. Implementación local independiente de ``src`` (referencia de prueba)."""
    n = len(elementos)
    if k < 1 or k > n:
        return
    if k == 1:
        yield [list(elementos)]
        return
    if k == n:
        yield [[e] for e in elementos]
        return
    primero, resto = elementos[0], elementos[1:]
    for parte in set_partitions_exact(resto, k - 1):
        yield [[primero]] + parte
    for parte in set_partitions_exact(resto, k):
        for i in range(len(parte)):
            yield parte[:i] + [[primero] + parte[i]] + parte[i + 1:]


def build_kq(tpm, bits: int, alc: str, mec: str, estado: str | None = None) -> KQNodes:
    """Crea un KQNodes con el subsistema ya preparado."""
    estado = estado or ("1" + "0" * (bits - 1))
    cond = "1" * bits
    kq = KQNodes(tpm)
    kq.sia_preparar_subsistema(estado, cond, alc, mec)
    return kq


def vertices(kq: KQNodes) -> list[tuple[int, int]]:
    """Conjunto base de vértices bitemporales V = futuros ∪ presentes."""
    futuros = [(EFFECT, int(c.indice)) for c in kq.sia_subsistema.ncubos]
    presentes = [(ACTUAL, int(d)) for d in kq.sia_subsistema.dims_ncubos]
    return futuros + presentes


def kbruteforce(kq: KQNodes, k: int) -> float:
    """Pérdida mínima EXACTA por fuerza bruta sobre todas las k-particiones de V.

    Evalúa cada candidato con la distribución completa + EMD-efecto (camino de
    código independiente de la búsqueda por separabilidad)."""
    base = kq.sia_dists_marginales
    mejor = math.inf
    for particion in set_partitions_exact(vertices(kq), k):
        dist = kq._distribucion_kparticion(particion)
        mejor = min(mejor, float(emd_efecto(dist, base)))
    return mejor


def kq_loss(tpm, bits, alc, mec, k, metodo="exacto", estado=None, **kw) -> float:
    """Pérdida de KQNodes con el método indicado."""
    estado = estado or ("1" + "0" * (bits - 1))
    cond = "1" * bits
    sol = KQNodes(tpm).aplicar_estrategia(estado, cond, alc, mec, k, metodo=metodo, **kw)
    return float(sol.perdida)


def bf_loss(tpm, bits, alc, mec) -> float:
    """Pérdida de la bipartición exacta (BruteForce)."""
    estado = "1" + "0" * (bits - 1)
    cond = "1" * bits
    return float(BruteForce(tpm).aplicar_estrategia(estado, cond, alc, mec).perdida)


def nueva_red(bits: int):
    """TPM ad-hoc para tamaños no cubiertos por las fixtures de conftest."""
    return Manager("1" + "0" * (bits - 1)).cargar_red()
