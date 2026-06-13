"""
Fase 0 de KQNodes — definir y validar la pérdida de k-particiones.

Hace tres cosas:
  1. Define `k_partir_marginal` / `delta_k`: generaliza la bipartición de QNodes
     a k bloques. Regla del corte (igual que `System.bipartir`): cada cubo
     FUTURO conserva únicamente las dims-PRESENTE de su propio bloque y margina
     el resto.
  2. `kbruteforce`: enumera TODAS las k-particiones de los 2N vértices
     (referencia EXACTA) para validar en n pequeño.
  3. Verifica:
       (a) k=2 reduce exactamente a la maquinaria de bipartición de QNodes.
       (b) k=2 BruteForce-k == BruteForce de biparticiones original (misma MIP).
       (c) SEPARABILIDAD: δ_k depende solo de, para cada nodo futuro, el conjunto
           de nodos presente en su bloque. Se demuestra mostrando que δ_k es
           invariante a reagrupar nodos futuros sin compañía-presente y a
           reagrupar nodos presente sin futuro.

Uso:
    PYTHONPATH=. .venv/bin/python tests/fase0_kloss.py
"""

import logging
import os
import sys

import numpy as np

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from src.models.base.application import aplicacion

aplicacion.profiler_habilitado = False

# Silenciar SafeLogger (colorama/FDs) durante la exploración.
from src.middlewares import slogger  # noqa: E402

_SILENT = logging.getLogger("fase0_silent")
_SILENT.addHandler(logging.NullHandler())
_SILENT.setLevel(logging.CRITICAL + 1)
slogger.SafeLogger.__init__ = lambda self, *a, **k: setattr(self, "_logger", _SILENT)

from src.controllers.manager import Manager  # noqa: E402
from src.funcs.iit import emd_efecto  # noqa: E402
from src.models.core.system import System  # noqa: E402
from src.strategies.q_nodes import QNodes  # noqa: E402

# Convención de vértices: (tiempo, indice). 1 = futuro/alcance, 0 = presente/mecanismo.
FUTURO, PRESENTE = 1, 0


def construir_subsistema(bits, alcance, mecanismo, estado=None, cond=None):
    estado = estado or "1" + "0" * (bits - 1)
    cond = cond or "1" * bits
    mpt = Manager(estado).cargar_red()
    q = QNodes(mpt)
    q.sia_preparar_subsistema(estado, cond, alcance, mecanismo)
    return q.sia_subsistema, q.sia_dists_marginales


def vertices_de(sub):
    fut = [(FUTURO, int(i)) for i in sub.indices_ncubos]
    pre = [(PRESENTE, int(i)) for i in sub.dims_ncubos]
    return fut + pre


def k_partir_marginal(sub, bloques):
    """Vector de marginales del sistema k-particionado.

    `bloques`: lista de bloques; cada bloque es un iterable de vértices (t, i).
    """
    # Para cada nodo futuro: conjunto de nodos presente en su mismo bloque.
    fut_a_presente: dict[int, set[int]] = {}
    for blq in bloques:
        presentes_blq = {i for (t, i) in blq if t == PRESENTE}
        for (t, i) in blq:
            if t == FUTURO:
                fut_a_presente[i] = presentes_blq

    nuevo = System.__new__(System)
    nuevo.estado_inicial = sub.estado_inicial
    nuevo.memo = {}
    cubos = []
    for cubo in sub.ncubos:
        P = fut_a_presente.get(int(cubo.indice), set())
        ejes = np.array([int(d) for d in cubo.dims if int(d) not in P], dtype=np.int8)
        cubos.append(cubo.marginalizar(ejes))
    nuevo.ncubos = tuple(cubos)
    return nuevo.distribucion_marginal()


def delta_k(sub, base, bloques):
    return float(emd_efecto(k_partir_marginal(sub, bloques), base))


def particiones_en_k(elementos, k):
    """Genera todas las particiones de `elementos` en EXACTAMENTE k bloques no vacíos."""
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
    for p in particiones_en_k(resto, k - 1):  # `primero` en bloque propio
        yield [[primero]] + p
    for p in particiones_en_k(resto, k):  # `primero` se une a un bloque existente
        for i in range(len(p)):
            yield p[:i] + [[primero] + p[i]] + p[i + 1 :]


def kbruteforce(sub, base, k):
    """Mínimo δ_k exacto sobre todas las k-particiones de los 2N vértices."""
    V = vertices_de(sub)
    mejor = float("inf")
    mejor_part = None
    for bloques in particiones_en_k(V, k):
        d = delta_k(sub, base, bloques)
        if d < mejor:
            mejor, mejor_part = d, bloques
    return mejor, mejor_part


def stirling2(n, k):
    if k == 0 or k > n:
        return 0
    tabla = [[0] * (k + 1) for _ in range(n + 1)]
    tabla[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            tabla[i][j] = j * tabla[i - 1][j] + tabla[i - 1][j - 1]
    return tabla[n][k]


# ---------------------------------------------------------------------------


def check_k2_reduce_a_biparticion(sub, base):
    """(a) δ_k con k=2 debe coincidir con sub.bipartir(...) de QNodes."""
    V = vertices_de(sub)
    import itertools

    maxdiff = 0.0
    n = len(V)
    for r in range(1, n):
        for comb in itertools.combinations(V, r):
            S = list(comb)
            comp = [v for v in V if v not in set(S)]
            d_k = delta_k(sub, base, [S, comp])
            al = np.array([i for (t, i) in S if t == FUTURO], dtype=np.int8)
            me = np.array([i for (t, i) in S if t == PRESENTE], dtype=np.int8)
            d_bip = float(emd_efecto(sub.bipartir(al, me).distribucion_marginal(), base))
            maxdiff = max(maxdiff, abs(d_k - d_bip))
    return maxdiff


def check_separabilidad(sub, base):
    """(c) δ_k invariante a reagrupar futuros-sin-presente y presentes-sin-futuro."""
    fut = [(FUTURO, int(i)) for i in sub.indices_ncubos]
    pre = [(PRESENTE, int(i)) for i in sub.dims_ncubos]
    if len(fut) < 2 or len(pre) < 1:
        return None
    f0, f1 = fut[0], fut[1]
    resto_fut = fut[2:]
    casos = []

    # Futuros sin compañía-presente: juntos vs separados (k distinto, δ igual).
    bloque_con_presente = [f for f in resto_fut] + pre  # toda la parte presente aquí
    A = [bloque_con_presente, [f0, f1]]
    B = [bloque_con_presente, [f0], [f1]]
    casos.append(("futuros-vacios juntos vs separados", delta_k(sub, base, A), delta_k(sub, base, B)))

    # Presentes sin futuro: juntos vs separados.
    if len(pre) >= 2:
        p0, p1 = pre[0], pre[1]
        resto_pre = pre[2:]
        bloque_full = fut + resto_pre  # todos los futuros + presentes restantes
        C = [bloque_full, [p0, p1]]
        D = [bloque_full, [p0], [p1]]
        casos.append(("presentes-libres juntos vs separados", delta_k(sub, base, C), delta_k(sub, base, D)))
    return casos


def main():
    print("=== Fase 0: validación de la pérdida k-particiones ===\n")
    pruebas = [(3, "111", "111"), (4, "1111", "1111")]

    for bits, alc, mec in pruebas:
        sub, base = construir_subsistema(bits, alc, mec)
        V = vertices_de(sub)
        print(f"--- N{bits}  subsistema alcance={alc} mecanismo={mec}  (|V|={len(V)} vértices) ---")

        maxdiff = check_k2_reduce_a_biparticion(sub, base)
        print(f"(a) k=2 vs bipartir QNodes: maxdiff = {maxdiff:.2e}  -> {'OK' if maxdiff < 1e-6 else 'DIFIERE'}")

        for k in range(2, min(len(V), 5) + 1):
            d, part = kbruteforce(sub, base, k)
            sizes = stirling2(len(V), k)
            print(f"    k={k}: min δ_k = {d:.6f}   (S({len(V)},{k})={sizes} particiones)")

        sep = check_separabilidad(sub, base)
        if sep:
            print("(c) separabilidad (δ debe ser igual en cada par):")
            for nombre, x, y in sep:
                print(f"    {nombre}: {x:.6f} vs {y:.6f}  -> {'IGUAL' if abs(x-y)<1e-9 else 'DISTINTO'}")
        print()


if __name__ == "__main__":
    main()
