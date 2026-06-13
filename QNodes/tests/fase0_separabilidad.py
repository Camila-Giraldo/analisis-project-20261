"""
Fase 0 — explorar la SEPARABILIDAD para un algoritmo exacto barato de k-MIP.

Hipótesis: como δ_k = Σ_i c_i(P_i), donde c_i(Q) es el coste del nodo futuro i
cuando los presentes de su bloque son Q, el k-MIP se puede resolver EXACTAMENTE
así:
  1. Particionar SOLO los nodos presente en r grupos (r = 1..k).
  2. Añadir (k - r) bloques "vacíos de presente" (present-set = ∅).
  3. Cada nodo futuro elige el bloque (grupo de presentes o ∅) que minimiza su
     coste; con la restricción de que cada bloque vacío reciba >=1 futuro.

Esto enumera Σ_r S(n_presente, r) candidatos en vez de S(2N, k). Validamos que
da el MISMO mínimo que `kbruteforce` (referencia exacta de la Fase 0) y medimos
la reducción de candidatos.

Uso:
    PYTHONPATH=. .venv/bin/python tests/fase0_separabilidad.py
"""

import logging
import os
import sys

import numpy as np

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from src.models.base.application import aplicacion  # noqa: E402

aplicacion.profiler_habilitado = False
from src.middlewares import slogger  # noqa: E402

_S = logging.getLogger("sep_silent")
_S.addHandler(logging.NullHandler())
_S.setLevel(logging.CRITICAL + 1)
slogger.SafeLogger.__init__ = lambda self, *a, **k: setattr(self, "_logger", _S)

from src.funcs.iit import seleccionar_estado  # noqa: E402
from tests.fase0_kloss import (  # noqa: E402
    PRESENTE,
    construir_subsistema,
    kbruteforce,
    particiones_en_k,
    stirling2,
    vertices_de,
)

INF = float("inf")


def costo_futuro(cubo, base_i, Q, estado_inicial, cache):
    """c_i(Q) = |marginal del cubo futuro i conservando presentes Q − base_i|."""
    clave = (int(cubo.indice), frozenset(Q))
    if clave in cache:
        return cache[clave]
    ejes = np.array([int(d) for d in cubo.dims if int(d) not in Q], dtype=np.int8)
    marg = cubo.marginalizar(ejes)
    if marg.dims.size:
        inicial = tuple(estado_inicial[j] for j in marg.dims)
        val = float(marg.data[seleccionar_estado(inicial)])
    else:
        val = float(marg.data)
    c = abs(val - base_i)
    cache[clave] = c
    return c


def exact_via_separabilidad(sub, base, k):
    """k-MIP exacto explotando la separabilidad. Devuelve (min_delta, candidatos)."""
    fut_cubos = list(sub.ncubos)
    pre = [int(i) for i in sub.dims_ncubos]
    m, n = len(fut_cubos), len(pre)
    estado = sub.estado_inicial
    cache: dict = {}
    mejor = INF
    candidatos = 0

    for r in range(1, k + 1):
        if r > n:
            break
        empties = k - r
        if empties > m:  # no hay suficientes futuros para llenar bloques vacíos
            continue
        allow_empty = empties > 0
        for present_part in particiones_en_k(pre, r):
            candidatos += 1
            grupos = [set(g) for g in present_part]
            total = 0.0
            deltas = []  # coste extra de forzar un futuro a bloque vacío
            empty_count = 0
            for idx, cubo in enumerate(fut_cubos):
                base_i = float(base[idx])
                best_present = min(
                    costo_futuro(cubo, base_i, g, estado, cache) for g in grupos
                )
                if allow_empty:
                    cempty = costo_futuro(cubo, base_i, set(), estado, cache)
                    if cempty <= best_present:
                        total += cempty
                        empty_count += 1
                    else:
                        total += best_present
                        deltas.append(cempty - best_present)
                else:
                    total += best_present
            if allow_empty and empty_count < empties:
                need = empties - empty_count
                if len(deltas) < need:
                    continue  # imposible cubrir todos los bloques vacíos
                deltas.sort()
                total += sum(deltas[:need])
            mejor = min(mejor, total)
    return mejor, candidatos


def main():
    print("=== Fase 0: algoritmo exacto por separabilidad vs kbruteforce ===\n")
    pruebas = [(3, "111", "111"), (4, "1111", "1111"), (5, "11111", "11111")]

    for bits, alc, mec in pruebas:
        sub, base = construir_subsistema(bits, alc, mec)
        twoN = len(vertices_de(sub))
        n_pre = int(sub.dims_ncubos.size)
        print(f"--- N{bits}  (|V|={twoN}, presentes={n_pre}) ---")
        for k in range(2, min(twoN, 6) + 1):
            d_bf, _ = kbruteforce(sub, base, k)
            d_sep, cand = exact_via_separabilidad(sub, base, k)
            full = stirling2(twoN, k)
            ok = "OK" if abs(d_bf - d_sep) < 1e-5 else "DIFIERE"  # tol float32
            print(
                f"  k={k}: bruteforce δ={d_bf:.6f}  separab δ={d_sep:.6f}  -> {ok}"
                f"   | candidatos: separab={cand} vs full={full}"
            )
        print()


if __name__ == "__main__":
    main()
