"""
Fase 1 — benchmark del método EXACTO de KQNodes.

Mide el tiempo de `find_kmip` (algoritmo exacto por separabilidad) en función del
tamaño del subsistema (subsistema completo: n presentes = m futuros = N) para
varios k, y proyecta el límite teórico de candidatos Σ_r S(n, r).

Uso:
    PYTHONPATH=. .venv/bin/python tests/fase1_bench.py
"""

import logging
import os
import sys
import time

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from src.models.base.application import aplicacion  # noqa: E402

aplicacion.profiler_habilitado = False
from src.middlewares import slogger  # noqa: E402

_S = logging.getLogger("bench_silent")
_S.addHandler(logging.NullHandler())
_S.setLevel(logging.CRITICAL + 1)
slogger.SafeLogger.__init__ = lambda self, *a, **k: setattr(self, "_logger", _S)

from src.controllers.manager import Manager  # noqa: E402
from src.strategies.KQNodes import KQNodes  # noqa: E402
from tests.fase0_kloss import stirling2  # noqa: E402

KS = (3, 4, 5)
LIMITE_SEG = 30.0  # si find_kmip supera esto, no se prueban tamaños mayores con esa k


def candidatos_teoricos(n, k):
    return sum(stirling2(n, r) for r in range(1, k + 1))


def bench():
    print("=== Benchmark KQNodes exacto (subsistema completo, n=m=N) ===\n")
    print(f"{'N':>3} {'prep(s)':>9} " + " ".join(f"k={k}:t(s)/cand" .rjust(18) for k in KS))
    saltar = set()
    for bits in range(4, 17):
        estado = "1" + "0" * (bits - 1)
        cond = "1" * bits
        try:
            mpt = Manager(estado).cargar_red()
        except MemoryError:
            print(f"{bits:>3}  (MemoryError al cargar la red)")
            break

        kq = KQNodes(mpt)
        t0 = time.perf_counter()
        kq.sia_preparar_subsistema(estado, cond, cond, cond)  # subsistema completo
        t_prep = time.perf_counter() - t0

        celdas = []
        for k in KS:
            if k in saltar:
                celdas.append("—".rjust(18))
                continue
            kq.memoria_costos.clear()
            t0 = time.perf_counter()
            kq.find_kmip(k)
            dt = time.perf_counter() - t0
            cand = candidatos_teoricos(bits, k)
            celdas.append(f"{dt:7.3f}/{cand:>8}".rjust(18))
            if dt > LIMITE_SEG:
                saltar.add(k)
        print(f"{bits:>3} {t_prep:9.3f} " + " ".join(celdas))
        sys.stdout.flush()
        if len(saltar) == len(KS):
            print("\n(todas las k superaron el límite de tiempo; fin)")
            break


def proyeccion():
    print("\n=== Proyección teórica de candidatos Σ_r S(n,r) (n = nº presentes) ===")
    print(f"{'n':>3} " + " ".join(f"k={k}".rjust(16) for k in KS))
    for n in list(range(5, 26)):
        print(f"{n:>3} " + " ".join(f"{candidatos_teoricos(n, k):>16,}" for k in KS))


if __name__ == "__main__":
    bench()
    proyeccion()
