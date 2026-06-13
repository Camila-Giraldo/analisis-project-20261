"""
Fase 2 — validación de las heurísticas de KQNodes (voraz y recocido simulado).

Parte A: exactitud. Sobre tamaños donde el método exacto es factible, compara
         δ_k de las heurísticas contra el exacto (tasa de acierto y gap medio).
Parte B: velocidad. Sobre un n donde el exacto sería impráctico, mide el tiempo
         de las heurísticas (deben resolver en segundos).

Uso:
    PYTHONPATH=. .venv/bin/python tests/fase2_heuristica.py
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

_S = logging.getLogger("fase2_silent")
_S.addHandler(logging.NullHandler())
_S.setLevel(logging.CRITICAL + 1)
slogger.SafeLogger.__init__ = lambda self, *a, **k: setattr(self, "_logger", _S)

from src.controllers.manager import Manager  # noqa: E402
from src.strategies.KQNodes import KQNodes  # noqa: E402

TOL = 1e-5


def parte_a():
    print("=== Parte A: exactitud heurísticas vs método exacto ===\n")
    pruebas = [
        (4, "1111", "1111"),
        (5, "11111", "11111"),
        (6, "111111", "111111"),
        (6, "111100", "110111"),
        (6, "101011", "111110"),
    ]
    aciertos_v = aciertos_r = total = 0
    gap_v_max = gap_r_max = 0.0

    for bits, alc, mec in pruebas:
        estado, cond = "1" + "0" * (bits - 1), "1" * bits
        mpt = Manager(estado).cargar_red()
        twoN = sum(c == "1" for c in alc) + sum(c == "1" for c in mec)
        print(f"--- N{bits} alc={alc} mec={mec}  (|V|≈{twoN}) ---")
        for k in range(2, min(twoN, 5) + 1):
            d_ex = float(KQNodes(mpt).aplicar_estrategia(estado, cond, alc, mec, k, metodo="exacto").perdida)
            d_vz = float(KQNodes(mpt).aplicar_estrategia(estado, cond, alc, mec, k, metodo="voraz").perdida)
            d_rc = float(KQNodes(mpt).aplicar_estrategia(estado, cond, alc, mec, k, metodo="recocido", iteraciones=2000).perdida)
            total += 1
            gap_v, gap_r = d_vz - d_ex, d_rc - d_ex
            aciertos_v += int(abs(gap_v) < TOL)
            aciertos_r += int(abs(gap_r) < TOL)
            gap_v_max, gap_r_max = max(gap_v_max, gap_v), max(gap_r_max, gap_r)
            print(
                f"  k={k}: exacto={d_ex:.6f} | voraz={d_vz:.6f} (gap {gap_v:+.5f}) "
                f"| recocido={d_rc:.6f} (gap {gap_r:+.5f})"
            )
        print()

    print(f"Voraz   : {aciertos_v}/{total} óptimos exactos | gap máx {gap_v_max:.5f}")
    print(f"Recocido: {aciertos_r}/{total} óptimos exactos | gap máx {gap_r_max:.5f}")


def parte_b():
    print("\n=== Parte B: velocidad en n grande (exacto impráctico) ===")
    bits = 14  # subsistema completo: 14 presentes (exacto ~ minutos/horas)
    estado, cond = "1" + "0" * (bits - 1), "1" * bits
    alc = mec = "1" * bits
    mpt = Manager(estado).cargar_red()
    for k in (3, 5):
        kq = KQNodes(mpt)
        t0 = time.perf_counter()
        sol = kq.aplicar_estrategia(estado, cond, alc, mec, k, metodo="recocido", iteraciones=4000)
        dt = time.perf_counter() - t0
        print(f"  k={k}: recocido δ={float(sol.perdida):.6f}  en {dt:.2f}s  (n=14 presentes)")


if __name__ == "__main__":
    parte_a()
    parte_b()
