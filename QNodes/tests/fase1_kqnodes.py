"""
Fase 1 — validación de la clase KQNodes.

Comprueba que `KQNodes.aplicar_estrategia(..., k)`:
  - coincide con la referencia exacta `kbruteforce` (Fase 0) para k = 2..5;
  - para k = 2 coincide con BruteForce (la bipartición exacta);
y muestra un ejemplo de solución formateada.

Uso:
    PYTHONPATH=. .venv/bin/python tests/fase1_kqnodes.py
"""

import logging
import math
import os
import sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from src.models.base.application import aplicacion  # noqa: E402

aplicacion.profiler_habilitado = False
from src.middlewares import slogger  # noqa: E402

_S = logging.getLogger("fase1_silent")
_S.addHandler(logging.NullHandler())
_S.setLevel(logging.CRITICAL + 1)
slogger.SafeLogger.__init__ = lambda self, *a, **k: setattr(self, "_logger", _S)

from src.controllers.manager import Manager  # noqa: E402
from src.strategies.force import BruteForce  # noqa: E402
from src.strategies.KQNodes import KQNodes  # noqa: E402
from tests.fase0_kloss import (  # noqa: E402
    construir_subsistema,
    kbruteforce,
    vertices_de,
)

TOL = 1e-5  # tolerancia float32


def main():
    print("=== Fase 1: validación de KQNodes vs referencia exacta ===\n")
    pruebas = [(3, "111", "111"), (4, "1111", "1111"), (5, "11111", "11111")]

    total = ok = 0
    for bits, alc, mec in pruebas:
        estado = "1" + "0" * (bits - 1)
        cond = "1" * bits
        mpt = Manager(estado).cargar_red()
        sub, base = construir_subsistema(bits, alc, mec)
        twoN = len(vertices_de(sub))
        print(f"--- N{bits}  (|V|={twoN}) ---")

        for k in range(2, min(twoN, 5) + 1):
            d_ref, _ = kbruteforce(sub, base, k)
            sol = KQNodes(mpt).aplicar_estrategia(estado, cond, alc, mec, k)
            d_kq = float(sol.perdida)
            coincide = abs(d_ref - d_kq) < TOL
            total += 1
            ok += int(coincide)
            print(
                f"  k={k}: kbruteforce δ={d_ref:.6f}  KQNodes δ={d_kq:.6f}"
                f"  -> {'OK' if coincide else 'DIFIERE'}"
            )
        print()

    # k=2 debe coincidir con BruteForce (bipartición exacta)
    print("--- k=2 vs BruteForce ---")
    for bits, alc, mec in pruebas:
        estado = "1" + "0" * (bits - 1)
        cond = "1" * bits
        mpt = Manager(estado).cargar_red()
        bf = float(BruteForce(mpt).aplicar_estrategia(estado, cond, alc, mec).perdida)
        kq = float(KQNodes(mpt).aplicar_estrategia(estado, cond, alc, mec, 2).perdida)
        coincide = abs(bf - kq) < TOL
        total += 1
        ok += int(coincide)
        print(f"  N{bits}: BruteForce={bf:.6f}  KQNodes(k=2)={kq:.6f}  -> {'OK' if coincide else 'DIFIERE'}")

    print(f"\nResultado: {ok}/{total} comprobaciones OK")

    # Ejemplo de solución formateada (k=3 sobre N4)
    print("\n=== Ejemplo: KQNodes k=3 sobre N4 ===")
    estado, cond, alc, mec = "1000", "1111", "1111", "1111"
    mpt = Manager(estado).cargar_red()
    sol = KQNodes(mpt).aplicar_estrategia(estado, cond, alc, mec, 3)
    print(f"δ_3 = {float(sol.perdida):.6f}")
    print("Partición:")
    print(sol.particion)


if __name__ == "__main__":
    main()
