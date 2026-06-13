"""
Baseline de regresión para la estrategia QNodes.

Objetivo
--------
Fijar (congelar) el comportamiento ACTUAL de QNodes antes de cualquier
limpieza / optimización / extensión a k-particiones. Si un cambio futuro
altera la salida de QNodes, este baseline lo detecta.

Adicionalmente compara QNodes contra BruteForce (referencia EXACTA: enumera
todas las biparticiones sobre el mismo `System.bipartir`) para documentar la
exactitud real de QNodes.

Reproducibilidad
----------------
Las TPMs se generan con `Manager.cargar_red()`, que usa la semilla numpy fija
(`aplicacion.semilla_numpy = 73`). Mismo camino que `exec.py` / `run_tests.py`.

Uso
---
    PYTHONPATH=. .venv/bin/python tests/baseline.py --update   # (re)genera el snapshot dorado
    PYTHONPATH=. .venv/bin/python tests/baseline.py            # verifica contra el snapshot

Artefacto: tests/baseline_qnodes.csv
"""

import argparse
import contextlib
import csv
import io
import math
import os
import sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import logging

from src.models.base.application import aplicacion

aplicacion.profiler_habilitado = False

# --- Neutralizar SafeLogger durante el baseline ---------------------------
# ColorFormatter llama colorama.init() en cada instanciación y re-envuelve
# sys.stdout; con cientos de loggers esto provoca RecursionError, además de
# fuga de descriptores por FileHandler reabiertos. Aquí lo silenciamos por
# completo (el arreglo de fondo es parte de la limpieza posterior).
from src.middlewares import slogger  # noqa: E402

_SILENT = logging.getLogger("qnodes_baseline_silent")
_SILENT.addHandler(logging.NullHandler())
_SILENT.setLevel(logging.CRITICAL + 1)
slogger.SafeLogger.__init__ = lambda self, *a, **k: setattr(self, "_logger", _SILENT)

from src.controllers.manager import Manager  # noqa: E402
from src.strategies.force import BruteForce  # noqa: E402
from src.strategies.q_nodes import QNodes  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(__file__), "baseline_qnodes.csv")

# Tamaños de red a cubrir (enumeración completa de subsistemas).
BITS = (3, 4, 5)
TOL = 1e-9
FIELDS = ["bits", "alcance", "mecanismo", "q_phi", "q_particion", "bf_phi", "status", "err"]


def _silencioso(fn, *a, **k):
    """Ejecuta suprimiendo stdout/stderr (logs ruidosos de las estrategias)."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def _ejecutar_caso(mpt, estado, cond, alcance, mecanismo):
    """Corre QNodes y BruteForce sobre un subsistema. Devuelve dict o None si falla."""
    try:
        bf = _silencioso(BruteForce(mpt).aplicar_estrategia, estado, cond, alcance, mecanismo)
        q = _silencioso(QNodes(mpt).aplicar_estrategia, estado, cond, alcance, mecanismo)
    except Exception:
        return None

    q_phi = float(q.perdida)
    bf_phi = float(bf.perdida)
    q_particion = " ".join(str(q.particion).split())  # normaliza saltos de línea

    if math.isinf(q_phi) or math.isnan(q_phi):
        status, err = "Q_INF", ""
    elif abs(q_phi - bf_phi) < TOL:
        status, err = "MATCH", 0.0
    else:
        status, err = "MISMATCH", q_phi - bf_phi

    return {
        "bits": estado.__len__(),
        "alcance": alcance,
        "mecanismo": mecanismo,
        "q_phi": q_phi,
        "q_particion": q_particion,
        "bf_phi": bf_phi,
        "status": status,
        "err": err,
    }


def generar_filas():
    """Itera todos los subsistemas válidos (cond=todo-1) para cada tamaño en BITS."""
    for bits in BITS:
        estado = "1" + "0" * (bits - 1)
        cond = "1" * bits
        mpt = _silencioso(Manager(estado).cargar_red)
        for a in range(1, 1 << bits):
            for m in range(1, 1 << bits):
                alcance = format(a, f"0{bits}b")
                mecanismo = format(m, f"0{bits}b")
                fila = _ejecutar_caso(mpt, estado, cond, alcance, mecanismo)
                if fila is not None:
                    yield fila


def _resumen(filas):
    por_bits = {}
    for f in filas:
        b = f["bits"]
        d = por_bits.setdefault(b, {"total": 0, "match": 0, "mismatch": 0, "inf": 0, "maxerr": 0.0})
        d["total"] += 1
        if f["status"] == "MATCH":
            d["match"] += 1
        elif f["status"] == "Q_INF":
            d["inf"] += 1
        else:
            d["mismatch"] += 1
            d["maxerr"] = max(d["maxerr"], float(f["err"]))
    print("\nResumen QNodes vs BruteForce (referencia exacta):")
    for b in sorted(por_bits):
        d = por_bits[b]
        fin = d["match"] + d["mismatch"]
        pct = (100 * d["match"] / fin) if fin else 0.0
        print(
            f"  N{b}: total={d['total']:4d} | inf(Q)={d['inf']:3d} | finitos={fin:4d}"
            f" match={d['match']:4d} ({pct:5.1f}%) mismatch={d['mismatch']:4d}"
            f" | maxerr={d['maxerr']:.5f}"
        )


def update():
    filas = list(generar_filas())
    with open(GOLDEN, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(filas)
    print(f"Snapshot dorado escrito: {GOLDEN}  ({len(filas)} casos)")
    _resumen(filas)


def verify():
    if not os.path.exists(GOLDEN):
        print(f"No existe {GOLDEN}. Corre primero con --update.")
        return 1
    with open(GOLDEN, newline="") as fh:
        esperado = {(r["bits"], r["alcance"], r["mecanismo"]): r for r in csv.DictReader(fh)}

    actuales = list(generar_filas())
    difs = 0
    for f in actuales:
        clave = (str(f["bits"]), f["alcance"], f["mecanismo"])
        ref = esperado.get(clave)
        if ref is None:
            print(f"  NUEVO caso no presente en golden: {clave}")
            difs += 1
            continue
        if abs(float(ref["q_phi"]) - f["q_phi"]) > TOL or ref["q_particion"] != f["q_particion"]:
            print(f"  CAMBIO en {clave}: golden phi={ref['q_phi']} part='{ref['q_particion']}'"
                  f"  ahora phi={f['q_phi']} part='{f['q_particion']}'")
            difs += 1

    if difs == 0:
        print(f"OK: QNodes reproduce el baseline ({len(actuales)} casos sin cambios).")
        _resumen(actuales)
        return 0
    print(f"\n{difs} diferencia(s) respecto al baseline.")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="(re)genera el snapshot dorado")
    args = ap.parse_args()
    if args.update:
        update()
        return 0
    return verify()


if __name__ == "__main__":
    sys.exit(main())
