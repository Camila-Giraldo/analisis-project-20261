"""
Experimentos KQNodes — generación de resultados para el informe técnico.

Produce en results/:
  accuracy_table.csv    — exactitud del método exacto vs kbruteforce (N3–N6).
  speedup_table.csv     — speedup teórico: candidatos KQNodes vs fuerza bruta (N3–N8).
  heuristic_table.csv   — exactitud voraz y recocido vs exacto (N4–N6).
  time_scaling.png      — tiempo de ejecución vs N para k=3,4,5.
  heuristic_accuracy.png — barras: aciertos exactos voraz vs recocido.

Uso (desde QNodes/):
    PYTHONPATH=. .venv/bin/python tests/experiments.py
"""

import logging
import os
import sys
import time
from pathlib import Path

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Silenciar loggers durante los experimentos.
from src.middlewares import slogger  # noqa: E402

_S = logging.getLogger("exp_silent")
_S.addHandler(logging.NullHandler())
_S.setLevel(logging.CRITICAL + 1)
slogger.SafeLogger.__init__ = lambda self, *a, **kw: setattr(self, "_logger", _S)

from src.models.base.application import aplicacion  # noqa: E402

aplicacion.profiler_habilitado = False

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.controllers.manager import Manager  # noqa: E402
from src.strategies.KQNodes import KQNodes  # noqa: E402
from tests.fase0_kloss import kbruteforce, construir_subsistema, stirling2  # noqa: E402

TOL = 1e-5
OUT = Path("results")

KS = [3, 4, 5]
BITS_EXACTO = [3, 4, 5, 6]
BITS_SPEED = [3, 4, 5, 6, 7, 8]
BITS_HEUR = [4, 5, 6]
BITS_TIEMPO = [4, 5, 6, 7, 8, 9, 10]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estado(bits): return "1" + "0" * (bits - 1)
def _cond(bits): return "1" * bits


def _kq_perdida(tpm, bits, alc, mec, k, metodo="exacto", **kw):
    estado, cond = _estado(bits), _cond(bits)
    return float(KQNodes(tpm).aplicar_estrategia(estado, cond, alc, mec, k, metodo=metodo, **kw).perdida)


def _kq_tiempo(tpm, bits, alc, mec, k, metodo="exacto", **kw):
    estado, cond = _estado(bits), _cond(bits)
    t0 = time.perf_counter()
    KQNodes(tpm).aplicar_estrategia(estado, cond, alc, mec, k, metodo=metodo, **kw)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Tabla 1: Exactitud exacto vs kbruteforce
# ---------------------------------------------------------------------------

def tabla_accuracy(bits_list=BITS_EXACTO, k_max=5):
    """N, k, δ_exacto, δ_kbruteforce, coincide, gap."""
    filas = []
    for bits in bits_list:
        alc = mec = "1" * bits
        tpm = Manager(_estado(bits)).cargar_red()
        sub, base = construir_subsistema(bits, alc, mec)
        twoN = len([i for i in sub.indices_ncubos]) + len([i for i in sub.dims_ncubos])
        for k in range(2, min(k_max, twoN) + 1):
            d_ref, _ = kbruteforce(sub, base, k)
            d_kq = _kq_perdida(tpm, bits, alc, mec, k)
            gap = d_kq - d_ref
            filas.append({
                "N": bits, "k": k,
                "δ_KQNodes": round(d_kq, 6),
                "δ_kbruteforce": round(d_ref, 6),
                "coincide": abs(gap) < TOL,
                "gap": round(gap, 8),
            })
            print(f"  N{bits} k={k}: KQNodes={d_kq:.6f}  ref={d_ref:.6f}  {'OK' if abs(gap)<TOL else 'DIF'}")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Tabla 2: Speedup teórico (candidatos KQNodes vs fuerza bruta sobre 2N vértices)
# ---------------------------------------------------------------------------

def tabla_speedup(bits_list=BITS_SPEED, k_max=5):
    """N, k, candidatos_KQNodes (Σ S(n,r)), candidatos_bf (S(2N,k)), speedup."""
    filas = []
    for bits in bits_list:
        for k in range(2, k_max + 1):
            n_pres = bits          # subsistema completo: n presentes = bits
            twoN = 2 * bits        # total de vértices
            cand_kq = sum(stirling2(n_pres, r) for r in range(1, k + 1))
            cand_bf = stirling2(twoN, k)
            speedup = cand_bf / cand_kq if cand_kq > 0 else float("inf")
            filas.append({
                "N": bits, "k": k,
                "candidatos_KQNodes": cand_kq,
                "candidatos_BruteForce": cand_bf,
                "speedup_teórico": round(speedup, 1),
            })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Tabla 3: Heurísticas vs exacto
# ---------------------------------------------------------------------------

def tabla_heuristicas(bits_list=BITS_HEUR, k_max=4):
    """N, k, δ_exacto, δ_voraz, δ_recocido, gap_v, gap_r."""
    filas = []
    for bits in bits_list:
        alc = mec = "1" * bits
        tpm = Manager(_estado(bits)).cargar_red()
        for k in range(2, k_max + 1):
            d_ex = _kq_perdida(tpm, bits, alc, mec, k, metodo="exacto")
            d_vz = _kq_perdida(tpm, bits, alc, mec, k, metodo="voraz")
            d_rc = _kq_perdida(tpm, bits, alc, mec, k, metodo="recocido", iteraciones=2000)
            gap_v, gap_r = d_vz - d_ex, d_rc - d_ex
            filas.append({
                "N": bits, "k": k,
                "δ_exacto": round(d_ex, 6),
                "δ_voraz": round(d_vz, 6),
                "δ_recocido": round(d_rc, 6),
                "gap_voraz": round(gap_v, 6),
                "gap_recocido": round(gap_r, 6),
                "voraz_óptimo": abs(gap_v) < TOL,
                "recocido_óptimo": abs(gap_r) < TOL,
            })
            print(f"  N{bits} k={k}: ex={d_ex:.5f} vz={d_vz:.5f}({gap_v:+.5f}) rc={d_rc:.5f}({gap_r:+.5f})")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Gráfica 1: tiempo vs N (log-scale)
# ---------------------------------------------------------------------------

def plot_time_scaling(bits_list=BITS_TIEMPO):
    """Mide tiempo de find_kmip para k=3,4,5 en función de N."""
    datos = {k: {"ns": [], "ts": []} for k in KS}
    for bits in bits_list:
        alc = mec = "1" * bits
        tpm = Manager(_estado(bits)).cargar_red()
        for k in KS:
            t = _kq_tiempo(tpm, bits, alc, mec, k, metodo="exacto")
            datos[k]["ns"].append(bits)
            datos[k]["ts"].append(t)
            print(f"  N{bits} k={k}: {t:.3f}s")

    fig, ax = plt.subplots(figsize=(7, 5))
    estilos = ["-o", "-s", "-^"]
    for i, k in enumerate(KS):
        ax.semilogy(datos[k]["ns"], datos[k]["ts"], estilos[i], label=f"k={k}", linewidth=1.8)
    ax.set_xlabel("N (nodos del sistema)")
    ax.set_ylabel("Tiempo (s) — escala log")
    ax.set_title("KQNodes exacto: tiempo vs N")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = OUT / "time_scaling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------------------
# Gráfica 2: barras de acierto heurísticas
# ---------------------------------------------------------------------------

def plot_heuristic_accuracy(df_heur: pd.DataFrame):
    """Barras agrupadas: fracción de óptimos exactos (voraz vs recocido) por N."""
    ns = sorted(df_heur["N"].unique())
    v_rates = [df_heur[df_heur["N"] == n]["voraz_óptimo"].mean() for n in ns]
    r_rates = [df_heur[df_heur["N"] == n]["recocido_óptimo"].mean() for n in ns]

    x = np.arange(len(ns))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - w / 2, [v * 100 for v in v_rates], w, label="Voraz", color="#4C8BE2")
    bars2 = ax.bar(x + w / 2, [r * 100 for r in r_rates], w, label="Recocido", color="#E27B4C")
    ax.set_xlabel("N (nodos del sistema)")
    ax.set_ylabel("Aciertos exactos (%)")
    ax.set_title("Heurísticas KQNodes: tasa de acierto exacto")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N{n}" for n in ns])
    ax.set_ylim(0, 110)
    ax.axhline(80, color="gray", linestyle="--", linewidth=1, label="Umbral 80 %")
    ax.legend()
    ax.bar_label(bars1, fmt="%.0f%%", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.0f%%", padding=2, fontsize=8)
    fig.tight_layout()
    path = OUT / "heuristic_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(exist_ok=True)

    print("\n=== Tabla 1: Exactitud KQNodes vs kbruteforce ===")
    df_acc = tabla_accuracy()
    df_acc.to_csv(OUT / "accuracy_table.csv", index=False)
    total = len(df_acc)
    ok = df_acc["coincide"].sum()
    print(f"  {ok}/{total} casos coinciden con kbruteforce")
    print(f"  Guardado: {OUT / 'accuracy_table.csv'}")

    print("\n=== Tabla 2: Speedup teórico ===")
    df_spd = tabla_speedup()
    df_spd.to_csv(OUT / "speedup_table.csv", index=False)
    print(f"  Guardado: {OUT / 'speedup_table.csv'}")

    print("\n=== Tabla 3: Heurísticas vs exacto ===")
    df_heur = tabla_heuristicas()
    df_heur.to_csv(OUT / "heuristic_table.csv", index=False)
    ok_v = df_heur["voraz_óptimo"].sum()
    ok_r = df_heur["recocido_óptimo"].sum()
    tot = len(df_heur)
    print(f"  Voraz:   {ok_v}/{tot} óptimos | Recocido: {ok_r}/{tot} óptimos")
    print(f"  Guardado: {OUT / 'heuristic_table.csv'}")

    print("\n=== Gráfica: tiempo vs N ===")
    plot_time_scaling()

    print("\n=== Gráfica: acierto heurísticas ===")
    plot_heuristic_accuracy(df_heur)

    print("\n✓ Todos los artefactos generados en results/")


if __name__ == "__main__":
    main()
