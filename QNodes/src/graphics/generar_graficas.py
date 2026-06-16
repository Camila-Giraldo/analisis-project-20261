"""
Visualizaciones comparativas de resultados QNodes.

Genera 2 figuras en esta misma carpeta (graficas/):
  fig1_escalabilidad_tiempo.png  — Tiempo mediano vs N por k (escalabilidad)
  fig2_phi_mediana_vs_N.png      — φ mediana vs N por k (calidad de partición)

Nota: QNodes completó el 100% de los experimentos sin timeouts
(dataset completo k=2..5 × N=10,15,20,22,25).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_DIR = Path(__file__).parent

PALETTE = {2: "#4C72B0", 3: "#DD8452", 4: "#55A868", 5: "#C44E52"}
K_VALS = [2, 3, 4, 5]
N_VALS = [10, 15, 20, 22, 25]

# K4A (N=4) es caso de prueba puntual; se excluye de las comparativas.
# K25A_k5 está vacío; se excluye automáticamente al filtrar filas.
FILE_MAP: dict[tuple[int, int], Path] = {
    (2, 10): RESULTS_DIR / "k2" / "K10A_k2_auto.csv",
    (2, 15): RESULTS_DIR / "k2" / "K15A_k2_auto.csv",
    (2, 20): RESULTS_DIR / "k2" / "K20A_k2_auto.csv",
    (2, 22): RESULTS_DIR / "k2" / "K22A_k2_auto.csv",
    (2, 25): RESULTS_DIR / "k2" / "K25A_k2_auto.csv",
    (3, 10): RESULTS_DIR / "k3" / "K10A_k3_auto.csv",
    (3, 15): RESULTS_DIR / "k3" / "K15A_k3_auto.csv",
    (3, 20): RESULTS_DIR / "k3" / "K20A_k3_auto.csv",
    (3, 22): RESULTS_DIR / "k3" / "K22A_k3_auto.csv",
    (3, 25): RESULTS_DIR / "k3" / "K25A_k3_auto.csv",
    (4, 10): RESULTS_DIR / "k4" / "K10A_k4_auto.csv",
    (4, 15): RESULTS_DIR / "k4" / "K15A_k4_auto.csv",
    (4, 20): RESULTS_DIR / "k4" / "K20A_k4_auto.csv",
    (4, 22): RESULTS_DIR / "k4" / "K22A_k4_auto.csv",
    (4, 25): RESULTS_DIR / "k4" / "K25A_k4_auto.csv",
    (5, 10): RESULTS_DIR / "k5" / "K10A_k5_auto.csv",
    (5, 15): RESULTS_DIR / "k5" / "K15A_k5_auto.csv",
    (5, 20): RESULTS_DIR / "k5" / "K20A_k5_auto.csv",
    (5, 22): RESULTS_DIR / "k5" / "K22A_k5_auto.csv",
    (5, 25): RESULTS_DIR / "k5" / "K25A_k5_auto.csv",
}


def load(k: int, n: int) -> pd.DataFrame | None:
    path = FILE_MAP.get((k, n))
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["phi"] = pd.to_numeric(df["Pérdida"], errors="coerce")
    df["tiempo"] = pd.to_numeric(df["Tiempo"], errors="coerce")
    df["k"] = k
    df["N"] = n
    return df


frames = [f for (k, n) in FILE_MAP if (f := load(k, n)) is not None]
all_data = pd.concat(frames, ignore_index=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F8F8",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ===========================================================================
# Figura 1 — Escalabilidad: tiempo mediano vs N por k
# ===========================================================================
def fig1_escalabilidad_tiempo():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for k in K_VALS:
        stats = (
            all_data[all_data["k"] == k]
            .groupby("N")["tiempo"]
            .median()
            .reset_index()
            .sort_values("N")
        )
        if stats.empty:
            continue
        ax.plot(
            stats["N"], stats["tiempo"],
            marker="o", linewidth=2.5, markersize=8,
            color=PALETTE[k], label=f"k = {k}",
        )
        peak = stats.loc[stats["tiempo"].idxmax()]
        ax.annotate(
            f"{peak['tiempo']:.1f} s",
            xy=(peak["N"], peak["tiempo"]),
            xytext=(5, 7), textcoords="offset points",
            fontsize=9, color=PALETTE[k], fontweight="bold",
        )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g} s"))
    ax.set_xticks(N_VALS)
    ax.set_xlabel("Tamaño de la red (N nodos)")
    ax.set_ylabel("Tiempo de ejecución mediano (s, escala log)")
    ax.set_title(
        "Escalabilidad de QNodes — Tiempo mediano vs N\n"
        "(cada punto = mediana sobre todos los subsistemas evaluados)"
    )
    ax.legend(title="Particiones (k)", framealpha=0.9)

    fig.tight_layout()
    out = OUT_DIR / "fig1_escalabilidad_tiempo.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Guardada: {out.name}")


# ===========================================================================
# Figura 2 — Calidad: φ mediana vs N por k
# ===========================================================================
def fig2_phi_mediana_vs_N():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for k in K_VALS:
        stats = (
            all_data[all_data["k"] == k]
            .groupby("N")["phi"]
            .median()
            .reset_index()
            .sort_values("N")
        )
        if stats.empty:
            continue
        ax.plot(
            stats["N"], stats["phi"],
            marker="s", linewidth=2.5, markersize=8,
            color=PALETTE[k], label=f"k = {k}",
        )

    ax.set_xticks(N_VALS)
    ax.set_xlabel("Tamaño de la red (N nodos)")
    ax.set_ylabel("Pérdida mediana (φ)")
    ax.set_title(
        "Calidad de la partición — φ mediana vs N\n"
        "(φ = pérdida de información en la MIP; mayor φ = mayor integración)"
    )
    ax.legend(title="Particiones (k)", framealpha=0.9)

    fig.tight_layout()
    out = OUT_DIR / "fig2_phi_mediana_vs_N.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Guardada: {out.name}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Datos cargados: {len(all_data)} filas desde {len(frames)} archivos\n")
    print("Generando figuras...")
    fig1_escalabilidad_tiempo()
    fig2_phi_mediana_vs_N()
    print("\nListo. Figuras en:", OUT_DIR)
