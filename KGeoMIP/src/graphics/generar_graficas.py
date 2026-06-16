"""
Visualizaciones comparativas de resultados KGeoMIP.

Genera 3 figuras en esta misma carpeta (graficas/):
  fig1_escalabilidad_tiempo.png  — Tiempo mediano vs N por k (escalabilidad)
  fig2_phi_mediana_vs_N.png      — φ mediana vs N por k (calidad de partición)
  fig3_timeouts.png              — Tasa de timeouts (%) por N×k (límites)
"""

import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = Path(__file__).parents[2] / "results"
OUT_DIR = Path(__file__).parent

PALETTE = {2: "#4C72B0", 3: "#DD8452", 4: "#55A868", 5: "#C44E52"}
K_VALS = [2, 3, 4, 5]
N_VALS = [10, 15, 20, 22, 25]

FILE_MAP: dict[tuple[int, int], Path] = {
    (2, 10): RESULTS_DIR / "k2" / "resultados_Geometric_10A2.xlsx",
    (2, 15): RESULTS_DIR / "k2" / "resultados_Geometric_15A2.xlsx",
    (2, 20): RESULTS_DIR / "k2" / "resultados_Geometric_20A2.xlsx",
    (2, 22): RESULTS_DIR / "k2" / "resultados_Geometric_22A2.xlsx",
    (2, 25): RESULTS_DIR / "k2" / "resultados_Geometric_25A2.xlsx",
    (3, 10): RESULTS_DIR / "k3" / "resultados_Geometric_10A3.xlsx",
    (3, 15): RESULTS_DIR / "k3" / "resultados_Geometric_15A3.xlsx",
    (3, 20): RESULTS_DIR / "k3" / "resultados_Geometric_20A3.xlsx",
    (3, 22): RESULTS_DIR / "k3" / "resultados_Geometric_22A3.xlsx",
    (3, 25): RESULTS_DIR / "k3" / "resultados_Geometric_25A3.xlsx",
    (4, 10): RESULTS_DIR / "k4" / "resultados_Geometric_10A4.xlsx",
    (4, 15): RESULTS_DIR / "k4" / "resultados_Geometric_15A4.xlsx",
    (4, 20): RESULTS_DIR / "k4" / "resultados_Geometric_20A4.xlsx",
    (4, 22): RESULTS_DIR / "k4" / "resultados_Geometric_22A4.xlsx",
    (4, 25): RESULTS_DIR / "k4" / "resultados_Geometric_25A4-1.xlsx",
    (5, 10): RESULTS_DIR / "k5" / "resultados_Geometric_10A5.xlsx",
    (5, 15): RESULTS_DIR / "k5" / "resultados_Geometric_15A5.xlsx",
    (5, 20): RESULTS_DIR / "k5" / "resultados_Geometric_20A5.xlsx",
    (5, 22): RESULTS_DIR / "k5" / "resultados_Geometric_22A5.xlsx",
    (5, 25): RESULTS_DIR / "k5" / "resultados_Geometric_25A5-1.xlsx",
}


def _to_seconds(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, datetime.time):
        return val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def load(k: int, n: int) -> pd.DataFrame | None:
    path = FILE_MAP.get((k, n))
    if path is None or not path.exists():
        return None
    df = pd.read_excel(path)
    df["phi"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
    df["tiempo"] = df.iloc[:, 6].map(_to_seconds)
    df["k"] = k
    df["N"] = n
    return df


frames = [load(k, n) for (k, n) in FILE_MAP if (p := FILE_MAP[(k, n)]).exists()]
frames = [f for f in frames if f is not None]
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
        ax.plot(
            stats["N"], stats["tiempo"],
            marker="o", linewidth=2.5, markersize=8,
            color=PALETTE[k], label=f"k = {k}",
        )
        # Etiqueta en el punto máximo
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
        "Escalabilidad de KGeoMIP — Tiempo mediano vs N\n"
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


# ===========================================================================
# Figura 3 — Límites: heatmap de timeouts por N×k
# ===========================================================================
def fig3_timeouts():
    matrix = np.zeros((len(K_VALS), len(N_VALS)))

    for i, k in enumerate(K_VALS):
        for j, n in enumerate(N_VALS):
            subset = all_data[(all_data["k"] == k) & (all_data["N"] == n)]
            if subset.empty:
                matrix[i, j] = np.nan
                continue
            nulos = subset["phi"].isna().sum()
            matrix[i, j] = (nulos / len(subset)) * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="#DDDDDD")
    masked = np.ma.masked_invalid(matrix)

    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=30)
    ax.set_xticks(range(len(N_VALS)))
    ax.set_xticklabels([f"N = {n}" for n in N_VALS])
    ax.set_yticks(range(len(K_VALS)))
    ax.set_yticklabels([f"k = {k}" for k in K_VALS])
    ax.set_title(
        "Tasa de timeouts (límite 3 600 s) por configuración N × k\n"
        "(gris = sin datos; rojo más intenso = más fallos)"
    )

    for i in range(len(K_VALS)):
        for j in range(len(N_VALS)):
            val = matrix[i, j]
            text = "—" if np.isnan(val) else f"{val:.0f}%"
            color = "white" if (not np.isnan(val) and val > 15) else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("% Timeouts")

    fig.tight_layout()
    out = OUT_DIR / "fig3_timeouts.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Guardada: {out.name}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Datos cargados: {len(all_data)} filas desde {len(frames)} archivos\n")
    print("Generando figuras...")
    fig1_escalabilidad_tiempo()
    fig2_phi_mediana_vs_N()
    fig3_timeouts()
    print("\nListo. Figuras en:", OUT_DIR)
