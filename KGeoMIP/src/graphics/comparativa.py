"""
Gráficas comparativas KGeoMIP vs QNodes para presentación.

Genera:
  comp1_tiempo_phi_vs_N.png   — Tiempo mediano y φ mediano vs N (ambos algoritmos)
  comp2_scatter_phi_tiempo.png — Scatter φ vs tiempo por subsistema (N=10 y N=25)
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
from matplotlib.lines import Line2D

OUT_DIR = Path(__file__).parent
KGEO_DIR = Path(__file__).parents[2] / "results"
QNODES_DIR = Path(__file__).parents[3] / "QNodes" / "src" / "results"

K_VALS = [2, 3, 4, 5]
N_VALS = [10, 15, 20, 22, 25]
PALETTE = {2: "#4C72B0", 3: "#DD8452", 4: "#55A868", 5: "#C44E52"}

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _to_seconds(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, datetime.time):
        return val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def load_kgeomip(k: int, n: int) -> pd.DataFrame | None:
    for suffix in [f"{n}A{k}", f"{n}A{k}-1"]:
        p = KGEO_DIR / f"k{k}" / f"resultados_Geometric_{suffix}.xlsx"
        if p.exists():
            df = pd.read_excel(p)
            df["phi"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
            df["tiempo"] = df.iloc[:, 6].map(_to_seconds)
            df["k"] = k
            df["N"] = n
            df["algo"] = "KGeoMIP"
            return df
    return None


def load_qnodes(k: int, n: int) -> pd.DataFrame | None:
    p = QNODES_DIR / f"k{k}" / f"K{n}A_k{k}_auto.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    df["phi"] = pd.to_numeric(df["Pérdida"], errors="coerce")
    df["tiempo"] = pd.to_numeric(df["Tiempo"], errors="coerce")
    df["k"] = k
    df["N"] = n
    df["algo"] = "QNodes"
    return df


frames = []
for k in K_VALS:
    for n in N_VALS:
        for fn in [load_kgeomip, load_qnodes]:
            f = fn(k, n)
            if f is not None:
                frames.append(f[["phi", "tiempo", "k", "N", "algo"]])

all_data = pd.concat(frames, ignore_index=True)

# ---------------------------------------------------------------------------
# Estilo global
# ---------------------------------------------------------------------------
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
# Figura 1 — Comparativa tiempo mediano y φ mediano vs N
# ===========================================================================
def comp1_tiempo_phi_vs_N():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel, title_suffix, use_log in [
        (axes[0], "tiempo", "Tiempo mediano (s, escala log)", "Escalabilidad temporal", True),
        (axes[1], "phi",    "φ mediana (pérdida de información)", "Calidad de partición",  False),
    ]:
        for k in K_VALS:
            for algo, ls, zorder in [("KGeoMIP", "-", 3), ("QNodes", "--", 2)]:
                sub = all_data[(all_data["algo"] == algo) & (all_data["k"] == k)]
                stats = (
                    sub.groupby("N")[metric]
                    .median()
                    .reset_index()
                    .sort_values("N")
                    .dropna()
                )
                if stats.empty:
                    continue
                ax.plot(
                    stats["N"], stats[metric],
                    linestyle=ls, linewidth=2.2, marker="o", markersize=7,
                    color=PALETTE[k], zorder=zorder,
                )

        if use_log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g} s"))

        ax.set_xticks(N_VALS)
        ax.set_xlabel("Tamaño de la red (N nodos)")
        ax.set_ylabel(ylabel)
        ax.set_title(title_suffix)

    # Leyenda unificada
    legend_k = [
        Line2D([0], [0], color=PALETTE[k], linewidth=2.5, label=f"k = {k}")
        for k in K_VALS
    ]
    legend_algo = [
        Line2D([0], [0], color="gray", linestyle="-",  linewidth=2, label="KGeoMIP"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="QNodes"),
    ]
    axes[1].legend(
        handles=legend_k + legend_algo,
        title="Particiones / Algoritmo",
        loc="upper left", framealpha=0.9, fontsize=10,
    )

    fig.suptitle(
        "KGeoMIP vs QNodes — Comparativa por tamaño de red",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "comp1_tiempo_phi_vs_N.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out.name}")


# ===========================================================================
# Figura 2 — Scatter φ vs tiempo (N=10 y N=25)
# ===========================================================================
def comp2_scatter_phi_tiempo():
    MARKERS = {"KGeoMIP": "o", "QNodes": "D"}
    ALPHA   = {"KGeoMIP": 0.65, "QNodes": 0.75}
    SIZE    = {"KGeoMIP": 45,   "QNodes": 40}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, n, use_log_x in [(axes[0], 10, False), (axes[1], 25, True)]:
        sub = all_data[(all_data["N"] == n)].dropna(subset=["phi", "tiempo"])

        for algo in ["KGeoMIP", "QNodes"]:
            for k in K_VALS:
                d = sub[(sub["algo"] == algo) & (sub["k"] == k)]
                if d.empty:
                    continue
                ax.scatter(
                    d["tiempo"], d["phi"],
                    color=PALETTE[k],
                    marker=MARKERS[algo],
                    alpha=ALPHA[algo],
                    s=SIZE[algo],
                    edgecolors="white" if algo == "KGeoMIP" else "none",
                    linewidths=0.4,
                )

        if use_log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g} s"))

        ax.set_xlabel("Tiempo de ejecución (s)" + (" — escala log" if use_log_x else ""))
        ax.set_ylabel("Pérdida φ")
        ax.set_title(f"N = {n} nodos")

    # Leyenda: colores = k, marcadores = algoritmo
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[k],
               markersize=9, label=f"k = {k}")
        for k in K_VALS
    ] + [
        Line2D([0], [0], marker="o", color="gray", markersize=9,
               markeredgecolor="white", markeredgewidth=0.8, label="KGeoMIP"),
        Line2D([0], [0], marker="D", color="gray", markersize=8, label="QNodes"),
    ]
    axes[1].legend(
        handles=legend_handles,
        title="Particiones / Algoritmo",
        loc="upper left", framealpha=0.9, fontsize=10,
    )

    fig.suptitle(
        "KGeoMIP vs QNodes — Scatter φ vs Tiempo por subsistema",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "comp2_scatter_phi_tiempo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out.name}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Datos: {len(all_data)} filas ({all_data['algo'].value_counts().to_dict()})\n")
    print("Generando figuras comparativas...")
    comp1_tiempo_phi_vs_N()
    comp2_scatter_phi_tiempo()
    print("\nListo. Figuras en:", OUT_DIR)
