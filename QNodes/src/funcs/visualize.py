"""
Visualización de k-particiones.

Función principal:
    plot_kparticion_hypercube(bloques, n_bits) → plt.Figure

    - n_bits = 2: cuadrado 2D (hipercubo exacto).
    - n_bits = 3: cubo 3D con Axes3D (hipercubo exacto).
    - n_bits ≥ 4: vista bipartita — futuros (t+1) arriba, presentes (t) abajo,
      coloreados por bloque. Más legible que proyectar 2^n vértices con PCA.

Ejemplo de uso:
    from src.controllers.manager import Manager
    from src.strategies.KQNodes import KQNodes
    from src.funcs.visualize import save_kparticion_hypercube

    tpm = Manager("1000").cargar_red()
    kq  = KQNodes(tpm)
    kq.aplicar_estrategia("1000", "1111", "1111", "1111", 3)
    save_kparticion_hypercube(kq.ultima_kparticion, 4, "particion_k3.png")
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

FUTURO, PRESENTE = 1, 0

_COLORES = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#6D4C41"]

# Abecedario local para no crear dependencia circular
_ABC = [chr(ord("A") + i) for i in range(26)]


# ---------------------------------------------------------------------------
# Coordenadas del hipercubo (n ≤ 3)
# ---------------------------------------------------------------------------

def _coordenadas_hipercubo(n: int) -> np.ndarray:
    indices = np.arange(2 ** n)
    return ((indices[:, None] >> np.arange(n)) & 1).astype(float)


def _aristas_hipercubo(n: int) -> list[tuple[int, int]]:
    aristas = []
    for u in range(2 ** n):
        for bit in range(n):
            v = u ^ (1 << bit)
            if v > u:
                aristas.append((u, v))
    return aristas


def _asignacion_bloques(
    bloques: list[list[tuple[int, int]]], n_bits: int
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    fut_bloque: dict[tuple[int, int], int] = {}
    pre_bloque: dict[tuple[int, int], int] = {}
    for b_idx, bloque in enumerate(bloques):
        for tiempo, idx in bloque:
            if tiempo == FUTURO:
                fut_bloque[(tiempo, idx)] = b_idx
            else:
                pre_bloque[(tiempo, idx)] = b_idx
    return fut_bloque, pre_bloque


# ---------------------------------------------------------------------------
# Dibujo hipercubo 2D / 3D
# ---------------------------------------------------------------------------

def _dibujar_2d(ax, coords2d, aristas, colores_nodos, marcadores, etiquetas_nodos):
    for u, v in aristas:
        ax.plot(
            [coords2d[u, 0], coords2d[v, 0]],
            [coords2d[u, 1], coords2d[v, 1]],
            color="lightgray", linewidth=1, zorder=1,
        )
    for i, (x, y) in enumerate(coords2d):
        ax.scatter(x, y, color=colores_nodos[i], marker=marcadores[i],
                   s=160, zorder=3, edgecolors="black", linewidths=1.2)
        ax.annotate(etiquetas_nodos[i], (x, y),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)


def _dibujar_3d(ax, coords3d, aristas, colores_nodos, marcadores, etiquetas_nodos):
    for u, v in aristas:
        ax.plot(
            [coords3d[u, 0], coords3d[v, 0]],
            [coords3d[u, 1], coords3d[v, 1]],
            [coords3d[u, 2], coords3d[v, 2]],
            color="lightgray", linewidth=0.8,
        )
    for i, (x, y, z) in enumerate(coords3d):
        ax.scatter(x, y, z, color=colores_nodos[i], marker=marcadores[i],
                   s=120, edgecolors="black", linewidths=1.0)
        ax.text(x, y, z + 0.05, etiquetas_nodos[i], fontsize=7)


# ---------------------------------------------------------------------------
# Vista bipartita (n ≥ 4)
# ---------------------------------------------------------------------------

def _dibujar_bipartito(
    ax,
    bloques: list[list[tuple[int, int]]],
) -> None:
    """
    Dibuja los nodos de la k-partición como grafo bipartito:
      - Fila superior: nodos futuros (t+1) — círculos.
      - Fila inferior: nodos presentes (t)  — cuadrados.
    Cada nodo se colorea según su bloque. Las líneas grises conectan
    nodos del mismo bloque entre filas.
    """
    fut_bloque: dict[int, int] = {}
    pre_bloque: dict[int, int] = {}
    for b_idx, bloque in enumerate(bloques):
        for tiempo, idx in bloque:
            if tiempo == FUTURO:
                fut_bloque[idx] = b_idx
            else:
                pre_bloque[idx] = b_idx

    fut_indices = sorted(fut_bloque)
    pre_indices = sorted(pre_bloque)
    n_fut = len(fut_indices)
    n_pre = len(pre_indices)

    def _x(pos, total):
        return pos / (total - 1) if total > 1 else 0.5

    pos_fut = {idx: (_x(i, n_fut), 1.0) for i, idx in enumerate(fut_indices)}
    pos_pre = {idx: (_x(i, n_pre), 0.0) for i, idx in enumerate(pre_indices)}

    # Líneas entre nodos del mismo bloque
    for b_idx, bloque in enumerate(bloques):
        color = _COLORES[b_idx % len(_COLORES)]
        futs_b = [idx for t, idx in bloque if t == FUTURO]
        pres_b = [idx for t, idx in bloque if t == PRESENTE]
        for fi in futs_b:
            for pi in pres_b:
                x1, y1 = pos_fut[fi]
                x2, y2 = pos_pre[pi]
                ax.plot([x1, x2], [y1, y2], color=color, alpha=0.20, lw=1.5, zorder=1)

    NODE_SIZE = 600

    # Nodos futuros — círculos con letra dentro
    for idx, b_idx in fut_bloque.items():
        x, y = pos_fut[idx]
        color = _COLORES[b_idx % len(_COLORES)]
        ax.scatter(x, y, color=color, s=NODE_SIZE, zorder=3,
                   edgecolors="black", linewidths=1.5, marker="o")
        label = _ABC[idx] if idx < len(_ABC) else str(idx)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)

    # Nodos presentes — cuadrados con letra minúscula dentro
    for idx, b_idx in pre_bloque.items():
        x, y = pos_pre[idx]
        color = _COLORES[b_idx % len(_COLORES)]
        ax.scatter(x, y, color=color, s=NODE_SIZE, zorder=3,
                   edgecolors="black", linewidths=1.5, marker="s")
        label = _ABC[idx].lower() if idx < len(_ABC) else str(idx)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=4)

    # Etiquetas de fila
    ax.text(-0.04, 1.0, "t+1", ha="right", va="center",
            fontsize=9, style="italic", color="#555555")
    ax.text(-0.04, 0.0, "t",   ha="right", va="center",
            fontsize=9, style="italic", color="#555555")

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.28, 1.28)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def plot_kparticion_hypercube(
    bloques: list[list[tuple[int, int]]],
    n_bits: int,
    titulo: str = "",
    ax=None,
) -> plt.Figure:
    """
    Dibuja la k-partición ``bloques``.

    Args:
        bloques: lista de bloques; cada bloque es lista de vértices
            ``(tiempo, índice)`` con ``tiempo`` ∈ {0=presente, 1=futuro}.
        n_bits: número de variables binarias del sistema.
        titulo: título opcional de la figura.
        ax: eje matplotlib existente (solo para n_bits=2).

    Returns:
        plt.Figure

    Notas:
        - n_bits ≤ 3: hipercubo exacto (cuadrado 2D o cubo 3D).
        - n_bits ≥ 4: vista bipartita — nodos futuros arriba, presentes abajo.
    """
    k = len(bloques)
    titulo_auto = titulo or f"k={k}-partición sobre N{n_bits}"

    if n_bits <= 3:
        # ── Hipercubo exacto ──────────────────────────────────────────────
        coords = _coordenadas_hipercubo(n_bits)
        aristas = _aristas_hipercubo(n_bits)
        fut_bloque, pre_bloque = _asignacion_bloques(bloques, n_bits)

        n_nodos = 2 ** n_bits
        etiquetas = [format(i, f"0{n_bits}b") for i in range(n_nodos)]
        colores: list[str] = []
        marcadores: list[str] = []
        for i in range(n_nodos):
            bloque_f = fut_bloque.get((FUTURO, i))
            bloque_p = pre_bloque.get((PRESENTE, i))
            if bloque_f is not None:
                b, m = bloque_f, "o"
            elif bloque_p is not None:
                b, m = bloque_p, "s"
            else:
                b, m = -1, "D"
            colores.append(_COLORES[b % len(_COLORES)] if b >= 0 else "#BDBDBD")
            marcadores.append(m)

        usar_3d = n_bits == 3
        if usar_3d:
            fig = plt.figure(figsize=(7, 6))
            ax3d = fig.add_subplot(111, projection="3d")
            _dibujar_3d(ax3d, coords, aristas, colores, marcadores, etiquetas)
            ax3d.set_xlabel("bit 0"); ax3d.set_ylabel("bit 1"); ax3d.set_zlabel("bit 2")
            ax3d.set_title(titulo_auto)
            leyenda_ax = ax3d
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 4))
            else:
                fig = ax.get_figure()
            _dibujar_2d(ax, coords, aristas, colores, marcadores, etiquetas)
            ax.set_title(titulo_auto)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            leyenda_ax = ax

        handles = [
            Patch(facecolor=_COLORES[b % len(_COLORES)], edgecolor="black",
                  label=f"Bloque {b}")
            for b in range(k)
        ]
        handles += [
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor="gray",
                          markeredgecolor="black", markersize=8, label="Nodo futuro"),
            mlines.Line2D([], [], marker="s", color="w", markerfacecolor="gray",
                          markeredgecolor="black", markersize=8, label="Nodo presente"),
        ]
        leyenda_ax.legend(handles=handles, loc="upper right", fontsize=7)

    else:
        # ── Vista bipartita ───────────────────────────────────────────────
        fig, ax_bp = plt.subplots(figsize=(max(6, n_bits * 0.7), 4))
        _dibujar_bipartito(ax_bp, bloques)
        ax_bp.set_title(titulo_auto, pad=10)

        handles = [
            Patch(facecolor=_COLORES[b % len(_COLORES)], edgecolor="black",
                  label=f"Bloque {b}")
            for b in range(k)
        ]
        handles += [
            mlines.Line2D([], [], marker="o", color="w", markerfacecolor="gray",
                          markeredgecolor="black", markersize=9, label="Nodo futuro  ○"),
            mlines.Line2D([], [], marker="s", color="w", markerfacecolor="gray",
                          markeredgecolor="black", markersize=9, label="Nodo presente □"),
        ]
        ax_bp.legend(handles=handles, loc="upper right", fontsize=8,
                     framealpha=0.9, borderpad=0.8)

    fig.tight_layout()
    return fig


def save_kparticion_hypercube(
    bloques: list[list[tuple[int, int]]],
    n_bits: int,
    path: str | Path,
    titulo: str = "",
    dpi: int = 150,
) -> None:
    """
    Genera la visualización de la k-partición y la guarda en ``path``.

    Args:
        bloques: k-partición (misma estructura que ``plot_kparticion_hypercube``).
        n_bits: dimensión del sistema.
        path: ruta del archivo de salida (PNG, PDF, etc.).
        titulo: título opcional.
        dpi: resolución de la imagen.
    """
    fig = plot_kparticion_hypercube(bloques, n_bits, titulo=titulo)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
