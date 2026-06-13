"""
Visualización de k-particiones sobre el hipercubo n-dimensional.

Función principal:
    plot_kparticion_hypercube(bloques, n_bits) → plt.Figure

    Dibuja el hipercubo binario de n_bits dimensiones con los nodos
    coloreados por bloque de la k-partición. Las aristas conectan vértices
    a distancia de Hamming 1. Los nodos presentes se muestran con relleno
    sólido; los futuros, con borde grueso.

Compatibilidad dimensional:
    - n_bits = 2: cuadrado 2D.
    - n_bits = 3: cubo 3D con Axes3D.
    - n_bits ≥ 4: proyección PCA 2D de las coordenadas binarias.

Ejemplo de uso:
    from src.controllers.manager import Manager
    from src.strategies.KQNodes import KQNodes
    from src.funcs.visualize import save_kparticion_hypercube

    tpm  = Manager("1000").cargar_red()
    kq   = KQNodes(tpm)
    sol  = kq.aplicar_estrategia("1000", "1111", "1111", "1111", 3)
    save_kparticion_hypercube(kq._last_bloques, 4, "particion_k3.png")
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — necesario para projection='3d'

# Tiempo convencional: 1 = futuro (alcance), 0 = presente (mecanismo).
FUTURO, PRESENTE = 1, 0

# Paleta de hasta 6 bloques (colores distinguibles).
_COLORES = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#6D4C41"]


# ---------------------------------------------------------------------------
# Coordenadas del hipercubo
# ---------------------------------------------------------------------------

def _coordenadas_hipercubo(n: int) -> np.ndarray:
    """
    Devuelve la matriz (2^n, n) de coordenadas binarias del hipercubo n-dimensional.
    El vértice i tiene coordenadas = representación binaria de i (little-endian).
    """
    indices = np.arange(2 ** n)
    coords = ((indices[:, None] >> np.arange(n)) & 1).astype(float)
    return coords


def _aristas_hipercubo(n: int) -> list[tuple[int, int]]:
    """Lista de pares (u, v) con distancia de Hamming 1."""
    aristas = []
    for u in range(2 ** n):
        for bit in range(n):
            v = u ^ (1 << bit)
            if v > u:
                aristas.append((u, v))
    return aristas


def _proyeccion_pca_2d(coords: np.ndarray) -> np.ndarray:
    """Proyecta coords (2^n, n) a 2D via PCA (sin dependencia sklearn)."""
    centro = coords.mean(axis=0)
    X = coords - centro
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


# ---------------------------------------------------------------------------
# Asignación bloque → color
# ---------------------------------------------------------------------------

def _asignacion_bloques(
    bloques: list[list[tuple[int, int]]], n_bits: int
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """
    Devuelve dos dicts: vértice → índice de bloque.
    Uno para nodos futuros (vértice = (FUTURO, i)) y otro para presentes.
    """
    fut_bloque: dict[tuple[int, int], int] = {}
    pre_bloque: dict[tuple[int, int], int] = {}
    for b_idx, bloque in enumerate(bloques):
        for tiempo, idx in bloque:
            clave = (tiempo, idx)
            if tiempo == FUTURO:
                fut_bloque[clave] = b_idx
            else:
                pre_bloque[clave] = b_idx
    return fut_bloque, pre_bloque


# ---------------------------------------------------------------------------
# Dibujado 2D
# ---------------------------------------------------------------------------

def _dibujar_2d(
    ax,
    coords2d: np.ndarray,
    aristas: list[tuple[int, int]],
    colores_nodos: list[str],
    marcadores: list[str],
    etiquetas_nodos: list[str],
):
    for u, v in aristas:
        ax.plot(
            [coords2d[u, 0], coords2d[v, 0]],
            [coords2d[u, 1], coords2d[v, 1]],
            color="lightgray", linewidth=1, zorder=1,
        )
    for i, (x, y) in enumerate(coords2d):
        ax.scatter(
            x, y,
            color=colores_nodos[i],
            marker=marcadores[i],
            s=160, zorder=3, edgecolors="black", linewidths=1.2,
        )
        ax.annotate(
            etiquetas_nodos[i], (x, y),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8,
        )


def _dibujar_3d(
    ax,
    coords3d: np.ndarray,
    aristas: list[tuple[int, int]],
    colores_nodos: list[str],
    marcadores: list[str],
    etiquetas_nodos: list[str],
):
    for u, v in aristas:
        ax.plot(
            [coords3d[u, 0], coords3d[v, 0]],
            [coords3d[u, 1], coords3d[v, 1]],
            [coords3d[u, 2], coords3d[v, 2]],
            color="lightgray", linewidth=0.8,
        )
    for i, (x, y, z) in enumerate(coords3d):
        ax.scatter(
            x, y, z,
            color=colores_nodos[i],
            marker=marcadores[i],
            s=120, edgecolors="black", linewidths=1.0,
        )
        ax.text(x, y, z + 0.05, etiquetas_nodos[i], fontsize=7)


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
    Dibuja la k-partición ``bloques`` sobre el hipercubo n-dimensional.

    Args:
        bloques: lista de bloques; cada bloque es una lista de vértices
            ``(tiempo, índice)`` donde ``tiempo`` ∈ {0=presente, 1=futuro}.
        n_bits: número de variables binarias (dimensión del hipercubo).
            Los nodos del hipercubo son los enteros 0..2^n_bits − 1.
        titulo: título opcional de la figura.
        ax: eje matplotlib existente (solo para n_bits=2; ignorado en 3D).

    Returns:
        plt.Figure: la figura generada (no se muestra automáticamente).

    Notas:
        - Para n_bits ≤ 3: dibuja el cubo 3D real (o cuadrado si n_bits=2).
        - Para n_bits ≥ 4: proyecta a 2D con PCA (orden de vértices preservado).
        - Nodos presentes: marcador cuadrado (``s``); futuros: círculo (``o``).
        - Colores: hasta 6 bloques (paleta _COLORES); bloques extras en gris.
    """
    coords = _coordenadas_hipercubo(n_bits)
    aristas = _aristas_hipercubo(n_bits)
    fut_bloque, pre_bloque = _asignacion_bloques(bloques, n_bits)

    n_nodos = 2 ** n_bits
    # Etiquetas: índice binario de cada nodo del hipercubo.
    etiquetas = [format(i, f"0{n_bits}b") for i in range(n_nodos)]

    # Asignar color y marcador a cada nodo del hipercubo.
    # Los nodos futuros se mapean por índice; los presentes también.
    colores: list[str] = []
    marcadores: list[str] = []
    for i in range(n_nodos):
        bloque_f = fut_bloque.get((FUTURO, i))
        bloque_p = pre_bloque.get((PRESENTE, i))
        if bloque_f is not None:
            b = bloque_f
            marcadores.append("o")
        elif bloque_p is not None:
            b = bloque_p
            marcadores.append("s")
        else:
            b = -1
            marcadores.append("D")
        colores.append(_COLORES[b % len(_COLORES)] if b >= 0 else "#BDBDBD")

    usar_3d = n_bits == 3

    if usar_3d:
        fig = plt.figure(figsize=(7, 6))
        ax3d = fig.add_subplot(111, projection="3d")
        _dibujar_3d(ax3d, coords, aristas, colores, marcadores, etiquetas)
        ax3d.set_xlabel("bit 0"); ax3d.set_ylabel("bit 1"); ax3d.set_zlabel("bit 2")
        ax3d.set_title(titulo or f"k-partición sobre hipercubo {n_bits}D")
    else:
        if n_bits == 2:
            coords2d = coords
        else:
            coords2d = _proyeccion_pca_2d(coords)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.get_figure()
        _dibujar_2d(ax, coords2d, aristas, colores, marcadores, etiquetas)
        ax.set_title(titulo or f"k-partición sobre hipercubo {n_bits}D")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    # Leyenda de bloques.
    leyenda_ax = ax3d if usar_3d else ax
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=_COLORES[b % len(_COLORES)], edgecolor="black",
              label=f"Bloque {b}")
        for b in range(len(bloques))
    ]
    marcador_leyenda = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=8, label="Nodo futuro"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=8, label="Nodo presente"),
    ]
    leyenda_ax.legend(handles=handles + marcador_leyenda,
                      loc="upper right", fontsize=7)
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
        n_bits: dimensión del hipercubo.
        path: ruta del archivo de salida (PNG, PDF, etc.).
        titulo: título opcional.
        dpi: resolución de la imagen.
    """
    fig = plot_kparticion_hypercube(bloques, n_bits, titulo=titulo)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
