import numpy as np
from typing import Tuple, List, Set
from numpy.typing import NDArray
from .tabla_t import TablaT


def warm_start(n: int, k: int, T: TablaT) -> Tuple[List[Set[int]], int]:
    """
    Construye partición inicial informada por T y detecta k_natural.

    Returns
    -------
    particion_inicial : List[Set[int]]
    k_natural         : int — particiones naturales según gap espectral
    """
    W = _grafo_afinidad(n, T)
    k_natural = _detectar_k_natural(W, k_max=min(n, 8))

    if abs(k - k_natural) > 2:
        print(
            f"[WarmStart] k={k} solicitado, k_natural={k_natural}. "
            "Particiones extra introducirán mayor pérdida de información."
        )

    particion = _clustering_espectral(W, k, n)
    particion = [g for g in particion if g]

    while len(particion) < k:
        particion = _subdividir_mayor(particion, W)

    return particion[:k], k_natural


def _grafo_afinidad(n: int, T: TablaT) -> NDArray[np.float64]:
    W = np.zeros((n, n))
    for vi in range(n):
        for vj in range(vi + 1, n):
            af = np.exp(-T.costos_cruzados(vi, vj))
            W[vi, vj] = af
            W[vj, vi] = af
    mx = W.max()
    if mx > 0:
        W /= mx
    return W


def _detectar_k_natural(W: NDArray[np.float64], k_max: int) -> int:
    n = W.shape[0]
    grados = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(grados + 1e-10))
    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    evals = np.sort(np.linalg.eigvalsh(L))
    k_max_real = min(k_max, n - 1)
    gaps = np.diff(evals[1:k_max_real + 1])
    if len(gaps) == 0:
        return 2
    return max(2, int(np.argmax(gaps)) + 2)


def _kmeans_numpy(X: NDArray[np.float64], k: int, n_init: int = 10) -> NDArray[np.int64]:
    """K-means de Lloyd implementado con numpy (sin sklearn)."""
    rng = np.random.default_rng(42)
    n = X.shape[0]
    best_labels = np.zeros(n, dtype=np.int64)
    best_inertia = float('inf')

    for _ in range(n_init):
        idx = rng.choice(n, size=k, replace=False)
        centers = X[idx].copy()

        for _ in range(300):
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = np.array([
                X[labels == c].mean(axis=0) if (labels == c).any() else centers[c]
                for c in range(k)
            ])
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        dists_final = np.linalg.norm(X - centers[labels], axis=1)
        inertia = (dists_final ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def _clustering_espectral(W: NDArray[np.float64], k: int, n: int) -> List[Set[int]]:
    if n <= 20:
        return _espectral_exacto(W, k)
    return _espectral_aproximado(W, k, bloque=n // 5)


def _espectral_exacto(W: NDArray[np.float64], k: int) -> List[Set[int]]:
    n = W.shape[0]
    k = min(k, n)
    grados = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(grados + 1e-10))
    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    _, vecs = np.linalg.eigh(L)
    U = vecs[:, :k]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U_norm = U / (norms + 1e-10)
    labels = _kmeans_numpy(U_norm, k)
    return [{i for i in range(n) if labels[i] == c} for c in range(k)]


def _espectral_aproximado(W: NDArray[np.float64], k: int, bloque: int) -> List[Set[int]]:
    n = W.shape[0]
    grupos: List[Set[int]] = []
    for inicio in range(0, n, bloque):
        fin = min(inicio + bloque, n)
        idx = list(range(inicio, fin))
        W_b = W[np.ix_(idx, idx)]
        k_b = max(1, k * len(idx) // n)
        for sg in _espectral_exacto(W_b, k_b):
            grupos.append({idx[i] for i in sg})

    while len(grupos) > k:
        mi, mj, ma = 0, 1, -1.0
        for i in range(len(grupos)):
            for j in range(i + 1, len(grupos)):
                af = sum(W[a, b] for a in grupos[i] for b in grupos[j])
                af /= max(len(grupos[i]) * len(grupos[j]), 1)
                if af > ma:
                    ma, mi, mj = af, i, j
        grupos[mi] |= grupos[mj]
        grupos.pop(mj)
    return grupos


def _subdividir_mayor(grupos: List[Set[int]], W: NDArray[np.float64]) -> List[Set[int]]:
    idx = max(range(len(grupos)), key=lambda i: len(grupos[i]))
    grupo = list(grupos[idx])
    if len(grupo) < 2:
        return grupos
    W_s = W[np.ix_(grupo, grupo)]
    sub = _espectral_exacto(W_s, 2)
    grupos.pop(idx)
    for sg in sub:
        grupos.append({grupo[i] for i in sg})
    return grupos
