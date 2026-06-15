"""
Estado incremental del recocido simulado para KQNodes.

Mantiene una matriz de costes ``coste[lab][i]`` —coste del futuro ``i``
respecto al bloque ``lab`` (``None`` para bloques sin presentes)— y el vector
constante ``c0`` (coste de cada futuro al bloque-vacío). Tras mover UN presente
de bloque solo se recalculan las ≤ 2 columnas afectadas en lugar de reconstruir
los grupos completos desde cero.
"""

from src.constants.base import INFTY_POS


class RecocidoIncremental:
    """
    Evaluador INCREMENTAL de ``δ_k`` para el recocido simulado de KQNodes.

    El ``δ`` producido coincide EXACTAMENTE con ``KQNodes._delta_grupos``
    (misma asignación voraz y misma regla de cubrimiento de bloques vacíos).
    La δ que ``find_kmip_recocido`` DEVUELVE se recalcula al final con
    ``_evaluar_grupos`` para garantizar consistencia con el método exacto.
    """

    __slots__ = (
        "_costo", "k", "m", "presentes", "cubos", "base_f", "c0",
        "etq", "miembros", "coste",
    )

    def __init__(self, costo_fn, etq, k, presentes, cubos_futuro, base_f, c0):
        self._costo = costo_fn
        self.k = k
        self.m = len(cubos_futuro)
        self.presentes = presentes
        self.cubos = cubos_futuro
        self.base_f = base_f
        self.c0 = c0  # list[float] (m,), constante
        self.etq = list(etq)
        self.miembros: list[set] = [set() for _ in range(k)]
        for pos, lab in enumerate(etq):
            self.miembros[lab].add(presentes[pos])
        self.coste: list = [None] * k
        for lab in range(k):
            self._recompute(lab)

    def _recompute(self, lab: int) -> None:
        """Recalcula la columna del bloque `lab` tras cambiar sus miembros.

        Reemplaza la lista completa (no la muta in situ), de modo que un
        snapshot puede conservar la referencia anterior sin copiarla.
        """
        miembros = self.miembros[lab]
        if miembros:
            q = frozenset(miembros)
            costo, cubos, base_f = self._costo, self.cubos, self.base_f
            self.coste[lab] = [costo(cubos[i], base_f[i], q) for i in range(self.m)]
        else:
            self.coste[lab] = None

    def delta(self) -> float:
        """δ_k actual: asignación voraz de cada futuro a su mejor bloque +
        cubrimiento de los ``k - r`` bloques vacíos."""
        coste = self.coste
        nonempty = [c for c in coste if c is not None]
        r = len(nonempty)
        vacios = self.k - r
        if vacios < 0 or vacios > self.m:
            return INFTY_POS

        total = 0.0
        if vacios == 0:
            for i in range(self.m):
                mejor = INFTY_POS
                for col in nonempty:
                    v = col[i]
                    if v < mejor:
                        mejor = v
                total += mejor
            return total

        c0 = self.c0
        n_en_vacio = 0
        sobrecostos: list[float] = []
        for i in range(self.m):
            mejor = INFTY_POS
            for col in nonempty:
                v = col[i]
                if v < mejor:
                    mejor = v
            c0i = c0[i]
            if c0i <= mejor:
                total += c0i
                n_en_vacio += 1
            else:
                total += mejor
                sobrecostos.append(c0i - mejor)
        faltan = vacios - n_en_vacio
        if faltan > 0:
            if len(sobrecostos) < faltan:
                return INFTY_POS
            sobrecostos.sort()
            total += sum(sobrecostos[:faltan])
        return total

    def _snapshot(self, labs, etq_entries):
        """Captura columnas/miembros de `labs` y entradas de etq para revertir.

        ``coste[lab]`` se reemplaza (no se muta) en ``_recompute``, así que
        basta guardar la referencia; ``miembros[lab]`` sí se muta, por lo que
        se copia.
        """
        return (
            {lab: (self.coste[lab], set(self.miembros[lab])) for lab in set(labs)},
            dict(etq_entries),
        )

    def restaurar(self, undo) -> None:
        """Deshace un movimiento (restaura columnas, miembros y etiquetas)."""
        cols, etq_entries = undo
        for lab, (col, miembros) in cols.items():
            self.coste[lab] = col
            self.miembros[lab] = miembros
        for pos, lab in etq_entries.items():
            self.etq[pos] = lab

    def mover(self, pos: int, nuevo_lab: int):
        """Reetiqueta el presente en `pos` a `nuevo_lab` (≠ actual). Devuelve token undo."""
        viejo = self.etq[pos]
        undo = self._snapshot((viejo, nuevo_lab), {pos: viejo})
        pid = self.presentes[pos]
        self.miembros[viejo].discard(pid)
        self.miembros[nuevo_lab].add(pid)
        self.etq[pos] = nuevo_lab
        self._recompute(viejo)
        self._recompute(nuevo_lab)
        return undo

    def swap(self, a: int, b: int):
        """Intercambia las etiquetas de los presentes en `a` y `b` (labs ≠). Token undo."""
        la, lb = self.etq[a], self.etq[b]
        undo = self._snapshot((la, lb), {a: la, b: lb})
        pa, pb = self.presentes[a], self.presentes[b]
        self.miembros[la].discard(pa)
        self.miembros[la].add(pb)
        self.miembros[lb].discard(pb)
        self.miembros[lb].add(pa)
        self.etq[a], self.etq[b] = lb, la
        self._recompute(la)
        self._recompute(lb)
        return undo
