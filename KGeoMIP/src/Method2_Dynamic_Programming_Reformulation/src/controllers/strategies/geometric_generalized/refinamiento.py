from typing import List, Set
from .tabla_t import TablaT
from .score import score_global
from .utils import variables_en_frontera


def refinar_fronteras(
    particion: List[Set[int]],
    T: TablaT,
    k: int,
    n: int,
    max_iter: int = None,
) -> List[Set[int]]:
    """
    Refinamiento local iterativo hasta convergencia.

    Para cada variable evalúa reasignación a todas las otras k-1 particiones
    y acepta el movimiento si mejora el score global.
    Compensa el beam search greedy (b=1) en sistemas grandes.
    """
    if max_iter is None:
        max_iter = n * k

    mejoro = True
    it = 0

    while mejoro and it < max_iter:
        mejoro = False
        score_act = score_global(particion, T, k, n)

        for variable in variables_en_frontera(particion, k):
            idx_orig = next(
                (i for i, Si in enumerate(particion) if variable in Si), None
            )
            if idx_orig is None or len(particion[idx_orig]) <= 1:
                continue

            mejor_p = None
            mejor_s = score_act

            for idx_dest in range(len(particion)):
                if idx_dest == idx_orig:
                    continue

                cand = [set(p) for p in particion]
                cand[idx_orig].remove(variable)
                cand[idx_dest].add(variable)
                cand = [p for p in cand if p]

                if len(cand) < 2:
                    continue

                s = score_global(cand, T, k, n)
                if s < mejor_s:
                    mejor_s = s
                    mejor_p = cand

            if mejor_p is not None:
                particion = mejor_p
                score_act = mejor_s
                mejoro = True

        it += 1

    return particion
