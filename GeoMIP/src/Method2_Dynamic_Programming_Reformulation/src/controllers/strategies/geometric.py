import heapq
from src.constants.error import ERROR_INCOMPATIBLE_SIZES
from src.models.core.system import System
from src.constants.base import NET_LABEL, STR_ZERO
from src.funcs.base import ABECEDARY
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto
from src.models.base.sia import SIA
from src.constants.base import (
    ACTUAL,
    EFECTO,
    TYPE_TAG,
)
from src.constants.models import (
    GEOMETRIC_ANALYSIS_TAG,
    GEOMETRIC_LABEL,
    GEOMETRIC_STRAREGY_TAG,
)
from src.controllers.manager import Manager
from src.funcs.format import fmt_biparte_q
from src.middlewares.profile import profiler_manager, profile
from src.models.core.solution import Solution
import numpy as np
import time
from typing import List, Dict, Tuple

from concurrent.futures import ThreadPoolExecutor
import itertools


class GeometricSIA(SIA):
    def __init__(self, gestor: Manager, k: int = 2):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.k = k
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)
        self.vertices: set[tuple]
        # Optimized storage: int key → numpy array of per-variable costs
        self.tabla_np: dict[int, np.ndarray] = {}
        self.memoria_particiones: dict[tuple, tuple[float, np.ndarray]] = {}
        # Precomputed in find_mip
        self._flat_matrix: np.ndarray
        self._powers: np.ndarray
        self._i0_int: int
        self._i_final_int: int

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: int = None,
    ):
        k = k if k is not None else self.k
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        futuro = tuple(
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )

        # Vectorized lookup matrix: shape (n_vars, 2^n_present)
        self._flat_matrix = np.stack(
            [ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos]
        )

        self.vertices = set(presente + futuro)
        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial

        mip_result = self.find_mip(k)

        if k == 2:
            fmt_mip = fmt_biparte_q(list(mip_result), self.nodes_complement(mip_result))
            return Solution(
                estrategia=GEOMETRIC_LABEL,
                perdida=self.memoria_particiones[mip_result][0],
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=self.memoria_particiones[mip_result][1],
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                particion=fmt_mip,
            )
        else:
            grupos_locales, emd_val, dist = mip_result
            return Solution(
                estrategia=GEOMETRIC_LABEL + f" k={k}",
                perdida=emd_val,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=dist,
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                particion=self._fmt_particion_k(grupos_locales),
            )

    def nodes_complement(self, nodes):
        return list(set(self.vertices) - set(nodes))

    def find_mip(self, k: int = 2):
        self.sia_logger.critic("empieza.")
        n = len(self.estado_inicial)
        n_vars = len(self.sia_subsistema.indices_ncubos)

        # Powers of 2: state[i] contributes 2^i to the integer index
        self._powers = 2 ** np.arange(n)
        self._i0_int = int(self.estado_inicial @ self._powers)
        self._i_final_int = int(self.estado_final @ self._powers)

        self.tabla_np = {}
        self.tabla_np[self._i0_int] = np.zeros(n_vars)
        self.caminos: Dict[int, List[List[int]]] = {0: [self.estado_inicial.tolist()]}

        for nivel in range(1, n + 1):
            self._calcular_costos_nivel(self.estado_final, nivel)

        if k == 2:
            self.memoria_particiones = {}
            return self._find_mip_k2()
        else:
            return self._find_mip_kn(k)

    def _calcular_costos_nivel(self, estado_final: np.ndarray, nivel: int):
        visitados: set[int] = set()
        self.caminos[nivel] = []
        for estado_prev in self.caminos[nivel - 1]:
            for bit in range(len(estado_final)):
                if estado_prev[bit] != estado_final[bit]:
                    nuevo = estado_prev.copy()
                    nuevo[bit] = estado_final[bit]
                    j_int = int(np.dot(nuevo, self._powers))
                    if j_int not in visitados:
                        self.caminos[nivel].append(nuevo)
                        self._calcular_costo(nuevo, j_int, nivel)
                        visitados.add(j_int)

    def _calcular_costo(self, j: list, j_int: int, d: int):
        """Calcula t_x(i0, j) para todas las variables simultáneamente."""
        raw = np.abs(
            self._flat_matrix[:, self._i0_int] - self._flat_matrix[:, j_int]
        )
        if d > 1:
            i0 = self.caminos[0][0]
            for bit, (jb, i0b) in enumerate(zip(j, i0)):
                if jb != i0b:
                    # Vecino de j un paso más cerca de i0 (XOR flip de ese bit)
                    k_int = j_int ^ (1 << bit)
                    raw = raw + self.tabla_np[k_int]
        self.tabla_np[j_int] = (0.5 ** d) * raw

    # ------------------------------------------------------------------
    # k = 2 path
    # ------------------------------------------------------------------

    def _find_mip_k2(self):
        """Genera y evalúa candidatos de bipartición (k=2)."""
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_pres = len(self.sia_subsistema.dims_ncubos)

        # Nivel 0: n_vars candidatos (uno por variable futura excluida)
        candidatos: list[tuple[list, list]] = []
        for idx in range(n_vars):
            presentes = list(range(n_pres))
            futuros = [i for i in range(n_vars) if i != idx]
            candidatos.append((presentes, futuros))

        # Niveles intermedios
        es_par = len(self.caminos) % 2 == 0
        mitad = len(self.caminos) // 2 if es_par else (len(self.caminos) // 2) + 1

        for nivel in range(1, mitad):
            costo_mejor = 1e5
            p_mejor: list = []
            f_mejor: list = []
            for estado in self.caminos[nivel]:
                i0 = self.caminos[0][0]
                estado_int = int(np.dot(estado, self._powers))
                estado_comp = (1 - np.array(estado)).tolist()
                comp_int = int(np.dot(estado_comp, self._powers))

                actual = self.tabla_np.get(estado_int)
                complementario = self.tabla_np.get(comp_int)
                if actual is None or complementario is None:
                    continue

                presentes = [i for i, (s, i0b) in enumerate(zip(estado, i0)) if s == i0b]
                futuros: list[int] = []
                costo = 0.0
                for idx in range(n_vars):
                    if actual[idx] <= complementario[idx]:
                        futuros.append(idx)
                        costo += actual[idx]
                    else:
                        costo += complementario[idx]

                if costo < costo_mejor:
                    costo_mejor = costo
                    p_mejor = presentes
                    f_mejor = futuros

            if f_mejor:
                candidatos.append((p_mejor, f_mejor))

        # Evaluar candidatos
        for presentes_local, futuros_local in candidatos:
            if not futuros_local:
                continue
            p_g = self.sia_subsistema.dims_ncubos[np.array(presentes_local, dtype=np.int8)]
            f_g = self.sia_subsistema.indices_ncubos[np.array(futuros_local, dtype=np.int8)]
            dist = self.sia_subsistema.bipartir(f_g, p_g).distribucion_marginal()
            emd_val = emd_efecto(dist, self.sia_dists_marginales)
            key = tuple([(0, n) for n in p_g] + [(1, n) for n in f_g])
            self.memoria_particiones[key] = (emd_val, dist)

        return min(self.memoria_particiones, key=lambda kk: self.memoria_particiones[kk][0])

    # ------------------------------------------------------------------
    # k > 2 path
    # ------------------------------------------------------------------

    def _find_mip_kn(self, k: int) -> tuple:
        """Genera y evalúa candidatos de k-partición para k>2."""
        candidatos = self._candidatos_k(k)
        mejor_emd = float("inf")
        mejor_grupos: list | None = None
        mejor_dist: np.ndarray | None = None

        for grupos in candidatos:
            try:
                emd_val, dist = self._evaluar_particion_k(grupos)
                if emd_val < mejor_emd:
                    mejor_emd = emd_val
                    mejor_grupos = grupos
                    mejor_dist = dist
            except Exception:
                continue

        if mejor_grupos is None:
            # Fallback: división equitativa
            n_vars = len(self.sia_subsistema.indices_ncubos)
            step = max(1, n_vars // k)
            grupos_fb = [list(range(i, min(i + step, n_vars))) for i in range(0, n_vars, step)][:k]
            mejor_emd, mejor_dist = self._evaluar_particion_k(grupos_fb)
            mejor_grupos = grupos_fb

        return mejor_grupos, mejor_emd, mejor_dist

    def _candidatos_k(self, k: int) -> list:
        """
        Genera candidatos de k-partición ordenando variables por costo en la
        tabla de transiciones y explorando distintos puntos de corte.
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        costos = self.tabla_np[self._i_final_int]
        orden = np.argsort(costos).tolist()       # variables ordenadas de menor a mayor costo
        costos_ord = costos[np.array(orden)]

        candidatos: list[list] = []

        def make_grupos(cortes_sorted: list[int]) -> list[list[int]] | None:
            gs, prev = [], 0
            for c in cortes_sorted:
                gs.append(orden[prev:c])
                prev = c
            gs.append(orden[prev:])
            return gs if len(gs) == k and all(g for g in gs) else None

        # 1. Cortes en los mayores saltos de costo (naturales)
        gaps = np.diff(costos_ord)
        if len(gaps) >= k - 1:
            top_cortes = sorted((np.argsort(gaps)[::-1][:k - 1] + 1).tolist())
            g = make_grupos(top_cortes)
            if g:
                candidatos.append(g)

            # Variaciones ±1, ±2 alrededor de cada punto de corte natural
            for shift_idx in range(len(top_cortes)):
                for delta in (-2, -1, 1, 2):
                    new_cortes = top_cortes.copy()
                    new_cortes[shift_idx] = max(1, min(n_vars - 1, new_cortes[shift_idx] + delta))
                    new_cortes_sorted = sorted(set(new_cortes))
                    if len(new_cortes_sorted) == k - 1:
                        g = make_grupos(new_cortes_sorted)
                        if g and g not in candidatos:
                            candidatos.append(g)

        # 2. División equitativa
        step = n_vars // k
        if step >= 1:
            eq_cortes = [step * i for i in range(1, k)]
            g = make_grupos(eq_cortes)
            if g and g not in candidatos:
                candidatos.append(g)

        # 3. Candidatos basados en niveles intermedios del camino óptimo
        es_par = len(self.caminos) % 2 == 0
        mitad = len(self.caminos) // 2 if es_par else (len(self.caminos) // 2) + 1

        for nivel in range(1, mitad):
            for estado in self.caminos[nivel]:
                i0 = self.caminos[0][0]
                flipeadas = [i for i in range(n_vars) if i < len(estado) and estado[i] != i0[i]]
                no_flip = [i for i in range(n_vars) if i < len(estado) and estado[i] == i0[i]]
                if not flipeadas or not no_flip:
                    continue
                all_vars = flipeadas + no_flip
                step_l = max(1, n_vars // k)
                g_lv = [all_vars[i:i + step_l] for i in range(0, n_vars, step_l)]
                g_lv = [g for g in g_lv if g]
                # Fusionar grupos excedentes en el último
                while len(g_lv) > k:
                    g_lv[-2].extend(g_lv[-1])
                    g_lv.pop()
                if len(g_lv) == k and g_lv not in candidatos:
                    candidatos.append(g_lv)
            if len(candidatos) >= 150:
                break

        return candidatos[:150]

    def _evaluar_particion_k(self, grupos_locales: list) -> tuple[float, np.ndarray]:
        """
        Evalúa EMD de una k-partición.
        grupos_locales: lista de listas con índices locales de variables futuras.
        """
        n_fut = len(self.sia_subsistema.indices_ncubos)
        n_pres = len(self.sia_subsistema.dims_ncubos)
        symmetric = (n_fut == n_pres)

        particion_k = []
        for grupo in grupos_locales:
            if not grupo:
                continue
            g_arr = np.array(sorted(grupo), dtype=np.int8)
            alc_global = self.sia_subsistema.indices_ncubos[g_arr]
            # Mecanismo: mismas posiciones locales si n_fut == n_pres;
            # si no, se asigna el mecanismo completo (corte unilateral).
            if symmetric:
                mec_global = self.sia_subsistema.dims_ncubos[g_arr]
            else:
                mec_global = self.sia_subsistema.dims_ncubos
            particion_k.append((alc_global, mec_global))

        dist = self.sia_subsistema.bipartir_k(particion_k).distribucion_marginal()
        emd_val = emd_efecto(dist, self.sia_dists_marginales)
        return emd_val, dist

    def _fmt_particion_k(self, grupos_locales: list) -> str:
        """Formatea la k-partición igual que fmt_biparte_q pero para k grupos."""
        abecedary_lower = [a.lower() for a in ABECEDARY]
        n_fut = len(self.sia_subsistema.indices_ncubos)
        n_pres = len(self.sia_subsistema.dims_ncubos)
        symmetric = (n_fut == n_pres)

        fut_parts = []
        pres_parts = []
        for grupo in grupos_locales:
            fut_labels = [
                ABECEDARY[int(self.sia_subsistema.indices_ncubos[j])]
                for j in sorted(grupo)
                if int(self.sia_subsistema.indices_ncubos[j]) < len(ABECEDARY)
            ]
            fut_parts.append(",".join(fut_labels) if fut_labels else "∅")

            if symmetric:
                pres_labels = [
                    abecedary_lower[int(self.sia_subsistema.dims_ncubos[j])]
                    for j in sorted(grupo)
                    if int(self.sia_subsistema.dims_ncubos[j]) < len(abecedary_lower)
                ]
                pres_parts.append(",".join(pres_labels) if pres_labels else "∅")
            else:
                pres_parts.append("∅")

        top = "| " + " || ".join(fut_parts) + " |"
        bot = "| " + " || ".join(pres_parts) + " |"
        return top + "\n" + bot

    def hamming(self, a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))
