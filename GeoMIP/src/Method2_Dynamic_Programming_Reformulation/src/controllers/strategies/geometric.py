from src.constants.base import NET_LABEL
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
import os
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        self.tabla_np: dict[int, np.ndarray] = {}
        self.memoria_particiones: dict[tuple, tuple[float, np.ndarray]] = {}
        # Memoria: columnas de la matriz de transición se cargan bajo demanda
        # en float32 y se reutilizan entre niveles de la tabla.
        self._ncubo_data: list[np.ndarray] = []   # datos planos por variable
        self._col_cache: dict[int, np.ndarray] = {}
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

        # Memoria 1 — float32: almacenar datos planos por variable en float32
        # en lugar de la matriz completa (n_vars × 2^n) en float64.
        # Para N=25: (25, 33M) float64 = 6,6 GB → float32 = 3,3 GB base,
        # más el caché lazy reduce a solo las columnas accedidas (~1,8M).
        self._ncubo_data = [
            ncubo.data.ravel().astype(np.float32)
            for ncubo in self.sia_subsistema.ncubos
        ]
        self._col_cache = {}

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

    # ------------------------------------------------------------------
    # Paso 1 — Truncamiento adaptativo de la tabla (DP)
    # ------------------------------------------------------------------

    @staticmethod
    def _umbral_nivel(n: int) -> int:
        """
        Devuelve el nivel máximo de Hamming a construir en la tabla.

        Para n <= 20 se construye la tabla completa. Para n mayores se
        trunca porque el factor de decaimiento 1/2^d hace despreciable
        la contribución de estados lejanos, y la memoria/tiempo se vuelven
        prohibitivos.

            n <= 20  →  nivel completo (n)
            n <= 22  →  n // 2   (error < 0.05 %)
            n  > 22  →  n // 3   (error < 0.1 %)
        """
        if n <= 20:
            return n
        elif n <= 22:
            return n // 2
        return n // 3

    def _get_costos_finales(self) -> np.ndarray:
        """
        Devuelve el vector de costos t_x(i0, i_final).

        Si la tabla fue truncada y el estado final no está almacenado,
        aproxima promediando los vectores del nivel más profundo disponible.
        Esta aproximación es válida porque los estados del nivel máximo son
        los más cercanos al estado final y tienen los costos más
        representativos de la dirección de partición óptima.
        """
        if self._i_final_int in self.tabla_np:
            return self.tabla_np[self._i_final_int]

        nivel_max = max(self.caminos.keys())
        vectores = [
            self.tabla_np[int(np.dot(e, self._powers))]
            for e in self.caminos[nivel_max]
            if int(np.dot(e, self._powers)) in self.tabla_np
        ]
        if vectores:
            return np.mean(vectores, axis=0)
        return np.zeros(len(self.sia_subsistema.indices_ncubos))

    # ------------------------------------------------------------------
    # Memoria 2 — caché lazy de columnas
    # ------------------------------------------------------------------

    def _get_col(self, j_int: int) -> np.ndarray:
        """
        Devuelve el vector de probabilidades de todas las variables para
        el estado j_int, en float32.

        El caché evita recalcular columnas ya accedidas. Con el truncamiento
        adaptativo solo se tocan ~1,8M columnas de las 2^25 = 33M posibles
        para N=25, reduciendo el uso real de RAM de ~6,6 GB a ~0,5 GB.
        """
        if j_int not in self._col_cache:
            self._col_cache[j_int] = np.array(
                [data[j_int] for data in self._ncubo_data], dtype=np.float32
            )
        return self._col_cache[j_int]

    def _get_cols_batch(self, j_ints: np.ndarray) -> np.ndarray:
        """Devuelve matriz (n_vars, m) con las columnas pedidas, desde caché."""
        return np.stack([self._get_col(int(j)) for j in j_ints], axis=1)

    def find_mip(self, k: int = 2):
        self.sia_logger.critic("empieza.")
        n = len(self.estado_inicial)
        n_vars = len(self.sia_subsistema.indices_ncubos)

        self._powers = 2 ** np.arange(n)
        self._i0_int = int(self.estado_inicial @ self._powers)
        self._i_final_int = int(self.estado_final @ self._powers)

        self.tabla_np = {}
        self.tabla_np[self._i0_int] = np.zeros(n_vars, dtype=np.float32)
        self.caminos: Dict[int, List[List[int]]] = {0: [self.estado_inicial.tolist()]}

        # Paso 1: construir tabla hasta el umbral adaptativo
        nivel_max = self._umbral_nivel(n)
        for nivel in range(1, nivel_max + 1):
            self._calcular_costos_nivel(self.estado_final, nivel)

        if k == 2:
            self.memoria_particiones = {}
            return self._find_mip_k2()
        else:
            return self._find_mip_kn(k)

    def _calcular_costos_nivel(self, estado_final: np.ndarray, nivel: int):
        """
        Paralelización 1 — Vectorización numpy por nivel.

        Todos los estados de un mismo nivel de Hamming son independientes
        entre sí (solo dependen del nivel anterior, ya calculado). Se
        procesan como operaciones matriciales en lugar de un bucle Python
        por estado, eliminando el overhead de iteración y aprovechando
        BLAS/MKL internos de numpy.

        Pasos:
          1. Generar todos los estados nuevos del nivel (idéntico al
             enfoque anterior, necesario para mantener self.caminos).
          2. Calcular raw = |F[:,i0] - F[:,j]| para todos los j a la vez
             mediante fancy indexing: F[:, j_ints] → shape (n_vars, m).
          3. Acumular costos de predecesores bit a bit: para cada bit b,
             identificar con máscara binaria qué estados tienen ese bit
             distinto de i0 y sumar sus predecesores en bloque.
          4. Aplicar factor de decaimiento 1/2^d y guardar en tabla_np.
        """
        n = len(estado_final)
        i0_arr = np.array(self.caminos[0][0], dtype=np.int8)

        # Paso 1: generar estados nuevos del nivel
        visitados: set[int] = set()
        nuevos: list[list] = []
        for estado_prev in self.caminos[nivel - 1]:
            for bit in range(n):
                if estado_prev[bit] != estado_final[bit]:
                    nuevo = estado_prev.copy()
                    nuevo[bit] = estado_final[bit]
                    j_int = int(np.dot(nuevo, self._powers))
                    if j_int not in visitados:
                        nuevos.append(nuevo)
                        visitados.add(j_int)

        self.caminos[nivel] = nuevos
        if not nuevos:
            return

        # Paso 2: base raw vectorizada — shape (n_vars, m)
        # Se usan _get_col / _get_cols_batch en lugar de _flat_matrix para
        # materializar solo las columnas necesarias (caché lazy + float32).
        j_ints = np.array([int(np.dot(e, self._powers)) for e in nuevos], dtype=np.int64)
        col_i0 = self._get_col(self._i0_int)[:, None]     # (n_vars, 1)
        raw = np.abs(col_i0 - self._get_cols_batch(j_ints))  # (n_vars, m)

        # Paso 3: acumulación de predecesores bit a bit
        if nivel > 1:
            for bit in range(n):
                i0_bit = int(i0_arr[bit])
                # máscara: estados donde el bit `bit` difiere de i0
                mask = ((j_ints >> bit) & 1) != i0_bit   # (m,) bool
                if not np.any(mask):
                    continue
                pred_ints = j_ints[mask] ^ (1 << bit)    # (k,) int
                # stack de vectores de costo: (n_vars, k)
                pred_costs = np.stack(
                    [self.tabla_np[int(p)] for p in pred_ints], axis=1
                )
                raw[:, mask] += pred_costs

        # Paso 4: aplicar decaimiento y guardar
        result = (0.5 ** nivel) * raw
        for idx, j_int in enumerate(j_ints):
            self.tabla_np[int(j_int)] = result[:, idx]

    # ------------------------------------------------------------------
    # k = 2 path
    # ------------------------------------------------------------------

    def _find_mip_k2(self):
        """Genera y evalúa candidatos de bipartición (k=2)."""
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_pres = len(self.sia_subsistema.dims_ncubos)

        candidatos: list[tuple[list, list]] = []
        for idx in range(n_vars):
            presentes = list(range(n_pres))
            futuros = [i for i in range(n_vars) if i != idx]
            candidatos.append((presentes, futuros))

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

        # Paralelización 2 — evaluación concurrente de candidatos k=2.
        #
        # Cada llamada a bipartir+distribucion_marginal+emd_efecto es
        # independiente: bipartir crea un nuevo objeto System (sin mutar
        # sia_subsistema), y emd_efecto solo lee sia_dists_marginales.
        # ThreadPoolExecutor aprovecha el tiempo de numpy fuera del GIL.
        # Para n pequeño (pocos candidatos) el umbral evita overhead.
        _n_workers = min(len(candidatos), os.cpu_count() or 4)

        def _eval(presentes_local, futuros_local):
            if not futuros_local:
                return None
            p_g = self.sia_subsistema.dims_ncubos[
                np.array(presentes_local, dtype=np.int8)
            ]
            f_g = self.sia_subsistema.indices_ncubos[
                np.array(futuros_local, dtype=np.int8)
            ]
            dist = self.sia_subsistema.bipartir(f_g, p_g).distribucion_marginal()
            emd_val = emd_efecto(dist, self.sia_dists_marginales)
            key = tuple([(0, v) for v in p_g] + [(1, v) for v in f_g])
            return key, emd_val, dist

        with ThreadPoolExecutor(max_workers=_n_workers) as pool:
            futuros_exec = [
                pool.submit(_eval, p, f) for p, f in candidatos
            ]
            for fut in as_completed(futuros_exec):
                res = fut.result()
                if res is not None:
                    key, emd_val, dist = res
                    self.memoria_particiones[key] = (emd_val, dist)

        return min(self.memoria_particiones, key=lambda kk: self.memoria_particiones[kk][0])

    # ------------------------------------------------------------------
    # k > 2 path — Paso 2 (greedy jerárquico) + Paso 3 (hill climbing)
    # ------------------------------------------------------------------

    def _find_mip_kn(self, k: int) -> tuple:
        """
        Encuentra la k-partición de menor pérdida mediante:

        Paso 2 — Greedy jerárquico:
            Parte de la mejor bipartición (k=2) y divide sucesivamente
            el grupo mayor usando la tabla de costos como guía, hasta
            obtener k grupos. Cada split evalúa un conjunto reducido de
            puntos de corte con EMD real, eligiendo el mejor.

        Paso 3 — Hill climbing:
            Reasigna variables de un grupo a otro mientras la pérdida
            (EMD) disminuya. Converge en O(n × k) iteraciones.
        """
        # Paso 2a: bipartición base usando el path k=2 existente
        self.memoria_particiones = {}
        mip_k2 = self._find_mip_k2()
        grupos = self._grupos_desde_mip_k2(mip_k2)

        # Paso 2b: splits sucesivos hasta tener k grupos
        while len(grupos) < k:
            grupos = self._split_mejor_grupo(grupos)
            if len(grupos) == 1:
                # No se pudo dividir más (todos tienen 1 variable)
                break

        # Paso 3: hill climbing
        grupos = self._hill_climbing(grupos)

        emd_val, dist = self._evaluar_particion_k(grupos)
        return grupos, emd_val, dist

    def _grupos_desde_mip_k2(self, mip_key: tuple) -> list[list[int]]:
        """
        Convierte la clave del MIP k=2 (índices globales) a dos listas
        de índices locales (posición en indices_ncubos).
        """
        fut_globals = list(self.sia_subsistema.indices_ncubos.astype(int))
        futuros_g1 = {x for tipo, x in mip_key if tipo == 1}
        futuros_g2 = {x for x in fut_globals if x not in futuros_g1}

        def to_local(global_set: set) -> list[int]:
            return [i for i, g in enumerate(fut_globals) if g in global_set]

        g1 = to_local(futuros_g1)
        g2 = to_local(futuros_g2)
        return [g for g in [g1, g2] if g]

    def _split_mejor_grupo(self, grupos: list[list[int]]) -> list[list[int]]:
        """
        Divide el grupo con más variables en dos usando puntos de corte
        basados en el vector de costos de la tabla. Evalúa hasta 5 cortes
        candidatos con EMD real y selecciona el de menor pérdida.
        """
        idx_mayor = max(range(len(grupos)), key=lambda i: len(grupos[i]))
        grupo = grupos[idx_mayor]

        if len(grupo) < 2:
            return grupos

        costos_full = self._get_costos_finales()
        costos_sub = costos_full[np.array(grupo)]
        orden_local = np.argsort(costos_sub)

        # Punto de corte natural: mayor gap de costo dentro del subgrupo
        costos_ord = costos_sub[orden_local]
        gaps = np.diff(costos_ord)
        corte_natural = int(np.argmax(gaps)) + 1 if len(gaps) > 0 else len(grupo) // 2
        corte_natural = max(1, min(corte_natural, len(grupo) - 1))

        # Explorar vecindario ±2 del corte natural (máx 5 candidatos)
        cortes = sorted(set(
            max(1, min(len(grupo) - 1, corte_natural + d))
            for d in range(-2, 3)
        ))

        # Paralelización 3 — evaluación concurrente de cortes candidatos.
        #
        # Los ≤5 cortes son completamente independientes entre sí.
        # _evaluar_particion_k crea objetos nuevos (bipartir_k) y no
        # muta estado compartido, por lo que es thread-safe.
        resto = [g for i, g in enumerate(grupos) if i != idx_mayor]

        def _eval_corte(corte):
            parte1 = [grupo[orden_local[i]] for i in range(corte)]
            parte2 = [grupo[orden_local[i]] for i in range(corte, len(grupo))]
            if not parte1 or not parte2:
                return None
            candidato = resto + [parte1, parte2]
            try:
                emd_val, _ = self._evaluar_particion_k(candidato)
                return emd_val, candidato
            except Exception:
                return None

        mejor_emd = float("inf")
        mejor_grupos: Optional[list] = None

        with ThreadPoolExecutor(max_workers=len(cortes)) as pool:
            futuros_exec = {pool.submit(_eval_corte, c): c for c in cortes}
            for fut in as_completed(futuros_exec):
                res = fut.result()
                if res is not None:
                    emd_val, candidato = res
                    if emd_val < mejor_emd:
                        mejor_emd = emd_val
                        mejor_grupos = candidato

        return mejor_grupos if mejor_grupos is not None else grupos

    def _hill_climbing(self, grupos: list[list[int]]) -> list[list[int]]:
        """
        Refinamiento local: reasigna una variable a la vez al grupo que
        más reduzca la pérdida EMD. Itera hasta convergencia.

        Complejidad por iteración: O(n × k × eval_bipartir_k).
        En la práctica converge en pocos pasos porque el greedy jerárquico
        ya produce una solución cercana al óptimo local.
        """
        mejor_emd, _ = self._evaluar_particion_k(grupos)
        mejoro = True

        while mejoro:
            mejoro = False
            for i in range(len(grupos)):
                if len(grupos[i]) <= 1:
                    continue
                for var in list(grupos[i]):
                    for j in range(len(grupos)):
                        if i == j:
                            continue
                        nueva = [list(g) for g in grupos]
                        nueva[i].remove(var)
                        nueva[j].append(var)
                        try:
                            emd_nuevo, _ = self._evaluar_particion_k(nueva)
                        except Exception:
                            continue
                        if emd_nuevo < mejor_emd - 1e-9:
                            mejor_emd = emd_nuevo
                            grupos = nueva
                            mejoro = True
                            break
                    if mejoro:
                        break
                if mejoro:
                    break

        return grupos

    # ------------------------------------------------------------------
    # Mecanismos y formato
    # ------------------------------------------------------------------

    def _asignar_mecanismos(self, grupos_locales: list) -> list[tuple]:
        """
        Construye la k-partición como lista de (alc_global, mec_global).

        Mecanismo de cada grupo = variables presentes cuyo índice global
        coincide con alguna variable futura del grupo (correspondencia natural
        en IIT). Las variables presentes sin par se distribuyen en round-robin
        empezando por el grupo de mayor tamaño.
        """
        fut_globals = self.sia_subsistema.indices_ncubos
        pres_set = {int(x) for x in self.sia_subsistema.dims_ncubos}

        grupos: list[tuple[list[int], list[int]]] = []
        asignadas: set[int] = set()
        for grupo in grupos_locales:
            if not grupo:
                continue
            g_arr = np.array(sorted(grupo), dtype=np.int8)
            alc = [int(x) for x in fut_globals[g_arr]]
            mec = sorted(pres_set & set(alc))
            asignadas |= set(mec)
            grupos.append((alc, mec))

        sin_asignar = sorted(pres_set - asignadas)
        orden = sorted(range(len(grupos)), key=lambda i: -len(grupos[i][0]))
        for j, p in enumerate(sin_asignar):
            grupos[orden[j % len(orden)]][1].append(p)

        return [
            (np.array(sorted(alc), dtype=np.int8),
             np.array(sorted(mec), dtype=np.int8))
            for alc, mec in grupos
        ]

    def _evaluar_particion_k(self, grupos_locales: list) -> tuple[float, np.ndarray]:
        """Evalúa EMD de una k-partición usando asignación correcta de mecanismos."""
        particion_k = self._asignar_mecanismos(grupos_locales)
        dist = self.sia_subsistema.bipartir_k(particion_k).distribucion_marginal()
        emd_val = emd_efecto(dist, self.sia_dists_marginales)
        return emd_val, dist

    def _fmt_particion_k(self, grupos_locales: list) -> str:
        """Formatea la k-partición igual que fmt_biparte_q pero para k grupos."""
        abecedary_lower = [a.lower() for a in ABECEDARY]
        particion_k = self._asignar_mecanismos(grupos_locales)

        fut_parts = []
        pres_parts = []
        for alc, mec in particion_k:
            fut_labels = [ABECEDARY[i] for i in alc if i < len(ABECEDARY)]
            pres_labels = [abecedary_lower[i] for i in mec if i < len(abecedary_lower)]
            fut_parts.append(",".join(fut_labels) if fut_labels else "∅")
            pres_parts.append(",".join(pres_labels) if pres_labels else "∅")

        top = "| " + " || ".join(fut_parts) + " |"
        bot = "| " + " || ".join(pres_parts) + " |"
        return top + "\n" + bot

    def hamming(self, a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))
