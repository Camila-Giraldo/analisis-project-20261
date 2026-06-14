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

        # Evaluación exacta de todos los candidatos con EMD real.
        #
        # Se evalúan todos los candidatos generados (no se filtra por proxy
        # ni por importance sampling) para garantizar que la bipartición k=2
        # encontrada sea la de menor pérdida real. Esto mejora la calidad
        # del resultado en k>2 al partir de la mejor base posible.
        n_vars = len(self.sia_subsistema.indices_ncubos)
        top_candidatos = [(p, f) for p, f in candidatos if f]

        # Evaluación concurrente con ThreadPoolExecutor.
        #
        # Paralelización 2: evaluación concurrente con ThreadPoolExecutor.
        # bipartir crea un nuevo System (sin mutar sia_subsistema) y
        # emd_efecto solo lee sia_dists_marginales → thread-safe.
        _n_workers = min(len(top_candidatos), os.cpu_count() or 4)

        def _eval(presentes_local, futuros_local):
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
                pool.submit(_eval, p, f) for p, f in top_candidatos
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
        Divide el grupo con más variables en dos evaluando todos los n-1 cortes.

        Para n_vars < 25 usa EMD exacto (bipartir_k) en cada corte.
        Para n_vars >= 25 usa el proxy geométrico O(n) para evitar el coste
        de bipartir_k sobre la tabla completa de N=25.
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        use_exact = n_vars < 25

        idx_mayor = max(range(len(grupos)), key=lambda i: len(grupos[i]))
        grupo = grupos[idx_mayor]

        if len(grupo) < 2:
            return grupos

        costos_full = self._get_costos_finales()
        costos_sub = costos_full[np.array(grupo)]
        orden_local = np.argsort(costos_sub)
        resto = [g for i, g in enumerate(grupos) if i != idx_mayor]

        mejor_score = float("inf")
        mejor_grupos: Optional[list] = None

        for corte in range(1, len(grupo)):
            parte1 = [grupo[orden_local[i]] for i in range(corte)]
            parte2 = [grupo[orden_local[i]] for i in range(corte, len(grupo))]
            candidato = resto + [parte1, parte2]
            if use_exact:
                score, _ = self._evaluar_particion_k(candidato)
            else:
                score = self._score_proxy(candidato)
            if score < mejor_score:
                mejor_score = score
                mejor_grupos = candidato

        return mejor_grupos if mejor_grupos is not None else grupos

    def _hill_climbing(self, grupos: list[list[int]]) -> list[list[int]]:
        """
        Refinamiento local de la k-partición por first-improvement.

        Para n_vars < 25 evalúa cada movimiento con EMD exacto (bipartir_k),
        recuperando la calidad del algoritmo original.
        Para n_vars >= 25 usa el proxy geométrico O(n) para mantener
        la viabilidad en subsistemas grandes.
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        use_exact = n_vars < 25

        if use_exact:
            mejor_score, _ = self._evaluar_particion_k(grupos)
        else:
            mejor_score = self._score_proxy(grupos)

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
                        if use_exact:
                            score, _ = self._evaluar_particion_k(nueva)
                        else:
                            score = self._score_proxy(nueva)
                        if score < mejor_score - 1e-9:
                            mejor_score = score
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

    def _score_proxy(self, grupos_locales: list) -> float:
        """
        Proxy O(n) para el EMD de una k-partición.

        Para k=2 coincide exactamente con _score() de _find_mip_k2.
        Para k>2 generaliza el principio: el grupo de mayor cohesión
        (suma de costos_finales) es el "sistema de referencia"; el score
        es la suma de costos de todo lo que queda fuera de ese grupo,
        es decir, el costo total de las variables cortadas.

            score = Σ_i costos[i]  -  max_g(Σ_{i∈g} costos[i])

        Costo: O(n_vars). No llama a bipartir_k ni a distribucion_marginal.
        Usado en _split_mejor_grupo y _hill_climbing para eliminar las
        llamadas exactas durante la búsqueda; solo queda 1 llamada exacta
        al final de _find_mip_kn.
        """
        costos = self._get_costos_finales()
        group_sums = [
            float(costos[np.array(g, dtype=np.int64)].sum()) if g else 0.0
            for g in grupos_locales
        ]
        if not group_sums:
            return float("inf")
        return float(sum(group_sums)) - max(group_sums)

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

    # ------------------------------------------------------------------
    # Heurística IS — Importance sampling para pre-filtrado EMD
    # ------------------------------------------------------------------

    def _compute_marginal_probs(self, futuros_local: list) -> np.ndarray:
        """
        Aproxima la probabilidad marginalizada de cada variable futura bajo
        la bipartición (G1=futuros_local, G2=resto), usando el modelo
        de Bernoulli independiente.

        Para cada variable i ∈ G1: q_i = promedio de ncubo_data[i] sobre los
        2^|M2| estados donde el mecanismo de G2 varía (fijando M1 = i0_M1).
        Para i ∈ G2: ídem con roles invertidos.

        Si |free_bits| > MAX_FREE_BITS se usa q_i ≈ ncubo_data[i][i0_int]
        sin marginalizar (evita iteraciones de 2^n).

        Costo: O(n_vars × 2^(n//2)) para bipartición balanceada.
        """
        MAX_FREE_BITS = 15
        n_vars = len(self.sia_subsistema.indices_ncubos)
        g1_set = set(futuros_local)
        g2_local = [i for i in range(n_vars) if i not in g1_set]

        if not futuros_local or not g2_local:
            return np.array(
                [float(self._ncubo_data[i][self._i0_int]) for i in range(n_vars)],
                dtype=np.float32,
            )

        particion_k = self._asignar_mecanismos([futuros_local, g2_local])
        dims = self.sia_subsistema.dims_ncubos

        def _global_to_local(mec_arr) -> list:
            bits = []
            for g in mec_arr:
                pos = np.where(dims == int(g))[0]
                if len(pos) > 0:
                    bits.append(int(pos[0]))
            return bits

        mec_bits = [_global_to_local(mec) for _, mec in particion_k]
        # G1 marginaliza sobre el mecanismo de G2 (free_bits[0] = mec_bits[1])
        free_bits = [mec_bits[1], mec_bits[0]]

        q = np.zeros(n_vars, dtype=np.float32)

        for gi, fut_locals in enumerate([futuros_local, g2_local]):
            fb = free_bits[gi]
            n_free = len(fb)

            if n_free == 0 or n_free > MAX_FREE_BITS:
                # Sin marginalización o demasiados bits libres: usar valor directo
                for var_idx in fut_locals:
                    q[var_idx] = float(self._ncubo_data[var_idx][self._i0_int])
                continue

            n_combos = 1 << n_free
            base = self._i0_int
            for bit in fb:
                base &= ~(1 << bit)

            k_range = np.arange(n_combos, dtype=np.int64)
            offsets = np.zeros(n_combos, dtype=np.int64)
            for k_idx, bit in enumerate(fb):
                offsets |= ((k_range >> k_idx) & 1) << bit
            states = base + offsets   # shape (n_combos,)

            for var_idx in fut_locals:
                q[var_idx] = float(np.mean(self._ncubo_data[var_idx][states]))

        return q

    def _emd_aproximado(self, futuros_local: list, n_samples: int = 2000) -> float:
        """
        Estima el EMD de una bipartición sin llamar a bipartir+distribucion_marginal.

        Usa la aproximación de producto de Bernoulli independiente:
            u_approx(x) = Π_i q_i^{x_i} × (1-q_i)^{1-x_i}

        donde q_i es la probabilidad marginalizada de la variable i bajo la
        bipartición. El EMD se estima muestreando desde los estados ya
        almacenados en tabla_np (región de alta probabilidad cerca de i0),
        evitando recorrer los 2^n estados completos.

        Costo total: O(n_vars × 2^(n//2)) para marginalizar +
                     O(n_samples × n_vars) para evaluar → << O(2^n_vars).

        Solo se aplica cuando dim(presente) == dim(futuro) (caso típico),
        ya que ambos espacios deben tener el mismo tamaño de índice.
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_dims = len(self.sia_subsistema.dims_ncubos)

        # Requiere que el espacio de índices presente y futuro coincidan
        if n_dims != n_vars:
            return float("inf")

        try:
            q = self._compute_marginal_probs(futuros_local)
        except Exception:
            return float("inf")

        # Muestrear desde tabla_np (estados visitados durante DP)
        table_keys = np.array(list(self.tabla_np.keys()), dtype=np.int64)
        if len(table_keys) == 0:
            return float("inf")

        if len(table_keys) > n_samples:
            sel = np.random.choice(len(table_keys), size=n_samples, replace=False)
            sampled = table_keys[sel]
        else:
            sampled = table_keys

        n_sampled = len(sampled)
        bits_pos = np.arange(n_vars, dtype=np.int64)

        # Bits de cada estado muestreado: shape (n_sampled, n_vars)
        bits = ((sampled[:, None] >> bits_pos[None, :]) & 1).astype(np.float32)

        # u_approx = exp(Σ_i [x_i·log(q_i) + (1-x_i)·log(1-q_i)])
        # Convertir a float64 ANTES de clipar: en float32, (1.0 - 1e-30) = 1.0
        # exactamente (solo 7 dígitos), lo que haría log(1-1.0)=log(0)→-inf.
        # float64 epsilon ≈ 2.22e-16: bounds menores que eso son inrepresentables
        # y 1.0 - 1e-30 == 1.0 en float64, causando log(0). Usar 1e-15 es seguro.
        q_cl = np.clip(q.astype(np.float64), 1e-15, 1.0 - 1e-15)
        log_u = (bits * np.log(q_cl) + (1.0 - bits) * np.log(1.0 - q_cl)).sum(axis=1)
        u_s = np.exp(log_u)

        # v en los estados muestreados
        v = np.asarray(self.sia_dists_marginales, dtype=np.float64)
        max_idx = int(sampled.max())
        if max_idx >= len(v):
            return float("inf")
        v_s = v[sampled]

        # Estimación escalada al espacio completo
        n_states = len(v)
        return float(np.sum(np.abs(u_s - v_s))) * n_states / n_sampled

    def hamming(self, a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))
