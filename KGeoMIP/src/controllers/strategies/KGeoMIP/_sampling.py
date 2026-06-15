"""
Importance sampling y estimación aproximada de EMD.

Responsabilidad única: aproximar la probabilidad marginalizada de cada
variable futura bajo una bipartición dada, y estimar el EMD sin construir
la distribución completa de 2^n estados.
"""

from typing import List

import numpy as np


class _SamplingMixin:

    def _compute_marginal_probs(self, futuros_local: list) -> np.ndarray:
        """
        Aproxima la probabilidad marginalizada de cada variable futura bajo
        la bipartición (G1=futuros_local, G2=resto), usando el modelo
        de Bernoulli independiente.

        Si |free_bits| > MAX_FREE_BITS usa q_i ≈ ncubo_data[i][i0_int]
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
        free_bits = [mec_bits[1], mec_bits[0]]

        q = np.zeros(n_vars, dtype=np.float32)

        for gi, fut_locals in enumerate([futuros_local, g2_local]):
            fb = free_bits[gi]
            n_free = len(fb)

            if n_free == 0 or n_free > MAX_FREE_BITS:
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
            states = base + offsets

            for var_idx in fut_locals:
                q[var_idx] = float(np.mean(self._ncubo_data[var_idx][states]))

        return q

    def _emd_aproximado(self, futuros_local: list, n_samples: int = 2000) -> float:
        """
        Estima el EMD de una bipartición sin llamar a bipartir+distribucion_marginal.

        Usa la aproximación de producto de Bernoulli independiente.
        Solo se aplica cuando dim(presente) == dim(futuro).

        Costo: O(n_vars × 2^(n//2)) para marginalizar +
               O(n_samples × n_vars) para evaluar → << O(2^n_vars).
        """
        n_vars = len(self.sia_subsistema.indices_ncubos)
        n_dims = len(self.sia_subsistema.dims_ncubos)

        if n_dims != n_vars:
            return float("inf")

        try:
            q = self._compute_marginal_probs(futuros_local)
        except Exception:
            return float("inf")

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
        bits = ((sampled[:, None] >> bits_pos[None, :]) & 1).astype(np.float32)

        # float64 antes de clipar: float32 epsilon ≈ 1.19e-7 haría 1-1e-8 == 1.0
        q_cl = np.clip(q.astype(np.float64), 1e-15, 1.0 - 1e-15)
        log_u = (bits * np.log(q_cl) + (1.0 - bits) * np.log(1.0 - q_cl)).sum(axis=1)
        u_s = np.exp(log_u)

        v = np.asarray(self.sia_dists_marginales, dtype=np.float64)
        max_idx = int(sampled.max())
        if max_idx >= len(v):
            return float("inf")
        v_s = v[sampled]

        n_states = len(v)
        return float(np.sum(np.abs(u_s - v_s))) * n_states / n_sampled

    def hamming(self, a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))
