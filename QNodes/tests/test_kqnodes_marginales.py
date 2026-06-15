"""
API pública ``marginal_distributions`` y despachador ``find_k_mip``.

Responsabilidad única: la distribución marginal de la k-partición ganadora es
coherente (forma/tipo float32) y reproduce la pérdida reportada; el despachador
``find_k_mip`` devuelve la misma δ que la fuerza bruta.

Reemplaza a la antigua TestTensorProduct (``tensor_product`` fue eliminado en la
refactorización modular).
"""

import numpy as np

from src.controllers.strategies.KQNodes import KQNodes
from src.funcs.iit import emd_efecto
from tests._kqnodes_helpers import TOL, build_kq, kbruteforce


class TestMarginalDistributions:

    def test_forma_y_tipo(self, tpm_n3):
        """marginal_distributions devuelve un vector float32 con un valor por futuro."""
        kq = KQNodes(tpm_n3)
        kq.aplicar_estrategia("100", "111", "111", "111", 2, metodo="exacto")
        marg = kq.marginal_distributions(kq.ultima_kparticion)
        assert isinstance(marg, np.ndarray)
        assert marg.dtype == np.float32
        assert marg.size == kq.sia_subsistema.indices_ncubos.size

    def test_consistente_con_perdida_reportada(self, tpm_n4):
        """La EMD entre la marginal de la k-partición ganadora y la base del
        subsistema reproduce la pérdida reportada por la solución."""
        kq = KQNodes(tpm_n4)
        sol = kq.aplicar_estrategia("1000", "1111", "1111", "1111", 3, metodo="exacto")
        marg = kq.marginal_distributions(kq.ultima_kparticion)
        emd = float(emd_efecto(marg, kq.sia_dists_marginales))
        assert abs(emd - float(sol.perdida)) < TOL

    def test_find_k_mip_despachador_consistente(self, tpm_n4):
        """find_k_mip(k, 'exacto') devuelve la misma δ que la fuerza bruta."""
        bits, alc, mec = 4, "1111", "1111"
        kq = build_kq(tpm_n4, bits, alc, mec)
        delta, bloques = kq.find_k_mip(3, metodo="exacto")
        assert bloques is not None
        ref = kbruteforce(kq, 3)
        assert abs(delta - ref) < TOL
