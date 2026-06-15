"""
Validación del MÉTODO EXACTO de KQNodes.

Responsabilidad única: el exacto por separabilidad debe coincidir, hasta la
precisión float32, con una fuerza bruta INDEPENDIENTE sobre todas las
k-particiones de los vértices (S(|V|, k)), y con la bipartición exacta de
``BruteForce`` para k=2.
"""

import pytest

from src.controllers.manager import Manager
from tests._kqnodes_helpers import (
    TOL, build_kq, vertices, kbruteforce, kq_loss, bf_loss,
)


class TestKQNodesExacto:
    """El método exacto debe coincidir con la fuerza bruta hasta float32."""

    @pytest.mark.parametrize("k", [2, 3])
    def test_coincide_con_kbruteforce_n3(self, tpm_n3, k):
        bits, alc, mec = 3, "111", "111"
        ref = kbruteforce(build_kq(tpm_n3, bits, alc, mec), k)
        assert abs(kq_loss(tpm_n3, bits, alc, mec, k) - ref) < TOL

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_coincide_con_kbruteforce_n4(self, tpm_n4, k):
        bits, alc, mec = 4, "1111", "1111"
        ref = kbruteforce(build_kq(tpm_n4, bits, alc, mec), k)
        assert abs(kq_loss(tpm_n4, bits, alc, mec, k) - ref) < TOL

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_coincide_con_kbruteforce_n5(self, tpm_n5, k):
        bits, alc, mec = 5, "11111", "11111"
        kq = build_kq(tpm_n5, bits, alc, mec)
        twoN = len(vertices(kq))
        if k > twoN:
            pytest.skip(f"k={k} > |V|={twoN}")
        ref = kbruteforce(kq, k)
        assert abs(kq_loss(tpm_n5, bits, alc, mec, k) - ref) < TOL

    # Validación obligatoria del spec: k=2 debe coincidir con BruteForce exacto.
    @pytest.mark.parametrize("bits,alc,mec", [
        (3, "111", "111"),
        (4, "1111", "1111"),
        (5, "11111", "11111"),
    ])
    def test_k2_coincide_con_bruteforce_biparticion(self, bits, alc, mec):
        tpm = Manager("1" + "0" * (bits - 1)).cargar_red()
        assert abs(kq_loss(tpm, bits, alc, mec, 2) - bf_loss(tpm, bits, alc, mec)) < TOL
